"""
Perceiver Ni-sweep benchmark (the first experiment to run on the GPU box).

Fixes N_latent at the Perceiver bottleneck and sweeps N_input over image
resolutions 784=28^2 .. 50176=224^2, timing warp / tiled / vectorized with
CUDA events (kernel-only, data resident -- not the legacy time.time() path).
warp is the baseline; it is expected to hit its shared-memory wall and report
'design-limited' at large N_input while tiled/vectorized scale -- that
truncation is the result.

Emits one CSV row per (impl, Ni) with latency, roofline metrics, and error vs
the fp32 reference, for plot_perceiver.py.

Run from anywhere; the script chdir's to the repo root so ./kernels.so and
data/ resolve.

    python experiments/bench_perceiver_ni.py
    python experiments/bench_perceiver_ni.py --n-latent 64 \\
        --n-input 784 3136 12544 50176 --impls warp tiled vectorized
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = 768
DEFAULT_NI = [784, 3136, 12544, 50176]
DEFAULT_IMPLS = ["warp", "tiled", "vectorized"]


def load_ref(nl, ni):
    import numpy as np
    sub = os.path.join("data", "perceiver", f"Nl{nl}_Ni{ni}")
    if not os.path.exists(os.path.join(sub, "Q_matrix.npy")):
        return None
    Q = np.load(os.path.join(sub, "Q_matrix.npy"))
    K = np.load(os.path.join(sub, "K_matrix.npy"))
    V = np.load(os.path.join(sub, "V_matrix.npy"))
    ref = np.load(os.path.join(sub, "output_reference.npy"))
    return Q, K, V, ref


def main():
    ap = argparse.ArgumentParser(description="Perceiver Ni-sweep benchmark (CUDA events)")
    ap.add_argument("--n-latent", type=int, default=64)
    ap.add_argument("--n-input", type=int, nargs="+", default=DEFAULT_NI)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--impls", nargs="+", default=DEFAULT_IMPLS)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-2)
    ap.add_argument("--csv", default="results/perceiver_ni.csv")
    args = ap.parse_args()

    os.chdir(ROOT)               # so ./kernels.so and data/ resolve
    sys.path.insert(0, HERE)     # roofline / gpu_specs / csvlog
    sys.path.insert(0, ROOT)     # cuda_wrappers

    import numpy as np
    import roofline
    import csvlog
    from gpu_specs import detect_gpu_name, raw_gpu_name, GPU_SPECS
    try:
        import cuda_wrappers
        from cuda_wrappers import (benchmark_kernel_events,
                                   benchmark_wmma_events,
                                   benchmark_splitk_events, DesignLimited)
    except Exception as e:
        print(f"Could not load CUDA kernels ({e}).")
        print("Build them first on the GPU box:  make")
        return

    # torch-free GPU detection (nvidia-smi). detect_gpu_name() returns a
    # GPU_SPECS key for %-peak math; if the card is unknown we still record its
    # raw name so the CSV is labeled.
    gpu = detect_gpu_name() or raw_gpu_name() or "unknown"
    sm_count = GPU_SPECS.get(gpu, {}).get("sm_count", 142)

    nl, B = args.n_latent, args.batch
    csv_path = os.path.join(ROOT, args.csv)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fh, w = csvlog.writer(csv_path)

    print("=" * 88)
    print(f"Perceiver Ni-sweep | GPU={gpu} | N_latent={nl} | B={B} | "
          f"warmup={args.warmup} iters={args.iters}")
    print("=" * 88)
    hdr = f"{'impl':<12}{'Ni':>7}{'res':>7}{'lat(ms)':>11}{'GFLOP/s':>11}" \
          f"{'GB/s':>9}{'%comp':>7}{'%bw':>7}{'bound':>8}{'maxerr':>10}{'status':>14}"
    print(hdr)
    print("-" * len(hdr))

    # warp baseline latency per Ni, for the speedup plot
    warp_lat = {}

    for ni in args.n_input:
        loaded = load_ref(nl, ni)
        if loaded is None:
            print(f"{'(all)':<12}{ni:>7}   missing reference -- run gen_perceiver_refs.py")
            continue
        Q, K, V, ref = loaded
        res = int(round(ni ** 0.5))
        res_s = f"{res}^2" if res * res == ni else ""

        for impl in args.impls:
            prec = "fp16" if impl == "wmma" else "fp32"
            row = {
                "experiment": "perceiver_ni", "gpu": gpu, "impl": impl,
                "precision": prec, "B": B, "N_latent": nl, "N_input": ni, "D": D,
            }
            try:
                if impl == "wmma":
                    out, mean_ms, std_ms = benchmark_wmma_events(
                        Q, K, V, B, nl, ni, D, args.warmup, args.iters)
                elif impl == "splitk":
                    out, mean_ms, std_ms, nsplits = benchmark_splitk_events(
                        Q, K, V, B, nl, ni, D, args.warmup, args.iters,
                        sm_count=sm_count)
                    row["note"] = f"splits={nsplits}"
                else:
                    out, mean_ms, std_ms = benchmark_kernel_events(
                        impl, Q, K, V, B, nl, ni, D, args.warmup, args.iters)
                # correctness vs fp32 reference (compare batch 0)
                o0 = out[0] if out.ndim == 3 else out
                r0 = ref.reshape(nl, D)
                max_abs = float(np.max(np.abs(o0 - r0)))
                rel = max_abs / (float(np.max(np.abs(r0))) + 1e-6)
                rl = roofline.roofline_row(B, nl, ni, D, mean_ms / 1e3, gpu,
                                           prec, materialized=False)
                status = "ok" if max_abs <= args.tol else "ACCURACY?"
                if impl == "warp":
                    warp_lat[ni] = mean_ms
                row.update({
                    "latency_ms_mean": f"{mean_ms:.4f}", "latency_ms_std": f"{std_ms:.4f}",
                    "achieved_gflops": f"{rl['achieved_gflops']:.1f}",
                    "achieved_gbs": f"{rl['achieved_gbs']:.1f}",
                    "pct_peak_compute": f"{rl.get('pct_peak_compute') or 0:.2f}",
                    "pct_peak_bw": f"{rl.get('pct_peak_bw') or 0:.2f}",
                    "arithmetic_intensity": f"{rl['arithmetic_intensity']:.1f}",
                    "ridge_point": f"{rl.get('ridge_point') or 0:.1f}",
                    "bound": rl.get("bound", ""),
                    "max_abs_err": f"{max_abs:.2e}", "rel_err": f"{rel:.2e}",
                    "status": status,
                })
                print(f"{impl:<12}{ni:>7}{res_s:>7}{mean_ms:>11.4f}"
                      f"{rl['achieved_gflops']:>11.1f}{rl['achieved_gbs']:>9.1f}"
                      f"{(rl.get('pct_peak_compute') or 0):>7.2f}"
                      f"{(rl.get('pct_peak_bw') or 0):>7.2f}{rl.get('bound',''):>8}"
                      f"{max_abs:>10.1e}{status:>14}")
            except DesignLimited as e:
                row.update({"status": "design-limited", "note": str(e)[:120]})
                print(f"{impl:<12}{ni:>7}{res_s:>7}{'--':>11}{'':>11}{'':>9}"
                      f"{'':>7}{'':>7}{'':>8}{'':>10}{'design-limited':>14}")
            except Exception as e:
                row.update({"status": "error", "note": str(e).splitlines()[0][:120]})
                print(f"{impl:<12}{ni:>7}{res_s:>7}   ERROR: {str(e).splitlines()[0][:60]}")
            w.writerow(row)

    fh.close()
    print("-" * len(hdr))
    print(f"CSV -> {csv_path}")
    print("Next: python experiments/plot_perceiver.py")


if __name__ == "__main__":
    main()
