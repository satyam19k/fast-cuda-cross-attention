"""
Perceiver Nl-sweep benchmark (the occupancy experiment, E2).

Fixes N_input at a small, L2-resident, warp-valid size (default 3136 = 56^2)
and sweeps N_latent. The launch is gridSize=(ceil(Nl/4), B), i.e. one warp per
latent / 4 latents per block, so the block count -- and thus how many SMs are
active -- scales with Nl. At small Nl the kernel cannot fill the GPU.

warp runs across the WHOLE sweep here because its shared-memory limit is on Ni
(fixed small), not Nl.

Emits experiment='perceiver_nl' rows for plot_perceiver_nl.py, and prints the
block count and active-SM estimate (the occupancy driver).

    python experiments/bench_perceiver_nl.py
    python experiments/bench_perceiver_nl.py --n-input 3136 \\
        --n-latent 16 32 64 128 256 512 1024 --impls warp tiled vectorized
"""

import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = 768
DEFAULT_NL = [16, 32, 64, 128, 256, 512, 1024]
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
    ap = argparse.ArgumentParser(description="Perceiver Nl-sweep (occupancy, CUDA events)")
    ap.add_argument("--n-input", type=int, default=3136)
    ap.add_argument("--n-latent", type=int, nargs="+", default=DEFAULT_NL)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--impls", nargs="+", default=DEFAULT_IMPLS)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-2)
    ap.add_argument("--csv", default="results/perceiver_nl.csv")
    args = ap.parse_args()

    os.chdir(ROOT)
    sys.path.insert(0, HERE)
    sys.path.insert(0, ROOT)

    import numpy as np
    import roofline
    import csvlog
    from gpu_specs import detect_gpu_name, raw_gpu_name, GPU_SPECS
    try:
        import cuda_wrappers  # noqa: F401
        from cuda_wrappers import benchmark_kernel_events, DesignLimited
    except Exception as e:
        print(f"Could not load CUDA kernels ({e}).")
        print("Build them first on the GPU box:  make")
        return

    gpu = detect_gpu_name() or raw_gpu_name() or "unknown"
    sm_count = GPU_SPECS.get(gpu, {}).get("sm_count")
    ni, B = args.n_input, args.batch
    csv_path = os.path.join(ROOT, args.csv)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fh, w = csvlog.writer(csv_path)

    sm_str = f"{sm_count} SMs" if sm_count else "SMs=?"
    print("=" * 92)
    print(f"Perceiver Nl-sweep | GPU={gpu} ({sm_str}) | N_input={ni} | B={B} | "
          f"warmup={args.warmup} iters={args.iters}")
    print("=" * 92)
    hdr = (f"{'impl':<12}{'Nl':>6}{'blocks':>8}{'actSM':>7}{'lat(ms)':>11}"
           f"{'GFLOP/s':>11}{'%comp':>7}{'%bw':>7}{'maxerr':>10}{'status':>14}")
    print(hdr)
    print("-" * len(hdr))

    for nl in args.n_latent:
        blocks = math.ceil(nl / 4) * B
        act_sm = min(blocks, sm_count) if sm_count else blocks
        loaded = load_ref(nl, ni)
        if loaded is None:
            print(f"{'(all)':<12}{nl:>6}   missing reference -- run gen_perceiver_refs.py")
            continue
        Q, K, V, ref = loaded

        for impl in args.impls:
            row = {
                "experiment": "perceiver_nl", "gpu": gpu, "impl": impl,
                "precision": "fp32", "B": B, "N_latent": nl, "N_input": ni, "D": D,
                "note": f"blocks={blocks};act_sm={act_sm}",
            }
            try:
                out, mean_ms, std_ms = benchmark_kernel_events(
                    impl, Q, K, V, B, nl, ni, D, args.warmup, args.iters)
                o0 = out[0] if out.ndim == 3 else out
                r0 = ref.reshape(nl, D)
                max_abs = float(np.max(np.abs(o0 - r0)))
                rel = max_abs / (float(np.max(np.abs(r0))) + 1e-6)
                rl = roofline.roofline_row(B, nl, ni, D, mean_ms / 1e3, gpu,
                                           "fp32", materialized=False)
                status = "ok" if max_abs <= args.tol else "ACCURACY?"
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
                print(f"{impl:<12}{nl:>6}{blocks:>8}{act_sm:>7}{mean_ms:>11.4f}"
                      f"{rl['achieved_gflops']:>11.1f}"
                      f"{(rl.get('pct_peak_compute') or 0):>7.2f}"
                      f"{(rl.get('pct_peak_bw') or 0):>7.2f}{max_abs:>10.1e}{status:>14}")
            except DesignLimited as e:
                row.update({"status": "design-limited", "note": row["note"] + ";" + str(e)[:80]})
                print(f"{impl:<12}{nl:>6}{blocks:>8}{act_sm:>7}{'--':>11}"
                      f"{'':>11}{'':>7}{'':>7}{'':>10}{'design-limited':>14}")
            except Exception as e:
                row.update({"status": "error", "note": str(e).splitlines()[0][:120]})
                print(f"{impl:<12}{nl:>6}   ERROR: {str(e).splitlines()[0][:60]}")
            w.writerow(row)

    fh.close()
    print("-" * len(hdr))
    print(f"CSV -> {csv_path}")
    print("Next: python experiments/plot_perceiver_nl.py")


if __name__ == "__main__":
    main()
