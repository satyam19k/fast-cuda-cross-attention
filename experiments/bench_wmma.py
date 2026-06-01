"""
E7 -- fp16 WMMA (tensor core) vs fp32 vectorized.

Two endpoints of the three-point decomposition (scalar fp32 -> cuBLAS-fp16 ->
WMMA-fused): the fp32 float4 kernel and the fused tensor-core kernel. The
cuBLAS-fp16 middle point (which isolates the tensor-core effect from fusion)
needs a cuBLAS path and is not included here.

Note on occupancy: the WMMA kernel uses 16 latents per block (grid = Nl/16),
vs Nl/4 for the others -- so it needs LARGER Nl to fill the GPU. Default Nl=256.

Requires Nl % 16 == 0 and Ni % 16 == 0 -> use square resolutions divisible by
16 (28^2, 56^2, 112^2, 224^2, ...). Verifies WMMA output against the fp32
reference with a loose fp16 tolerance.

    python experiments/bench_wmma.py
    python experiments/bench_wmma.py --n-latent 256 --n-input 784 3136 12544 50176
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = 768


def main():
    ap = argparse.ArgumentParser(description="E7: fp16 WMMA vs fp32 vectorized")
    ap.add_argument("--n-latent", type=int, default=256)
    ap.add_argument("--n-input", type=int, nargs="+",
                    default=[784, 3136, 12544, 50176])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--rtol", type=float, default=5e-2)
    ap.add_argument("--csv", default="results/e7_wmma.csv")
    args = ap.parse_args()

    os.chdir(ROOT)
    sys.path.insert(0, HERE)
    sys.path.insert(0, ROOT)
    import numpy as np
    import roofline
    import csvlog
    from gpu_specs import detect_gpu_name, raw_gpu_name
    try:
        from cuda_wrappers import (benchmark_kernel_events,
                                   benchmark_wmma_events, DesignLimited)
    except Exception as e:
        print(f"Could not load CUDA kernels ({e}). Build with kernel_wmma.cu.")
        return

    gpu = detect_gpu_name() or raw_gpu_name() or "unknown"
    nl = args.n_latent
    csv_path = os.path.join(ROOT, args.csv)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fh, w = csvlog.writer(csv_path)

    print("=" * 92)
    print(f"E7 WMMA fp16 vs vectorized fp32 | GPU={gpu} | N_latent={nl}")
    print("=" * 92)
    hdr = (f"{'impl':<16}{'prec':>5}{'Ni':>7}{'lat(ms)':>11}{'GFLOP/s':>11}"
           f"{'%peak':>8}{'speedup':>9}{'relerr':>10}{'status':>12}")
    print(hdr); print("-" * len(hdr))

    for ni in args.n_input:
        sub = os.path.join("data", "perceiver", f"Nl{nl}_Ni{ni}")
        if not os.path.exists(os.path.join(sub, "Q_matrix.npy")):
            print(f"{'(missing ref)':<16}{'':>5}{ni:>7}  -> gen_perceiver_refs.py "
                  f"--n-latent {nl} --n-input {ni}")
            continue
        Q = np.load(os.path.join(sub, "Q_matrix.npy"))
        K = np.load(os.path.join(sub, "K_matrix.npy"))
        V = np.load(os.path.join(sub, "V_matrix.npy"))
        ref = np.load(os.path.join(sub, "output_reference.npy")).reshape(nl, D)

        # fp32 vectorized baseline
        vec_ms = None
        try:
            out, vec_ms, _ = benchmark_kernel_events(
                "vectorized", Q, K, V, 1, nl, ni, D, args.warmup, args.iters)
            rl = roofline.roofline_row(1, nl, ni, D, vec_ms / 1e3, gpu, "fp32")
            print(f"{'vectorized':<16}{'fp32':>5}{ni:>7}{vec_ms:>11.4f}"
                  f"{rl['achieved_gflops']:>11.1f}"
                  f"{(rl.get('pct_peak_compute') or 0):>8.2f}{'1.0×':>9}"
                  f"{'--':>10}{'ok':>12}")
            w.writerow(dict(experiment="e7_wmma", gpu=gpu, impl="vectorized",
                precision="fp32", B=1, N_latent=nl, N_input=ni, D=D,
                latency_ms_mean=f"{vec_ms:.4f}",
                achieved_gflops=f"{rl['achieved_gflops']:.1f}",
                pct_peak_compute=f"{rl.get('pct_peak_compute') or 0:.2f}",
                status="ok"))
        except Exception as e:
            print(f"{'vectorized':<16}{'fp32':>5}{ni:>7}  ERROR {str(e)[:40]}")

        # fp16 WMMA
        try:
            out_w, w_ms, _ = benchmark_wmma_events(
                Q, K, V, 1, nl, ni, D, args.warmup, args.iters)
            o0 = out_w[0] if out_w.ndim == 3 else out_w
            max_abs = float(np.max(np.abs(o0 - ref)))
            relerr = max_abs / (float(np.max(np.abs(ref))) + 1e-6)
            rl = roofline.roofline_row(1, nl, ni, D, w_ms / 1e3, gpu, "fp16")
            speed = (vec_ms / w_ms) if vec_ms else 0
            status = "ok" if relerr <= args.rtol else "ACCURACY?"
            print(f"{'wmma_fused':<16}{'fp16':>5}{ni:>7}{w_ms:>11.4f}"
                  f"{rl['achieved_gflops']:>11.1f}"
                  f"{(rl.get('pct_peak_compute') or 0):>8.2f}{speed:>8.2f}×"
                  f"{relerr:>10.1e}{status:>12}")
            w.writerow(dict(experiment="e7_wmma", gpu=gpu, impl="wmma_fused",
                precision="fp16", B=1, N_latent=nl, N_input=ni, D=D,
                latency_ms_mean=f"{w_ms:.4f}",
                achieved_gflops=f"{rl['achieved_gflops']:.1f}",
                pct_peak_compute=f"{rl.get('pct_peak_compute') or 0:.2f}",
                rel_err=f"{relerr:.2e}", max_abs_err=f"{max_abs:.2e}",
                status=status))
        except DesignLimited as e:
            print(f"{'wmma_fused':<16}{'fp16':>5}{ni:>7}  design-limited: {str(e)[:40]}")
        except Exception as e:
            print(f"{'wmma_fused':<16}{'fp16':>5}{ni:>7}  ERROR {str(e).splitlines()[0][:50]}")

    fh.close()
    print("-" * len(hdr))
    print(f"CSV -> {csv_path}")


if __name__ == "__main__":
    main()
