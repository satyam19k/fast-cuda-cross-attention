"""
E0 -- SDPA backend identification.

Q: which backend does PyTorch's scaled_dot_product_attention actually use at
   head_dim = 768, and which backends even support it?

Why it matters: FlashAttention's forward kernel caps head_dim at 256, so at
D=768 it is ineligible -- meaning the "real" PyTorch baseline our kernels race
is mem-efficient or the math fallback, not flash. This script forces each
backend in turn and records run/error so we know which one we are comparing to.

Anchor A = (N_latent=256, N_input=4096, B=1), single head, head_dim=768.
We probe fp32 (math/efficient eligible) and fp16 (flash/cudnn eligible).

Runs on the GPU box. Off-GPU it prints a clear message and exits 0 so the
script is still import/lint-clean locally.

Usage:
    python experiments/e0_sdpa_backend.py
    python experiments/e0_sdpa_backend.py --n-latent 256 --n-input 4096 --csv results/e0.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csvlog  # noqa: E402

# Anchor A
DEFAULT_NL = 256
DEFAULT_NI = 4096
D = 768
WARMUP = 20
ITERS = 100


def _backends():
    """Return [(label, SDPBackend, precisions_to_try)] for available backends."""
    from torch.nn.attention import SDPBackend
    out = [
        ("sdpa_flash", SDPBackend.FLASH_ATTENTION, ["fp16"]),
        ("sdpa_efficient", SDPBackend.EFFICIENT_ATTENTION, ["fp32", "fp16"]),
        ("sdpa_math", SDPBackend.MATH, ["fp32", "fp16"]),
    ]
    # CUDNN_ATTENTION exists in newer torch; include if present.
    if hasattr(SDPBackend, "CUDNN_ATTENTION"):
        out.append(("sdpa_cudnn", SDPBackend.CUDNN_ATTENTION, ["fp16"]))
    return out


def _time_sdpa(q, k, v, scale, backend):
    """Force `backend`, run warmup+timed loop, return (mean_ms, std_ms).

    Raises whatever SDPA raises if the backend is unsupported for this shape.
    """
    import torch
    from torch.nn.attention import sdpa_kernel
    import torch.nn.functional as F

    with sdpa_kernel(backend):
        # One call outside the timing loop will raise immediately if the
        # backend rejects this shape/dtype -- that is the signal we want.
        _ = F.scaled_dot_product_attention(q, k, v, scale=scale)
        for _ in range(WARMUP):
            _ = F.scaled_dot_product_attention(q, k, v, scale=scale)
        torch.cuda.synchronize()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        for i in range(ITERS):
            starts[i].record()
            _ = F.scaled_dot_product_attention(q, k, v, scale=scale)
            ends[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]

    import statistics
    return statistics.mean(times), (statistics.pstdev(times) if len(times) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser(description="E0: identify SDPA backend at head_dim=768")
    ap.add_argument("--n-latent", type=int, default=DEFAULT_NL)
    ap.add_argument("--n-input", type=int, default=DEFAULT_NI)
    ap.add_argument("--csv", default="results/e0_sdpa_backend.csv")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("PyTorch not installed; E0 requires torch. Skipping.")
        return

    if not torch.cuda.is_available():
        print("=" * 72)
        print("E0 -- SDPA backend identification")
        print("No CUDA device available (this is expected off the GPU box).")
        print("SDPA backend selection (flash/efficient/math) is CUDA-specific,")
        print("so run this on the 4070/L40/H200. Script is otherwise ready.")
        print("=" * 72)
        return

    import math
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gpu_specs import detect_gpu_name

    gpu = detect_gpu_name() or torch.cuda.get_device_name(0)
    Nl, Ni = args.n_latent, args.n_input
    scale = 1.0 / math.sqrt(D)

    print("=" * 72)
    print("E0 -- SDPA backend identification")
    print(f"GPU: {torch.cuda.get_device_name(0)}  (key={gpu})")
    print(f"Anchor: B=1, N_latent={Nl}, N_input={Ni}, head_dim=D={D}, heads=1")
    print(f"torch {torch.__version__}")
    print("=" * 72)

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    fh, w = csvlog.writer(args.csv)

    results = []
    for label, backend, precisions in _backends():
        for prec in precisions:
            dtype = torch.float32 if prec == "fp32" else torch.float16
            # SDPA layout: [B, num_heads, seq, head_dim]; single head.
            q = torch.randn(1, 1, Nl, D, device="cuda", dtype=dtype)
            k = torch.randn(1, 1, Ni, D, device="cuda", dtype=dtype)
            v = torch.randn(1, 1, Ni, D, device="cuda", dtype=dtype)
            try:
                mean_ms, std_ms = _time_sdpa(q, k, v, scale, backend)
                status, note = "ok", ""
                print(f"  {label:16s} {prec:4s}: OK   "
                      f"{mean_ms:7.3f} +/- {std_ms:.3f} ms")
            except Exception as e:
                mean_ms = std_ms = ""
                msg = str(e).strip().splitlines()[0][:160]
                status, note = "unsupported", msg
                print(f"  {label:16s} {prec:4s}: ERR  {msg}")
            results.append((label, prec, status))
            w.writerow({
                "experiment": "e0", "gpu": gpu, "impl": label,
                "precision": prec, "B": 1, "N_latent": Nl, "N_input": Ni,
                "D": D, "latency_ms_mean": mean_ms, "latency_ms_std": std_ms,
                "status": status, "note": note,
            })
            del q, k, v
            torch.cuda.empty_cache()
    fh.close()

    print("-" * 72)
    ok = [f"{l}/{p}" for (l, p, s) in results if s == "ok"]
    bad = [f"{l}/{p}" for (l, p, s) in results if s != "ok"]
    print(f"Runs:   {', '.join(ok) if ok else '(none)'}")
    print(f"Errors: {', '.join(bad) if bad else '(none)'}")
    print(f"Takeaway: at head_dim={D}, the usable SDPA backend(s) define the "
          f"true PyTorch baseline for E1/E6.")
    print(f"CSV -> {args.csv}")


if __name__ == "__main__":
    main()
