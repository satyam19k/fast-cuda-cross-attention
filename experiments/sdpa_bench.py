"""
PyTorch SDPA timing for the matched comparison (E1/E6 baseline).

Single-head, head_dim=D=768 -- the exact computation the CUDA kernels do --
forced onto a specific SDPA backend, CUDA-event timed. Returns fp32 output for
the correctness check against the reference.

Backends at head_dim=768 (per E0): 'math' (materialized cuBLAS bmm+softmax+bmm)
and 'efficient' (fused mem-efficient) run; 'flash'/'cudnn' are unavailable.
"""

import math
import statistics

import numpy as np


def _prep(a, n_rows, batch):
    a = np.ascontiguousarray(a, dtype=np.float32)
    if a.ndim == 2:
        a = a[None]
    if a.shape[0] == 1 and batch > 1:
        a = np.repeat(a, batch, axis=0)
    return a


def benchmark_sdpa_events(Q, K, V, batch_size, N_latent, N_input, D,
                          backend="math", precision="fp32",
                          warmup=20, iters=100):
    """Time torch SDPA on a forced backend. Returns (out_np[B,Nl,D], mean, std).

    Raises whatever SDPA raises if the backend is unsupported for this shape
    (e.g. flash at head_dim=768) -- the caller records it.
    """
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import torch.nn.functional as Fn

    dtype = torch.float32 if precision == "fp32" else torch.float16
    bk = {
        "math": SDPBackend.MATH,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "flash": SDPBackend.FLASH_ATTENTION,
    }[backend]

    # SDPA layout [B, heads=1, seq, head_dim].
    q = torch.from_numpy(_prep(Q, N_latent, batch_size)).to("cuda", dtype).unsqueeze(1)
    k = torch.from_numpy(_prep(K, N_input, batch_size)).to("cuda", dtype).unsqueeze(1)
    v = torch.from_numpy(_prep(V, N_input, batch_size)).to("cuda", dtype).unsqueeze(1)
    scale = 1.0 / math.sqrt(D)

    with sdpa_kernel(bk):
        out = Fn.scaled_dot_product_attention(q, k, v, scale=scale)  # may raise
        for _ in range(warmup):
            out = Fn.scaled_dot_product_attention(q, k, v, scale=scale)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            out = Fn.scaled_dot_product_attention(q, k, v, scale=scale)
            ends[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]

    out_np = out.squeeze(1).float().cpu().numpy()  # [B, Nl, D]
    del q, k, v, out
    torch.cuda.empty_cache()
    return (out_np, statistics.mean(times),
            statistics.pstdev(times) if len(times) > 1 else 0.0)
