"""
Roofline math for cross-attention: softmax(Q @ K^T / sqrt(d)) @ V.

Shapes: Q [B, Nl, D], K [B, Ni, D], V [B, Ni, D], out [B, Nl, D].

FLOPs (forward, the two matmuls dominate; softmax is O(B*Nl*Ni) and ignored
to first order, matching the doc's 4*B*Nl*Ni*D convention):
    QK^T : 2 * B * Nl * Ni * D
    AV   : 2 * B * Nl * Ni * D
    total: 4 * B * Nl * Ni * D

Byte models (HBM traffic, one pass):
    fused (FlashAttention-style, scores never leave SMEM/registers):
        read  Q (B*Nl*D) + K (B*Ni*D) + V (B*Ni*D)
        write out (B*Nl*D)
        = (2*Nl*D + 2*Ni*D) * B * bytes_per_elem
    materialized (S = Q@K^T written to HBM, read back for softmax@V):
        fused traffic + write S (B*Nl*Ni) + read S (B*Nl*Ni)
        = fused_bytes + 2 * B * Nl * Ni * bytes_per_elem
"""

BYTES_PER_ELEM = {"fp32": 4, "fp16": 2, "fp16_tc": 2, "fp8": 1}


def flops(B, Nl, Ni, D):
    """Forward FLOPs for the two matmuls (4*B*Nl*Ni*D)."""
    return 4.0 * B * Nl * Ni * D


def bytes_moved(B, Nl, Ni, D, precision="fp32", materialized=False):
    """HBM bytes moved for the given byte model."""
    be = BYTES_PER_ELEM[precision]
    fused = (2.0 * Nl * D + 2.0 * Ni * D) * B * be
    if not materialized:
        return fused
    return fused + 2.0 * B * Nl * Ni * be


def arithmetic_intensity(B, Nl, Ni, D, precision="fp32", materialized=False):
    """FLOP per byte."""
    return flops(B, Nl, Ni, D) / bytes_moved(B, Nl, Ni, D, precision, materialized)


def achieved_flops(B, Nl, Ni, D, latency_s):
    """Achieved FLOP/s from measured latency (seconds)."""
    return flops(B, Nl, Ni, D) / latency_s


def achieved_bw(B, Nl, Ni, D, latency_s, precision="fp32", materialized=False):
    """Achieved HBM bytes/s from measured latency (seconds)."""
    return bytes_moved(B, Nl, Ni, D, precision, materialized) / latency_s


def roofline_row(B, Nl, Ni, D, latency_s, gpu_key, precision="fp32",
                 materialized=False):
    """Full roofline record for one measured point.

    Returns a dict of derived metrics (% peak compute, % peak BW, AI, ridge,
    bound-type). gpu_key must be a key in gpu_specs.GPU_SPECS; pass None to
    skip the %-peak fields (e.g. on an unknown card).
    """
    from gpu_specs import GPU_SPECS, PRECISION_PEAK_KEY, ridge_point

    f = flops(B, Nl, Ni, D)
    byt = bytes_moved(B, Nl, Ni, D, precision, materialized)
    ai = f / byt
    a_flops = f / latency_s
    a_bw = byt / latency_s

    row = {
        "flops": f,
        "bytes": byt,
        "arithmetic_intensity": ai,
        "achieved_gflops": a_flops / 1e9,
        "achieved_gbs": a_bw / 1e9,
    }
    if gpu_key is not None and gpu_key in GPU_SPECS:
        spec = GPU_SPECS[gpu_key]
        peak_flops = spec[PRECISION_PEAK_KEY[precision]]
        peak_bw = spec["peak_bw_bytes"]
        rp = ridge_point(gpu_key, precision)
        row["pct_peak_compute"] = (
            100.0 * a_flops / peak_flops if peak_flops else None)
        row["pct_peak_bw"] = 100.0 * a_bw / peak_bw
        row["ridge_point"] = rp
        row["bound"] = (
            "compute" if (rp is not None and ai >= rp) else "memory")
    return row


if __name__ == "__main__":
    # Sanity: anchor A = (Nl=256, Ni=4096, B=1) at a hypothetical 1 ms.
    r = roofline_row(1, 256, 4096, 768, 1e-3, "H200", "fp32", materialized=False)
    for k, v in r.items():
        print(f"  {k}: {v}")
