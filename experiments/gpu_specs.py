"""
Per-GPU hardware ceilings for roofline analysis.

Numbers are vendor spec-sheet peaks (verify against the actual card before
quoting in the writeup -- boost clocks and SKU variants shift these by a few %).
All FLOP/s are in FLOP/s (not TFLOP/s); all bandwidths are in bytes/s.

ridge_point = peak_flops / peak_bw  (FLOP/byte). An operation is compute-bound
when its arithmetic intensity (AI) exceeds the ridge point, memory-bound below.
"""

TFLOP = 1.0e12
GBs = 1.0e9  # GB/s -> bytes/s

GPU_SPECS = {
    # RTX 4070 (Ada, AD104). 46 SM. ~504 GB/s. ~100 KB smem/SM.
    "4070": {
        "arch": "Ada",
        "sm_count": 46,
        "smem_per_sm_kb": 100,
        "peak_fp32_flops": 29.1 * TFLOP,     # non-tensor FP32
        "peak_fp16_tc_flops": 116.0 * TFLOP, # FP16 tensor core (no sparsity)
        "peak_fp8_tc_flops": None,           # no FP8 on Ada consumer
        "peak_bw_bytes": 504.0 * GBs,
    },
    # L40 (Ada, AD102). 142 SM. ~864 GB/s. ~100 KB smem/SM.
    "L40": {
        "arch": "Ada",
        "sm_count": 142,
        "smem_per_sm_kb": 100,
        "peak_fp32_flops": 90.5 * TFLOP,
        "peak_fp16_tc_flops": 181.0 * TFLOP,
        "peak_fp8_tc_flops": 362.0 * TFLOP,
        "peak_bw_bytes": 864.0 * GBs,
    },
    # L40S (Ada, AD102). 142 SM. ~864 GB/s. ~100 KB smem/SM. L2 = 96 MB.
    # Higher clocks than L40 -> higher tensor throughput.
    "L40S": {
        "arch": "Ada",
        "sm_count": 142,
        "smem_per_sm_kb": 100,
        "peak_fp32_flops": 91.6 * TFLOP,
        "peak_fp16_tc_flops": 362.0 * TFLOP,   # FP16 tensor, no sparsity
        "peak_fp8_tc_flops": 733.0 * TFLOP,
        "peak_bw_bytes": 864.0 * GBs,
    },
    # H200 (Hopper, GH100). 132 SM. ~4.8 TB/s. ~228 KB smem/SM.
    "H200": {
        "arch": "Hopper",
        "sm_count": 132,
        "smem_per_sm_kb": 228,
        "peak_fp32_flops": 67.0 * TFLOP,
        "peak_fp16_tc_flops": 989.0 * TFLOP,  # FP16/BF16 tensor core, no sparsity
        "peak_fp8_tc_flops": 1979.0 * TFLOP,  # FP8 tensor core, no sparsity
        "peak_bw_bytes": 4800.0 * GBs,
    },
}

# Precision -> which peak_*_flops key to use for %-of-compute-peak.
PRECISION_PEAK_KEY = {
    "fp32": "peak_fp32_flops",
    "fp16": "peak_fp16_tc_flops",
    "fp16_tc": "peak_fp16_tc_flops",
    "fp8": "peak_fp8_tc_flops",
}


def raw_gpu_name():
    """Raw device name via nvidia-smi (no torch); None if unavailable."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            first = out.stdout.strip().splitlines()
            if first:
                return first[0].strip()
    except Exception:
        pass
    # fall back to torch if it happens to be present
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def detect_gpu_name():
    """GPU_SPECS key for the current device, or None if unknown/off-GPU.

    Uses nvidia-smi first (no torch dependency). L40S is checked before L40
    since 'L40S' contains 'L40'.
    """
    name = raw_gpu_name()
    if not name:
        return None
    upper = name.upper()
    if "4070" in upper:
        return "4070"
    if "L40S" in upper:
        return "L40S"
    if "L40" in upper:
        return "L40"
    if "H200" in upper:
        return "H200"
    return None  # unknown card -> caller falls back to raw name


def ridge_point(gpu_key, precision="fp32"):
    spec = GPU_SPECS[gpu_key]
    peak = spec[PRECISION_PEAK_KEY[precision]]
    if peak is None:
        return None
    return peak / spec["peak_bw_bytes"]


if __name__ == "__main__":
    for k, s in GPU_SPECS.items():
        for p in ("fp32", "fp16"):
            rp = ridge_point(k, p)
            rp_s = f"{rp:.1f}" if rp is not None else "n/a"
            print(f"{k:5s} {p:5s} ridge={rp_s} FLOP/byte  "
                  f"(peak {s[PRECISION_PEAK_KEY[p]]/TFLOP:.0f} TFLOP/s, "
                  f"BW {s['peak_bw_bytes']/GBs:.0f} GB/s, {s['sm_count']} SM)")
