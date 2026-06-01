"""
One-row-per-measurement CSV logging, shared across all experiments E0-E9.

Schema is fixed so every experiment's output concatenates into one frame for
plotting. Unused fields are left blank. Append-safe: writes the header only
when the file is new/empty.
"""

import csv
import os

FIELDS = [
    "experiment",   # e0..e9
    "gpu",          # 4070 | L40 | H200 | <raw name>
    "impl",         # naive|warp|tiled|vectorized|cublas_softmax|sdpa_flash|sdpa_efficient|sdpa_math|wmma_fused|key_parallel
    "precision",    # fp32|fp16|fp8
    "B", "N_latent", "N_input", "D",
    "latency_ms_mean", "latency_ms_std",
    "achieved_gflops", "achieved_gbs",
    "pct_peak_compute", "pct_peak_bw",
    "arithmetic_intensity", "ridge_point", "bound",
    "max_abs_err", "rel_err",
    "status",       # ok | error | unsupported | oom
    "note",
]


def writer(path):
    """Return (file_handle, csv.DictWriter), writing header if file is new."""
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    fh = open(path, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
    if new:
        w.writeheader()
    return fh, w


def write_row(path, **kwargs):
    """Append a single row; missing fields blank. Convenience for one-offs."""
    fh, w = writer(path)
    w.writerow(kwargs)
    fh.close()
