"""
Plots for the Perceiver Ni-sweep, from results/perceiver_ni.csv.

Produces (in --outdir):
  1. perceiver_latency_vs_ni.png   log-log latency vs input length; warp's line
                                   truncates at its shared-memory wall.
  2. perceiver_speedup_vs_warp.png speedup of tiled/vectorized over warp.
  3. perceiver_gflops_vs_ni.png    achieved GFLOP/s vs Ni (LINEAR y) -- shows
                                   the occupancy plateau and the L2 drop.
  4. perceiver_pct_peak_vs_ni.png  % of the GPU's fp32 compute & BW peak.
  5. perceiver_roofline.png        achieved GFLOP/s vs arithmetic intensity
                                   against the card's compute & BW ceilings.

Pure matplotlib (Agg), no GPU needed:
    python experiments/plot_perceiver.py [--csv results/perceiver_ni.csv]
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Presentable, consistent defaults.
plt.rcParams.update({
    "figure.figsize": (8, 5.2), "figure.dpi": 150, "savefig.dpi": 150,
    "savefig.bbox": "tight", "font.size": 12, "axes.titlesize": 15,
    "axes.titleweight": "bold", "axes.labelsize": 12.5, "legend.fontsize": 11,
    "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.4,
    "axes.axisbelow": True, "figure.facecolor": "white",
})

COLORS = {"warp": "#1f77b4", "tiled": "#2ca02c", "vectorized": "#7a2fbf",
          "naive": "#d62728", "wmma": "#ff7f0e", "splitk": "#000000",
          "wmma_splitk": "#e377c2"}
MARKERS = {"warp": "s", "tiled": "^", "vectorized": "D", "naive": "o",
           "wmma": "P", "splitk": "*", "wmma_splitk": "X"}
LABELS = {"warp": "Warp parallel", "tiled": "Tiled", "vectorized": "Vectorized",
          "naive": "Naive", "wmma": "WMMA fp16",
          "splitk": "Split-K (key-parallel)",
          "wmma_splitk": "Tensor-core split-K"}
# L2 cache size per card (MB) -- for the capacity-crossover marker.
L2_MB = {"4070": 36, "L40": 96, "L40S": 96, "H200": 50}


def load(csv_path):
    by_impl, meta = {}, {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r.get("experiment") != "perceiver_ni":
                continue
            by_impl.setdefault(r["impl"], {})[int(r["N_input"])] = r
            meta.setdefault("gpu", r.get("gpu", "?"))
            meta.setdefault("nl", r.get("N_latent", "?"))
            meta.setdefault("precision", r.get("precision", "fp32"))
    return by_impl, meta


def _ok(row):
    return row and row.get("status") in ("ok", "ACCURACY?")


def _fnum(row, key):
    if not _ok(row):
        return None
    try:
        return float(row.get(key, ""))
    except (ValueError, TypeError):
        return None


def res_label(ni):
    r = int(round(ni ** 0.5))
    return f"{ni}\n({r}²)" if r * r == ni else str(ni)


def subtitle(meta):
    return (f"{meta.get('gpu','?')}  ·  N_latent={meta.get('nl','?')}  ·  "
            f"{meta.get('precision','fp32')}  ·  baseline = warp")


def _xticks(ax, nis):
    ax.set_xscale("log")
    ax.set_xticks(nis)
    ax.set_xticklabels([res_label(n) for n in nis], fontsize=9)
    ax.minorticks_off()


def _l2_marker(ax, meta):
    """Vertical line where K+V (2*Ni*D*4 bytes) exceeds the L2 cache."""
    gpu = meta.get("gpu")
    if gpu not in L2_MB:
        return
    ni_cross = L2_MB[gpu] * 1e6 / (2 * 768 * 4)
    ax.axvline(ni_cross, color="gray", ls=":", lw=1.4, alpha=0.8)
    ax.text(ni_cross, ax.get_ylim()[1], f" L2 = {L2_MB[gpu]} MB",
            color="gray", fontsize=8.5, va="top", ha="left", rotation=90)


def plot_latency(by_impl, nis, meta, outdir):
    fig, ax = plt.subplots()
    for impl, rows in by_impl.items():
        xs = [n for n in nis if _fnum(rows.get(n), "latency_ms_mean") is not None]
        ys = [_fnum(rows[n], "latency_ms_mean") for n in xs]
        if xs:
            ax.plot(xs, ys, marker=MARKERS.get(impl, "o"), color=COLORS.get(impl),
                    label=LABELS.get(impl, impl), lw=2.2, ms=9)
        if impl == "warp":
            lim = [n for n in nis if rows.get(n, {}).get("status") == "design-limited"]
            if lim and xs:
                ax.annotate("warp shared-memory wall\n(design-limited beyond here)",
                            xy=(xs[-1], ys[-1]), xytext=(0.52, 0.30),
                            textcoords="axes fraction", color=COLORS["warp"],
                            fontsize=9.5, ha="left",
                            arrowprops=dict(arrowstyle="->", color=COLORS["warp"]))
    ax.set_yscale("log")
    _xticks(ax, nis)
    _l2_marker(ax, meta)
    ax.set_xlabel("Input length  N_input  (image resolution)")
    ax.set_ylabel("Execution time (ms, log)")
    ax.set_title("Perceiver cross-attention: latency vs input length")
    ax.text(0.5, 1.02, subtitle(meta), transform=ax.transAxes, ha="center",
            fontsize=10, color="0.35")
    ax.legend(frameon=True)
    p = os.path.join(outdir, "perceiver_latency_vs_ni.png")
    fig.savefig(p); plt.close(fig)
    return p


def plot_speedup(by_impl, nis, meta, outdir):
    warp = by_impl.get("warp", {})
    fig, ax = plt.subplots()
    valid_ni = [n for n in nis if _fnum(warp.get(n), "latency_ms_mean")]
    for impl, rows in by_impl.items():
        if impl == "warp":
            continue
        xs, ys = [], []
        for n in valid_ni:
            wt = _fnum(warp.get(n), "latency_ms_mean")
            kt = _fnum(rows.get(n), "latency_ms_mean")
            if wt and kt:
                xs.append(n); ys.append(wt / kt)
        if xs:
            ax.plot(xs, ys, marker=MARKERS.get(impl, "o"), color=COLORS.get(impl),
                    label=LABELS.get(impl, impl), lw=2.2, ms=9)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.1f}×", (x, y), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8.5,
                            color=COLORS.get(impl))
    ax.axhline(1.0, color="gray", ls=":", alpha=0.7)
    if valid_ni:
        _xticks(ax, valid_ni)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Input length  N_input  (where warp still runs)")
    ax.set_ylabel("Speedup vs warp (×)")
    ax.set_title("Speedup over the warp baseline")
    ax.text(0.5, 1.02, subtitle(meta), transform=ax.transAxes, ha="center",
            fontsize=10, color="0.35")
    ax.legend(frameon=True)
    p = os.path.join(outdir, "perceiver_speedup_vs_warp.png")
    fig.savefig(p); plt.close(fig)
    return p


def plot_gflops(by_impl, nis, meta, outdir):
    fig, ax = plt.subplots()
    for impl, rows in by_impl.items():
        xs = [n for n in nis if _fnum(rows.get(n), "achieved_gflops") is not None]
        ys = [_fnum(rows[n], "achieved_gflops") for n in xs]
        if xs:
            ax.plot(xs, ys, marker=MARKERS.get(impl, "o"), color=COLORS.get(impl),
                    label=LABELS.get(impl, impl), lw=2.2, ms=9)
    _xticks(ax, nis)
    ax.set_ylim(bottom=0)
    _l2_marker(ax, meta)
    ax.set_xlabel("Input length  N_input  (image resolution)")
    ax.set_ylabel("Achieved throughput (GFLOP/s)")
    ax.set_title("Compute throughput vs input length")
    ax.text(0.5, 1.02, subtitle(meta), transform=ax.transAxes, ha="center",
            fontsize=10, color="0.35")
    ax.legend(frameon=True)
    p = os.path.join(outdir, "perceiver_gflops_vs_ni.png")
    fig.savefig(p); plt.close(fig)
    return p


def plot_pct_peak(by_impl, nis, meta, outdir):
    fig, ax = plt.subplots()
    drew = False
    for impl, rows in by_impl.items():
        xs = [n for n in nis if _fnum(rows.get(n), "pct_peak_compute") is not None]
        ys = [_fnum(rows[n], "pct_peak_compute") for n in xs]
        if xs and any(y for y in ys):
            ax.plot(xs, ys, marker=MARKERS.get(impl, "o"), color=COLORS.get(impl),
                    label=LABELS.get(impl, impl), lw=2.2, ms=9)
            drew = True
    if not drew:
        plt.close(fig)
        return None
    _xticks(ax, nis)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Input length  N_input  (image resolution)")
    ax.set_ylabel("% of fp32 compute peak")
    ax.set_title("Fraction of peak compute achieved (occupancy-limited)")
    ax.text(0.5, 1.02, subtitle(meta), transform=ax.transAxes, ha="center",
            fontsize=10, color="0.35")
    ax.legend(frameon=True)
    p = os.path.join(outdir, "perceiver_pct_peak_vs_ni.png")
    fig.savefig(p); plt.close(fig)
    return p


def plot_roofline(by_impl, nis, meta, outdir):
    """Achieved GFLOP/s vs arithmetic intensity, against the card ceilings."""
    gpu = meta.get("gpu")
    try:
        from gpu_specs import GPU_SPECS
        spec = GPU_SPECS.get(gpu)
    except Exception:
        spec = None

    fig, ax = plt.subplots()
    # ceilings
    if spec:
        peak_c = spec["peak_fp32_flops"] / 1e9          # GFLOP/s
        peak_bw = spec["peak_bw_bytes"]                  # bytes/s
        ais = [2 ** i for i in range(0, 11)]             # 1 .. 1024 FLOP/byte
        roof = [min(peak_c, peak_bw * ai / 1e9) for ai in ais]
        ax.plot(ais, roof, color="0.25", lw=2,
                label=f"{gpu} fp32 roofline")
        ax.axhline(peak_c, color="0.55", ls="--", lw=1,
                   label=f"fp32 compute peak {peak_c/1e3:.0f} TFLOP/s")
        peak_fp16 = spec.get("peak_fp16_tc_flops")
        if peak_fp16:
            ax.axhline(peak_fp16 / 1e9, color="#ff7f0e", ls="--", lw=1,
                       label=f"fp16 TC peak {peak_fp16/1e12:.0f} TFLOP/s")
        ridge = peak_c * 1e9 / peak_bw
        ax.axvline(ridge, color="0.7", ls=":", lw=1)
        ax.text(ridge, peak_c, f" ridge {ridge:.0f}", color="0.4",
                fontsize=8.5, va="bottom")

    for impl, rows in by_impl.items():
        pts = [(_fnum(rows[n], "arithmetic_intensity"),
                _fnum(rows[n], "achieved_gflops"))
               for n in nis
               if _fnum(rows.get(n), "achieved_gflops") is not None]
        pts = [(a, g) for a, g in pts if a and g]
        if pts:
            xs, ys = zip(*sorted(pts))
            ax.plot(xs, ys, marker=MARKERS.get(impl, "o"), color=COLORS.get(impl),
                    label=LABELS.get(impl, impl), lw=2, ms=8)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("Achieved throughput (GFLOP/s, log)")
    ax.set_title("Roofline: how far below peak the kernels sit")
    ax.text(0.5, 1.02, subtitle(meta), transform=ax.transAxes, ha="center",
            fontsize=10, color="0.35")
    ax.legend(frameon=True, fontsize=9, loc="lower right")
    p = os.path.join(outdir, "perceiver_roofline.png")
    fig.savefig(p); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description="Plot the Perceiver Ni-sweep")
    ap.add_argument("--csv", default="results/perceiver_ni.csv")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"No CSV at {args.csv}. Run bench_perceiver_ni.py first.")
        return
    by_impl, meta = load(args.csv)
    if not by_impl:
        print("No perceiver_ni rows in CSV.")
        return
    nis = sorted({n for rows in by_impl.values() for n in rows})
    os.makedirs(args.outdir, exist_ok=True)

    made = [
        plot_latency(by_impl, nis, meta, args.outdir),
        plot_speedup(by_impl, nis, meta, args.outdir),
        plot_gflops(by_impl, nis, meta, args.outdir),
        plot_pct_peak(by_impl, nis, meta, args.outdir),
        plot_roofline(by_impl, nis, meta, args.outdir),
    ]
    print("Wrote:")
    for p in made:
        if p:
            print(f"  {p}")


if __name__ == "__main__":
    main()
