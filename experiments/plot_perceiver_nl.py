"""
Plots for the Perceiver Nl-sweep (occupancy experiment), from
results/perceiver_nl.csv.

Produces (in --outdir):
  1. perceiver_nl_gflops.png   achieved GFLOP/s vs N_latent -- throughput rises
                               as more blocks fill the GPU, then saturates.
  2. perceiver_nl_latency.png  latency vs N_latent (log-log).
  3. perceiver_nl_occupancy.png blocks (= ceil(Nl/4)) vs N_latent with the SM
                               count drawn in -- the occupancy driver.

    python experiments/plot_perceiver_nl.py [--csv results/perceiver_nl.csv]
"""

import argparse
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

plt.rcParams.update({
    "figure.figsize": (8, 5.2), "figure.dpi": 150, "savefig.dpi": 150,
    "savefig.bbox": "tight", "font.size": 12, "axes.titlesize": 15,
    "axes.titleweight": "bold", "axes.labelsize": 12.5, "legend.fontsize": 11,
    "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.4,
    "axes.axisbelow": True, "figure.facecolor": "white",
})

COLORS = {"warp": "#1f77b4", "tiled": "#2ca02c", "vectorized": "#7a2fbf"}
MARKERS = {"warp": "s", "tiled": "^", "vectorized": "D"}
LABELS = {"warp": "Warp parallel", "tiled": "Tiled", "vectorized": "Vectorized"}


def load(csv_path):
    by_impl, meta = {}, {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r.get("experiment") != "perceiver_nl":
                continue
            by_impl.setdefault(r["impl"], {})[int(r["N_latent"])] = r
            meta.setdefault("gpu", r.get("gpu", "?"))
            meta.setdefault("ni", r.get("N_input", "?"))
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


def sm_count(gpu):
    try:
        from gpu_specs import GPU_SPECS
        return GPU_SPECS.get(gpu, {}).get("sm_count")
    except Exception:
        return None


def subtitle(meta):
    return (f"{meta.get('gpu','?')}  ·  N_input={meta.get('ni','?')}  ·  fp32  "
            f"·  grid = ceil(Nl/4) blocks")


def _line(ax, by_impl, nls, key):
    for impl, rows in by_impl.items():
        xs = [n for n in nls if _fnum(rows.get(n), key) is not None]
        ys = [_fnum(rows[n], key) for n in xs]
        if xs:
            ax.plot(xs, ys, marker=MARKERS.get(impl, "o"), color=COLORS.get(impl),
                    label=LABELS.get(impl, impl), lw=2.2, ms=9)


def plot_gflops(by_impl, nls, meta, outdir):
    fig, ax = plt.subplots()
    _line(ax, by_impl, nls, "achieved_gflops")
    ax.set_xscale("log", base=2)
    ax.set_xticks(nls); ax.set_xticklabels(nls); ax.minorticks_off()
    ax.set_ylim(bottom=0)
    smc = sm_count(meta.get("gpu"))
    if smc:
        nl_fill = smc * 4  # ceil(Nl/4) >= SM count
        if nls[0] <= nl_fill <= nls[-1]:
            ax.axvline(nl_fill, color="gray", ls=":", lw=1.4)
            ax.text(nl_fill, ax.get_ylim()[1], f" Nl≈{nl_fill} fills {smc} SMs",
                    color="gray", fontsize=8.5, va="top", ha="left", rotation=90)
    ax.set_xlabel("Number of latents  N_latent")
    ax.set_ylabel("Achieved throughput (GFLOP/s)")
    ax.set_title("Throughput vs latents: filling the GPU")
    ax.text(0.5, 1.02, subtitle(meta), transform=ax.transAxes, ha="center",
            fontsize=10, color="0.35")
    ax.legend(frameon=True)
    p = os.path.join(outdir, "perceiver_nl_gflops.png")
    fig.savefig(p); plt.close(fig)
    return p


def plot_latency(by_impl, nls, meta, outdir):
    fig, ax = plt.subplots()
    _line(ax, by_impl, nls, "latency_ms_mean")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(nls); ax.set_xticklabels(nls); ax.minorticks_off()
    ax.set_xlabel("Number of latents  N_latent")
    ax.set_ylabel("Execution time (ms, log)")
    ax.set_title("Latency vs latents")
    ax.text(0.5, 1.02, subtitle(meta), transform=ax.transAxes, ha="center",
            fontsize=10, color="0.35")
    ax.legend(frameon=True)
    p = os.path.join(outdir, "perceiver_nl_latency.png")
    fig.savefig(p); plt.close(fig)
    return p


def plot_occupancy(by_impl, nls, meta, outdir):
    fig, ax = plt.subplots()
    blocks = [math.ceil(n / 4) for n in nls]
    ax.plot(nls, blocks, marker="o", color="#444", lw=2.2, ms=8,
            label="blocks = ceil(Nl/4)")
    smc = sm_count(meta.get("gpu"))
    if smc:
        ax.axhline(smc, color="#d62728", ls="--", lw=1.8,
                   label=f"{meta.get('gpu')} = {smc} SMs (1 block/SM)")
        ax.axhline(2 * smc, color="#d62728", ls=":", lw=1.2, alpha=0.7,
                   label=f"{2*smc} blocks (2 blocks/SM)")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(nls); ax.set_xticklabels(nls); ax.minorticks_off()
    ax.set_xlabel("Number of latents  N_latent")
    ax.set_ylabel("Thread blocks launched (log)")
    ax.set_title("Why small Nl starves the GPU: blocks vs SM count")
    ax.text(0.5, 1.02, subtitle(meta), transform=ax.transAxes, ha="center",
            fontsize=10, color="0.35")
    ax.legend(frameon=True, fontsize=9)
    p = os.path.join(outdir, "perceiver_nl_occupancy.png")
    fig.savefig(p); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description="Plot the Perceiver Nl-sweep")
    ap.add_argument("--csv", default="results/perceiver_nl.csv")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"No CSV at {args.csv}. Run bench_perceiver_nl.py first.")
        return
    by_impl, meta = load(args.csv)
    if not by_impl:
        print("No perceiver_nl rows in CSV.")
        return
    nls = sorted({n for rows in by_impl.values() for n in rows})
    os.makedirs(args.outdir, exist_ok=True)

    for p in (plot_gflops(by_impl, nls, meta, args.outdir),
              plot_latency(by_impl, nls, meta, args.outdir),
              plot_occupancy(by_impl, nls, meta, args.outdir)):
        print(f"  {p}")


if __name__ == "__main__":
    main()
