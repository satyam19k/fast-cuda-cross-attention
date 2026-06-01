"""
Recreate the comparison plots for the Perceiver Ni-sweep from the CSV.

Reads results/perceiver_ni.csv (from bench_perceiver_ni.py) and writes:
  - perceiver_latency_vs_ni.png : execution time vs input length (log-y).
        warp's line truncates where it goes 'design-limited' (SMEM wall);
        tiled/vectorized continue to 224^2. This is the headline plot.
  - perceiver_speedup_vs_warp.png : speedup of tiled/vectorized over warp,
        at the Ni where warp still runs.

Pure matplotlib; no GPU needed. Run after the benchmark:
    python experiments/plot_perceiver.py [--csv results/perceiver_ni.csv]
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"warp": "#1f5fd0", "tiled": "#2a9d3a", "vectorized": "#7a2fbf",
          "naive": "#d62728"}
MARKERS = {"warp": "s", "tiled": "^", "vectorized": "D", "naive": "o"}


def load(csv_path):
    """Return rows grouped by impl: {impl: {ni: row}}."""
    by_impl = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r.get("experiment") != "perceiver_ni":
                continue
            by_impl.setdefault(r["impl"], {})[int(r["N_input"])] = r
    return by_impl


def _fnum(row, key):
    """Float value if present and the row succeeded, else None."""
    if not row or row.get("status") not in ("ok", "ACCURACY?"):
        return None
    v = row.get(key, "")
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def res_label(ni):
    r = int(round(ni ** 0.5))
    return f"{ni}\n({r}²)" if r * r == ni else str(ni)


def plot_latency(by_impl, all_ni, outdir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for impl, rows in by_impl.items():
        xs, ys = [], []
        for ni in all_ni:
            y = _fnum(rows.get(ni), "latency_ms_mean")
            if y is not None:
                xs.append(ni); ys.append(y)
        if xs:
            ax.plot(xs, ys, marker=MARKERS.get(impl, "o"),
                    color=COLORS.get(impl), label=impl, linewidth=2, markersize=8)
        # annotate the warp SMEM wall
        if impl == "warp":
            limited = [ni for ni in all_ni
                       if rows.get(ni, {}).get("status") == "design-limited"]
            if limited and xs:
                ax.annotate("warp SMEM wall\n(design-limited)",
                            xy=(xs[-1], ys[-1]), xytext=(10, 18),
                            textcoords="offset points", color=COLORS["warp"],
                            fontsize=9, arrowprops=dict(arrowstyle="->",
                            color=COLORS["warp"]))
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks(all_ni)
    ax.set_xticklabels([res_label(n) for n in all_ni], fontsize=8)
    ax.set_xlabel("Input length  N_input  (image resolution)")
    ax.set_ylabel("Execution time (ms, log)")
    ax.set_title("Perceiver cross-attention: latency vs input length")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    p = os.path.join(outdir, "perceiver_latency_vs_ni.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


def plot_speedup(by_impl, all_ni, outdir):
    warp = by_impl.get("warp", {})
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for impl, rows in by_impl.items():
        if impl == "warp":
            continue
        xs, ys = [], []
        for ni in all_ni:
            wt = _fnum(warp.get(ni), "latency_ms_mean")
            kt = _fnum(rows.get(ni), "latency_ms_mean")
            if wt and kt:
                xs.append(ni); ys.append(wt / kt)
        if xs:
            ax.plot(xs, ys, marker=MARKERS.get(impl, "o"),
                    color=COLORS.get(impl), label=impl, linewidth=2, markersize=8)
            plotted = True
    ax.axhline(1.0, color="gray", ls=":", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xticks([n for n in all_ni if _fnum(warp.get(n), "latency_ms_mean")])
    ax.set_xticklabels([res_label(n) for n in all_ni
                        if _fnum(warp.get(n), "latency_ms_mean")], fontsize=8)
    ax.set_xlabel("Input length  N_input  (where warp still runs)")
    ax.set_ylabel("Speedup vs warp parallel (×)")
    ax.set_title("Speedup over warp baseline")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    if plotted:
        ax.legend()
    fig.tight_layout()
    p = os.path.join(outdir, "perceiver_speedup_vs_warp.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description="Plot the Perceiver Ni-sweep")
    ap.add_argument("--csv", default="results/perceiver_ni.csv")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"No CSV at {args.csv}. Run bench_perceiver_ni.py first.")
        return
    by_impl = load(args.csv)
    if not by_impl:
        print("No perceiver_ni rows in CSV.")
        return
    all_ni = sorted({ni for rows in by_impl.values() for ni in rows})
    os.makedirs(args.outdir, exist_ok=True)

    p1 = plot_latency(by_impl, all_ni, args.outdir)
    p2 = plot_speedup(by_impl, all_ni, args.outdir)
    print(f"Wrote:\n  {p1}\n  {p2}")


if __name__ == "__main__":
    main()
