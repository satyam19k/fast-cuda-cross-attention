"""
The punchline figure: my best kernel vs PyTorch's eligible SDPA backends.

Reads a perceiver_ni CSV that contains split-K and SDPA rows (run
bench_perceiver_ni.py with --impls splitk sdpa_math sdpa_math16 sdpa_eff) and
makes the three figures that carry the materialization-vs-fusion story:

  1. pytorch_latency_vs_ni.png  log-log latency vs Ni for my split-K and the
                                PyTorch backends -- the head-to-head.
  2. pytorch_bar_anchor.png     grouped bars at one Ni (default 4096): the
                                single-slide punchline with the key ratios.
  3. pytorch_slowdown_vs_math.png  each impl's latency / SDPA-math latency vs Ni
                                -- "x off the materialized backend" across the
                                sweep (does the gap grow with Ni?).

Pure matplotlib; no GPU needed.
    python experiments/plot_pytorch_compare.py --csv results/ni_nl256.csv --anchor 4096
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (8, 5.2), "figure.dpi": 150, "savefig.dpi": 150,
    "savefig.bbox": "tight", "font.size": 12, "axes.titlesize": 15,
    "axes.titleweight": "bold", "axes.labelsize": 12.5, "legend.fontsize": 10.5,
    "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.4,
    "axes.axisbelow": True, "figure.facecolor": "white",
})

# Consistent with plot_perceiver.py; "mine" (split-K) is bold black.
STYLE = {
    "splitk":      ("#000000", "*", "Split-K  (mine, fused)"),
    "vectorized":  ("#7a2fbf", "D", "Vectorized (mine, fused)"),
    "sdpa_math":   ("#8c564b", "v", "PyTorch SDPA-math  (cuBLAS, materialized)"),
    "sdpa_math16": ("#c49c94", "v", "PyTorch SDPA-math fp16"),
    "sdpa_eff":    ("#17becf", "<", "PyTorch SDPA-efficient  (fused)"),
}
DEFAULT_BASELINES = ["sdpa_math", "sdpa_math16", "sdpa_eff"]


def load(csv_path):
    by_impl, meta = {}, {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r.get("experiment") != "perceiver_ni":
                continue
            by_impl.setdefault(r["impl"], {})[int(r["N_input"])] = r
            meta.setdefault("gpu", r.get("gpu", "?"))
            meta.setdefault("nl", r.get("N_latent", "?"))
    return by_impl, meta


def lat(row):
    if not row or row.get("status") not in ("ok", "ACCURACY?"):
        return None
    try:
        return float(row.get("latency_ms_mean", ""))
    except (ValueError, TypeError):
        return None


def res_label(ni):
    r = int(round(ni ** 0.5))
    return f"{ni}\n({r}²)" if r * r == ni else str(ni)


def sub(meta):
    return (f"{meta.get('gpu','?')}  ·  N_latent={meta.get('nl','?')}  ·  "
            f"single head, head_dim=768")


def style(impl):
    return STYLE.get(impl, ("#888888", "o", impl))


def plot_latency(by_impl, impls, nis, meta, outdir):
    fig, ax = plt.subplots()
    for impl in impls:
        rows = by_impl.get(impl, {})
        xs = [n for n in nis if lat(rows.get(n)) is not None]
        ys = [lat(rows[n]) for n in xs]
        if not xs:
            continue
        c, m, lab = style(impl)
        lw = 2.6 if impl in ("splitk", "vectorized") else 2.0
        ax.plot(xs, ys, marker=m, color=c, label=lab, lw=lw, ms=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(nis); ax.set_xticklabels([res_label(n) for n in nis], fontsize=9)
    ax.minorticks_off()
    ax.set_xlabel("Input length  N_input  (image resolution)")
    ax.set_ylabel("Latency (ms, log)")
    ax.set_title("My best kernel vs PyTorch's eligible backends")
    ax.text(0.5, 1.02, sub(meta), transform=ax.transAxes, ha="center",
            fontsize=10, color="0.35")
    ax.legend(frameon=True)
    p = os.path.join(outdir, "pytorch_latency_vs_ni.png")
    fig.savefig(p); plt.close(fig)
    return p


def plot_bar(by_impl, impls, anchor, meta, outdir):
    items = [(impl, lat(by_impl.get(impl, {}).get(anchor))) for impl in impls]
    items = [(i, v) for i, v in items if v is not None]
    if not items:
        return None
    items.sort(key=lambda t: t[1])  # fastest first
    labels = [style(i)[2].split("  ")[0] for i, _ in items]
    vals = [v for _, v in items]
    colors = [style(i)[0] for i, _ in items]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(range(len(items)), vals, color=colors, width=0.62,
                  edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f} ms",
                ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Latency (ms)")
    ax.set_ylim(top=max(vals) * 1.18)
    r = int(round(anchor ** 0.5))
    res = f" ({r}²)" if r * r == anchor else ""
    ax.set_title(f"The punchline: materialized beats fused (Ni={anchor}{res})")
    ax.text(0.5, 1.02, sub(meta), transform=ax.transAxes, ha="center",
            fontsize=10, color="0.35")

    # annotate the two key ratios if the impls are present
    d = {i: v for i, v in items}
    msgs = []
    if "splitk" in d and "sdpa_math" in d:
        msgs.append(f"split-K is {d['splitk']/d['sdpa_math']:.1f}× the "
                    f"materialized cuBLAS time")
    if "splitk" in d and "sdpa_eff" in d:
        msgs.append(f"...but {d['sdpa_eff']/d['splitk']:.1f}× faster than "
                    f"PyTorch's fused backend")
    if msgs:
        ax.text(0.97, 0.97, "\n".join(msgs), transform=ax.transAxes,
                ha="right", va="top", fontsize=10.5, color="0.15",
                bbox=dict(boxstyle="round", fc="#f4f4f4", ec="0.7"))
    p = os.path.join(outdir, "pytorch_bar_anchor.png")
    fig.savefig(p); plt.close(fig)
    return p


def plot_slowdown(by_impl, impls, nis, meta, outdir):
    math_rows = by_impl.get("sdpa_math", {})
    fig, ax = plt.subplots()
    drew = False
    for impl in impls:
        if impl == "sdpa_math":
            continue
        rows = by_impl.get(impl, {})
        xs, ys = [], []
        for n in nis:
            mt = lat(math_rows.get(n)); it = lat(rows.get(n))
            if mt and it:
                xs.append(n); ys.append(it / mt)
        if xs:
            c, m, lab = style(impl)
            ax.plot(xs, ys, marker=m, color=c, label=lab, lw=2.2, ms=9)
            drew = True
    if not drew:
        plt.close(fig)
        return None
    ax.axhline(1.0, color="#8c564b", ls="--", lw=1.6,
               label="PyTorch SDPA-math (= 1.0)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(nis); ax.set_xticklabels([res_label(n) for n in nis], fontsize=9)
    ax.minorticks_off()
    ax.set_xlabel("Input length  N_input")
    ax.set_ylabel("Latency / SDPA-math latency  (× slower)")
    ax.set_title("Gap to the materialized backend across the sweep")
    ax.text(0.5, 1.02, sub(meta), transform=ax.transAxes, ha="center",
            fontsize=10, color="0.35")
    ax.legend(frameon=True)
    p = os.path.join(outdir, "pytorch_slowdown_vs_math.png")
    fig.savefig(p); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description="Plot my best kernel vs PyTorch SDPA")
    ap.add_argument("--csv", default="results/ni_nl256.csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--mine", default="splitk", help="my best kernel impl name")
    ap.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES)
    ap.add_argument("--anchor", type=int, default=4096,
                    help="Ni for the bar-chart punchline slide")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"No CSV at {args.csv}. Run bench_perceiver_ni.py with splitk + "
              f"sdpa_* impls first.")
        return
    by_impl, meta = load(args.csv)
    impls = [args.mine] + [b for b in args.baselines if b != args.mine]
    present = [i for i in impls if i in by_impl]
    if args.mine not in by_impl:
        print(f"'{args.mine}' not in CSV; have: {sorted(by_impl)}")
    nis = sorted({n for i in present for n in by_impl[i]})
    os.makedirs(args.outdir, exist_ok=True)

    made = [
        plot_latency(by_impl, present, nis, meta, args.outdir),
        plot_bar(by_impl, present, args.anchor, meta, args.outdir),
        plot_slowdown(by_impl, present, nis, meta, args.outdir),
    ]
    print("Wrote:")
    for p in made:
        if p:
            print(f"  {p}")


if __name__ == "__main__":
    main()
