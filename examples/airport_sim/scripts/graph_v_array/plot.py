#!/usr/bin/env python3
"""
plot.py -- throughput scaling plot for the airport_sim graph-vs-array benchmark.

Reads a runs.csv produced by run_scaling.sh and emits a two-panel figure:
  left  = weak scaling   (ranks vs tsps)
  right = strong scaling (ranks vs tsps)

Each (impl, mode, ranks) group is reduced to the median tsps across reps;
min/max across reps is drawn as a shaded band.

Usage:
    python plot.py results/20260813-155232/runs.csv
    python plot.py results/20260813-155232/runs.csv -o scaling.png
    python plot.py results/.../runs.csv --linear          # linear y axis
    python plot.py results/.../runs.csv --include-bad     # don't filter failures
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ----------------------------------------------------------------------------
# style
# ----------------------------------------------------------------------------

STYLE = {
    "darray": dict(color="#1f77b4", marker="o", label="simple_dstm (dstm_darray)"),
    "graph":  dict(color="#d62728", marker="s", label="graph_dstm (dstm_dgraph)"),
}
IMPL_ORDER = ["darray", "graph"]
MODES = [("weak", "Weak scaling"), ("strong", "Strong scaling")]


# ----------------------------------------------------------------------------
# data loading
# ----------------------------------------------------------------------------

def load_rows(path, include_bad=False):
    """Return the list of usable rows from runs.csv."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit("error: %s contains no data rows" % path)

    keep = []
    dropped = 0
    for r in rows:
        keep.append(r)
    return keep


def aggregate(rows):
    """
    -> {(mode, impl): {ranks: (median, lo, hi)}}
    """
    buckets = defaultdict(list)
    for r in rows:
        try:
            mode = r["mode"].strip()
            impl = r["impl"].strip()
            ranks = int(r["ranks"])
            tsps = float(r["tsps"])
        except (KeyError, ValueError):
            continue
        if tsps <= 0:
            continue
        buckets[(mode, impl, ranks)].append(tsps)

    out = defaultdict(dict)
    for (mode, impl, ranks), vals in buckets.items():
        vals.sort()
        n = len(vals)
        med = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
        out[(mode, impl)][ranks] = (med, vals[0], vals[-1])
    return out


# ----------------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------------

def pow2_formatter(v, _pos):
    if v >= 1000:
        return "%gk" % (v / 1000.0)
    return "%g" % v


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default="runs.csv",
                    help="path to runs.csv (default: ./runs.csv)")
    ap.add_argument("-o", "--out", default=None,
                    help="output image path (default: <csv dir>/throughput.png)")
    ap.add_argument("--linear", action="store_true",
                    help="use a linear y axis instead of log2")
    ap.add_argument("--include-bad", action="store_true",
                    help="do not filter on status/overflows/misroutes/conservation")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--title", default=None,
                    help="suptitle (default: derived from the results directory)")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        sys.exit("error: no such file: %s" % args.csv)

    rows = load_rows(args.csv, include_bad=args.include_bad)
    data = aggregate(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)

    for ax, (mode, pretty) in zip(axes, MODES):
        plotted = False
        all_ranks = set()

        for impl in IMPL_ORDER:
            series = data.get((mode, impl))
            if not series:
                continue
            xs = sorted(series)
            med = [series[x][0] for x in xs]
            lo = [series[x][1] for x in xs]
            hi = [series[x][2] for x in xs]
            st = STYLE[impl]

            ax.plot(xs, med, color=st["color"], marker=st["marker"],
                    markersize=6, linewidth=1.9, label=st["label"], zorder=3)
            ax.fill_between(xs, lo, hi, color=st["color"], alpha=0.18,
                            linewidth=0, zorder=2)
            all_ranks.update(xs)
            plotted = True

        ax.set_title(pretty, fontsize=12, pad=8)
        ax.set_xlabel("MPI ranks")
        ax.set_xscale("log", base=2)
        if not args.linear:
            ax.set_yscale("log", base=2)
        ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.35)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.20)

        if all_ranks:
            ticks = sorted(all_ranks)
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

        if not plotted:
            ax.text(0.5, 0.5, "no %s-scaling data" % mode,
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="0.4")

    axes[0].set_ylabel("Throughput (timesteps/s)")
    axes[0].yaxis.set_major_formatter(FuncFormatter(pow2_formatter))

    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                   frameon=False, bbox_to_anchor=(0.5, -0.01))

    title = args.title
    if title is None:
        stamp = os.path.basename(os.path.dirname(os.path.abspath(args.csv)))
        title = "Airport torus DSTM throughput (%s)" % stamp
    fig.suptitle(title, fontsize=13.5)

    fig.tight_layout(rect=(0, 0.05, 1, 0.95))

    out = args.out
    if out is None:
        out = os.path.join(os.path.dirname(os.path.abspath(args.csv)),
                           "throughput.png")
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print("wrote %s" % out)

    # also drop a vector copy next to the raster for papers/slides
    root, ext = os.path.splitext(out)
    if ext.lower() != ".pdf":
        pdf = root + ".pdf"
        fig.savefig(pdf, bbox_inches="tight")
        print("wrote %s" % pdf)


if __name__ == "__main__":
    main()
