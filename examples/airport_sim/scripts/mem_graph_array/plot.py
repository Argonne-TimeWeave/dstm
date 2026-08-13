#!/usr/bin/env python3
"""
plot.py -- peak-memory comparison for the airport_sim graph-vs-array benchmark.

Reads the mem.csv produced by run_memory.sh and emits a two-panel clustered
bar chart:
  left  = weak scaling   (ranks vs peak RSS per rank)
  right = strong scaling (ranks vs peak RSS per rank)

darray ("simple_dstm") is blue, graph ("graph_dstm") is red.

The two backends differ by ~2 orders of magnitude (16 MB vs 3800 MB), so each
panel is drawn with a BROKEN y-axis: the split point is found automatically as
the largest multiplicative gap in the data, and both segments are drawn to
scale with diagonal break marks between them. Use --log for a single
log-scaled axis instead, or --no-break to force one linear axis.

Usage:
    python plot.py results/20260813-182758/mem.csv
    python plot.py results/.../mem.csv -o memory.png
    python plot.py results/.../mem.csv --metric max      # worst rank, not mean
    python plot.py results/.../mem.csv --log             # log axis, no break
    python plot.py results/.../mem.csv --include-bad     # keep non-ok rows
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# style
# ----------------------------------------------------------------------------

STYLE = {
    "darray": dict(color="#1f77b4", edge="#12405f",
                   label="simple_dstm (dstm_darray)"),
    "graph":  dict(color="#d62728", edge="#7d1618",
                   label="graph_dstm (dstm_dgraph)"),
}
IMPL_ORDER = ["darray", "graph"]
MODES = [("weak", "Weak scaling  (na=64/rank, total grows)"),
         ("strong", "Strong scaling  (total=4096, na shrinks)")]

METRIC_COL = {
    "mean": ("rss_mean_mb", "Peak RSS per rank (MB, median of reps)"),
    "max":  ("rss_max_mb",  "Peak RSS, worst rank (MB, median of reps)"),
    "sum":  ("rss_sum_mb",  "Peak RSS, all ranks summed (MB, median of reps)"),
    "vsz":  ("vsz_max_mb",  "Peak virtual size, worst rank (MB)"),
}

BAR_W = 0.38
BREAK_RATIO = 5.0        # min gap ratio before a break is inserted


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

def load(path, variant, include_bad):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit("error: %s has no data rows" % path)

    keep, dropped = [], 0
    for r in rows:
        if r.get("variant", "base").strip() != variant:
            continue
        if not include_bad and r.get("status", "").strip() != "ok":
            dropped += 1
            continue
        keep.append(r)

    if dropped:
        print("note: dropped %d row(s) with status != ok" % dropped,
              file=sys.stderr)
    if not keep:
        sys.exit("error: nothing left after filtering (try --include-bad "
                 "or --variant)")
    return keep


def value_of(row, col):
    """Metric in MB, with a fallback for a missing rss_mean_mb column."""
    v = row.get(col, "")
    if v not in (None, "", "nan", "-1"):
        try:
            return float(v)
        except ValueError:
            pass
    if col == "rss_mean_mb":
        try:
            s = float(row["rss_sum_mb"])
            n = float(row["n_procs_seen"])
            if n > 0:
                return s / n
        except (KeyError, ValueError, ZeroDivisionError):
            pass
    return None


def aggregate(rows, col):
    """-> {mode: {ranks: {impl: median_mb}}}, {mode: {ranks: 'NxP'}}"""
    buckets = defaultdict(list)
    layout = {}
    for r in rows:
        try:
            mode = r["mode"].strip()
            impl = r["impl"].strip()
            ranks = int(r["ranks"])
        except (KeyError, ValueError):
            continue
        v = value_of(r, col)
        if v is None or v <= 0:
            continue
        buckets[(mode, ranks, impl)].append(v)
        if r.get("nodes") and r.get("ppn"):
            layout[(mode, ranks)] = "%sx%s" % (r["nodes"], r["ppn"])

    out = defaultdict(lambda: defaultdict(dict))
    for (mode, ranks, impl), vals in buckets.items():
        vals.sort()
        n = len(vals)
        med = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
        out[mode][ranks][impl] = med
    return out, layout


def find_break(values, thresh=BREAK_RATIO):
    """Largest multiplicative gap in the data -> (low_max, high_min) or None."""
    vals = sorted({v for v in values if v > 0})
    if len(vals) < 2:
        return None
    best, best_i = 1.0, -1
    for i in range(len(vals) - 1):
        ratio = vals[i + 1] / vals[i]
        if ratio > best:
            best, best_i = ratio, i
    if best < thresh:
        return None
    return vals[best_i], vals[best_i + 1]


# ----------------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------------

def fmt(v):
    if v >= 1000:
        return "%,.0f".replace(",", ",") % v if False else format(v, ",.0f")
    if v >= 100:
        return "%.0f" % v
    return "%.1f" % v


def draw_bars(ax, xs, series, impls):
    """Draw the clustered bars for one panel onto one axis."""
    handles = {}
    n = len(impls)
    for k, impl in enumerate(impls):
        off = (k - (n - 1) / 2.0) * BAR_W
        heights, positions = [], []
        for i, r in enumerate(xs):
            v = series.get(r, {}).get(impl)
            if v is None:
                continue
            positions.append(i + off)
            heights.append(v)
        if not positions:
            continue
        st = STYLE[impl]
        b = ax.bar(positions, heights, width=BAR_W, color=st["color"],
                   edgecolor=st["edge"], linewidth=0.7, zorder=3,
                   label=st["label"])
        handles[impl] = b
    return handles


def label_bars(ax_top, ax_bot, xs, series, impls, split):
    """Put a value label above each bar, on whichever axis shows its top."""
    n = len(impls)
    for k, impl in enumerate(impls):
        off = (k - (n - 1) / 2.0) * BAR_W
        for i, r in enumerate(xs):
            v = series.get(r, {}).get(impl)
            if v is None:
                continue
            if ax_top is not None and split is not None and v >= split:
                ax, lim = ax_top, ax_top.get_ylim()
            else:
                ax, lim = ax_bot, ax_bot.get_ylim()
            pad = 0.025 * (lim[1] - lim[0])
            ax.text(i + off, v + pad, fmt(v), ha="center", va="bottom",
                    fontsize=8, rotation=0, zorder=5,
                    color=STYLE[impl]["edge"])


def break_marks(ax_top, ax_bot):
    d = 0.5
    kw = dict(marker=[(-1, -d), (1, d)], markersize=8, linestyle="none",
              color="0.25", mec="0.25", mew=1.0, clip_on=False)
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kw)
    ax_bot.plot([0, 1], [1, 1], transform=ax_bot.transAxes, **kw)


def style_axis(ax):
    ax.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.30, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default="mem.csv",
                    help="path to mem.csv (default: ./mem.csv)")
    ap.add_argument("-o", "--out", default=None,
                    help="output image (default: <csv dir>/memory.png)")
    ap.add_argument("--metric", choices=sorted(METRIC_COL), default="mean",
                    help="which footprint number to plot (default: mean)")
    ap.add_argument("--variant", default="base", help="variant to plot")
    ap.add_argument("--log", action="store_true",
                    help="single log-scaled y axis instead of a break")
    ap.add_argument("--no-break", dest="allow_break", action="store_false",
                    help="force one linear axis even if the data is split")
    ap.add_argument("--break-ratio", type=float, default=BREAK_RATIO,
                    help="min gap ratio to trigger a break (default %.1f)"
                         % BREAK_RATIO)
    ap.add_argument("--per-panel-scale", action="store_true",
                    help="scale each panel independently (default: shared)")
    ap.add_argument("--no-labels", dest="labels", action="store_false",
                    help="omit the value label above each bar")
    ap.add_argument("--include-bad", action="store_true",
                    help="do not filter on status == ok")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        sys.exit("error: no such file: %s" % args.csv)

    col, ylabel = METRIC_COL[args.metric]
    rows = load(args.csv, args.variant, args.include_bad)
    data, layout = aggregate(rows, col)

    panels = [(m, t) for m, t in MODES if data.get(m)]
    if not panels:
        sys.exit("error: no weak/strong rows found")

    # ---- decide the break, shared across panels unless asked otherwise ----
    def vals_for(mode):
        return [v for per in data[mode].values() for v in per.values()]

    if args.log or not args.allow_break:
        splits = {m: None for m, _ in panels}
    elif args.per_panel_scale:
        splits = {m: find_break(vals_for(m), args.break_ratio)
                  for m, _ in panels}
    else:
        allv = [v for m, _ in panels for v in vals_for(m)]
        shared = find_break(allv, args.break_ratio)
        splits = {m: shared for m, _ in panels}

    # ---- shared limits ----------------------------------------------------
    if args.per_panel_scale:
        limsrc = {m: vals_for(m) for m, _ in panels}
    else:
        allv = [v for m, _ in panels for v in vals_for(m)]
        limsrc = {m: allv for m, _ in panels}

    # ---- figure -----------------------------------------------------------
    fig = plt.figure(figsize=(13.0, 6.2))
    outer = fig.add_gridspec(1, len(panels), wspace=0.16,
                             left=0.075, right=0.985, top=0.885, bottom=0.135)

    legend_handles, legend_labels = [], []

    for pi, (mode, pretty) in enumerate(panels):
        series = data[mode]
        xs = sorted(series)
        impls = [i for i in IMPL_ORDER
                 if any(i in series[r] for r in xs)]
        brk = splits[mode]
        vmax = max(limsrc[mode])
        vmin = min(limsrc[mode])

        if brk is None:
            sub = outer[0, pi].subgridspec(1, 1)
            ax_bot = fig.add_subplot(sub[0, 0])
            ax_top = None
            if args.log:
                ax_bot.set_yscale("log")
                ax_bot.set_ylim(max(vmin * 0.35, 1e-3), vmax * 2.2)
            else:
                ax_bot.set_ylim(0, vmax * 1.16)
            axes = [ax_bot]
            split_val = None
        else:
            low_max, high_min = brk
            sub = outer[0, pi].subgridspec(2, 1, height_ratios=[1.9, 1.0],
                                           hspace=0.09)
            ax_top = fig.add_subplot(sub[0, 0])
            ax_bot = fig.add_subplot(sub[1, 0])
            ax_top.set_ylim(high_min * 0.72, vmax * 1.14)
            ax_bot.set_ylim(0, low_max * 1.42)
            axes = [ax_top, ax_bot]
            split_val = high_min * 0.72

        for ax in axes:
            h = draw_bars(ax, xs, series, impls)
            style_axis(ax)
            ax.set_xlim(-0.6, len(xs) - 0.4)
            if not legend_handles:
                for impl in impls:
                    if impl in h:
                        legend_handles.append(h[impl])
                        legend_labels.append(STYLE[impl]["label"])

        # x tick labels live on the bottom axis only
        ticks = []
        for r in xs:
            lay = layout.get((mode, r))
            ticks.append("%d\n%s" % (r, lay) if lay else str(r))
        ax_bot.set_xticks(range(len(xs)))
        ax_bot.set_xticklabels(ticks, fontsize=9)
        ax_bot.set_xlabel("MPI ranks  (nodes x ranks-per-node)", labelpad=6)

        if ax_top is not None:
            ax_top.set_xticks(range(len(xs)))
            ax_top.set_xticklabels([])
            ax_top.spines["bottom"].set_visible(False)
            ax_top.tick_params(axis="x", length=0)
            ax_bot.spines["top"].set_visible(False)
            break_marks(ax_top, ax_bot)
            ax_top.set_title(pretty, fontsize=11.5, pad=9)
        else:
            ax_bot.set_title(pretty, fontsize=11.5, pad=9)

        if args.labels:
            label_bars(ax_top, ax_bot, xs, series, impls, split_val)

        # ratio callout under each cluster, when both impls are present
        if len(impls) == 2:
            for i, r in enumerate(xs):
                a = series[r].get("darray")
                b = series[r].get("graph")
                if not a or not b:
                    continue
                ax_bot.annotate("%.0fx" % (a / b), xy=(i, 0),
                                xytext=(0, -30), textcoords="offset points",
                                ha="center", va="top", fontsize=8,
                                color="0.35", annotation_clip=False)

    fig.text(0.012, 0.52, ylabel, rotation=90, va="center", ha="left",
             fontsize=11)
    fig.text(0.5, 0.035,
             "grey values below the axis = darray / graph footprint ratio",
             ha="center", fontsize=8.5, color="0.35")

    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper center",
                   ncol=len(legend_handles), frameon=False,
                   bbox_to_anchor=(0.5, 0.975), fontsize=10)

    title = args.title
    if title is None:
        stamp = os.path.basename(os.path.dirname(os.path.abspath(args.csv)))
        title = "Airport torus DSTM peak memory (%s, %s)" % (stamp, args.metric)
    fig.suptitle(title, fontsize=13.5, y=0.995)

    out = args.out
    if out is None:
        out = os.path.join(os.path.dirname(os.path.abspath(args.csv)),
                           "memory.png")
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print("wrote %s" % out)

    root, ext = os.path.splitext(out)
    if ext.lower() != ".pdf":
        fig.savefig(root + ".pdf", bbox_inches="tight")
        print("wrote %s" % (root + ".pdf"))


if __name__ == "__main__":
    main()
