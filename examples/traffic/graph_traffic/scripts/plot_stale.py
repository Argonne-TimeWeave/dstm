#!/usr/bin/env python3
"""
plot_stale.py -- plot mean trip duration vs. stale-delay for graph_traffic.

Reads a runs.csv produced by scale_stale.sh and draws:
  * a line of mean_dur vs. stale_delay for the swept `stale` mode
  * horizontal dotted baselines for the `oracle` and `blind` modes
    (these ignore stale_delay, so they appear as flat reference lines)

If a mode has multiple reps per stale_delay, the median is plotted.

USAGE
    python plot_stale.py runs.csv
    python plot_stale.py runs.csv --out duration.png
    python plot_stale.py runs.csv --show
"""

import argparse
import csv
import sys
from collections import defaultdict
from statistics import median


def load_rows(path):
    """Read the CSV, keeping only rows with status == ok."""
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            if r.get("status") != "ok":
                continue
            rows.append(r)
    if not rows:
        sys.exit(f"error: no rows with status=ok found in {path}")
    return rows


def median_by_delay(rows, mode):
    """
    For the given mode, return (xs, ys) sorted by stale_delay, where ys are
    the median mean_dur across reps at each stale_delay.
    """
    buckets = defaultdict(list)
    for r in rows:
        if r["mode"] != mode:
            continue
        try:
            sd = int(r["stale_delay"])
            dur = float(r["mean_dur"])
        except (ValueError, KeyError):
            continue
        buckets[sd].append(dur)
    xs = sorted(buckets)
    ys = [median(buckets[x]) for x in xs]
    return xs, ys


def baseline_median(rows, mode):
    """Median mean_dur across all reps for a mode that ignores stale_delay."""
    durs = []
    for r in rows:
        if r["mode"] != mode:
            continue
        try:
            durs.append(float(r["mean_dur"]))
        except (ValueError, KeyError):
            continue
    return median(durs) if durs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="runs.csv produced by scale_stale.sh")
    ap.add_argument("--out", default=None,
                    help="output image path (default: <csv-dir>/mean_dur_vs_stale.png)")
    ap.add_argument("--show", action="store_true", help="display the plot window")
    ap.add_argument("--dpi", type=int, default=150, help="output DPI (default 150)")
    args = ap.parse_args()

    # Import here so --help works even without matplotlib installed.
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_rows(args.csv)

    # Swept stale line.
    xs, ys = median_by_delay(rows, "stale")
    if not xs:
        sys.exit("error: no `stale` rows to plot")

    # Baselines (may be absent).
    oracle = baseline_median(rows, "oracle")
    blind = baseline_median(rows, "blind")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(xs, ys, marker="o", linewidth=2, color="tab:blue",
            label="stale (swept)")

    xmin, xmax = min(xs), max(xs)
    if oracle is not None:
        ax.hlines(oracle, xmin, xmax, colors="tab:green", linestyles=":",
                  linewidth=2, label=f"oracle (always-fresh) = {oracle:.1f}")
    if blind is not None:
        ax.hlines(blind, xmin, xmax, colors="tab:red", linestyles=":",
                  linewidth=2, label=f"blind (never-merge) = {blind:.1f}")

    ax.set_xlabel("stale-delay (ticks between merges)")
    ax.set_ylabel("mean trip duration")
    ax.set_title("graph_traffic: mean trip duration vs. stale-delay\n(lower is better)")
    ax.set_xticks(xs)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    if args.show:
        plt.show()

    out = args.out
    if out is None:
        import os
        out = os.path.join(os.path.dirname(os.path.abspath(args.csv)),
                           "mean_dur_vs_stale.png")
    fig.savefig(out, dpi=args.dpi)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
