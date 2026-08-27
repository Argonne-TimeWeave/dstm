#!/usr/bin/env python3
"""
plot_scale.py -- plot run time vs. rank count for graph_traffic.

Reads a runs.csv produced by the scaling sweep and draws two lines:
  * stale  -- run_time vs. ranks
  * oracle -- run_time vs. ranks

If a mode has multiple reps at a given rank count, the median run_time is
plotted. The x-axis is ranks (log2), the y-axis is run time.

USAGE
    python plot_scale.py runs.csv
    python plot_scale.py runs.csv --out scaling.png
    python plot_scale.py runs.csv --show
    python plot_scale.py runs.csv --linear-x
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


def median_by_ranks(rows, mode):
    """
    For the given mode, return (xs, ys) sorted by ranks, where ys are the
    median run_time across reps at each rank count.
    """
    buckets = defaultdict(list)
    for r in rows:
        if r["mode"] != mode:
            continue
        try:
            ranks = int(r["ranks"])
            t = float(r["run_time"])
        except (ValueError, KeyError):
            continue
        buckets[ranks].append(t)
    xs = sorted(buckets)
    ys = [median(buckets[x]) for x in xs]
    return xs, ys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="runs.csv produced by the scaling sweep")
    ap.add_argument("--out", default=None,
                    help="output image path (default: <csv-dir>/run_time_vs_ranks.png)")
    ap.add_argument("--show", action="store_true", help="display the plot window")
    ap.add_argument("--linear-x", action="store_true",
                    help="use a linear x-axis instead of log2(ranks)")
    ap.add_argument("--dpi", type=int, default=150, help="output DPI (default 150)")
    args = ap.parse_args()

    # Import here so --help works even without matplotlib installed.
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_rows(args.csv)

    stale_x, stale_y = median_by_ranks(rows, "stale")
    oracle_x, oracle_y = median_by_ranks(rows, "oracle")

    if not stale_x and not oracle_x:
        sys.exit("error: no `stale` or `oracle` rows to plot")

    fig, ax = plt.subplots(figsize=(8, 5))

    if stale_x:
        ax.plot(stale_x, stale_y, marker="o", linewidth=2, color="tab:blue",
                label="stale")
    if oracle_x:
        ax.plot(oracle_x, oracle_y, marker="s", linewidth=2, color="tab:green",
                label="oracle")

    # All rank counts present, for consistent ticks.
    all_ranks = sorted(set(stale_x) | set(oracle_x))

    if not args.linear_x:
        ax.set_xscale("log", base=2)
    ax.set_xticks(all_ranks)
    ax.set_xticklabels([str(r) for r in all_ranks])

    ax.set_xlabel("ranks")
    ax.set_ylabel("run time (s)")
    ax.set_title("graph_traffic: run time vs. ranks\n(lower is better)")
    ax.grid(True, linestyle="--", alpha=0.4, which="both")
    ax.legend()
    fig.tight_layout()

    if args.show:
        plt.show()

    out = args.out
    if out is None:
        import os
        out = os.path.join(os.path.dirname(os.path.abspath(args.csv)),
                           "run_time_vs_ranks.png")
    fig.savefig(out, dpi=args.dpi)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
