#!/usr/bin/env python3
"""
plot_scale.py -- plot run time vs. rank count for graph_traffic.

Reads a runs.csv produced by the scaling sweep and draws one line per
replica configuration:
  * stale d=<delay>  -- one line per --stale-delay value, in shades of blue
  * oracle           -- run_time vs. ranks (green)
  * blind            -- run_time vs. ranks (gray)

If a configuration has multiple reps at a given rank count, the median
run_time is plotted. The x-axis is ranks (log2 by default), the y-axis is
run time.

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


def median_by_ranks(rows, mode, delay=None):
    """
    For the given mode (and optionally a specific stale_delay), return
    (xs, ys) sorted by ranks, where ys are the median run_time across reps
    at each rank count.

    delay=None means "don't filter on delay" (used for blind/oracle).
    """
    buckets = defaultdict(list)
    for r in rows:
        if r["mode"] != mode:
            continue
        if delay is not None and r.get("stale_delay") != delay:
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


def stale_delays(rows):
    """Return the sorted numeric list of distinct stale_delay values."""
    vals = set()
    for r in rows:
        if r["mode"] != "stale":
            continue
        d = r.get("stale_delay")
        try:
            vals.add(int(d))
        except (TypeError, ValueError):
            continue
    return sorted(vals)


def blue_shades(plt, n):
    """
    Return n distinct shades of blue, light -> dark, avoiding the very
    lightest end of the colormap so all lines stay visible.
    """
    if n <= 0:
        return []
    cmap = plt.cm.Blues
    if n == 1:
        return [cmap(0.75)]
    lo, hi = 0.45, 0.95
    return [cmap(lo + (hi - lo) * i / (n - 1)) for i in range(n)]


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

    delays = stale_delays(rows)
    shades = blue_shades(plt, len(delays))

    fig, ax = plt.subplots(figsize=(8, 5))

    all_ranks = set()
    plotted_any = False

    # One blue line per stale delay (light = small delay, dark = large delay).
    for delay, color in zip(delays, shades):
        xs, ys = median_by_ranks(rows, "stale", delay=str(delay))
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", linewidth=2, color=color,
                label=f"stale d={delay}")
        all_ranks.update(xs)
        plotted_any = True

    # Oracle (green) and blind (gray) -- delay is "na", so don't filter on it.
    oracle_x, oracle_y = median_by_ranks(rows, "oracle")
    if oracle_x:
        ax.plot(oracle_x, oracle_y, marker="s", linewidth=2, color="tab:green",
                label="oracle")
        all_ranks.update(oracle_x)
        plotted_any = True

    blind_x, blind_y = median_by_ranks(rows, "blind")
    if blind_x:
        ax.plot(blind_x, blind_y, marker="^", linewidth=2, color="tab:gray",
                label="blind")
        all_ranks.update(blind_x)
        plotted_any = True

    if not plotted_any:
        sys.exit("error: no stale/oracle/blind rows to plot")

    all_ranks = sorted(all_ranks)

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
