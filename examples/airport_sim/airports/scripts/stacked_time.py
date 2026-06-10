#!/usr/bin/env python3
"""
Plot a stacked bar chart of where time is spent (get / put / other)
for each rank count in a weak-scaling CSV produced by
weak_scaling_dstm.sh.

Usage:
    python3 plot_weak_scaling.py <csv> [-o out.png] [--show]

Expected columns:
    ranks, elapsed_s, get_avg_s, put_avg_s
(extra columns are ignored). Multiple trials per rank count are averaged.
"""

import argparse
import csv
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    # rank -> list of (elapsed, get_avg, put_avg)
    rows = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"ranks", "elapsed_s", "get_avg_s", "put_avg_s"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"ERROR: CSV is missing columns: {sorted(missing)}")

        for row in reader:
            try:
                r = int(row["ranks"])
                elapsed = float(row["elapsed_s"])
                get_s = float(row["get_avg_s"] or 0.0)
                put_s = float(row["put_avg_s"] or 0.0)
            except ValueError:
                # skip rows with empty/garbled numeric fields
                continue
            rows[r].append((elapsed, get_s, put_s))

    if not rows:
        sys.exit("ERROR: no usable rows found in CSV")

    ranks = sorted(rows.keys())
    elapsed = np.array([np.mean([t[0] for t in rows[r]]) for r in ranks])
    get_s   = np.array([np.mean([t[1] for t in rows[r]]) for r in ranks])
    put_s   = np.array([np.mean([t[2] for t in rows[r]]) for r in ranks])
    other_s = np.maximum(elapsed - get_s - put_s, 0.0)
    return ranks, elapsed, get_s, put_s, other_s


def main():
    ap = argparse.ArgumentParser(description="Stacked bar chart of get/put/other time per rank.")
    ap.add_argument("csv", help="Path to weak_scaling_dstm_*.csv")
    ap.add_argument("-o", "--output", default=None,
                    help="Output image path (default: <csv-stem>.png)")
    ap.add_argument("--show", action="store_true", help="Open an interactive window")
    ap.add_argument("--title", default=None, help="Override plot title")
    args = ap.parse_args()

    ranks, elapsed, get_s, put_s, other_s = load_csv(args.csv)

    x = np.arange(len(ranks))
    width = 0.65

    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(ranks) + 3), 5.5))

    b_get   = ax.bar(x, get_s,   width, label="get_or_default", color="#4C72B0")
    b_put   = ax.bar(x, put_s,   width, bottom=get_s,
                     label="put", color="#DD8452")
    b_other = ax.bar(x, other_s, width, bottom=get_s + put_s,
                     label="other", color="#BFBFBF")

    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Per-rank average time (s)")
    title = args.title or f"Time breakdown vs. ranks  ({args.csv})"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate each bar with total elapsed time.
    ymax = elapsed.max()
    pad = ymax * 0.01
    for xi, total in zip(x, elapsed):
        ax.text(xi, total + pad, f"{total:.3f}s",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, ymax * 1.12)

    fig.tight_layout()

    out = args.output
    if out is None:
        stem = args.csv.rsplit(".", 1)[0]
        out = f"{stem}.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
