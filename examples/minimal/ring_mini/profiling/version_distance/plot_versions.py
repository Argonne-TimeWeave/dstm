#!/usr/bin/env python3
"""
Plot v_avg (avg active versions) vs dist, from the version_distance sweep CSV.

Usage:
  python plot_dist_versions.py <summary_csv> [-o out.png]
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load(csv_path):
    dists = []
    v_avgs = []
    v_mins = []
    v_maxs = []
    skipped = 0
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("status", "").startswith("ok"):
                skipped += 1
                continue
            try:
                dists.append(int(row["dist"]))
                v_avgs.append(float(row["v_avg"]))
                v_mins.append(int(row["v_min"]))
                v_maxs.append(int(row["v_max"]))
            except (KeyError, ValueError):
                skipped += 1
                continue
    return dists, v_avgs, v_mins, v_maxs, skipped


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path, help="summary CSV from run_versions.sh")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output image path (default: <csv stem>.png)")
    ap.add_argument("--title", default=None, help="plot title override")
    ap.add_argument("--show", action="store_true", help="show interactively")
    args = ap.parse_args(argv[1:])

    if not args.csv.is_file():
        print("ERROR: not a file: {}".format(args.csv), file=sys.stderr)
        return 1

    dists, v_avgs, v_mins, v_maxs, skipped = load(args.csv)
    if not dists:
        print("ERROR: no usable rows in CSV.", file=sys.stderr)
        return 1
    if skipped:
        print("note: skipped {} non-ok / unparseable row(s)".format(skipped),
              file=sys.stderr)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Shade the min–max range
    ax.fill_between(dists, v_mins, v_maxs, alpha=0.15, color="steelblue",
                    label="min–max range")
    # Plot v_avg as the primary line
    ax.plot(dists, v_avgs, marker="o", color="steelblue", linewidth=2,
            label="v_avg")

    ax.set_xticks(dists)
    ax.set_xlabel("dist (--dist parameter)")
    ax.set_ylabel("Active versions (per rank-tick)")
    ax.set_title(args.title or
                 "Active versions vs communication distance\n{}".format(
                     args.csv.name))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    out = args.out or args.csv.with_suffix(".png")
    fig.savefig(out, dpi=150)
    print("wrote {}".format(out))
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
