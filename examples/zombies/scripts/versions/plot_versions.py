#!/usr/bin/env python3
"""
Plot v_avg (avg active versions per rank-tick) vs rank count, one line per build.

Usage:
  python plot_versions.py <summary_csv> [-o out.png]
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


def load(csv_path):
    series = defaultdict(list)  # (build, ranks) -> list of v_avg
    skipped = 0
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("status", "").startswith("ok"):
                skipped += 1
                continue
            try:
                build = row["build"]
                ranks = int(row["ranks"])
                v_avg = float(row["v_avg"])
            except (KeyError, ValueError):
                skipped += 1
                continue
            series[(build, ranks)].append(v_avg)
    return series, skipped


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path, help="summary CSV from run_versions_*.sh")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output image path (default: <csv stem>.png)")
    ap.add_argument("--title", default=None, help="plot title override")
    ap.add_argument("--show", action="store_true", help="show interactively")
    args = ap.parse_args(argv[1:])

    if not args.csv.is_file():
        print("ERROR: not a file: {}".format(args.csv), file=sys.stderr)
        return 1

    series, skipped = load(args.csv)
    if not series:
        print("ERROR: no usable rows in CSV.", file=sys.stderr)
        return 1
    if skipped:
        print("note: skipped {} non-ok / unparseable row(s)".format(skipped),
              file=sys.stderr)

    builds = defaultdict(list)  # build -> list of (ranks, avg, err)
    for (build, ranks), vals in series.items():
        avg = mean(vals)
        err = stdev(vals) if len(vals) > 1 else 0.0
        builds[build].append((ranks, avg, err))
    for b in builds:
        builds[b].sort(key=lambda t: t[0])

    fig, ax = plt.subplots(figsize=(8, 5))
    for build in sorted(builds):
        pts = builds[build]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        errs = [p[2] for p in pts]
        if any(e > 0 for e in errs):
            ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=3, label=build)
        else:
            ax.plot(xs, ys, marker="o", label=build)

    ax.set_xscale("log", base=2)
    all_ranks = sorted({r for pts in builds.values() for r, _, _ in pts})
    ax.set_xticks(all_ranks)
    ax.set_xticklabels([str(r) for r in all_ranks])

    ax.set_xlabel("Ranks")
    ax.set_ylabel("Average active versions (per rank-tick)")
    ax.set_title(args.title or "Active versions vs rank count\n{}".format(args.csv.name))
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(title="build")

    fig.tight_layout()
    out = args.out or args.csv.with_suffix(".png")
    fig.savefig(out, dpi=150)
    print("wrote {}".format(out))
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
