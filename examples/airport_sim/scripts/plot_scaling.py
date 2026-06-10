#!/usr/bin/env python3
"""
Plot scaling results from a CSV produced by the airport scaling sweeps.

Usage:
    python plot_scaling.py <csv_file> [--out-prefix OUT]

Produces:
    <prefix>_elapsed.png   — elapsed time vs ranks (log-log)
    <prefix>_throughput.png — timesteps/sec vs ranks (log-log)
"""

import argparse
import csv
import sys
from collections import defaultdict

import matplotlib.pyplot as plt

# Consistent styling per build
STYLE = {
    "dstm":         {"color": "#d62728", "marker": "o", "label": "DSTM"},
    "mpi_alltoall": {"color": "#1f77b4", "marker": "s", "label": "MPI Alltoallv"},
    "mpi_sendrecv": {"color": "#2ca02c", "marker": "^", "label": "MPI Sendrecv (per-pair)"},
}


def load_csv(path):
    """Return {build: [(ranks, elapsed_mean, sps_mean), ...]} sorted by ranks."""
    # builds[build][ranks] = list of (elapsed, sps)
    builds = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":
                continue
            try:
                elapsed = float(row["elapsed_s"])
                sps = float(row["steps_per_s"])
                ranks = int(row["ranks"])
                build = row["build"]
            except (ValueError, KeyError):
                continue
            builds[build][ranks].append((elapsed, sps))

    # Average across trials
    result = {}
    for build, by_rank in builds.items():
        series = []
        for ranks in sorted(by_rank):
            trials = by_rank[ranks]
            n = len(trials)
            mean_elapsed = sum(t[0] for t in trials) / n
            mean_sps = sum(t[1] for t in trials) / n
            series.append((ranks, mean_elapsed, mean_sps))
        result[build] = series
    return result


def plot_series(data, ycol, ylabel, title, outfile):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for build, series in data.items():
        if not series:
            continue
        style = STYLE.get(build, {"color": "gray", "marker": "x", "label": build})
        xs = [s[0] for s in series]
        ys = [s[ycol] for s in series]
        ax.plot(xs, ys,
                color=style["color"],
                marker=style["marker"],
                markersize=7,
                linewidth=2,
                label=style["label"])

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("MPI Ranks (P)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    # Show actual rank values on x-axis (not powers like 2^4)
    all_ranks = sorted({s[0] for series in data.values() for s in series})
    ax.set_xticks(all_ranks)
    ax.set_xticklabels([str(r) for r in all_ranks])

    ax.legend(loc="best", fontsize=11, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"  wrote {outfile}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="path to weak/strong scaling CSV")
    ap.add_argument("--out-prefix", default=None,
                    help="output filename prefix (default: csv basename)")
    args = ap.parse_args()

    data = load_csv(args.csv)
    if not data:
        print(f"No usable rows in {args.csv}", file=sys.stderr)
        sys.exit(1)

    prefix = args.out_prefix or args.csv.rsplit(".", 1)[0]

    plot_series(data, ycol=1,
                ylabel="Elapsed time (s)",
                title="Scaling: elapsed time vs ranks",
                outfile=f"{prefix}_elapsed.png")

    plot_series(data, ycol=2,
                ylabel="Throughput (timesteps / sec)",
                title="Scaling: throughput vs ranks",
                outfile=f"{prefix}_throughput.png")


if __name__ == "__main__":
    main()
