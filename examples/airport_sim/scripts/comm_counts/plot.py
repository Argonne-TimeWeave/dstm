#!/usr/bin/env python3
"""
Plot stacked-bar comm counts from a comm_counts.csv produced by counts.sh.

Each x-tick is a rank count. At each tick, one bar per build (dstm, mpi)
is drawn side-by-side. Within each bar, the comm-op categories are stacked.

Usage:
    python plot_counts.py <comm_counts.csv>
    python plot_counts.py <comm_counts.csv> -o counts.png
    python plot_counts.py <comm_counts.csv> --log-y
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Columns we treat as comm-op counts (must match parse.py's csv_header()).
# TOTAL is intentionally excluded -- it's the sum of the others.
OP_COLS = ["Recv", "Irecv", "Isend", "Send_h", "Send_b", "Send", "alltoall"]

# Fixed visual order (bottom -> top) and a consistent color per op so
# stacks are comparable across builds and runs.
STACK_ORDER = ["Recv", "Irecv", "Isend", "Send", "Send_h", "Send_b", "alltoall"]


def load_rows(path):
    """Return list of dicts, one per CSV row, with int-coerced count fields."""
    rows = []
    with open(path, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if row.get("status", "") != "ok":
                continue
            try:
                row["ranks"] = int(row["ranks"])
            except (KeyError, ValueError):
                continue
            for op in OP_COLS:
                v = row.get(op, "")
                try:
                    row[op] = int(v) if v != "" else 0
                except ValueError:
                    row[op] = 0
            rows.append(row)
    return rows


def organize(rows):
    """
    Return:
      ranks  : sorted list of unique rank counts
      builds : sorted list of unique build names
      data   : dict[(build, ranks)] -> dict[op] -> count
    """
    data = {}
    rank_set = set()
    build_set = set()
    for r in rows:
        key = (r["build"], r["ranks"])
        rank_set.add(r["ranks"])
        build_set.add(r["build"])
        # If duplicates exist (e.g. you reran a single point), last write wins.
        data[key] = {op: r.get(op, 0) for op in OP_COLS}

    # Put dstm before mpi if both present; otherwise alphabetical.
    preferred = ["dstm", "mpi"]
    builds = [b for b in preferred if b in build_set]
    builds += sorted(b for b in build_set if b not in preferred)

    return sorted(rank_set), builds, data


def plot(ranks, builds, data, out_path, log_y, title):
    n_groups = len(ranks)
    n_builds = len(builds)
    if n_groups == 0 or n_builds == 0:
        print("Nothing to plot (no ok rows in CSV).", file=sys.stderr)
        return 1

    # X positions per group, then per-build offsets within the group.
    x = np.arange(n_groups, dtype=float)
    group_width = 0.8
    bar_width = group_width / n_builds
    # Offsets center the cluster on the tick.
    offsets = (np.arange(n_builds) - (n_builds - 1) / 2.0) * bar_width

    # Consistent color per op across builds/runs. tab10 has 10 distinct colors.
    cmap = plt.get_cmap("tab10")
    op_color = {op: cmap(i % 10) for i, op in enumerate(STACK_ORDER)}

    fig, ax = plt.subplots(figsize=(max(8, 1.0 * n_groups + 2), 6))

    # For the legend we want each op to appear once, and each build to appear once.
    op_handles = {}
    build_handles = {}

    # Patterns to visually distinguish builds (in addition to position).
    build_hatches = {"dstm": "", "mpi": "//"}
    # Edge color helps when hatches are off.
    edge_color = "black"

    for bi, build in enumerate(builds):
        xpos = x + offsets[bi]
        hatch = build_hatches.get(build, "")
        bottom = np.zeros(n_groups)

        for op in STACK_ORDER:
            heights = np.array(
                [data.get((build, r), {}).get(op, 0) for r in ranks],
                dtype=float,
            )
            if not np.any(heights):
                continue
            bars = ax.bar(
                xpos, heights, bar_width,
                bottom=bottom,
                color=op_color[op],
                edgecolor=edge_color,
                linewidth=0.4,
                hatch=hatch,
            )
            bottom += heights
            if op not in op_handles:
                # Plain patch (no hatch) for the op legend entry.
                op_handles[op] = plt.Rectangle(
                    (0, 0), 1, 1,
                    facecolor=op_color[op],
                    edgecolor=edge_color,
                    linewidth=0.4,
                )

        # Build-level legend swatch: gray w/ this build's hatch.
        if build not in build_handles:
            build_handles[build] = plt.Rectangle(
                (0, 0), 1, 1,
                facecolor="lightgray",
                edgecolor=edge_color,
                linewidth=0.4,
                hatch=hatch,
            )

        # Annotate totals above each bar so small ops aren't lost.
        totals = bottom  # final cumulative height
        for xi, t in zip(xpos, totals):
            if t > 0:
                ax.text(
                    xi, t, "{:g}".format(t),
                    ha="center", va="bottom",
                    fontsize=7, rotation=0,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Communication count")
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
        ax.set_ylim(bottom=1)

    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    # Two legends: one for ops (stack categories), one for builds (clusters).
    op_legend = ax.legend(
        list(op_handles.values()), list(op_handles.keys()),
        title="Op", loc="upper left", fontsize=8, title_fontsize=9,
    )
    ax.add_artist(op_legend)
    ax.legend(
        list(build_handles.values()), list(build_handles.keys()),
        title="Build", loc="upper right", fontsize=8, title_fontsize=9,
    )

    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        print("Wrote {}".format(out_path))
    else:
        plt.show()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="Path to comm_counts.csv")
    ap.add_argument("-o", "--output",
                    help="Output image path (e.g. counts.png). If omitted, opens an interactive window.")
    ap.add_argument("--log-y", action="store_true",
                    help="Use log scale on the y axis (helps when counts span orders of magnitude across P).")
    ap.add_argument("--title", default=None,
                    help="Override plot title (default: derived from CSV path).")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    ranks, builds, data = organize(rows)

    title = args.title or "Comm counts vs ranks  ({})".format(Path(args.csv).parent.name or args.csv)
    return plot(ranks, builds, data, args.output, args.log_y, title)


if __name__ == "__main__":
    sys.exit(main())
