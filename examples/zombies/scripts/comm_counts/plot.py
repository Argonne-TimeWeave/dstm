#!/usr/bin/env python3
"""
Plot stacked-bar comm counts from a comm_counts.csv produced by counts.sh.

Each x-tick is a rank count. At each tick, one bar per build (dstm, mpi)
is drawn side-by-side. Within each bar, the comm-op categories are stacked.

By default counts are normalized to *per-rank averages* (raw count / num_ranks)
so weak-scaling comparisons aren't dominated by the linear growth in rank
count. Use --totals to plot raw aggregate counts instead.

Usage:
    python plot_counts.py <comm_counts.csv>
    python plot_counts.py <comm_counts.csv> -o counts.png
    python plot_counts.py <comm_counts.csv> --totals     # raw, not per-rank
    python plot_counts.py <comm_counts.csv> --title "..."
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Columns we treat as comm-op counts (must match parse.py's csv_header()).
# TOTAL is intentionally excluded -- it's the sum of the others.
OP_COLS = ["Recv", "Irecv", "Isend", "Send_h", "Send_b", "Send", "alltoall", "Waitall"]

# Fixed visual order (bottom -> top) and a consistent color per op so
# stacks are comparable across builds and runs. Recvs at the bottom,
# sends in the middle, collectives on top.
STACK_ORDER = ["Recv", "Irecv", "Isend", "Send", "Send_h", "Send_b", "alltoall", "Waitall"]


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
            if row["ranks"] <= 0:
                continue
            for op in OP_COLS:
                v = row.get(op, "")
                try:
                    row[op] = int(v) if v != "" else 0
                except ValueError:
                    row[op] = 0
            rows.append(row)
    return rows


def organize(rows, per_rank):
    """
    Return:
      ranks  : sorted list of unique rank counts
      builds : sorted list of unique build names (dstm first if present)
      data   : dict[(build, ranks)] -> dict[op] -> count (per-rank or total)
    """
    data = {}
    rank_set = set()
    build_set = set()
    for r in rows:
        key = (r["build"], r["ranks"])
        rank_set.add(r["ranks"])
        build_set.add(r["build"])
        # Per-rank normalization happens here. We divide every raw aggregate
        # count by the rank count for that run. Counts become floats; that's
        # fine since matplotlib handles them and a non-integer per-rank value
        # is itself a useful diagnostic (would indicate straggler skew).
        denom = r["ranks"] if per_rank else 1
        data[key] = {op: r.get(op, 0) / denom for op in OP_COLS}

    preferred = ["dstm", "mpi"]
    builds = [b for b in preferred if b in build_set]
    builds += sorted(b for b in build_set if b not in preferred)

    return sorted(rank_set), builds, data


def plot(ranks, builds, data, out_path, per_rank, title):
    n_groups = len(ranks)
    n_builds = len(builds)
    if n_groups == 0 or n_builds == 0:
        print("Nothing to plot (no ok rows in CSV).", file=sys.stderr)
        return 1

    # X positions per group, then per-build offsets within the group.
    x = np.arange(n_groups, dtype=float)
    group_width = 0.8
    bar_width = group_width / n_builds
    offsets = (np.arange(n_builds) - (n_builds - 1) / 2.0) * bar_width

    cmap = plt.get_cmap("tab10")
    op_color = {op: cmap(i % 10) for i, op in enumerate(STACK_ORDER)}

    fig, ax = plt.subplots(figsize=(max(8, 1.0 * n_groups + 2), 6))

    op_handles = {}
    build_handles = {}

    build_hatches = {"dstm": "", "mpi": "//"}
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
            ax.bar(
                xpos, heights, bar_width,
                bottom=bottom,
                color=op_color[op],
                edgecolor=edge_color,
                linewidth=0.4,
                hatch=hatch,
            )
            bottom += heights
            if op not in op_handles:
                op_handles[op] = plt.Rectangle(
                    (0, 0), 1, 1,
                    facecolor=op_color[op],
                    edgecolor=edge_color,
                    linewidth=0.4,
                )

        if build not in build_handles:
            build_handles[build] = plt.Rectangle(
                (0, 0), 1, 1,
                facecolor="lightgray",
                edgecolor=edge_color,
                linewidth=0.4,
                hatch=hatch,
            )

        # Annotate totals above each bar so small ops aren't lost.
        # Use a format that's readable for both integers (totals) and
        # fractional per-rank averages.
        totals = bottom  # final cumulative height
        for xi, t in zip(xpos, totals):
            if t > 0:
                if per_rank:
                    label = "{:.1f}".format(t) if t < 100 else "{:.0f}".format(t)
                else:
                    label = "{:g}".format(t)
                ax.text(
                    xi, t, label,
                    ha="center", va="bottom",
                    fontsize=7, rotation=0,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("MPI ranks")
    if per_rank:
        ax.set_ylabel("Communication count per rank (avg)")
    else:
        ax.set_ylabel("Communication count (total across ranks)")
    # Linear y only; log scaling on stacks misrepresents proportions.
    ax.set_ylim(bottom=0)
    ax.set_title(title)

    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

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
                    help="Output image path (e.g. counts.png). "
                         "If omitted, opens an interactive window.")
    ap.add_argument("--totals", action="store_true",
                    help="Plot raw aggregate counts (sum across all ranks) "
                         "instead of the default per-rank average.")
    ap.add_argument("--title", default=None,
                    help="Override plot title (default: derived from CSV path).")
    args = ap.parse_args()

    per_rank = not args.totals

    rows = load_rows(args.csv)
    ranks, builds, data = organize(rows, per_rank=per_rank)

    suffix = "per rank" if per_rank else "totals"
    default_title = "Zombies: comm counts ({}) vs ranks  ({})".format(
        suffix, Path(args.csv).parent.name or args.csv
    )
    title = args.title or default_title

    return plot(ranks, builds, data, args.output, per_rank, title)


if __name__ == "__main__":
    sys.exit(main())
