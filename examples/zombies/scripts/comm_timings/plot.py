#!/usr/bin/env python3
"""
Plot stacked-bar comm timings from a comm_times.csv produced by run_times.sh.

Each x-tick is a rank count. At each tick, one bar per build (dstm, mpi)
is drawn side-by-side. Bar height = total application runtime (elapsed_s).
The bar is stacked by per-op communication time (seconds per rank), with
the leftover time shown as OTHER (compute + everything not in a tracked op).

Usage:
    python plot_times.py <comm_times.csv>
    python plot_times.py <comm_times.csv> -o times.png
    python plot_times.py <comm_times.csv> --pct          # normalize bars to 100%
    python plot_times.py <comm_times.csv> --title "..."  # custom title
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Comm-op columns we expect from parse_times.py (seconds per rank).
# ISEND/IRECV are included since the zombies MPI version tracks them.
OP_COLS = ["PUT", "GET", "ISEND", "IRECV", "ALLTOALL", "WAITALL"]

# Fixed visual order (bottom -> top) and a consistent color per op.
# OTHER is appended on top to fill the bar up to elapsed_s.
STACK_ORDER = ["PUT", "GET", "ISEND", "IRECV", "ALLTOALL", "WAITALL", "OTHER"]


def _to_float(v):
    try:
        return float(v) if v not in ("", None) else 0.0
    except ValueError:
        return 0.0


def load_rows(path):
    """Return list of dicts, one per CSV row, with float-coerced timing fields."""
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
            row["elapsed_s"] = _to_float(row.get("elapsed_s", ""))
            for op in OP_COLS:
                row[op] = _to_float(row.get("{}_s_per_rank".format(op.lower()), ""))
            rows.append(row)
    return rows


def organize(rows):
    """
    Return:
      ranks  : sorted list of unique rank counts
      builds : list of unique build names (dstm first, mpi second, rest sorted)
      data   : dict[(build, ranks)] -> {"elapsed": float, "ops": {op: seconds}}
    """
    data = {}
    rank_set = set()
    build_set = set()
    for r in rows:
        key = (r["build"], r["ranks"])
        rank_set.add(r["ranks"])
        build_set.add(r["build"])
        # Last write wins on duplicates.
        data[key] = {
            "elapsed": r["elapsed_s"],
            "ops": {op: r[op] for op in OP_COLS},
        }

    preferred = ["dstm", "mpi"]
    builds = [b for b in preferred if b in build_set]
    builds += sorted(b for b in build_set if b not in preferred)

    return sorted(rank_set), builds, data


def plot(ranks, builds, data, out_path, pct_mode, title):
    n_groups = len(ranks)
    n_builds = len(builds)
    if n_groups == 0 or n_builds == 0:
        print("Nothing to plot (no ok rows in CSV).", file=sys.stderr)
        return 1

    x = np.arange(n_groups, dtype=float)
    group_width = 0.8
    bar_width = group_width / n_builds
    offsets = (np.arange(n_builds) - (n_builds - 1) / 2.0) * bar_width

    # Consistent color per stack category. OTHER gets a muted gray.
    # We use tab10 for the tracked ops and reserve gray for OTHER.
    cmap = plt.get_cmap("tab10")
    op_color = {op: cmap(i % 10) for i, op in enumerate(OP_COLS)}
    op_color["OTHER"] = (0.78, 0.78, 0.78, 1.0)

    # Slight visual tuning so DSTM vs MPI bars are distinguishable beyond color.
    build_hatches = {"dstm": "", "mpi": "//"}
    edge_color = "black"

    fig, ax = plt.subplots(figsize=(max(8, 1.0 * n_groups + 2), 6))

    op_handles = {}
    build_handles = {}

    for bi, build in enumerate(builds):
        xpos = x + offsets[bi]
        hatch = build_hatches.get(build, "")

        # Build per-category height arrays for this build across all rank groups.
        elapsed_arr = np.array(
            [data.get((build, r), {}).get("elapsed", 0.0) for r in ranks],
            dtype=float,
        )
        op_arrays = {}
        comm_sum = np.zeros(n_groups)
        for op in OP_COLS:
            arr = np.array(
                [data.get((build, r), {}).get("ops", {}).get(op, 0.0) for r in ranks],
                dtype=float,
            )
            op_arrays[op] = arr
            comm_sum += arr

        # OTHER = elapsed - sum(comm ops). Clamp at 0 in case comm overshoots
        # elapsed (can happen if MPI_Wtime regions overlap or measure wall time
        # on a noisy rank).
        other_arr = np.maximum(elapsed_arr - comm_sum, 0.0)
        op_arrays["OTHER"] = other_arr

        # Optional percent normalization: scale every category by 100/elapsed.
        if pct_mode:
            with np.errstate(divide="ignore", invalid="ignore"):
                scale = np.where(elapsed_arr > 0, 100.0 / elapsed_arr, 0.0)
            for k in op_arrays:
                op_arrays[k] = op_arrays[k] * scale

        bottom = np.zeros(n_groups)
        for op in STACK_ORDER:
            heights = op_arrays[op]
            if not np.any(heights > 0):
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

        # Annotate bar tops. In time mode show seconds; in pct mode show the
        # comm fraction (everything below OTHER) for readability.
        for xi, r in zip(xpos, ranks):
            entry = data.get((build, r))
            if not entry or entry["elapsed"] <= 0:
                continue
            if pct_mode:
                comm_pct = sum(entry["ops"].values()) / entry["elapsed"] * 100.0
                ax.text(
                    xi, 100.5, "{:.0f}%".format(comm_pct),
                    ha="center", va="bottom", fontsize=7,
                )
            else:
                ax.text(
                    xi, entry["elapsed"], "{:.3g}s".format(entry["elapsed"]),
                    ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("MPI ranks")
    if pct_mode:
        ax.set_ylabel("Share of runtime (%)")
        ax.set_ylim(0, 115)
    else:
        ax.set_ylabel("Time per rank (s)")
        # Linear y only; log scaling on stacks is misleading.
        ax.set_ylim(bottom=0)

    ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    # Two legends: which op (color) and which build (hatch).
    op_legend = ax.legend(
        [op_handles[op] for op in STACK_ORDER if op in op_handles],
        [op for op in STACK_ORDER if op in op_handles],
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
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("csv", help="Path to comm_times.csv")
    ap.add_argument("-o", "--output",
                    help="Output image path (e.g. times.png). "
                         "If omitted, opens an interactive window.")
    ap.add_argument("--pct", action="store_true",
                    help="Normalize each bar to 100%% of its run's elapsed time "
                         "instead of plotting absolute seconds.")
    ap.add_argument("--title", default=None,
                    help="Override plot title (default: derived from CSV path).")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    ranks, builds, data = organize(rows)

    default_title = "Zombies: comm time vs ranks  ({})".format(
        Path(args.csv).parent.name or args.csv
    )
    if args.pct:
        default_title = "Zombies: comm time share vs ranks  ({})".format(
            Path(args.csv).parent.name or args.csv
        )
    title = args.title or default_title

    return plot(ranks, builds, data, args.output, args.pct, title)


if __name__ == "__main__":
    sys.exit(main())
