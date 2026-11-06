#!/usr/bin/env python3
"""
Plot weak scaling results for zombies DSTM vs MPI.

Reads a CSV with columns:
    build,ranks,ppn,nodes_used,nl,nh,work,timesteps,trial,elapsed_s,steps_per_s,status

Averages over trials and produces a 1x2 figure:
    (1) elapsed time vs ranks (log-log)
    (2) timesteps/second vs ranks (log-log)

Usage:
    python plot_weak_scaling.py [csv_path] [-o output.png]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BUILD_STYLE = {
    "dstm":  {"color": "#1f77b4", "marker": "o", "label": "DSTM"},
    "fixed": {"color": "red",     "marker": "x", "label": "DSTM_fixed"},
    "mpi":   {"color": "#d62728", "marker": "s", "label": "MPI (idiomatic)"},
}


def load_and_aggregate(csv_path: Path) -> pd.DataFrame:
    """Load CSV, filter to ok rows, and aggregate over trials."""
    df = pd.read_csv(csv_path)

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        sys.exit(f"No ok rows in {csv_path}")

    for col in ["elapsed_s", "steps_per_s", "ranks", "trial"]:
        ok[col] = pd.to_numeric(ok[col], errors="coerce")
    ok = ok.dropna(subset=["elapsed_s", "steps_per_s", "ranks"])

    agg = (
        ok.groupby(["build", "ranks"], as_index=False)
        .agg(
            elapsed_mean=("elapsed_s", "mean"),
            elapsed_std=("elapsed_s", "std"),
            steps_mean=("steps_per_s", "mean"),
            steps_std=("steps_per_s", "std"),
            n_trials=("trial", "count"),
        )
        .sort_values(["build", "ranks"])
        .reset_index(drop=True)
    )
    # std is NaN when only one trial; replace with 0 for plotting.
    agg["elapsed_std"] = agg["elapsed_std"].fillna(0.0)
    agg["steps_std"] = agg["steps_std"].fillna(0.0)
    return agg


def plot_elapsed(ax, agg: pd.DataFrame):
    for build, style in BUILD_STYLE.items():
        sub = agg[agg["build"] == build]
        if sub.empty:
            continue
        ax.errorbar(
            sub["ranks"], sub["elapsed_mean"], yerr=sub["elapsed_std"],
            color=style["color"], marker=style["marker"], label=style["label"],
            capsize=3, linewidth=1.6, markersize=6,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Elapsed time (s)")
    ax.set_title("Wall-clock time (weak scaling)")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend()


def plot_throughput(ax, agg: pd.DataFrame):
    for build, style in BUILD_STYLE.items():
        sub = agg[agg["build"] == build]
        if sub.empty:
            continue
        ax.errorbar(
            sub["ranks"], sub["steps_mean"], yerr=sub["steps_std"],
            color=style["color"], marker=style["marker"], label=style["label"],
            capsize=3, linewidth=1.6, markersize=6,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Timesteps / second")
    ax.set_title("Throughput (higher is better)")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", nargs="?", default="weak_scaling_zombies.csv",
                    type=Path, help="input CSV (default: %(default)s)")
    ap.add_argument("-o", "--out", default="weak_scaling_zombies.png",
                    type=Path, help="output figure path (default: %(default)s)")
    ap.add_argument("--show", action="store_true",
                    help="also display the plot interactively")
    args = ap.parse_args()

    if not args.csv.is_file():
        sys.exit(f"CSV not found: {args.csv}")

    agg = load_and_aggregate(args.csv)

    print("Aggregated results:")
    with pd.option_context("display.max_rows", None,
                           "display.float_format", "{:.4f}".format):
        print(agg.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Zombies weak scaling: DSTM vs idiomatic MPI",
                 fontsize=14, fontweight="bold")

    plot_elapsed(axes[0], agg)
    plot_throughput(axes[1], agg)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nsaved: {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
