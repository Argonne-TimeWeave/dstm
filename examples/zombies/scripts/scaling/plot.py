#!/usr/bin/env python3
"""
Plot weak scaling results for zombies DSTM vs MPI.

Reads a CSV with columns:
    build,ranks,ppn,nodes_used,nl,nh,work,timesteps,trial,elapsed_s,steps_per_s,status

Averages over trials and produces a 2x2 figure:
    (1) elapsed time vs ranks (log-log)
    (2) timesteps/second vs ranks (log-log)
    (3) speedup of MPI relative to DSTM vs ranks
    (4) trial-to-trial variability (coefficient of variation) vs ranks

Usage:
    python plot_weak_scaling.py [csv_path] [-o output.png]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BUILD_STYLE = {
    "dstm": {"color": "#1f77b4", "marker": "o", "label": "DSTM"},
    "mpi":  {"color": "#d62728", "marker": "s", "label": "MPI (idiomatic)"},
}


def load_and_aggregate(csv_path: Path) -> pd.DataFrame:
    """Load CSV, filter to ok rows, and aggregate over trials."""
    df = pd.read_csv(csv_path)

    # Keep only successful runs.
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        sys.exit(f"No ok rows in {csv_path}")

    # Coerce numeric.
    for col in ["elapsed_s", "steps_per_s", "ranks", "trial"]:
        ok[col] = pd.to_numeric(ok[col], errors="coerce")
    ok = ok.dropna(subset=["elapsed_s", "steps_per_s", "ranks"])

    # Aggregate over trials: mean + std for both metrics.
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
    # Coefficient of variation (relative noise) — handy diagnostic.
    agg["elapsed_cv"] = np.where(
        agg["elapsed_mean"] > 0,
        agg["elapsed_std"] / agg["elapsed_mean"],
        0.0,
    )
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


def plot_relative_speedup(ax, agg: pd.DataFrame):
    """Plot DSTM elapsed / MPI elapsed. >1 means MPI is faster."""
    dstm = agg[agg["build"] == "dstm"].set_index("ranks")
    mpi = agg[agg["build"] == "mpi"].set_index("ranks")
    common = sorted(set(dstm.index) & set(mpi.index))
    if not common:
        ax.text(0.5, 0.5, "no common rank counts",
                ha="center", va="center", transform=ax.transAxes)
        return

    ratio = np.array([dstm.loc[r, "elapsed_mean"] / mpi.loc[r, "elapsed_mean"]
                      for r in common])

    ax.axhline(1.0, color="grey", ls="--", linewidth=1, label="parity")
    ax.plot(common, ratio, color="#2ca02c", marker="D",
            linewidth=1.8, markersize=7, label="DSTM time / MPI time")

    # Shade who wins.
    ax.fill_between(common, ratio, 1.0,
                    where=(ratio > 1.0), alpha=0.15, color="#d62728",
                    label="MPI faster")
    ax.fill_between(common, ratio, 1.0,
                    where=(ratio < 1.0), alpha=0.15, color="#1f77b4",
                    label="DSTM faster")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("DSTM elapsed / MPI elapsed")
    ax.set_title("Relative cost (>1: MPI wins, <1: DSTM wins)")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(loc="best", fontsize=8)


def plot_variability(ax, agg: pd.DataFrame):
    """Coefficient of variation across trials — diagnostic for noise."""
    for build, style in BUILD_STYLE.items():
        sub = agg[agg["build"] == build]
        if sub.empty:
            continue
        ax.plot(
            sub["ranks"], sub["elapsed_cv"] * 100.0,
            color=style["color"], marker=style["marker"], label=style["label"],
            linewidth=1.6, markersize=6,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Trial-to-trial CV (%)")
    ax.set_title("Run-to-run variability (lower = more stable)")
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

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Zombies weak scaling: DSTM vs idiomatic MPI",
                 fontsize=14, fontweight="bold")

    plot_elapsed(axes[0, 0], agg)
    plot_throughput(axes[0, 1], agg)
    plot_relative_speedup(axes[1, 0], agg)
    plot_variability(axes[1, 1], agg)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nsaved: {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
