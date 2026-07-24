#!/usr/bin/env python3
"""
Ring-topology drift experiment: max time-step distance vs. number of ranks.

The `p` column (send probability) does not meaningfully correlate with drift,
so we treat rows sharing the same `n` as independent trials and plot
mean ± spread across them.

Usage:
    python plot_ring_drift.py results.csv [-o out.png]
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="results.csv from run_sims.sh")
    ap.add_argument("-o", "--output", default=None,
                    help="output PNG (default: <csv-basename>_drift.png)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    for col in ("n", "avg_max_distance", "max_of_runs"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["n", "avg_max_distance"])

    # Treat each (p) row as an independent trial for a given n.
    agg = (df.groupby("n")["avg_max_distance"]
             .agg(mean="mean", std="std", lo="min", hi="max")
             .reset_index()
             .sort_values("n"))
    # Also track the worst single run observed across all trials at each n.
    worst = df.groupby("n")["max_of_runs"].max().reset_index()
    agg = agg.merge(worst, on="n")

    fig, ax = plt.subplots(figsize=(11, 3.2))

    # Mean drift with min/max whiskers across trials.
    yerr = np.vstack([agg["mean"] - agg["lo"], agg["hi"] - agg["mean"]])
    ax.errorbar(agg["n"], agg["mean"], yerr=yerr,
                marker="o", markersize=7, linewidth=2,
                capsize=4, color="#1f77b4",
                label="mean max drift (± across trials)")

    # Reference line: y = n/2 (ring half-diameter, natural upper bound).
    ns = np.array(sorted(df["n"].unique()))
    ax.plot(ns, ns / 2.0, linestyle=":", color="gray", linewidth=1.2,
            label="n / 2  (ring half-diameter)")

    ax.set_xlabel("ranks in ring (n)")
    ax.set_ylabel("max time-step distance")
    ax.set_xticks(ns)
    ax.grid(True, alpha=0.3)
    # Legend outside the axes on the right so the plot stays wide and short.
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=9, frameon=False)
    ax.set_title("DSTM drift on a ring topology")

    fig.tight_layout()

    if args.output:
        out = args.output
    else:
        base = os.path.splitext(args.csv)[0]
        out = f"{base}_drift.png"

    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
