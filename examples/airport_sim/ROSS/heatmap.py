#!/usr/bin/env python3
"""
Heatmap of ROSS optimistic-mode event-processing efficiency (%)
across (ranks x planes-per-airport).

Efficiency reported by ROSS is (net committed events) / (total events processed) * 100.
Negative values indicate more rollback work than committed work.

Usage:
    python plot_ross_efficiency_heatmap.py <csv> [-o out.png] [--title "..."]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def load_efficiency(csv_path):
    df = pd.read_csv(csv_path)

    # Keep only ROSS optimistic, successful, non-warmup runs.
    df = df[(df["build"] == "ross") & (df["sync"] == "opt")]
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    if "warmup" in df.columns:
        df["warmup"] = pd.to_numeric(df["warmup"], errors="coerce").fillna(0).astype(int)
        df = df[df["warmup"] == 0]

    for col in ("ranks", "planes", "ross_efficiency"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ranks", "planes", "ross_efficiency"])

    if df.empty:
        sys.exit("ERROR: no ROSS(opt) rows with efficiency found in CSV")

    # Median across trials.
    pivot = (df.groupby(["ranks", "planes"])["ross_efficiency"]
               .median()
               .reset_index()
               .pivot(index="ranks", columns="planes", values="ross_efficiency"))
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    return pivot


def plot_heatmap(pivot, out_path, title):
    data = pivot.values.astype(float)

    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
    # Diverging colormap centered at 0. Guard against degenerate ranges.
    vmin_eff = min(vmin, -1.0)
    vmax_eff = max(vmax, 1.0)
    norm = TwoSlopeNorm(vmin=vmin_eff, vcenter=0.0, vmax=vmax_eff)
    cmap = plt.get_cmap("RdYlGn")

    n_rows, n_cols = data.shape
    fig, ax = plt.subplots(figsize=(1.0 * n_cols + 2.2, 0.7 * n_rows + 1.8))

    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)

    # Ticks / labels.
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([str(int(c)) for c in pivot.columns])
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([str(int(r)) for r in pivot.index])
    ax.set_xlabel("planes per airport")
    ax.set_ylabel("ranks")
    ax.set_title(title)

    # Cell annotations. Pick text color for contrast against cell shade.
    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=9, color="black")
                continue
            # Normalized value in [0,1] via the same norm as the image.
            nv = norm(v)
            # Darken text on very light middle cells, lighten on saturated ends.
            text_color = "black" if 0.25 < nv < 0.85 else "white"
            # Round to nearest int for compactness; use one decimal if tiny.
            label = f"{v:.0f}" if abs(v) >= 10 else f"{v:.1f}"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=9, color=text_color)

    # Gridlines between cells.
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("efficiency (%)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="raw crossover CSV")
    ap.add_argument("-o", "--output", default=None,
                    help="output PNG (default: <csv-basename>_ross_opt_efficiency.png)")
    ap.add_argument("--title",
                    default="ROSS(opt) event-processing efficiency (%)",
                    help="figure title")
    args = ap.parse_args()

    pivot = load_efficiency(args.csv)

    if args.output:
        out = args.output
    else:
        base = os.path.splitext(args.csv)[0]
        out = f"{base}_ross_opt_efficiency.png"

    plot_heatmap(pivot, out, args.title)


if __name__ == "__main__":
    main()
