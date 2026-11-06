#!/usr/bin/env python3
"""
plot_sync_comm_stacked.py — Stacked bar chart of sync/comm vs work time.

Reads the summary.csv files from weak_timing and strong_timing results
and produces a two-panel figure (weak | strong) with grouped stacked bars:
  - DSTM (left bar):   comm_per_rank_s  vs  (elapsed_s - comm_per_rank_s)
  - Repast (right bar): sync_per_rank_s vs  (elapsed_s - sync_per_rank_s)

Usage:
    python plot_sync_comm_stacked.py \
        --weak  results/weak_timing_*/summary.csv \
        --strong results/strong_timing_*/summary.csv \
        [--font-scale 1.0] \
        [-o overhead_stacked.pdf]
"""

import argparse
import sys

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Defaults ────────────────────────────────────────────────────────────
BASE_FONT = 11

COLORS = {
    "comm":  "#d62728",   # red   — DSTM communication
    "sync":  "#1f77b4",   # blue  — Repast synchronization
    "work":  "#2ca02c",   # green — useful work / other
}

FW_STYLE = {
    "dstm":   {"label": "DSTM",   "color": "#d62728"},
    "timing": {"label": "Repast", "color": "#1f77b4"},
}


# ── Data loading ────────────────────────────────────────────────────────
def load_and_average(csv_path: str) -> pd.DataFrame:
    """Load a summary.csv and return per-ranks means for both frameworks."""
    df = pd.read_csv(csv_path)

    # Coerce numeric columns (NA strings -> NaN).
    for col in ["elapsed_s", "sync_per_rank_s", "comm_per_rank_s"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    rows = []
    for fw in ["dstm", "timing"]:
        sub = df[df["framework"] == fw].copy()
        if sub.empty:
            continue
        grp = sub.groupby("ranks", as_index=False).agg(
            elapsed_s=("elapsed_s", "mean"),
            sync_per_rank_s=("sync_per_rank_s", "mean"),
            comm_per_rank_s=("comm_per_rank_s", "mean"),
        )
        grp["framework"] = fw
        rows.append(grp)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ── Drawing ─────────────────────────────────────────────────────────────
def draw_panel(ax, avg_df, ranks_vals, title: str):
    """One panel: grouped stacked bars per rank count, y = time (seconds)."""
    n_ranks = len(ranks_vals)
    x = np.arange(n_ranks, dtype=float)
    width = 0.36

    stack_totals = []
    tag_fontsize = plt.rcParams["font.size"] * 0.75

    for fw, offset in [("dstm", -width / 2 - 0.02),
                       ("timing", +width / 2 + 0.02)]:
        sub = avg_df[avg_df["framework"] == fw].set_index("ranks").reindex(ranks_vals)

        elapsed = sub["elapsed_s"].fillna(0.0).to_numpy()

        if fw == "dstm":
            overhead = sub["comm_per_rank_s"].fillna(0.0).to_numpy()
            overhead_color = COLORS["comm"]
            overhead_label = "Communication"
        else:
            overhead = sub["sync_per_rank_s"].fillna(0.0).to_numpy()
            overhead_color = COLORS["sync"]
            overhead_label = "Synchronization"

        work = np.maximum(elapsed - overhead, 0.0)

        # Stack: work on bottom, overhead on top.
        ax.bar(x + offset, work, width=width, bottom=0,
               color=COLORS["work"], edgecolor="black", linewidth=0.6)
        ax.bar(x + offset, overhead, width=width, bottom=work,
               color=overhead_color, edgecolor="black", linewidth=0.6)

        tops = work + overhead
        stack_totals.extend(tops.tolist())

        # Framework tag above each stack.
        for xi, total in enumerate(tops):
            if total > 0:
                ax.text(x[xi] + offset, total + max(stack_totals) * 0.015,
                        FW_STYLE[fw]["label"][0],
                        ha="center", va="bottom", fontsize=tag_fontsize,
                        color=FW_STYLE[fw]["color"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(r)) for r in ranks_vals])
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Time (seconds)")

    if stack_totals:
        ax.set_ylim(0, max(stack_totals) * 1.25)

    ax.grid(True, axis="y", ls="--", alpha=0.35)
    ax.set_title(title)


def build_legend(fig):
    """Shared legend across both panels."""
    handles = [
        mpatches.Patch(facecolor=COLORS["comm"], edgecolor="black",
                       label="DSTM communication"),
        mpatches.Patch(facecolor=COLORS["sync"], edgecolor="black",
                       label="Repast synchronization"),
        mpatches.Patch(facecolor=COLORS["work"], edgecolor="black",
                       label="Work / other"),
    ]
    fig.legend(handles=handles, loc="upper center",
               ncol=3, frameon=True,
               bbox_to_anchor=(0.5, 1.02))


# ── Main ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Stacked bar chart: sync/comm overhead vs work time.")
    ap.add_argument("--weak", required=True,
                    help="Path to weak-scaling summary.csv")
    ap.add_argument("--strong", required=True,
                    help="Path to strong-scaling summary.csv")
    ap.add_argument("-o", "--output", default="overhead_stacked.png",
                    help="Output file (default: overhead_stacked.png)")
    ap.add_argument("--font-scale", type=float, default=1.0,
                    help=f"multiplier on base font size ({BASE_FONT}pt); "
                         "use e.g. 1.2 to go even bigger, 0.8 for smaller")
    args = ap.parse_args()

    # Global font scaling.
    scaled = BASE_FONT * args.font_scale
    plt.rcParams.update({
        "font.size":        scaled,
        "axes.titlesize":   scaled * 1.15,
        "axes.labelsize":   scaled,
        "xtick.labelsize":  scaled * 0.9,
        "ytick.labelsize":  scaled * 0.9,
        "legend.fontsize":  scaled * 0.85,
    })

    weak_df = load_and_average(args.weak)
    strong_df = load_and_average(args.strong)

    if weak_df.empty and strong_df.empty:
        print("No data found in either CSV.", file=sys.stderr)
        sys.exit(1)

    # Determine rank values present in each dataset.
    weak_ranks = np.sort(weak_df["ranks"].unique()) if not weak_df.empty else np.array([])
    strong_ranks = np.sort(strong_df["ranks"].unique()) if not strong_df.empty else np.array([])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if len(weak_ranks) > 0:
        draw_panel(axes[0], weak_df, weak_ranks, "Weak scaling")
    else:
        axes[0].set_visible(False)

    if len(strong_ranks) > 0:
        draw_panel(axes[1], strong_df, strong_ranks, "Strong scaling")
    else:
        axes[1].set_visible(False)

    build_legend(fig)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
