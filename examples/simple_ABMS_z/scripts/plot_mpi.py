#!/usr/bin/env python3
"""
Plot MPI activity: totals on the left, composition on the right.

Two panels:
  Left  : line chart of total MPI calls per rank per tick vs. ranks
          for each framework.
  Right : stacked bar chart of MPI calls per rank per tick, broken
          down by call category. Grouped bars per rank count, with
          DSTM and Repast side by side. Linear y-axis so the size
          difference between the two stacks is visually literal.

Repast's recv counts are 0 in the wrapper output because Boost.MPI's
internal receive path bypasses the standard MPI_Irecv/MPI_Recv symbols.
The Repast stack should be read as a LOWER BOUND on true MPI activity.

Usage:
    python plot_mpi.py <mpi_summary.csv> [--outdir DIR]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


CATEGORY_STYLE = {
    "p2p_send":    {"label": "P2P send",     "color": "#1f77b4"},
    "p2p_recv":    {"label": "P2P recv",     "color": "#6baed6"},
    "barrier":     {"label": "Barrier",      "color": "#ff7f0e"},
    "collective":  {"label": "Collectives",  "color": "#2ca02c"},
    "progress":    {"label": "Progress",     "color": "#7f7f7f"},
}

FW_STYLE = {
    "dstm":   {"label": "DSTM",       "color": "#1f77b4", "marker": "o"},
    "repast": {"label": "Repast HPC", "color": "#d62728", "marker": "s"},
}


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric = ("ranks", "ppn", "nodes", "nh", "nz", "time", "elapsed_s",
               "isend", "irecv", "send", "recv", "wait", "waitall", "test",
               "barrier", "reduce", "allreduce", "bcast",
               "gather", "allgather", "alltoall",
               "bytes_sent", "bytes_recvd",
               "total_calls", "calls_per_rank_per_tick",
               "bytes_per_rank_per_tick", "n_ranks_seen")
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ranks", "elapsed_s", "calls_per_rank_per_tick"])
    return df


def per_rank_per_tick(row, cols):
    denom = row["n_ranks_seen"] * row["time"]
    if denom <= 0:
        return 0.0
    return sum(row[c] for c in cols) / denom


def compute_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["p2p_send"]   = df.apply(lambda r: per_rank_per_tick(r, ["isend", "send"]),           axis=1)
    df["p2p_recv"]   = df.apply(lambda r: per_rank_per_tick(r, ["irecv", "recv"]),           axis=1)
    df["barrier"]    = df.apply(lambda r: per_rank_per_tick(r, ["barrier"]),                 axis=1)
    df["collective"] = df.apply(lambda r: per_rank_per_tick(r, ["reduce", "allreduce",
                                                                "bcast", "gather",
                                                                "allgather", "alltoall"]),   axis=1)
    df["progress"]   = df.apply(lambda r: per_rank_per_tick(r, ["wait", "waitall", "test"]), axis=1)

    cats = list(CATEGORY_STYLE.keys())
    g = df.groupby(["framework", "ranks"])[cats].mean().reset_index()
    return g


def agg(df: pd.DataFrame, ycol: str) -> pd.DataFrame:
    g = df.groupby(["framework", "ranks"])[ycol].agg(["mean", "min", "max"]).reset_index()
    g["err_lo"] = g["mean"] - g["min"]
    g["err_hi"] = g["max"] - g["mean"]
    return g


def draw_totals(ax, df, ranks_vals):
    """Left panel: total calls/rank/tick vs. ranks."""
    a = agg(df, "calls_per_rank_per_tick")

    for fw, style in FW_STYLE.items():
        sub = a[a["framework"] == fw].sort_values("ranks")
        if sub.empty:
            continue
        ax.errorbar(sub["ranks"], sub["mean"],
                    yerr=[sub["err_lo"], sub["err_hi"]],
                    label=style["label"], color=style["color"],
                    marker=style["marker"], markersize=8, linewidth=1.8,
                    capsize=3)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ranks_vals)
    ax.set_xticklabels([str(int(r)) for r in ranks_vals])
    ax.set_xlabel("MPI ranks (log$_2$ scale)")
    ax.set_ylabel("MPI calls per rank per tick (log scale)")
    ax.set_title("Total MPI calls per rank per tick\n"
                 f"scaling from {int(min(ranks_vals))} to {int(max(ranks_vals))} ranks")
    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.legend(loc="upper left", frameon=True)


def draw_stacked(ax, cat_df, ranks_vals):
    """Right panel: grouped stacked bars per rank count, LINEAR y-axis."""
    n_ranks = len(ranks_vals)
    x = np.arange(n_ranks, dtype=float)
    width = 0.36

    stack_totals = []

    for fw, offset in [("dstm", -width/2 - 0.02),
                       ("repast", +width/2 + 0.02)]:
        sub = cat_df[cat_df["framework"] == fw].set_index("ranks").reindex(ranks_vals)
        bottom = np.zeros(n_ranks)

        for cat, style in CATEGORY_STYLE.items():
            vals = sub[cat].fillna(0.0).to_numpy()
            hatch = "////" if (fw == "repast" and cat == "p2p_recv") else None
            ax.bar(x + offset, vals, width=width, bottom=bottom,
                   color=style["color"], edgecolor="black", linewidth=0.4,
                   hatch=hatch)
            bottom += vals

        stack_totals.extend(bottom.tolist())

        # "D" / "R" tags above each stack, positioned linearly.
        for xi, total in enumerate(bottom):
            if total > 0:
                ax.text(x[xi] + offset, total + max(stack_totals) * 0.015,
                        FW_STYLE[fw]["label"][0],
                        ha="center", va="bottom", fontsize=8,
                        color=FW_STYLE[fw]["color"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(r)) for r in ranks_vals])
    ax.set_xlabel("MPI ranks (D = DSTM (left), R = Repast (right))")
    ax.set_ylabel("MPI calls per rank per tick")

    # Linear y-axis with a bit of headroom above the tallest stack.
    if stack_totals:
        ax.set_ylim(0, max(stack_totals) * 1.15)

    ax.grid(True, axis="y", ls="--", alpha=0.35)

    handles = [mpatches.Patch(facecolor=s["color"], edgecolor="black",
                              label=s["label"])
               for s in CATEGORY_STYLE.values()]
    handles.append(mpatches.Patch(facecolor="#6baed6", edgecolor="black",
                                  hatch="////",
                                  label="Repast recv (undercounted*)"))
    ax.legend(handles=handles, loc="upper left", fontsize=8,
              frameon=True, ncol=2)

    ax.set_title("Per-tick MPI call composition by framework\n"
                 "(stacked by call category; linear scale)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    if not args.csv.is_file():
        sys.exit(f"CSV not found: {args.csv}")

    df = load(args.csv)
    if df.empty:
        sys.exit("no usable rows")

    outdir = args.outdir or args.csv.parent
    outdir.mkdir(parents=True, exist_ok=True)
    base = args.csv.parent.name if args.csv.name == "mpi_summary.csv" else args.csv.stem

    ranks_vals = sorted(int(r) for r in df["ranks"].dropna().unique())
    cat_df = compute_categories(df)

    fixed = []
    for col, fmt in (("nh", "nh={}"), ("nz", "nz={}"),
                     ("time", "T={}"), ("density", "d={}")):
        vals = df[col].dropna().unique() if col in df.columns else []
        if len(vals) == 1:
            v = vals[0]
            if isinstance(v, float) and float(v).is_integer():
                v = int(v)
            fixed.append(fmt.format(v))
    param_note = "   ".join(fixed)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={"width_ratios": [1.0, 1.35]})

    draw_totals(axes[0], df, ranks_vals)
    draw_stacked(axes[1], cat_df, ranks_vals)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.14, wspace=0.28)

    out_path = outdir / f"{base}_mpi_activity.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")

    # Companion numeric table.
    print()
    header = f"{'ranks':>6}  {'fw':>7}  " + \
             "  ".join(f"{c:>10}" for c in CATEGORY_STYLE) + \
             f"  {'total':>10}  {'elapsed_s':>10}"
    print(header)
    for r in ranks_vals:
        for fw in ("dstm", "repast"):
            row = cat_df[(cat_df["framework"] == fw) & (cat_df["ranks"] == r)]
            if row.empty:
                continue
            row = row.iloc[0]
            elapsed = df[(df["framework"] == fw) & (df["ranks"] == r)]["elapsed_s"].mean()
            total = sum(row[c] for c in CATEGORY_STYLE)
            print(f"{r:>6}  {fw:>7}  " +
                  "  ".join(f"{row[c]:>10.3f}" for c in CATEGORY_STYLE) +
                  f"  {total:>10.3f}  {elapsed:>10.4f}")


if __name__ == "__main__":
    main()
