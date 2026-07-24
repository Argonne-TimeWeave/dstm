#!/usr/bin/env python3
"""
Plot total MPI communications vs. ranks (single panel, tall aspect).

This script reads mpi_summary.csv and produces a single line plot:
  y = total number of MPI calls (mean with min/max error bars)
  x = MPI ranks (log2 scale)

Usage:
    python plot_mpi_totals.py <mpi_summary.csv> [--outdir DIR] [--font-scale 1.0]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

FW_STYLE = {
    "dstm":   {"label": "DSTM",       "color": "#1f77b4", "marker": "o"},
    "repast": {"label": "Repast HPC", "color": "#d62728", "marker": "s"},
}

BASE_FONT = 18


def configure_fonts(scale: float = 1.0):
    s = BASE_FONT * scale
    plt.rcParams.update({
        "font.size":         s,
        "axes.titlesize":    s * 1.05,
        "axes.labelsize":    s * 1.05,
        "xtick.labelsize":   s * 0.95,
        "ytick.labelsize":   s * 0.95,
        "legend.fontsize":   s * 0.85,
        "legend.title_fontsize": s * 0.95,
        "figure.titlesize":  s * 1.15,
        "lines.linewidth":   2.4,
        "lines.markersize":  10,
        "axes.linewidth":    1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size":  6,
        "ytick.major.size":  6,
        "xtick.minor.width": 0.9,
        "ytick.minor.width": 0.9,
    })


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    for c in ("ranks", "total_calls", "elapsed_s"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    need = ["framework", "ranks", "total_calls"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"CSV missing required columns: {', '.join(missing)}")

    df = df.dropna(subset=["framework", "ranks", "total_calls"])
    return df


def agg(df: pd.DataFrame, ycol: str) -> pd.DataFrame:
    g = df.groupby(["framework", "ranks"])[ycol].agg(["mean", "min", "max"]).reset_index()
    g["err_lo"] = g["mean"] - g["min"]
    g["err_hi"] = g["max"] - g["mean"]
    return g


def draw_total_calls(ax, df: pd.DataFrame, ranks_vals):
    a = agg(df, "total_calls")

    for fw, style in FW_STYLE.items():
        sub = a[a["framework"] == fw].sort_values("ranks")
        if sub.empty:
            continue
        ax.errorbar(
            sub["ranks"], sub["mean"],
            yerr=[sub["err_lo"], sub["err_hi"]],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            markersize=plt.rcParams["lines.markersize"],
            linewidth=plt.rcParams["lines.linewidth"],
            capsize=5, capthick=1.5, elinewidth=1.4,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ranks_vals)
    ax.set_xticklabels([str(int(r)) for r in ranks_vals])

    ax.set_xlabel("MPI ranks (log$_2$ scale)")
    ax.set_ylabel("Total MPI calls (log scale)")
    ax.set_title("Total MPI communications")
    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.legend(loc="upper right", frameon=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--font-scale", type=float, default=1.0,
                    help=f"multiplier on base font size ({BASE_FONT}pt)")
    args = ap.parse_args()

    if not args.csv.is_file():
        sys.exit(f"CSV not found: {args.csv}")

    configure_fonts(args.font_scale)

    df = load(args.csv)
    if df.empty:
        sys.exit("no usable rows")

    outdir = args.outdir or args.csv.parent
    outdir.mkdir(parents=True, exist_ok=True)
    base = args.csv.parent.name if args.csv.name == "mpi_summary.csv" else args.csv.stem

    ranks_vals = sorted(int(r) for r in df["ranks"].dropna().unique())

    # Taller than wide:
    fig, ax = plt.subplots(1, 1, figsize=(9, 10))
    draw_total_calls(ax, df, ranks_vals)

    fig.tight_layout(pad=1.2)
    out_path = outdir / f"{base}_mpi_total_calls.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")

    a = agg(df, "total_calls")
    print()
    print(f"{'ranks':>6}  {'fw':>7}  {'mean_total_calls':>17}  {'min':>12}  {'max':>12}")
    for r in ranks_vals:
        for fw in ("dstm", "repast"):
            row = a[(a["framework"] == fw) & (a["ranks"] == r)]
            if row.empty:
                continue
            row = row.iloc[0]
            print(f"{int(r):>6}  {fw:>7}  {row['mean']:>17.3f}  {row['min']:>12.3f}  {row['max']:>12.3f}")


if __name__ == "__main__":
    main()
