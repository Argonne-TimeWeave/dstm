#!/usr/bin/env python3
"""
Plot weak-scaling results from the zombie compare.sh summary CSV.

Usage:
    python plot_weak.py <csv-file> [--outdir DIR] [--linear-y]

Expected CSV columns (produced by compare.sh):
    framework,ranks,ppn,nodes,nh,nz,time,density,nl,seed,repeat,
    elapsed_s,tps,status

Produces two PNGs next to the CSV (or in --outdir):
    <base>_elapsed.png   : wall-clock time vs ranks
    <base>_tps.png       : ticks/second vs ranks

Aggregates across repeats with mean; error bars show min/max spread.
Failed rows (status != ok) are dropped with a warning.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Consistent styling per framework so multiple CSVs plot the same.
STYLE = {
    "dstm":   {"label": "DSTM",        "color": "#1f77b4", "marker": "o"},
    "repast": {"label": "Repast HPC",  "color": "#d62728", "marker": "s"},
}


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop failed rows, warn user.
    bad = df[df["status"] != "ok"]
    if not bad.empty:
        print(f"WARNING: dropping {len(bad)} failed row(s):", file=sys.stderr)
        for _, row in bad.iterrows():
            print(
                f"  framework={row['framework']} ranks={row['ranks']} "
                f"nh={row['nh']} nz={row['nz']} time={row['time']} "
                f"repeat={row['repeat']} status={row['status']}",
                file=sys.stderr,
            )
    df = df[df["status"] == "ok"].copy()

    # Coerce numerics (parse-fail rows would have blanks or 'NA').
    for col in ("elapsed_s", "tps", "ranks", "ppn", "nodes",
                "nh", "nz", "time", "density", "repeat"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["elapsed_s", "tps", "ranks"])

    return df


def aggregate(df: pd.DataFrame, ycol: str) -> pd.DataFrame:
    """Group by (framework, ranks); return mean plus min/max for error bars."""
    g = df.groupby(["framework", "ranks"])[ycol].agg(["mean", "min", "max"])
    g = g.reset_index()
    g["err_lo"] = g["mean"] - g["min"]
    g["err_hi"] = g["max"] - g["mean"]
    return g


def plot_metric(df: pd.DataFrame, ycol: str, ylabel: str, title: str,
                out_path: Path, log_y: bool = True):
    agg = aggregate(df, ycol)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Iterate frameworks in a stable order so the legend is deterministic.
    seen = set()
    for framework, style in STYLE.items():
        sub = agg[agg["framework"] == framework].sort_values("ranks")
        if sub.empty:
            continue
        seen.add(framework)
        ax.errorbar(
            sub["ranks"],
            sub["mean"],
            yerr=[sub["err_lo"], sub["err_hi"]],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            markersize=7,
            linewidth=1.8,
            capsize=3,
        )

    # Any frameworks in the CSV that we don't have styling for.
    others = set(agg["framework"]) - seen
    for framework in sorted(others):
        sub = agg[agg["framework"] == framework].sort_values("ranks")
        ax.errorbar(
            sub["ranks"], sub["mean"],
            yerr=[sub["err_lo"], sub["err_hi"]],
            label=str(framework),
            marker="x", markersize=7, linewidth=1.8, capsize=3,
        )

    ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel("MPI ranks (log2)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # X-axis: show every rank count present as its own tick.
    all_ranks = sorted(agg["ranks"].unique())
    ax.set_xticks(all_ranks)
    ax.set_xticklabels([str(int(r)) for r in all_ranks])

    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def build_title(csv_path: Path, df: pd.DataFrame, metric_label: str) -> str:
    """Encode fixed experiment params in the plot title."""
    parts = [metric_label]
    # These columns should be fixed across all rows in a weak-scaling sweep.
    for col, fmt in (
        ("nh",      "nh={}"),
        ("nz",      "nz={}"),
        ("time",    "T={}"),
        ("density", "d={}"),
    ):
        if col in df.columns:
            vals = df[col].dropna().unique()
            if len(vals) == 1:
                v = vals[0]
                # Print ints without trailing .0
                if float(v).is_integer():
                    v = int(v)
                parts.append(fmt.format(v))
    parts.append(f"({csv_path.stem})")
    return "  |  ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, help="input CSV (summary.csv)")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="output directory (default: same as CSV)")
    ap.add_argument("--linear-y", action="store_true",
                    help="use a linear y-axis instead of log")
    args = ap.parse_args()

    if not args.csv.is_file():
        sys.exit(f"error: CSV not found: {args.csv}")

    outdir = args.outdir or args.csv.parent
    outdir.mkdir(parents=True, exist_ok=True)
    base = args.csv.parent.name if args.csv.name == "summary.csv" else args.csv.stem

    df = load_csv(args.csv)
    if df.empty:
        sys.exit("error: no usable rows after filtering; nothing to plot")

    log_y = not args.linear_y

    plot_metric(
        df,
        ycol="elapsed_s",
        ylabel="Elapsed wall time (s)" + (", log scale" if log_y else ""),
        title=build_title(args.csv, df, "Weak scaling: wall-clock time"),
        out_path=outdir / f"{base}_elapsed.png",
        log_y=log_y,
    )
    plot_metric(
        df,
        ycol="tps",
        ylabel="Ticks / second" + (", log scale" if log_y else ""),
        title=build_title(args.csv, df, "Weak scaling: throughput"),
        out_path=outdir / f"{base}_tps.png",
        log_y=log_y,
    )


if __name__ == "__main__":
    main()
