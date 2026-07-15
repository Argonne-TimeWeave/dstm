#!/usr/bin/env python3
"""
Plot strong- vs weak-scaling zombie results side-by-side.

Consumes two summary CSVs (produced by compare.sh / strong_scale.sh) and
emits two PNGs, each with a (strong, weak) subplot pair:

    <outdir>/scaling_elapsed.png    : wall-clock time vs ranks
    <outdir>/scaling_tps.png        : ticks/second   vs ranks

Usage:
    python plot_scaling.py \
        --strong results/strong_20260715/summary.csv \
        --weak   results/20260710_190100/summary.csv \
        [--outdir DIR] [--linear-y]

Expected CSV columns (produced by compare.sh / strong_scale.sh):
    framework,ranks,ppn,nodes,nh,nz,time,density,nl,seed,repeat,
    elapsed_s,tps,status

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


def load_csv(path: Path, tag: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop failed rows, warn user.
    bad = df[df["status"] != "ok"]
    if not bad.empty:
        print(f"WARNING [{tag}]: dropping {len(bad)} failed row(s):",
              file=sys.stderr)
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


def draw_on_axis(ax: plt.Axes, df: pd.DataFrame, ycol: str,
                 ylabel: str, subplot_title: str, log_y: bool):
    """Draw one scaling curve (per framework) on a given Axes."""
    agg = aggregate(df, ycol)

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

    # Frameworks in the CSV that we don't have styling for.
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
    ax.set_title(subplot_title)

    # X-axis: show every rank count present as its own tick.
    all_ranks = sorted(agg["ranks"].unique())
    if all_ranks:
        ax.set_xticks(all_ranks)
        ax.set_xticklabels([str(int(r)) for r in all_ranks])

    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.legend(loc="best", frameon=True)


def summarise_params(df: pd.DataFrame, mode: str) -> str:
    """
    Build a short parameter string for a subplot title.

    For weak scaling, per-rank counts (nh, nz) are fixed across ranks.
    For strong scaling, per-rank counts vary but total (nh+nz)*ranks is fixed.
    """
    def _unique(col):
        if col not in df.columns:
            return None
        vals = df[col].dropna().unique()
        return vals if len(vals) else None

    parts = []

    if mode == "weak":
        for col, fmt in (("nh", "nh/rank={}"),
                         ("nz", "nz/rank={}")):
            vals = _unique(col)
            if vals is not None and len(vals) == 1:
                v = vals[0]
                if float(v).is_integer():
                    v = int(v)
                parts.append(fmt.format(v))
    else:  # strong
        # Compute total agents = (nh + nz) * ranks per row; report if constant.
        if {"nh", "nz", "ranks"}.issubset(df.columns):
            total = ((df["nh"] + df["nz"]) * df["ranks"]).dropna().unique()
            if len(total) == 1:
                parts.append(f"N={int(total[0])}")
        # nz per rank is usually still fixed in strong scaling.
        vals = _unique("nz")
        if vals is not None and len(vals) == 1:
            v = vals[0]
            if float(v).is_integer():
                v = int(v)
            parts.append(f"nz/rank={v}")

    for col, fmt in (("time", "T={}"),
                     ("density", "d={}")):
        vals = _unique(col)
        if vals is not None and len(vals) == 1:
            v = vals[0]
            if float(v).is_integer():
                v = int(v)
            parts.append(fmt.format(v))

    return "  ".join(parts)


def plot_pair(strong_df: pd.DataFrame, weak_df: pd.DataFrame,
              ycol: str, ylabel: str, sup_title: str,
              out_path: Path, log_y: bool):
    fig, (ax_strong, ax_weak) = plt.subplots(
        1, 2, figsize=(14, 5.5), sharey=False
    )

    strong_params = summarise_params(strong_df, mode="strong")
    weak_params   = summarise_params(weak_df,   mode="weak")

    draw_on_axis(
        ax_strong, strong_df, ycol,
        ylabel=ylabel,
        subplot_title=f"Strong scaling\n{strong_params}",
        log_y=log_y,
    )
    draw_on_axis(
        ax_weak, weak_df, ycol,
        ylabel=ylabel,
        subplot_title=f"Weak scaling\n{weak_params}",
        log_y=log_y,
    )

    fig.suptitle(sup_title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strong", type=Path, required=True,
                    help="strong-scaling summary.csv")
    ap.add_argument("--weak",   type=Path, required=True,
                    help="weak-scaling summary.csv")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="output directory (default: parent of --strong)")
    ap.add_argument("--linear-y", action="store_true",
                    help="use a linear y-axis instead of log")
    args = ap.parse_args()

    for label, path in (("strong", args.strong), ("weak", args.weak)):
        if not path.is_file():
            sys.exit(f"error: {label} CSV not found: {path}")

    outdir = args.outdir or args.strong.parent
    outdir.mkdir(parents=True, exist_ok=True)

    strong_df = load_csv(args.strong, tag="strong")
    weak_df   = load_csv(args.weak,   tag="weak")

    if strong_df.empty:
        sys.exit("error: no usable rows in strong CSV after filtering")
    if weak_df.empty:
        sys.exit("error: no usable rows in weak CSV after filtering")

    log_y = not args.linear_y
    y_suffix = " (log scale)" if log_y else ""

    plot_pair(
        strong_df, weak_df,
        ycol="elapsed_s",
        ylabel=f"Elapsed wall time (s){y_suffix}",
        sup_title="Strong vs Weak scaling — elapsed time",
        out_path=outdir / "scaling_elapsed.png",
        log_y=log_y,
    )
    plot_pair(
        strong_df, weak_df,
        ycol="tps",
        ylabel=f"Ticks / second{y_suffix}",
        sup_title="Strong vs Weak scaling — throughput",
        out_path=outdir / "scaling_tps.png",
        log_y=log_y,
    )


if __name__ == "__main__":
    main()
