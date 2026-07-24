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
        [--outdir DIR] [--linear-y] [--font-scale 1.0]

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


# Base font sizes tuned for a double-column paper column width (~3.5 in wide
# per column). Every text element scales from BASE_FONT via --font-scale.
BASE_FONT = 18


def configure_fonts(scale: float = 1.0):
    """Set global matplotlib rcParams so all text renders large enough
    to be readable when the figure is shrunk into a double-column layout."""
    s = BASE_FONT * scale
    plt.rcParams.update({
        "font.size":         s,
        "axes.titlesize":    s * 1.05,
        "axes.labelsize":    s * 1.05,
        "xtick.labelsize":   s * 0.95,
        "ytick.labelsize":   s * 0.95,
        "legend.fontsize":   s * 0.90,
        "legend.title_fontsize": s * 0.95,
        "figure.titlesize":  s * 1.15,
        # Thicker lines + bigger ticks/markers so they read at the scaled
        # font sizes; matplotlib defaults look wispy next to large text.
        "lines.linewidth":   2.4,
        "lines.markersize":  9,
        "axes.linewidth":    1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size":  6,
        "ytick.major.size":  6,
        "xtick.minor.width": 0.9,
        "ytick.minor.width": 0.9,
    })


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
            markersize=plt.rcParams["lines.markersize"],
            linewidth=plt.rcParams["lines.linewidth"],
            capsize=5,
            capthick=1.5,
            elinewidth=1.4,
        )

    # Frameworks in the CSV that we don't have styling for.
    others = set(agg["framework"]) - seen
    for framework in sorted(others):
        sub = agg[agg["framework"] == framework].sort_values("ranks")
        ax.errorbar(
            sub["ranks"], sub["mean"],
            yerr=[sub["err_lo"], sub["err_hi"]],
            label=str(framework),
            marker="x",
            markersize=plt.rcParams["lines.markersize"],
            linewidth=plt.rcParams["lines.linewidth"],
            capsize=5, capthick=1.5, elinewidth=1.4,
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
    # Larger canvas so the enlarged text has room to breathe without
    # overlapping. Preserves the ~2.55:1 aspect ratio of the original.
    fig, (ax_strong, ax_weak) = plt.subplots(
        1, 2, figsize=(18, 7.5), sharey=False
    )

    strong_params = summarise_params(strong_df, mode="strong")
    weak_params   = summarise_params(weak_df,   mode="weak")

    draw_on_axis(
        ax_strong, strong_df, ycol,
        ylabel=ylabel,
        subplot_title=f"Strong scaling",
        log_y=log_y,
    )
    draw_on_axis(
        ax_weak, weak_df, ycol,
        ylabel=ylabel,
        subplot_title=f"Weak scaling",
        log_y=log_y,
    )

    # fig.suptitle(sup_title, fontsize=plt.rcParams["figure.titlesize"])
    fig.tight_layout(rect=[0, 0, 1, 0.97], pad=1.5, w_pad=3.0)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
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
    ap.add_argument("--font-scale", type=float, default=1.0,
                    help=f"multiplier on base font size ({BASE_FONT}pt); "
                         "use e.g. 1.2 to go even bigger, 0.8 for smaller")
    args = ap.parse_args()

    for label, path in (("strong", args.strong), ("weak", args.weak)):
        if not path.is_file():
            sys.exit(f"error: {label} CSV not found: {path}")

    outdir = args.outdir or args.strong.parent
    outdir.mkdir(parents=True, exist_ok=True)

    # Apply the global font scaling BEFORE any figures are created.
    configure_fonts(args.font_scale)

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
