#!/usr/bin/env python3
"""
Plot DSTM vs ROSS scaling curves as three side-by-side panels:

  (1) Weak scaling      -- from weak CSV,   x = ranks
  (2) Strong scaling    -- from strong CSV, x = ranks
  (3) Airport scaling   -- from strong CSV, x = planes at fixed ranks (default 256)

Both inputs may be either the raw scale_crossover.sh CSV or its .summary.csv;
each file is auto-detected independently.

Usage:
    python3 plot_scaling.py <weak_csv> <strong_csv> [--output prefix]
                            [--metric hops_per_sec|hops_per_sec_per_rank]
                            [--weak-planes N]  [--strong-planes N]
                            [--airport-fixed-ranks 256]
                            [--font-scale 1.0]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator


ENGINE_STYLE = {
    ("dstm", "bsp"):  dict(label="DSTM (BSP)",          color="#1f77b4", marker="o"),
    ("ross", "opt"):  dict(label="ROSS (optimistic)",   color="#d62728", marker="s"),
    ("ross", "cons"): dict(label="ROSS (conservative)", color="#ff9f2b", marker="^"),
}

METRIC_LABEL = {
    "hops_per_sec":          "plane-hops / second",
    "hops_per_sec_per_rank": "plane-hops / second / rank",
}

BASE_FONT = 18


# ---------------------------------------------------------------------------
# Font / style config
# ---------------------------------------------------------------------------

def configure_fonts(scale: float = 1.0):
    s = BASE_FONT * scale
    plt.rcParams.update({
        "font.size":             s,
        "axes.titlesize":        s * 1.05,
        "axes.labelsize":        s * 1.05,
        "xtick.labelsize":       s * 0.95,
        "ytick.labelsize":       s * 0.95,
        "legend.fontsize":       s * 0.85,
        "legend.title_fontsize": s * 0.95,
        "figure.titlesize":      s * 1.15,
        "lines.linewidth":       2.4,
        "lines.markersize":      10,
        "axes.linewidth":        1.2,
        "xtick.major.width":     1.2,
        "ytick.major.width":     1.2,
        "xtick.major.size":      6,
        "ytick.major.size":      6,
        "xtick.minor.width":     0.9,
        "ytick.minor.width":     0.9,
    })


# ---------------------------------------------------------------------------
# CSV loading (same logic as plot_crossover.py, works for raw or summary)
# ---------------------------------------------------------------------------

def is_summary_csv(df):
    return "median_hops_per_sec" in df.columns


def load(csv_path, metric):
    """
    Returns a long-form DataFrame with columns:
        engine (tuple), ranks (int), planes (int),
        median (float), lo (float), hi (float)
    """
    df = pd.read_csv(csv_path)

    if is_summary_csv(df):
        median_col = f"median_{metric}"
        if median_col not in df.columns:
            sys.exit(f"ERROR: {csv_path}: summary CSV missing column "
                     f"'{median_col}'. Available: {list(df.columns)}")
        df = df.copy()
        df["ranks"]  = pd.to_numeric(df["ranks"],  errors="coerce").astype("Int64")
        df["planes"] = pd.to_numeric(df["planes"], errors="coerce").astype("Int64")
        df["median"] = pd.to_numeric(df[median_col], errors="coerce")
        df = df.dropna(subset=["ranks", "planes", "median"])
        df["engine"] = list(zip(df["build"], df["sync"]))
        df["lo"] = df["median"]
        df["hi"] = df["median"]
        return df[["engine", "ranks", "planes", "median", "lo", "hi"]]

    if metric not in df.columns:
        sys.exit(f"ERROR: {csv_path}: raw CSV missing column '{metric}'. "
                 f"Available: {list(df.columns)}")

    df = df[df["status"] == "ok"].copy()
    if "warmup" in df.columns:
        df["warmup"] = pd.to_numeric(df["warmup"], errors="coerce").fillna(0).astype(int)
        df = df[df["warmup"] == 0]

    df["ranks"]  = pd.to_numeric(df["ranks"],  errors="coerce").astype("Int64")
    df["planes"] = pd.to_numeric(df["planes"], errors="coerce").astype("Int64")
    df[metric]   = pd.to_numeric(df[metric],   errors="coerce")
    df = df.dropna(subset=["ranks", "planes", metric])

    df["engine"] = list(zip(df["build"], df["sync"]))

    agg = (df.groupby(["engine", "ranks", "planes"])[metric]
             .agg(median="median", lo="min", hi="max")
             .reset_index())
    return agg


# ---------------------------------------------------------------------------
# Slice helpers for weak / strong / airport panels
# ---------------------------------------------------------------------------

def slice_scaling(agg, forced_planes=None):
    """
    Reduce a weak-or-strong DataFrame to one row per (engine, ranks).

    If `forced_planes` is given, filter to that planes value.
    Otherwise, for each (engine, ranks) pick the row with the largest `planes`.
    """
    if forced_planes is not None:
        out = agg[agg["planes"] == forced_planes].copy()
        if out.empty:
            avail = sorted(agg["planes"].unique().tolist())
            print(f"WARNING: no rows with planes={forced_planes}; "
                  f"available: {avail}", file=sys.stderr)
        return out

    # Pick the max-planes row per (engine, ranks).
    idx = (agg.sort_values("planes")
              .groupby(["engine", "ranks"], as_index=False)
              .tail(1).index)
    return agg.loc[idx].copy()


def slice_airport(agg, fixed_ranks):
    out = agg[agg["ranks"] == fixed_ranks].copy()
    if out.empty:
        avail = sorted(agg["ranks"].unique().tolist())
        print(f"WARNING: no rows with ranks={fixed_ranks}; "
              f"available: {avail}", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_one(ax, sub, engines, x_col, x_label, y_label, title,
              show_ylabel=True, show_legend=True):
    """Draw one scaling panel. `sub` is already filtered to the slice."""
    any_data = False
    x_values = set()

    for eng in engines:
        e_sub = sub[sub["engine"] == eng].sort_values(x_col)
        if e_sub.empty:
            continue
        any_data = True
        x_values.update(int(v) for v in e_sub[x_col].tolist())

        style = ENGINE_STYLE.get(eng, dict(label=f"{eng[0]}/{eng[1]}",
                                           color="gray", marker="x"))
        yerr = np.vstack([e_sub["median"] - e_sub["lo"],
                          e_sub["hi"]     - e_sub["median"]])
        ax.errorbar(e_sub[x_col], e_sub["median"],
                    yerr=yerr,
                    capsize=5, capthick=1.5, elinewidth=1.4,
                    linewidth=plt.rcParams["lines.linewidth"],
                    markersize=plt.rcParams["lines.markersize"],
                    **style)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    if show_ylabel:
        ax.set_ylabel(y_label)
    ax.grid(True, which="both", alpha=0.3)

    # X ticks pinned to sampled data points, printed as plain integers.
    if x_values:
        xt = sorted(x_values)
        ax.xaxis.set_major_locator(FixedLocator(xt))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{int(v)}"))

    # Y axis: leave matplotlib's default log formatting (10^k) alone.

    if any_data and show_legend:
        ax.legend(loc="best")
    elif not any_data:
        ax.text(0.5, 0.5, "no data for this slice",
                ha="center", va="center", transform=ax.transAxes)


def plot(weak_agg, strong_agg, out_prefix, metric,
         weak_planes, strong_planes, airport_fixed_ranks):
    y_label = METRIC_LABEL.get(metric, metric)

    # Union engines across both CSVs, DSTM first.
    engines = sorted(set(weak_agg["engine"]).union(strong_agg["engine"]),
                     key=lambda e: (e[0] != "dstm", e[0], e[1]))

    weak_sub    = slice_scaling(weak_agg,   forced_planes=weak_planes)
    strong_sub  = slice_scaling(strong_agg, forced_planes=strong_planes)
    airport_sub = slice_airport(strong_agg, fixed_ranks=airport_fixed_ranks)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), sharey=True)
    ax_w, ax_s, ax_a = axes

    weak_title = "Weak scaling"
    if weak_planes is not None:
        weak_title += f" (planes = {weak_planes})"

    strong_title = "Strong scaling"
    if strong_planes is not None:
        strong_title += f" (planes = {strong_planes})"

    _plot_one(ax_w, weak_sub, engines,
              x_col="ranks",
              x_label="ranks",
              y_label=y_label,
              title=weak_title,
              show_ylabel=True,  show_legend=True)

    _plot_one(ax_s, strong_sub, engines,
              x_col="ranks",
              x_label="ranks",
              y_label=y_label,
              title=strong_title,
              show_ylabel=False, show_legend=False)

    _plot_one(ax_a, airport_sub, engines,
              x_col="planes",
              x_label="planes per airport",
              y_label=y_label,
              title=f"Airport scaling (ranks = {airport_fixed_ranks})",
              show_ylabel=False, show_legend=False)

    fig.tight_layout(pad=1.5, w_pad=3.0, h_pad=2.0)

    out = f"{out_prefix}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weak_csv",
                    help="weak-scaling CSV (raw or .summary.csv)")
    ap.add_argument("strong_csv",
                    help="strong-scaling CSV (raw or .summary.csv)")
    ap.add_argument("--output", "-o", default=None,
                    help="output prefix (default: derived from strong CSV basename)")
    ap.add_argument("--metric", default="hops_per_sec",
                    choices=["hops_per_sec", "hops_per_sec_per_rank"],
                    help="throughput metric to plot (default: hops_per_sec)")
    ap.add_argument("--weak-planes", type=int, default=None,
                    help="force a specific planes value for the weak-scaling panel "
                         "(default: auto-pick largest available per rank)")
    ap.add_argument("--strong-planes", type=int, default=None,
                    help="force a specific planes value for the strong-scaling panel "
                         "(default: auto-pick largest available per rank)")
    ap.add_argument("--airport-fixed-ranks", type=int, default=256,
                    help="ranks slice for the airport-scaling panel (default: 256)")
    ap.add_argument("--font-scale", type=float, default=1.0,
                    help=f"multiplier on base font size ({BASE_FONT}pt)")
    args = ap.parse_args()

    configure_fonts(args.font_scale)

    weak_agg   = load(args.weak_csv,   args.metric)
    strong_agg = load(args.strong_csv, args.metric)
    if weak_agg.empty and strong_agg.empty:
        print("No usable rows in either CSV", file=sys.stderr)
        sys.exit(1)

    if args.output:
        prefix = args.output
    else:
        base = os.path.splitext(args.strong_csv)[0]
        if base.endswith(".summary"):
            base = base[:-len(".summary")]
        prefix = base

    plot(weak_agg, strong_agg, prefix, args.metric,
         weak_planes=args.weak_planes,
         strong_planes=args.strong_planes,
         airport_fixed_ranks=args.airport_fixed_ranks)


if __name__ == "__main__":
    main()
