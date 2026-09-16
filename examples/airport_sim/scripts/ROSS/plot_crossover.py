#!/usr/bin/env python3
"""
Plot DSTM vs ROSS throughput from a scale_crossover.sh output.

Accepts EITHER:
  * the raw CSV (weak per-run rows, includes trials + warmup), OR
  * the .summary.csv (one row per (build, sync, ranks, planes) with medians)

Auto-detects which one was passed.

One subplot per rank count. X axis: planes per airport (log).
Y axis: plane-hops per second (log). One line per engine.
When using raw CSV, error bars show trial min/max; with the summary CSV,
error bars are omitted (only the median is available).

Usage:
    python3 plot_crossover.py <csv_or_summary_csv> [--output prefix]
                              [--metric hops_per_sec|hops_per_sec_per_rank]
                              [--font-scale 1.0]
"""

import argparse
import os
import sys
import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ENGINE_STYLE = {
    ("dstm", "bsp"):  dict(label="DSTM (BSP)",          color="#1f77b4", marker="o"),
    ("ross", "opt"):  dict(label="ROSS (optimistic)",   color="#d62728", marker="s"),
    ("ross", "cons"): dict(label="ROSS (conservative)", color="#ff9f2b", marker="^"),
}

# Human-readable axis labels per metric.
METRIC_LABEL = {
    "hops_per_sec":          "plane-hops / second",
    "hops_per_sec_per_rank": "plane-hops / second / rank",
}


# Base font size tuned for a double-column paper column width.
# Every text element scales from BASE_FONT via --font-scale.
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
        "legend.fontsize":   s * 0.85,
        "legend.title_fontsize": s * 0.95,
        "figure.titlesize":  s * 1.15,
        # Thicker lines + bigger ticks/markers so they read at the scaled
        # font sizes; matplotlib defaults look wispy next to large text.
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


def is_summary_csv(df):
    """A summary CSV has median_* columns and no per-trial `trial` column."""
    return "median_hops_per_sec" in df.columns


def load(csv_path, metric):
    """
    Returns a DataFrame with columns:
        engine (tuple), ranks (int), planes (int),
        median (float), lo (float), hi (float)
    Regardless of whether the input was the raw or summary CSV.
    """
    df = pd.read_csv(csv_path)

    if is_summary_csv(df):
        # Summary CSV path: already aggregated, one row per point.
        median_col = f"median_{metric}"
        if median_col not in df.columns:
            sys.exit(f"ERROR: summary CSV missing column '{median_col}'. "
                     f"Available: {list(df.columns)}")
        df = df.copy()
        df["ranks"]  = pd.to_numeric(df["ranks"],  errors="coerce").astype("Int64")
        df["planes"] = pd.to_numeric(df["planes"], errors="coerce").astype("Int64")
        df["median"] = pd.to_numeric(df[median_col], errors="coerce")
        df = df.dropna(subset=["ranks", "planes", "median"])
        df["engine"] = list(zip(df["build"], df["sync"]))
        # No per-trial spread available — set lo/hi = median so error bars vanish.
        df["lo"] = df["median"]
        df["hi"] = df["median"]
        return df[["engine", "ranks", "planes", "median", "lo", "hi"]]

    # Raw CSV path: filter, aggregate.
    if metric not in df.columns:
        sys.exit(f"ERROR: raw CSV missing column '{metric}'. "
                 f"Available: {list(df.columns)}")

    # Drop failed runs and warm-up trials.
    df = df[df["status"] == "ok"].copy()
    if "warmup" in df.columns:
        df["warmup"] = pd.to_numeric(df["warmup"], errors="coerce").fillna(0).astype(int)
        df = df[df["warmup"] == 0]

    df["ranks"]  = pd.to_numeric(df["ranks"],  errors="coerce").astype("Int64")
    df["planes"] = pd.to_numeric(df["planes"], errors="coerce").astype("Int64")
    df[metric]   = pd.to_numeric(df[metric],   errors="coerce")
    df = df.dropna(subset=["ranks", "planes", metric])

    df["engine"] = list(zip(df["build"], df["sync"]))

    # Median + min/max across trials.
    agg = (df.groupby(["engine", "ranks", "planes"])[metric]
             .agg(median="median", lo="min", hi="max")
             .reset_index())
    return agg


def plot(agg, out_prefix, metric, title_suffix=""):
    y_label = METRIC_LABEL.get(metric, metric)

    ranks_list = sorted(agg["ranks"].unique())
    engines    = sorted(agg["engine"].unique(),
                        key=lambda e: (e[0] != "dstm", e[0], e[1]))
    n = len(ranks_list)

    cols = min(n, 2)
    rows = math.ceil(n / cols)

    # Larger canvas per subplot so the enlarged text has room to breathe.
    fig, axes = plt.subplots(rows, cols,
                             figsize=(7.5 * cols, 6.0 * rows),
                             squeeze=False, sharex=True)

    for i, p in enumerate(ranks_list):
        ax = axes[i // cols][i % cols]
        sub = agg[agg["ranks"] == p]

        for eng in engines:
            e_sub = sub[sub["engine"] == eng].sort_values("planes")
            if e_sub.empty:
                continue

            style = ENGINE_STYLE.get(eng, dict(label=f"{eng[0]}/{eng[1]}",
                                               color="gray", marker="x"))
            yerr = np.vstack([e_sub["median"] - e_sub["lo"],
                              e_sub["hi"]     - e_sub["median"]])
            # If summary CSV was used, lo==hi==median, so yerr is all zeros
            # and errorbar() just draws the line/markers.
            ax.errorbar(e_sub["planes"], e_sub["median"],
                        yerr=yerr,
                        capsize=5, capthick=1.5, elinewidth=1.4,
                        linewidth=plt.rcParams["lines.linewidth"],
                        markersize=plt.rcParams["lines.markersize"],
                        **style)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"P = {p} ranks")
        ax.set_xlabel("planes per airport")
        ax.set_ylabel(y_label)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")

    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")

    fig.tight_layout(pad=1.5, w_pad=3.0, h_pad=2.0)

    out = f"{out_prefix}_crossover.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")

    # ---------- Speedup companion figure ----------
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    dstm_key = ("dstm", "bsp")
    ross_key = ("ross", "opt")
    cmap = plt.cm.viridis
    ranks_norm = plt.Normalize(vmin=math.log2(min(ranks_list)),
                               vmax=math.log2(max(ranks_list)))
    for p in ranks_list:
        sub = agg[agg["ranks"] == p]
        d = sub[sub["engine"] == dstm_key].sort_values("planes")
        r = sub[sub["engine"] == ross_key].sort_values("planes")
        if d.empty or r.empty:
            continue
        merged = pd.merge(d[["planes", "median"]],
                          r[["planes", "median"]],
                          on="planes", suffixes=("_d", "_r"))
        if merged.empty:
            print(f"  (P={p}: no common planes values between DSTM and ROSS)",
                  file=sys.stderr)
            continue
        merged["speedup"] = merged["median_d"] / merged["median_r"]
        ax2.plot(merged["planes"], merged["speedup"],
                 marker="o",
                 linewidth=plt.rcParams["lines.linewidth"],
                 markersize=plt.rcParams["lines.markersize"],
                 color=cmap(ranks_norm(math.log2(p))),
                 label=f"P={p}")
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.4, alpha=0.6)
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("planes per airport")
    ax2.set_ylabel("DSTM throughput  /  ROSS(opt) throughput")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(title="ranks")
    fig2.tight_layout()

    out2 = f"{out_prefix}_speedup.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"wrote {out2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv",
                    help="raw CSV or .summary.csv from scale_crossover.sh")
    ap.add_argument("--output", "-o", default=None,
                    help="output prefix (default: derived from CSV basename)")
    ap.add_argument("--metric", default="hops_per_sec",
                    choices=["hops_per_sec", "hops_per_sec_per_rank"],
                    help="which throughput metric to plot (default: hops_per_sec)")
    ap.add_argument("--font-scale", type=float, default=1.0,
                    help=f"multiplier on base font size ({BASE_FONT}pt); "
                         "use e.g. 1.2 to go even bigger, 0.8 for smaller")
    args = ap.parse_args()

    # Apply font scaling BEFORE any figures are created.
    configure_fonts(args.font_scale)

    agg = load(args.csv, args.metric)
    if agg.empty:
        print("No usable rows in CSV", file=sys.stderr)
        sys.exit(1)

    if args.output:
        prefix = args.output
    else:
        # Strip trailing .summary and .csv so both foo.csv and foo.summary.csv
        # produce clean prefix "foo".
        base = os.path.splitext(args.csv)[0]
        if base.endswith(".summary"):
            base = base[:-len(".summary")]
        prefix = base

    plot(agg, prefix, args.metric,
         title_suffix=f" ({METRIC_LABEL[args.metric]})")


if __name__ == "__main__":
    main()
