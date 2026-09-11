#!/usr/bin/env python3
"""
plot_stale.py -- plot routing quality, run time, and data movement vs. stale-delay.

Reads a runs.csv produced by scale_stale.sh and draws FIVE figures:

  1. mean_dur_vs_stale.png
       * mean duration among arrived early-cohort cars
       * lower is better

  2. mean_hops_vs_stale.png
       * mean hop count among arrived early-cohort cars
       * lower is better

  3. arrived_pct_vs_stale.png
       * percentage of the early-departure cohort that arrived
       * higher is better

  4. run_time_vs_stale.png
       * total program run time
       * uses a broken y-axis when the oracle baseline is an outlier

  5. mib_tick_vs_stale.png
       * per-rank data movement in MiB/tick

Each plot contains a line for the swept `stale` mode and horizontal
baselines for the `oracle` and `blind` modes. If a configuration has
multiple repetitions, its median is plotted. Undefined `nan` routing
means are omitted.

The routing cohort consists of cars departing during the first 10% of
simulation timestamps. mean_dur and mean_hops are conditional on cohort
arrival; arrived_pct exposes the cohort completion rate.

USAGE
    python plot_stale.py runs.csv
    python plot_stale.py runs.csv --out duration.png
    python plot_stale.py runs.csv --hops-out hops.png
    python plot_stale.py runs.csv --arrival-out arrival.png
    python plot_stale.py runs.csv --time-out timing.png
    python plot_stale.py runs.csv --move-out movement.png
    python plot_stale.py runs.csv --show
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from math import isfinite
from statistics import median


def load_rows(path):
    """Read the CSV, keeping only rows with status == ok."""
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            if r.get("status") != "ok":
                continue
            rows.append(r)
    if not rows:
        sys.exit(f"error: no rows with status=ok found in {path}")
    return rows


def median_by_delay(rows, mode, col):
    """
    For the given mode, return (xs, ys) sorted by stale_delay, where ys are
    the median of column `col` across reps at each stale_delay.
    """
    buckets = defaultdict(list)
    for r in rows:
        if r.get("mode") != mode:
            continue
        try:
            sd = int(r["stale_delay"])
            val = float(r[col])
        except (TypeError, ValueError, KeyError):
            continue
        if not isfinite(val):
            continue
        buckets[sd].append(val)

    xs = sorted(buckets)
    ys = [median(buckets[x]) for x in xs]
    return xs, ys


def baseline_median(rows, mode, col):
    """
    Return the median of `col` for a mode, ignoring stale_delay.
    """
    vals = []
    for r in rows:
        if r.get("mode") != mode:
            continue
        try:
            val = float(r[col])
        except (TypeError, ValueError, KeyError):
            continue
        if isfinite(val):
            vals.append(val)

    return median(vals) if vals else None


def plot_duration(plt, rows, out, dpi, show):
    """Mean duration among arrived early-cohort cars vs. stale-delay."""
    xs, ys = median_by_delay(rows, "stale", "mean_dur")
    if not xs:
        sys.exit("error: no `stale` rows to plot")

    oracle = baseline_median(rows, "oracle", "mean_dur")
    blind = baseline_median(rows, "blind", "mean_dur")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, marker="o", linewidth=2, color="tab:blue",
            label="stale (swept)")

    xmin, xmax = min(xs), max(xs)
    if oracle is not None:
        ax.hlines(oracle, xmin, xmax, colors="tab:green", linestyles=":",
                  linewidth=2, label=f"oracle (always-fresh) = {oracle:.1f}")
    if blind is not None:
        ax.hlines(blind, xmin, xmax, colors="tab:red", linestyles=":",
                  linewidth=2, label=f"blind (never-merge) = {blind:.1f}")

    ax.set_xlabel("stale-delay (ticks between merges)")
    ax.set_ylabel("cohort mean duration (ticks)")
    ax.set_title(
        "graph_traffic: early-cohort mean duration vs. stale-delay\n"
        "(arrived cohort cars only; lower is better)"
    )
    ax.set_xticks(xs)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout(rect=(0.04, 0, 1, 1))

    if show:
        plt.show()
    fig.savefig(out, dpi=dpi)
    print(f"wrote {out}")
    return fig


def plot_routing_metric(
        plt, rows, col, out, dpi, show, ylabel, title, better, percent=False):
    """
    Plot one early-departure cohort routing metric vs. stale-delay.

    The stale mode is plotted over numeric stale-delay values. Oracle and
    blind ignore stale-delay and therefore appear as horizontal baselines.
    """
    xs, ys = median_by_delay(rows, "stale", col)
    if not xs:
        sys.exit(f"error: no finite `stale` values for {col}")

    oracle = baseline_median(rows, "oracle", col)
    blind = baseline_median(rows, "blind", col)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        xs,
        ys,
        marker="o",
        linewidth=2,
        color="tab:blue",
        label="stale (swept)",
    )

    xmin, xmax = min(xs), max(xs)
    value_fmt = ".1f" if percent else ".2f"

    if oracle is not None:
        ax.hlines(
            oracle,
            xmin,
            xmax,
            colors="tab:green",
            linestyles=":",
            linewidth=2,
            label=f"oracle (always-fresh) = {oracle:{value_fmt}}",
        )

    if blind is not None:
        ax.hlines(
            blind,
            xmin,
            xmax,
            colors="tab:red",
            linestyles=":",
            linewidth=2,
            label=f"blind (never-merge) = {blind:{value_fmt}}",
        )

    ax.set_xlabel("stale-delay (ticks between merges)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"graph_traffic: {title} vs. stale-delay\n({better})")
    ax.set_xticks(xs)

    if percent:
        ax.set_ylim(0.0, 100.0)

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()

    fig.savefig(out, dpi=dpi)
    print(f"wrote {out}")
    return fig


def plot_timing(plt, rows, out, dpi, show):
    """
    Run time vs. stale-delay with a BROKEN y-axis.

    The oracle baseline (~200s) dwarfs everything else (~5-25s), so a single
    linear axis would flatten the stale line into the floor. We stack two
    axes: the top frames the oracle band, the bottom frames the rest, with
    diagonal break marks between them.
    """
    xs, ys = median_by_delay(rows, "stale", "run_time")
    if not xs:
        sys.exit("error: no `stale` rows to plot for timing")

    oracle = baseline_median(rows, "oracle", "run_time")
    blind = baseline_median(rows, "blind", "run_time")

    xmin, xmax = min(xs), max(xs)

    # -- Decide the two y-windows -----------------------------------------
    # "lower" band: everything except the oracle outlier.
    lower_vals = list(ys)
    if blind is not None:
        lower_vals.append(blind)
    lo_min, lo_max = min(lower_vals), max(lower_vals)
    lo_pad = 0.10 * (lo_max - lo_min) if lo_max > lo_min else max(1.0, 0.1 * lo_max)
    lo_lo, lo_hi = max(0.0, lo_min - lo_pad), lo_max + lo_pad

    # Only bother with a broken axis if oracle really is an outlier well
    # above the lower band. Otherwise fall back to a single axis.
    broken = oracle is not None and oracle > lo_hi * 1.5

    def draw_series(ax):
        ax.plot(xs, ys, marker="o", linewidth=2, color="tab:blue",
                label="stale (swept)")
        if oracle is not None:
            ax.hlines(oracle, xmin, xmax, colors="tab:green", linestyles=":",
                      linewidth=2, label=f"oracle (always-fresh) = {oracle:.1f}s")
        if blind is not None:
            ax.hlines(blind, xmin, xmax, colors="tab:red", linestyles=":",
                      linewidth=2, label=f"blind (never-merge) = {blind:.1f}s")

    if not broken:
        # Simple single-axis fallback (keeps the tool robust to other data).
        fig, ax = plt.subplots(figsize=(8, 5))
        draw_series(ax)
        ax.set_xlabel("stale-delay (ticks between merges)")
        ax.set_ylabel("total run time (s)")
        ax.set_title("graph_traffic: total run time vs. stale-delay\n(lower is better)")
        ax.set_xticks(xs)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        if show:
            plt.show()
        fig.savefig(out, dpi=dpi)
        print(f"wrote {out}")
        return fig

    # -- Broken-axis layout -----------------------------------------------
    # Top panel is short (just the oracle band); bottom panel is taller.
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 5.5),
        gridspec_kw={"height_ratios": [1, 3], "hspace": 0.08})

    # Draw the full series on BOTH axes; the y-limits crop each panel.
    draw_series(ax_top)
    draw_series(ax_bot)

    # Top panel: tight band around oracle.
    o_pad = 0.15 * oracle if oracle > 0 else 1.0
    ax_top.set_ylim(oracle - o_pad, oracle + o_pad)

    # Bottom panel: the stale line + blind baseline.
    ax_bot.set_ylim(lo_lo, lo_hi)

    # Hide the spines between the two panels.
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False)
    ax_bot.xaxis.tick_bottom()

    # Diagonal "break" marks on the facing spines.
    d = 0.012  # size of the diagonal lines in axes coords
    kw = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1)
    ax_top.plot((-d, +d), (-d, +d), **kw)              # bottom-left of top
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kw)        # bottom-right of top
    kw.update(transform=ax_bot.transAxes)
    # scale d for the bottom panel's taller height so the marks look even
    db = d * 3
    ax_bot.plot((-d, +d), (1 - db, 1 + db), **kw)      # top-left of bottom
    ax_bot.plot((1 - d, 1 + d), (1 - db, 1 + db), **kw)  # top-right of bottom

    # Grid on both panels; ticks/labels on the bottom.
    for ax in (ax_top, ax_bot):
        ax.grid(True, linestyle="--", alpha=0.4)
    ax_bot.set_xticks(xs)
    ax_bot.set_xlabel("stale-delay (ticks between merges)")

    # One shared y-label centered across both panels.
    fig.text(0.015, 0.5, "total run time (s)",
             va="center", rotation="vertical")
    ax_top.set_title("graph_traffic: total run time vs. stale-delay\n(lower is better)")

    # Legend once (on the bottom panel where the stale line lives).
    ax_bot.legend(loc="best")

    fig.tight_layout()
    if show:
        plt.show()
    fig.savefig(out, dpi=dpi)
    print(f"wrote {out}")
    return fig


def plot_movement(plt, rows, out, dpi, show):
    """
    Data movement vs. stale-delay (single axis).

    Plots per-rank data movement (mib_tick_per_rank) for the swept `stale`
    mode, with oracle/blind medians as flat dotted baselines. More frequent
    merges (smaller stale-delay) move more data per tick, so this line should
    fall as stale-delay grows.
    """
    col = "mib_tick_per_rank"
    xs, ys = median_by_delay(rows, "stale", col)
    if not xs:
        sys.exit("error: no `stale` rows to plot for data movement")

    oracle = baseline_median(rows, "oracle", col)
    blind = baseline_median(rows, "blind", col)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, marker="o", linewidth=2, color="tab:blue",
            label="stale (swept)")

    xmin, xmax = min(xs), max(xs)
    if oracle is not None:
        ax.hlines(oracle, xmin, xmax, colors="tab:green", linestyles=":",
                  linewidth=2, label=f"oracle (always-fresh) = {oracle:.3f}")
    if blind is not None:
        ax.hlines(blind, xmin, xmax, colors="tab:red", linestyles=":",
                  linewidth=2, label=f"blind (never-merge) = {blind:.3f}")

    ax.set_xlabel("stale-delay (ticks between merges)")
    ax.set_ylabel("data movement (MiB/tick per rank)")
    ax.set_title("graph_traffic: data movement vs. stale-delay\n(lower is less traffic)")
    ax.set_xticks(xs)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()
    fig.savefig(out, dpi=dpi)
    print(f"wrote {out}")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="runs.csv produced by scale_stale.sh")
    ap.add_argument("--out", default=None, 
        help="duration image path (default: <csv-dir>/mean_dur_vs_stale.png)",
    )
    ap.add_argument("--hops-out", dest="hops_out", default=None, 
        help="hop image path (default: <csv-dir>/mean_hops_vs_stale.png)",
    )
    ap.add_argument("--arrival-out", dest="arrival_out", default=None,
        help=(
            "arrival-percentage image path "
            "(default: <csv-dir>/arrived_pct_vs_stale.png)"
        ),
    )
    ap.add_argument("--time-out", dest="time_out", default=None,
        help="timing image path (default: <csv-dir>/run_time_vs_stale.png)",
    )
    ap.add_argument("--move-out", dest="move_out", default=None,
        help=(
            "data-movement image path "
            "(default: <csv-dir>/mib_tick_vs_stale.png)"
        ),
    )
    ap.add_argument("--show", action="store_true", help="display the plot windows")
    ap.add_argument("--dpi", type=int, default=150, help="output DPI (default 150)")
    args = ap.parse_args()

    # Import here so --help works even without matplotlib installed.
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_rows(args.csv)

    csv_dir = os.path.dirname(os.path.abspath(args.csv))

    dur_out = args.out or os.path.join(csv_dir, "mean_dur_vs_stale.png")
    hops_out = args.hops_out or os.path.join(csv_dir, "mean_hops_vs_stale.png")
    arrival_out = args.arrival_out or os.path.join(csv_dir, "arrived_pct_vs_stale.png")
    time_out = args.time_out or os.path.join(csv_dir, "run_time_vs_stale.png")
    move_out = args.move_out or os.path.join(csv_dir, "mib_tick_vs_stale.png")

    plot_duration(plt, rows, dur_out, args.dpi, args.show)

    plot_routing_metric(
        plt,
        rows,
        col="mean_hops",
        out=hops_out,
        dpi=args.dpi,
        show=args.show,
        ylabel="cohort mean hops",
        title="early-cohort mean hops",
        better="arrived cohort cars only; lower is better",
    )

    plot_routing_metric(
        plt,
        rows,
        col="arrived_pct",
        out=arrival_out,
        dpi=args.dpi,
        show=args.show,
        ylabel="cohort arrived (%)",
        title="early-cohort arrival percentage",
        better="higher is better",
        percent=True,
    )

    plot_timing(plt, rows, time_out, args.dpi, args.show)
    plot_movement(plt, rows, move_out, args.dpi, args.show)


if __name__ == "__main__":
    main()
