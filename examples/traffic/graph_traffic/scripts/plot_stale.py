#!/usr/bin/env python3
"""
plot_stale.py -- analyze trip-level routing quality vs. stale delay.

Reads a runs.csv produced by scale_stale.sh. Each successful runs.csv row
must contain a `trip_table` column pointing to the per-trip CSV written by
traffic.

The script produces seven figures:

  1. prediction_gap_by_length_vs_stale.png
       Mean experienced_tt - routed_tt for short, medium, and long trips.
       Positive values mean the routed estimate was optimistic.

  2. excess_tt_by_length_vs_stale.png
       Mean experienced travel-time difference against the matched oracle
       run, paired by trip_id.

  3. excess_vht_vs_stale.png
       Total excess vehicle-hours against oracle over trips completed in
       both runs.

  4. excess_vmt_vs_stale.png
       Total excess vehicle-miles against oracle over trips completed in
       both runs.

  5. trip_arrived_pct_vs_stale.png
       Percentage of all trips in each trip table that arrived.

  6. run_time_vs_stale.png
       Total program run time.

  7. mib_tick_vs_stale.png
       Per-rank data movement in MiB/tick.

Short/medium/long classes are global free-flow-time terciles. The same
thresholds are used for every mode and stale delay.

A tested run is paired with the oracle run having the same rank count and
repetition number. If no exact repetition match exists, the first oracle
run with the same rank count is used.

Only trips that arrived in both a tested run and its paired oracle run are
used for excess-TT, excess-VHT, and excess-VMT calculations. The separate
arrival-percentage plot must be consulted when interpreting these paired
completed-trip metrics.

USAGE
    python plot_stale.py runs.csv
    python plot_stale.py runs.csv --show
    python plot_stale.py runs.csv --out-dir figures
    python plot_stale.py runs.csv --gap-out gap.png
    python plot_stale.py runs.csv --excess-tt-out excess_tt.png
    python plot_stale.py runs.csv --vht-out excess_vht.png
    python plot_stale.py runs.csv --vmt-out excess_vmt.png
    python plot_stale.py runs.csv --arrival-out arrivals.png
    python plot_stale.py runs.csv --time-out timing.png
    python plot_stale.py runs.csv --move-out movement.png
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from math import isfinite
from statistics import mean, median


TRIP_CLASSES = ("short", "medium", "long")
CLASS_COLORS = {
    "short": "tab:blue",
    "medium": "tab:orange",
    "long": "tab:red",
}


def finite_float(value):
    """Return a finite float, or None for missing/invalid/non-finite data."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def integer(value):
    """Return an integer, or None for missing/invalid data."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_rows(path):
    """Read runs.csv, retaining only rows with status=ok."""
    rows = []

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)

        required = {
            "ranks",
            "mode",
            "stale_delay",
            "rep",
            "status",
            "run_time",
            "mib_tick_per_rank",
            "trip_table",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            names = ", ".join(sorted(missing))
            sys.exit(f"error: {path} is missing CSV columns: {names}")

        for row in reader:
            if row.get("status") == "ok":
                rows.append(row)

    if not rows:
        sys.exit(f"error: no rows with status=ok found in {path}")

    return rows


def resolve_trip_path(runs_csv, trip_table):
    """Resolve a trip-table path relative to the runs.csv directory."""
    if not trip_table:
        return None
    if os.path.isabs(trip_table):
        return trip_table
    return os.path.join(
        os.path.dirname(os.path.abspath(runs_csv)),
        trip_table,
    )


def load_trip_table(path):
    """Load one per-trip CSV and return records keyed by trip_id."""
    trips = {}

    if path is None or not os.path.isfile(path):
        raise FileNotFoundError(path or "<missing trip_table path>")

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)

        required = {
            "trip_id",
            "arrived",
            "free_flow_tt",
            "routed_tt",
            "experienced_tt",
            "prediction_gap",
            "vmt",
            "vht",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{path} is missing trip columns: {names}")

        for row in reader:
            trip_id = integer(row.get("trip_id"))
            if trip_id is None:
                continue

            trips[trip_id] = {
                "trip_id": trip_id,
                "arrived": integer(row.get("arrived")) == 1,
                "free_flow_tt": finite_float(row.get("free_flow_tt")),
                "routed_tt": finite_float(row.get("routed_tt")),
                "experienced_tt": finite_float(row.get("experienced_tt")),
                "prediction_gap": finite_float(row.get("prediction_gap")),
                "vmt": finite_float(row.get("vmt")),
                "vht": finite_float(row.get("vht")),
            }

    return trips


def attach_trip_tables(rows, runs_csv):
    """Load and cache the trip table referenced by every successful run."""
    cache = {}

    for row in rows:
        path = resolve_trip_path(runs_csv, row.get("trip_table"))

        if path not in cache:
            try:
                cache[path] = load_trip_table(path)
            except (OSError, ValueError) as exc:
                sys.exit(f"error loading trip table for a successful run: {exc}")

        row["_trip_path"] = path
        row["_trips"] = cache[path]


def free_flow_thresholds(rows):
    """
    Calculate global free-flow-time tercile thresholds.

    A trip ID contributes once. Oracle is preferred as the source when
    available, but free_flow_tt should be identical across configurations.
    """
    by_id = {}

    ordered_rows = sorted(
        rows,
        key=lambda row: 0 if row.get("mode") == "oracle" else 1,
    )

    for row in ordered_rows:
        for trip_id, trip in row["_trips"].items():
            ff = trip["free_flow_tt"]
            if ff is not None and trip_id not in by_id:
                by_id[trip_id] = ff

    values = sorted(by_id.values())
    if not values:
        sys.exit("error: no finite free_flow_tt values in trip tables")

    short_index = (len(values) - 1) // 3
    long_index = (2 * (len(values) - 1)) // 3

    return values[short_index], values[long_index]


def trip_class(free_flow_tt, thresholds):
    """Classify a trip using global free-flow-time terciles."""
    if free_flow_tt is None:
        return None

    short_max, medium_max = thresholds
    if free_flow_tt <= short_max:
        return "short"
    if free_flow_tt <= medium_max:
        return "medium"
    return "long"


def matched_oracle(run, rows):
    """
    Find the oracle run corresponding to `run`.

    Prefer the same rank count and repetition. Fall back to any oracle run
    with the same rank count.
    """
    exact = [
        row
        for row in rows
        if row.get("mode") == "oracle"
        and row.get("ranks") == run.get("ranks")
        and row.get("rep") == run.get("rep")
    ]
    if exact:
        return exact[0]

    same_ranks = [
        row
        for row in rows
        if row.get("mode") == "oracle"
        and row.get("ranks") == run.get("ranks")
    ]
    return same_ranks[0] if same_ranks else None


def class_prediction_gaps(run, thresholds):
    """Return mean signed prediction gap for each trip-length class."""
    values = defaultdict(list)

    for trip in run["_trips"].values():
        gap = trip["prediction_gap"]
        cls = trip_class(trip["free_flow_tt"], thresholds)

        if trip["arrived"] and gap is not None and cls is not None:
            values[cls].append(gap)

    return {
        cls: mean(values[cls]) if values[cls] else None
        for cls in TRIP_CLASSES
    }


def paired_trip_differences(run, oracle, thresholds):
    """
    Calculate paired differences against oracle.

    Only trip IDs completed in both runs are included.
    """
    by_class = defaultdict(list)
    excess_vht = 0.0
    excess_vmt = 0.0
    paired_count = 0

    if oracle is None:
        return {
            "by_class": {cls: None for cls in TRIP_CLASSES},
            "excess_vht": None,
            "excess_vmt": None,
            "paired_count": 0,
        }

    for trip_id, trip in run["_trips"].items():
        oracle_trip = oracle["_trips"].get(trip_id)
        if oracle_trip is None:
            continue
        if not trip["arrived"] or not oracle_trip["arrived"]:
            continue

        actual = trip["experienced_tt"]
        oracle_actual = oracle_trip["experienced_tt"]
        vht = trip["vht"]
        oracle_vht = oracle_trip["vht"]
        vmt = trip["vmt"]
        oracle_vmt = oracle_trip["vmt"]

        if actual is None or oracle_actual is None:
            continue

        ff = trip["free_flow_tt"]
        if ff is None:
            ff = oracle_trip["free_flow_tt"]

        cls = trip_class(ff, thresholds)
        if cls is not None:
            by_class[cls].append(actual - oracle_actual)

        if vht is not None and oracle_vht is not None:
            excess_vht += vht - oracle_vht

        if vmt is not None and oracle_vmt is not None:
            excess_vmt += vmt - oracle_vmt

        paired_count += 1

    return {
        "by_class": {
            cls: mean(by_class[cls]) if by_class[cls] else None
            for cls in TRIP_CLASSES
        },
        "excess_vht": excess_vht if paired_count else None,
        "excess_vmt": excess_vmt if paired_count else None,
        "paired_count": paired_count,
    }


def trip_arrival_percentage(run):
    """Return percentage of all trip-table rows that arrived."""
    trips = list(run["_trips"].values())
    if not trips:
        return None

    arrived = sum(1 for trip in trips if trip["arrived"])
    return 100.0 * arrived / len(trips)


def median_values(values):
    """Median of finite values, or None if no finite values exist."""
    finite = [value for value in values if value is not None and isfinite(value)]
    return median(finite) if finite else None


def stale_series(run_metrics, metric_getter):
    """Return stale-delay x values and median metric values."""
    buckets = defaultdict(list)

    for run, metrics in run_metrics:
        if run.get("mode") != "stale":
            continue

        delay = integer(run.get("stale_delay"))
        value = metric_getter(metrics)
        if delay is None or value is None or not isfinite(value):
            continue

        buckets[delay].append(value)

    xs = sorted(buckets)
    ys = [median(buckets[x]) for x in xs]
    return xs, ys


def mode_baseline(run_metrics, mode, metric_getter):
    """Median metric for one non-swept mode."""
    values = []

    for run, metrics in run_metrics:
        if run.get("mode") != mode:
            continue

        value = metric_getter(metrics)
        if value is not None and isfinite(value):
            values.append(value)

    return median(values) if values else None


def aggregate_runs(rows, thresholds):
    """Calculate all trip-level metrics once per run."""
    result = []

    for run in rows:
        oracle = matched_oracle(run, rows)

        metrics = {
            "prediction_gap": class_prediction_gaps(run, thresholds),
            "arrival_pct": trip_arrival_percentage(run),
            "paired": paired_trip_differences(run, oracle, thresholds),
        }
        result.append((run, metrics))

    return result


def plot_class_metric(
        plt,
        run_metrics,
        extractor,
        out,
        ylabel,
        title,
        oracle_zero=False,
):
    """Plot short/medium/long metrics in three aligned panels."""
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4.8),
        sharex=True,
        sharey=True,
    )

    plotted = False

    for ax, cls in zip(axes, TRIP_CLASSES):
        xs, ys = stale_series(
            run_metrics,
            lambda metrics, c=cls: extractor(metrics, c),
        )

        if xs:
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                color=CLASS_COLORS[cls],
                label="stale",
            )
            plotted = True

        if oracle_zero:
            oracle = 0.0
        else:
            oracle = mode_baseline(
                run_metrics,
                "oracle",
                lambda metrics, c=cls: extractor(metrics, c),
            )

        blind = mode_baseline(
            run_metrics,
            "blind",
            lambda metrics, c=cls: extractor(metrics, c),
        )

        if xs:
            xmin, xmax = min(xs), max(xs)

            if oracle is not None:
                ax.hlines(
                    oracle,
                    xmin,
                    xmax,
                    colors="tab:green",
                    linestyles=":",
                    linewidth=2,
                    label=f"oracle = {oracle:.2f}",
                )

            if blind is not None:
                ax.hlines(
                    blind,
                    xmin,
                    xmax,
                    colors="tab:gray",
                    linestyles="--",
                    linewidth=2,
                    label=f"blind = {blind:.2f}",
                )

            ax.set_xticks(xs)

        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
        ax.set_title(f"{cls} trips")
        ax.set_xlabel("stale delay (ticks)")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=8)

    if not plotted:
        sys.exit(f"error: no finite stale trip metrics available for {title}")

    axes[0].set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out)
    print(f"wrote {out}")
    return fig


def plot_scalar_trip_metric(
        plt,
        run_metrics,
        extractor,
        out,
        ylabel,
        title,
        oracle_zero=False,
        percent=False,
):
    """Plot one scalar trip-level metric against stale delay."""
    xs, ys = stale_series(run_metrics, extractor)
    if not xs:
        sys.exit(f"error: no finite stale values available for {title}")

    oracle = (
        0.0
        if oracle_zero
        else mode_baseline(run_metrics, "oracle", extractor)
    )
    blind = mode_baseline(run_metrics, "blind", extractor)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        xs,
        ys,
        marker="o",
        linewidth=2,
        color="tab:blue",
        label="stale",
    )

    xmin, xmax = min(xs), max(xs)

    if oracle is not None:
        ax.hlines(
            oracle,
            xmin,
            xmax,
            colors="tab:green",
            linestyles=":",
            linewidth=2,
            label=f"oracle = {oracle:.3f}",
        )

    if blind is not None:
        ax.hlines(
            blind,
            xmin,
            xmax,
            colors="tab:gray",
            linestyles="--",
            linewidth=2,
            label=f"blind = {blind:.3f}",
        )

    if oracle_zero:
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)

    if percent:
        ax.set_ylim(0.0, 100.0)

    ax.set_xticks(xs)
    ax.set_xlabel("stale delay (ticks)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")
    return fig


def aggregate_csv_series(rows, mode, column):
    """Median runs.csv value by stale delay for one mode."""
    buckets = defaultdict(list)

    for row in rows:
        if row.get("mode") != mode:
            continue

        delay = integer(row.get("stale_delay"))
        value = finite_float(row.get(column))

        if delay is None or value is None:
            continue

        buckets[delay].append(value)

    xs = sorted(buckets)
    ys = [median(buckets[x]) for x in xs]
    return xs, ys


def aggregate_csv_baseline(rows, mode, column):
    """Median runs.csv value for a non-swept mode."""
    values = [
        value
        for row in rows
        if row.get("mode") == mode
        for value in [finite_float(row.get(column))]
        if value is not None
    ]
    return median(values) if values else None


def plot_timing(plt, rows, out):
    """Plot run time against stale delay, using a broken axis if useful."""
    xs, ys = aggregate_csv_series(rows, "stale", "run_time")
    if not xs:
        sys.exit("error: no stale run-time values")

    oracle = aggregate_csv_baseline(rows, "oracle", "run_time")
    blind = aggregate_csv_baseline(rows, "blind", "run_time")
    xmin, xmax = min(xs), max(xs)

    lower_values = list(ys)
    if blind is not None:
        lower_values.append(blind)

    lower_min = min(lower_values)
    lower_max = max(lower_values)
    lower_pad = (
        0.10 * (lower_max - lower_min)
        if lower_max > lower_min
        else max(1.0, 0.1 * lower_max)
    )
    lower_lo = max(0.0, lower_min - lower_pad)
    lower_hi = lower_max + lower_pad

    broken = oracle is not None and oracle > lower_hi * 1.5

    def draw(ax):
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2,
            color="tab:blue",
            label="stale",
        )

        if oracle is not None:
            ax.hlines(
                oracle,
                xmin,
                xmax,
                colors="tab:green",
                linestyles=":",
                linewidth=2,
                label=f"oracle = {oracle:.1f}s",
            )

        if blind is not None:
            ax.hlines(
                blind,
                xmin,
                xmax,
                colors="tab:gray",
                linestyles="--",
                linewidth=2,
                label=f"blind = {blind:.1f}s",
            )

    if not broken:
        fig, ax = plt.subplots(figsize=(8, 5))
        draw(ax)
        ax.set_xticks(xs)
        ax.set_xlabel("stale delay (ticks)")
        ax.set_ylabel("total run time (s)")
        ax.set_title("graph_traffic: total run time vs. stale delay")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out)
        print(f"wrote {out}")
        return fig

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(8, 5.5),
        gridspec_kw={"height_ratios": [1, 3], "hspace": 0.08},
    )

    draw(ax_top)
    draw(ax_bottom)

    oracle_pad = 0.15 * oracle if oracle > 0 else 1.0
    ax_top.set_ylim(oracle - oracle_pad, oracle + oracle_pad)
    ax_bottom.set_ylim(lower_lo, lower_hi)

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False)
    ax_bottom.xaxis.tick_bottom()

    diagonal = 0.012
    kwargs = {
        "transform": ax_top.transAxes,
        "color": "black",
        "clip_on": False,
        "linewidth": 1,
    }
    ax_top.plot((-diagonal, diagonal), (-diagonal, diagonal), **kwargs)
    ax_top.plot(
        (1 - diagonal, 1 + diagonal),
        (-diagonal, diagonal),
        **kwargs,
    )

    kwargs["transform"] = ax_bottom.transAxes
    bottom_diagonal = diagonal * 3
    ax_bottom.plot(
        (-diagonal, diagonal),
        (1 - bottom_diagonal, 1 + bottom_diagonal),
        **kwargs,
    )
    ax_bottom.plot(
        (1 - diagonal, 1 + diagonal),
        (1 - bottom_diagonal, 1 + bottom_diagonal),
        **kwargs,
    )

    for ax in (ax_top, ax_bottom):
        ax.grid(True, linestyle="--", alpha=0.4)

    ax_bottom.set_xticks(xs)
    ax_bottom.set_xlabel("stale delay (ticks)")
    ax_bottom.legend(loc="best")

    fig.text(
        0.015,
        0.5,
        "total run time (s)",
        va="center",
        rotation="vertical",
    )
    ax_top.set_title("graph_traffic: total run time vs. stale delay")

    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")
    return fig


def plot_movement(plt, rows, out):
    """Plot per-rank data movement against stale delay."""
    column = "mib_tick_per_rank"
    xs, ys = aggregate_csv_series(rows, "stale", column)
    if not xs:
        sys.exit("error: no stale data-movement values")

    oracle = aggregate_csv_baseline(rows, "oracle", column)
    blind = aggregate_csv_baseline(rows, "blind", column)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        xs,
        ys,
        marker="o",
        linewidth=2,
        color="tab:blue",
        label="stale",
    )

    xmin, xmax = min(xs), max(xs)

    if oracle is not None:
        ax.hlines(
            oracle,
            xmin,
            xmax,
            colors="tab:green",
            linestyles=":",
            linewidth=2,
            label=f"oracle = {oracle:.3f}",
        )

    if blind is not None:
        ax.hlines(
            blind,
            xmin,
            xmax,
            colors="tab:gray",
            linestyles="--",
            linewidth=2,
            label=f"blind = {blind:.3f}",
        )

    ax.set_xticks(xs)
    ax.set_xlabel("stale delay (ticks)")
    ax.set_ylabel("data movement (MiB/tick per rank)")
    ax.set_title("graph_traffic: data movement vs. stale delay")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")
    return fig


def output_path(explicit, output_dir, default_name):
    """Return an explicit output path or a default under output_dir."""
    return explicit or os.path.join(output_dir, default_name)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "csv",
        help="runs.csv produced by scale_stale.sh",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="default output directory; defaults to the runs.csv directory",
    )
    parser.add_argument(
        "--gap-out",
        default=None,
        help="prediction-gap image path",
    )
    parser.add_argument(
        "--excess-tt-out",
        default=None,
        help="paired excess-travel-time image path",
    )
    parser.add_argument(
        "--vht-out",
        default=None,
        help="paired excess-VHT image path",
    )
    parser.add_argument(
        "--vmt-out",
        default=None,
        help="paired excess-VMT image path",
    )
    parser.add_argument(
        "--arrival-out",
        default=None,
        help="trip arrival-percentage image path",
    )
    parser.add_argument(
        "--time-out",
        default=None,
        help="run-time image path",
    )
    parser.add_argument(
        "--move-out",
        default=None,
        help="data-movement image path",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="display plot windows after writing images",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="output DPI; default 150",
    )
    args = parser.parse_args()

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    rows = load_rows(args.csv)
    attach_trip_tables(rows, args.csv)

    thresholds = free_flow_thresholds(rows)
    short_max, medium_max = thresholds

    print(
        "free-flow trip classes: "
        f"short <= {short_max:g}, "
        f"medium <= {medium_max:g}, "
        f"long > {medium_max:g} ticks"
    )

    run_metrics = aggregate_runs(rows, thresholds)

    output_dir = args.out_dir
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(output_dir, exist_ok=True)

    gap_out = output_path(
        args.gap_out,
        output_dir,
        "prediction_gap_by_length_vs_stale.png",
    )
    excess_tt_out = output_path(
        args.excess_tt_out,
        output_dir,
        "excess_tt_by_length_vs_stale.png",
    )
    vht_out = output_path(
        args.vht_out,
        output_dir,
        "excess_vht_vs_stale.png",
    )
    vmt_out = output_path(
        args.vmt_out,
        output_dir,
        "excess_vmt_vs_stale.png",
    )
    arrival_out = output_path(
        args.arrival_out,
        output_dir,
        "trip_arrived_pct_vs_stale.png",
    )
    time_out = output_path(
        args.time_out,
        output_dir,
        "run_time_vs_stale.png",
    )
    move_out = output_path(
        args.move_out,
        output_dir,
        "mib_tick_vs_stale.png",
    )

    figures = []

    figures.append(
        plot_class_metric(
            plt,
            run_metrics,
            extractor=lambda metrics, cls: metrics["prediction_gap"][cls],
            out=gap_out,
            ylabel="mean experienced − routed time (ticks)",
            title=(
                "graph_traffic: routed vs. experienced travel-time gap\n"
                "positive values indicate optimistic route estimates"
            ),
            oracle_zero=False,
        )
    )

    figures.append(
        plot_class_metric(
            plt,
            run_metrics,
            extractor=lambda metrics, cls: metrics["paired"]["by_class"][cls],
            out=excess_tt_out,
            ylabel="mean experienced TT − oracle TT (ticks)",
            title=(
                "graph_traffic: paired excess travel time vs. oracle\n"
                "only trips completed in both compared runs"
            ),
            oracle_zero=True,
        )
    )

    figures.append(
        plot_scalar_trip_metric(
            plt,
            run_metrics,
            extractor=lambda metrics: metrics["paired"]["excess_vht"],
            out=vht_out,
            ylabel="total excess vehicle-hours",
            title=(
                "graph_traffic: paired excess VHT vs. oracle\n"
                "only trips completed in both compared runs"
            ),
            oracle_zero=True,
        )
    )

    figures.append(
        plot_scalar_trip_metric(
            plt,
            run_metrics,
            extractor=lambda metrics: metrics["paired"]["excess_vmt"],
            out=vmt_out,
            ylabel="total excess vehicle-miles",
            title=(
                "graph_traffic: paired excess VMT vs. oracle\n"
                "only trips completed in both compared runs"
            ),
            oracle_zero=True,
        )
    )

    figures.append(
        plot_scalar_trip_metric(
            plt,
            run_metrics,
            extractor=lambda metrics: metrics["arrival_pct"],
            out=arrival_out,
            ylabel="all trips arrived (%)",
            title="graph_traffic: trip completion vs. stale delay",
            oracle_zero=False,
            percent=True,
        )
    )

    figures.append(plot_timing(plt, rows, time_out))
    figures.append(plot_movement(plt, rows, move_out))

    for figure in figures:
        figure.set_dpi(args.dpi)

    # Re-save with the requested DPI. The plotting functions save once so
    # failures are reported close to their source; this pass applies the
    # common command-line DPI consistently.
    for figure, path in zip(
        figures,
        (
            gap_out,
            excess_tt_out,
            vht_out,
            vmt_out,
            arrival_out,
            time_out,
            move_out,
        ),
    ):
        figure.savefig(path, dpi=args.dpi)

    if args.show:
        plt.show()

    for figure in figures:
        plt.close(figure)


if __name__ == "__main__":
    main()
