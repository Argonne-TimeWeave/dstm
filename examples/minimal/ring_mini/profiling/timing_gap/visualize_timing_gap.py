#!/usr/bin/env python3
"""
Create an animated GIF showing rank-local simulation-time drift.

Each GIF frame corresponds to one "Rank N: time step T" record in the
order it appears in the log. Ranks are arranged vertically, and each
rank's marker moves horizontally as its local simulation time advances.

Compatible with Python 3.6.15.

Dependencies:
    pip install matplotlib pillow

Example:
    python3 visualize_timing_gap.py \
        data/mini_16_30_run1.log \
        -o mini_16_30_run1.gif \
        --fps 12
"""

import argparse
import os
import re
import sys

import matplotlib

# Use a non-interactive backend so the script works on remote systems.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


TIME_STEP_RE = re.compile(
    r"Rank\s+(?P<rank>\d+)\s*:\s*time\s+step\s+(?P<time>-?\d+)",
    re.IGNORECASE,
)

PREFIX_RANK_COUNT_RE = re.compile(
    r"normal\s*:\s*\d+\s*/\s*(?P<count>\d+)\s*:",
    re.IGNORECASE,
)

TOPOLOGY_RANK_COUNT_RE = re.compile(
    r"Ring\s+topology\s*:\s*(?P<count>\d+)\s+nodes",
    re.IGNORECASE,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Create a GIF showing rank-local simulation times and the "
            "resulting timing gap."
        )
    )

    parser.add_argument(
        "log_file",
        help="Path to the ring simulation log file.",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="timing_gap.gif",
        help="Output GIF path. Default: timing_gap.gif",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=12.0,
        help="Frames per second. Default: 12",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Output resolution in dots per inch. Default: 120",
    )

    parser.add_argument(
        "--width",
        type=float,
        default=10.0,
        help="Figure width in inches. Default: 10",
    )

    parser.add_argument(
        "--height",
        type=float,
        default=None,
        help=(
            "Figure height in inches. If omitted, height is selected "
            "from the number of ranks."
        ),
    )

    parser.add_argument(
        "--ranks",
        type=int,
        default=None,
        help=(
            "Override the number of ranks. By default, the script reads "
            "the rank count from the log or infers it from rank IDs."
        ),
    )

    parser.add_argument(
        "--title",
        default="Distributed Simulation Timing Drift",
        help="Title displayed above the visualization.",
    )

    parser.add_argument(
        "--no-rank-times",
        action="store_true",
        help="Do not print the current timestamp next to each rank marker.",
    )

    return parser.parse_args()


def parse_log(log_path):
    """
    Parse timestamp updates in their exact log-file order.

    Returns:
        events:
            List of (rank, time, line_number) tuples.

        declared_rank_counts:
            Rank counts discovered in log prefixes or topology messages.
    """
    events = []
    declared_rank_counts = []

    with open(log_path, "r") as log_file:
        for line_number, line in enumerate(log_file, 1):
            prefix_match = PREFIX_RANK_COUNT_RE.search(line)
            if prefix_match:
                declared_rank_counts.append(
                    int(prefix_match.group("count"))
                )

            topology_match = TOPOLOGY_RANK_COUNT_RE.search(line)
            if topology_match:
                declared_rank_counts.append(
                    int(topology_match.group("count"))
                )

            step_match = TIME_STEP_RE.search(line)
            if not step_match:
                continue

            rank = int(step_match.group("rank"))
            timestamp = int(step_match.group("time"))
            events.append((rank, timestamp, line_number))

    return events, declared_rank_counts


def determine_rank_count(events, declared_rank_counts, override):
    inferred_count = max(event[0] for event in events) + 1

    if override is not None:
        if override <= 0:
            raise ValueError("--ranks must be greater than zero")

        if inferred_count > override:
            raise ValueError(
                "Log contains rank {0}, but --ranks was set to {1}".format(
                    inferred_count - 1,
                    override,
                )
            )

        return override

    if declared_rank_counts:
        declared_count = max(declared_rank_counts)
        return max(declared_count, inferred_count)

    return inferred_count


def build_snapshots(events, rank_count):
    """
    Build the rank-local time state after every log update.

    The snapshot at index i contains the current timestamp for every rank
    after applying events[i]. A value of NaN means that rank has not yet
    appeared in the log.
    """
    snapshots = np.full(
        (len(events), rank_count),
        np.nan,
        dtype=float,
    )

    current_times = np.full(rank_count, np.nan, dtype=float)
    previous_times = {}
    regressions = []

    for event_index, event in enumerate(events):
        rank, timestamp, line_number = event

        if rank in previous_times and timestamp < previous_times[rank]:
            regressions.append(
                (
                    rank,
                    previous_times[rank],
                    timestamp,
                    line_number,
                )
            )

        previous_times[rank] = timestamp
        current_times[rank] = timestamp
        snapshots[event_index, :] = current_times

    return snapshots, regressions


def make_animation(
    events,
    snapshots,
    rank_count,
    output_path,
    fps,
    dpi,
    width,
    height,
    title,
    show_rank_times,
):
    all_timestamps = [event[1] for event in events]
    minimum_timestamp = min(all_timestamps)
    maximum_timestamp = max(all_timestamps)

    # Most simulations begin at zero. Preserve negative timestamps if the
    # input contains them.
    timeline_origin = min(0, minimum_timestamp)

    time_span = maximum_timestamp - timeline_origin
    if time_span <= 0:
        time_span = 1

    left_padding = max(0.5, time_span * 0.04)
    right_padding = max(1.5, time_span * 0.12)
    label_offset = max(0.10, time_span * 0.012)

    x_min = timeline_origin - left_padding
    x_max = maximum_timestamp + right_padding

    if height is None:
        # Keep small cases readable without making large-rank figures huge.
        height = max(5.0, min(12.0, 2.5 + rank_count * 0.34))

    fig, ax = plt.subplots(figsize=(width, height))

    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7f8fa")

    rank_positions = np.arange(rank_count)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(rank_count - 0.5, -0.5)

    ax.set_xlabel("Local simulation timestamp", fontsize=11)
    ax.set_ylabel("Rank", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=26)

    ax.set_yticks(rank_positions)
    ax.set_yticklabels(
        ["Rank {0}".format(rank) for rank in range(rank_count)]
    )

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(
        axis="x",
        color="#c8cdd3",
        linestyle="--",
        linewidth=0.8,
        alpha=0.75,
    )

    for y_position in rank_positions:
        ax.axhline(
            y=y_position,
            color="#e2e5e9",
            linewidth=0.8,
            zorder=0,
        )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # This shaded rectangle spans the slowest-to-fastest current timestamp.
    drift_band = Rectangle(
        (timeline_origin, -0.5),
        0.0,
        rank_count,
        facecolor="#4c78a8",
        edgecolor="none",
        alpha=0.10,
        zorder=1,
    )
    ax.add_patch(drift_band)

    # Highlight the rank updated by the current log record.
    active_rank_band = Rectangle(
        (x_min, -0.4),
        x_max - x_min,
        0.8,
        facecolor="#f2c14e",
        edgecolor="none",
        alpha=0.18,
        zorder=1,
    )
    ax.add_patch(active_rank_band)

    minimum_line = ax.axvline(
        timeline_origin,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.4,
        alpha=0.9,
        zorder=2,
    )

    maximum_line = ax.axvline(
        timeline_origin,
        color="#d62728",
        linestyle="--",
        linewidth=1.4,
        alpha=0.9,
        zorder=2,
    )

    progress_lines = []
    timestamp_labels = []

    for rank in range(rank_count):
        progress_line, = ax.plot(
            [],
            [],
            color="#4c78a8",
            linewidth=3.0,
            solid_capstyle="round",
            alpha=0.72,
            zorder=3,
        )
        progress_lines.append(progress_line)

        timestamp_label = ax.text(
            timeline_origin,
            rank,
            "",
            va="center",
            ha="left",
            fontsize=8,
            color="#333333",
            zorder=5,
        )
        timestamp_labels.append(timestamp_label)

    rank_markers = ax.scatter(
        np.empty((0, 2))[:, 0],
        np.empty((0, 2))[:, 1],
        s=64,
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )

    status_text = ax.text(
        0.5,
        1.025,
        "",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#333333",
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="#4c78a8",
            markerfacecolor="#1f77b4",
            markeredgecolor="white",
            linewidth=3,
            markersize=8,
            label="Rank progress",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#ff8c00",
            markeredgecolor="white",
            markersize=8,
            label="Rank updated this frame",
        ),
        Line2D(
            [0],
            [0],
            color="#2ca02c",
            linestyle="--",
            linewidth=1.4,
            label="Slowest rank",
        ),
        Line2D(
            [0],
            [0],
            color="#d62728",
            linestyle="--",
            linewidth=1.4,
            label="Fastest rank",
        ),
        Patch(
            facecolor="#4c78a8",
            alpha=0.10,
            label="Timing gap",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
        fontsize=8,
    )

    def reset_artists():
        rank_markers.set_offsets(np.empty((0, 2)))

        for line in progress_lines:
            line.set_data([], [])

        for label in timestamp_labels:
            label.set_text("")

        drift_band.set_x(timeline_origin)
        drift_band.set_width(0.0)

        active_rank_band.set_y(-0.4)

        minimum_line.set_xdata([timeline_origin, timeline_origin])
        maximum_line.set_xdata([timeline_origin, timeline_origin])

        status_text.set_text("")

        return []

    def draw_frame(frame_index):
        current_times = snapshots[frame_index]
        updated_rank, updated_time, line_number = events[frame_index]

        initialized_ranks = np.where(~np.isnan(current_times))[0]
        initialized_times = current_times[initialized_ranks]

        slowest_time = float(np.min(initialized_times))
        fastest_time = float(np.max(initialized_times))
        timing_gap = fastest_time - slowest_time

        marker_offsets = []
        marker_colors = []

        for rank in range(rank_count):
            timestamp = current_times[rank]

            if np.isnan(timestamp):
                progress_lines[rank].set_data([], [])
                timestamp_labels[rank].set_text("")
                continue

            progress_lines[rank].set_data(
                [timeline_origin, timestamp],
                [rank, rank],
            )

            marker_offsets.append([timestamp, rank])

            if rank == updated_rank:
                marker_colors.append("#ff8c00")
            else:
                marker_colors.append("#1f77b4")

            if show_rank_times:
                timestamp_labels[rank].set_position(
                    (timestamp + label_offset, rank)
                )
                timestamp_labels[rank].set_text(
                    "t={0:g}".format(timestamp)
                )
            else:
                timestamp_labels[rank].set_text("")

        if marker_offsets:
            rank_markers.set_offsets(np.asarray(marker_offsets))
            rank_markers.set_facecolors(marker_colors)
        else:
            rank_markers.set_offsets(np.empty((0, 2)))

        minimum_line.set_xdata([slowest_time, slowest_time])
        maximum_line.set_xdata([fastest_time, fastest_time])

        drift_band.set_x(slowest_time)
        drift_band.set_width(timing_gap)

        active_rank_band.set_y(updated_rank - 0.4)

        status_text.set_text(
            "Update {0}/{1}: Rank {2} advanced to t={3}  |  "
            "slowest={4:g}, fastest={5:g}, gap={6:g}  |  "
            "active ranks={7}/{8}".format(
                frame_index + 1,
                len(events),
                updated_rank,
                updated_time,
                slowest_time,
                fastest_time,
                timing_gap,
                len(initialized_ranks),
                rank_count,
            )
        )

        return []

    fig.subplots_adjust(
        left=0.13,
        right=0.97,
        top=0.87,
        bottom=0.20,
    )

    frame_interval_ms = 1000.0 / fps

    gif_animation = animation.FuncAnimation(
        fig,
        draw_frame,
        frames=len(events),
        init_func=reset_artists,
        interval=frame_interval_ms,
        blit=False,
        repeat=False,
    )

    output_directory = os.path.dirname(os.path.abspath(output_path))
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    writer = animation.PillowWriter(
        fps=fps,
        metadata={
            "title": title,
            "artist": "DSTM timing-gap visualization",
        },
    )

    gif_animation.save(
        output_path,
        writer=writer,
        dpi=dpi,
    )

    plt.close(fig)


def main():
    args = parse_arguments()

    if args.fps <= 0:
        print("error: --fps must be greater than zero", file=sys.stderr)
        return 2

    if args.dpi <= 0:
        print("error: --dpi must be greater than zero", file=sys.stderr)
        return 2

    if not os.path.isfile(args.log_file):
        print(
            "error: log file does not exist: {0}".format(args.log_file),
            file=sys.stderr,
        )
        return 2

    try:
        events, declared_rank_counts = parse_log(args.log_file)
    except (IOError, OSError) as error:
        print(
            "error: could not read log file: {0}".format(error),
            file=sys.stderr,
        )
        return 2

    if not events:
        print(
            "error: no 'Rank N: time step T' records were found in {0}".format(
                args.log_file
            ),
            file=sys.stderr,
        )
        return 2

    try:
        rank_count = determine_rank_count(
            events,
            declared_rank_counts,
            args.ranks,
        )
    except ValueError as error:
        print("error: {0}".format(error), file=sys.stderr)
        return 2

    snapshots, regressions = build_snapshots(events, rank_count)

    if regressions:
        print(
            "warning: found {0} timestamp regression(s). "
            "The visualization preserves them exactly.".format(
                len(regressions)
            ),
            file=sys.stderr,
        )

        for regression in regressions[:10]:
            rank, previous_time, new_time, line_number = regression
            print(
                "  line {0}: Rank {1} moved from t={2} to t={3}".format(
                    line_number,
                    rank,
                    previous_time,
                    new_time,
                ),
                file=sys.stderr,
            )

        if len(regressions) > 10:
            print(
                "  ... {0} additional regression(s) omitted".format(
                    len(regressions) - 10
                ),
                file=sys.stderr,
            )

    duration = len(events) / args.fps

    print("Parsed log:       {0}".format(args.log_file))
    print("Detected ranks:   {0}".format(rank_count))
    print("Animation frames: {0}".format(len(events)))
    print("GIF duration:     {0:.2f} seconds".format(duration))
    print("Writing:          {0}".format(args.output))

    try:
        make_animation(
            events=events,
            snapshots=snapshots,
            rank_count=rank_count,
            output_path=args.output,
            fps=args.fps,
            dpi=args.dpi,
            width=args.width,
            height=args.height,
            title=args.title,
            show_rank_times=not args.no_rank_times,
        )
    except ImportError as error:
        print(
            "error: could not create GIF: {0}\n"
            "Install dependencies with:\n"
            "  pip install matplotlib pillow".format(error),
            file=sys.stderr,
        )
        return 1
    except (IOError, OSError) as error:
        print(
            "error: could not write GIF: {0}".format(error),
            file=sys.stderr,
        )
        return 1

    print("Finished writing {0}".format(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
