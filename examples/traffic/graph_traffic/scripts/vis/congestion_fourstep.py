#!/usr/bin/env python3
"""
Render a single ACM-ready PDF figure showing whole-network traffic congestion
at four points in time, from a one-rank graph_traffic run.

Why one rank
============
graph_traffic block-partitions vertices across MPI ranks and each rank only
holds a *stale, diffused* replica of edges it does not own. With a single rank
(-n 1) rank 0 owns every vertex, so its congestion replica is the true, fully
up-to-date state of the entire network. That makes the one-rank snapshot the
right source for a "here is the congestion of the whole network" figure shown
*before* staleness is introduced.

What it does
============
For each of four requested ticks it runs (or reuses) a --snapshot-dump of
rank 0's replica, which the C code writes as:

    staleness-<mode>-t<TICK>-rank0.csv

It then draws one panel per tick in a 1x4 row, each panel coloring undirected
road segments by congestion (the stored EWMA load of the more-congested
direction). Edges never observed are drawn as faint gray background roads.

Output is a single PDF with:
  * no Type 3 fonts (pdf/ps fonttype 42, TrueType-embedded),
  * no figure title or panel subtitles (explain in the caption),
  * font sizes sized for an ACM two-column paper.

Snapshot CSV format (from vis.c, record_type column)
    node: node_id, original_node_id, x, y, owner, ...
    edge: edge_id, src, dst, stamp, age, value, known, ...
src and dst on edge rows are dense node ids that index the node rows.

Example
=======
    python3 congestion_fourstep.py \
        --ranks 1 --mode stale --steps 80 \
        --ticks 10 30 50 70 \
        --rate 32 --cars 0 --cap 3 --seed 42 \
        --output congestion-fourstep.pdf

Reuse already-dumped CSVs (no MPI run):
    python3 congestion_fourstep.py --skip-run --ticks 10 30 50 70 \
        --output congestion-fourstep.pdf

Compatible with Python 3.6+.
"""

import argparse
import copy
import csv
import math
from pathlib import Path
import shlex
import subprocess
import sys

import matplotlib

# Headless compute nodes.
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# ACM / camera-ready rendering rules.
#
# fonttype 42 embeds TrueType outlines and guarantees NO Type 3 fonts, which
# the ACM submission system rejects. Set before pyplot is used.
# ---------------------------------------------------------------------------
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["pdf.compression"] = 9
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 7
matplotlib.rcParams["axes.titlesize"] = 7
matplotlib.rcParams["axes.labelsize"] = 7
matplotlib.rcParams["xtick.labelsize"] = 6
matplotlib.rcParams["ytick.labelsize"] = 6
matplotlib.rcParams["legend.fontsize"] = 6
matplotlib.rcParams["figure.titlesize"] = 7

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import PowerNorm
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
# This script is intended to live anywhere; resolve the executable/data
# relative to a graph_traffic layout only as a convenience default.
DEFAULT_GRAPH_TRAFFIC_DIR = SCRIPT_DIR / ".." / ".."
DEFAULT_EXECUTABLE = DEFAULT_GRAPH_TRAFFIC_DIR / "traffic"
DEFAULT_DATA_DIR = DEFAULT_GRAPH_TRAFFIC_DIR / "data" / "bloomington"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Render a 1x4 PDF of whole-network congestion at four ticks "
            "from a one-rank graph_traffic run."
        )
    )

    parser.add_argument(
        "--executable",
        type=Path,
        default=DEFAULT_EXECUTABLE,
        help="traffic executable (default: {})".format(DEFAULT_EXECUTABLE),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=(
            "directory containing node.csv and link.csv "
            "(default: {})".format(DEFAULT_DATA_DIR)
        ),
    )
    parser.add_argument(
        "--ranks",
        type=int,
        default=1,
        help=(
            "number of MPI ranks; use 1 so rank 0 owns the whole graph "
            "and the snapshot is the true global congestion (default: 1)"
        ),
    )
    parser.add_argument(
        "--launcher",
        default="mpirun",
        help="MPI launcher executable (default: mpirun)",
    )
    parser.add_argument(
        "--launcher-args",
        default="",
        help="additional MPI launcher arguments, e.g. '--bind-to core'",
    )
    parser.add_argument(
        "--mode",
        choices=("stale", "oracle", "blind"),
        default="stale",
        help=(
            "replica mode; on one rank stale and oracle are equivalent "
            "(default: stale)"
        ),
    )
    parser.add_argument(
        "--stale-delay",
        type=int,
        default=2,
        help="stale replica merge cadence (default: 2)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=80,
        help="last simulation timestamp (default: 80)",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        nargs=4,
        default=[10, 30, 50, 70],
        metavar=("T0", "T1", "T2", "T3"),
        help="exactly four timestamps to snapshot (default: 10 30 50 70)",
    )
    parser.add_argument(
        "--cars",
        type=int,
        default=0,
        help="initial global trip count (default: 0, use --rate instead)",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=32,
        help=(
            "global trips spawned per tick; continuous demand builds "
            "steady-state congestion (default: 32)"
        ),
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=3,
        help="maximum cars sent per edge per tick (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="simulation random seed (default: 42)",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help=(
            "percentile of congestion used as the shared upper color "
            "limit across all four panels (default: 99)"
        ),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help=(
            "power-normalization exponent; <1 emphasizes light "
            "congestion (default: 0.5)"
        ),
    )
    parser.add_argument(
        "--figwidth",
        type=float,
        default=7.0,
        help=(
            "figure width in inches; 7.0 fills an ACM two-column text "
            "block (default: 7.0)"
        ),
    )
    parser.add_argument(
        "--edge-width",
        type=float,
        default=0.35,
        help="road line width (default: 0.35)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="raster DPI for the embedded line art (default: 600)",
    )
    parser.add_argument(
        "--cmap",
        default="inferno",
        help="matplotlib colormap name for congestion (default: inferno)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("congestion-fourstep.pdf"),
        help="output PDF path (default: congestion-fourstep.pdf)",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="do not run MPI; reuse existing snapshot CSV files",
    )
    parser.add_argument(
        "--keep-csv",
        action="store_true",
        help="retain generated snapshot CSV files after rendering",
    )

    args = parser.parse_args()

    if args.output.suffix.lower() != ".pdf":
        parser.error("--output must end in .pdf")
    if len(set(args.ticks)) != 4:
        parser.error("--ticks must be four distinct timestamps")
    if any(tick < 0 for tick in args.ticks):
        parser.error("--ticks cannot be negative")
    if any(tick > args.steps for tick in args.ticks):
        parser.error("every tick in --ticks must be <= --steps")
    if args.ranks < 1:
        parser.error("--ranks must be at least 1")
    if args.stale_delay < 1:
        parser.error("--stale-delay must be at least 1")
    if args.cars < 0:
        parser.error("--cars cannot be negative")
    if args.rate < 0:
        parser.error("--rate cannot be negative")
    if args.cap < 1:
        parser.error("--cap must be at least 1")
    if not 0.0 < args.percentile <= 100.0:
        parser.error("--percentile must be in (0, 100]")
    if args.gamma <= 0.0:
        parser.error("--gamma must be positive")
    if args.figwidth <= 0.0:
        parser.error("--figwidth must be positive")
    if args.edge_width <= 0.0:
        parser.error("--edge-width must be positive")
    if args.dpi < 1:
        parser.error("--dpi must be at least 1")

    return args


def require_file(path, description):
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("{} not found: {}".format(description, resolved))
    return resolved


def require_data_directory(path):
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError("data directory not found: {}".format(resolved))
    node_path = resolved / "node.csv"
    link_path = resolved / "link.csv"
    if not node_path.is_file():
        raise FileNotFoundError("node.csv not found: {}".format(node_path))
    if not link_path.is_file():
        raise FileNotFoundError("link.csv not found: {}".format(link_path))
    return resolved


def snapshot_path_for(mode, tick):
    return Path.cwd() / "staleness-{}-t{}-rank0.csv".format(mode, tick)


def run_single_snapshot(args, tick):
    """Run graph_traffic once with --snapshot-dump TICK and return the CSV.

    The C code names the snapshot staleness-<mode>-t<TICK>-rank0.csv and only
    rank 0 writes it, so one run produces exactly one snapshot.
    """
    executable = require_file(args.executable, "traffic executable")
    data_dir = require_data_directory(args.data)
    snapshot_path = snapshot_path_for(args.mode, tick)

    # A stale file from a previous run must not masquerade as this run.
    if snapshot_path.exists():
        snapshot_path.unlink()

    command = [args.launcher]
    command.extend(shlex.split(args.launcher_args))
    command.extend([
        "-n", str(args.ranks),
        str(executable),
        "--data", str(data_dir),
        "--steps", str(args.steps),
        "--cars", str(args.cars),
        "--rate", str(args.rate),
        "--cap", str(args.cap),
        "--seed", str(args.seed),
        "--stale-delay", str(args.stale_delay),
        "--replica-mode", args.mode,
        "--snapshot-dump", str(tick),
        "--quiet",
    ])

    print("Running tick {}:".format(tick))
    print("  " + " ".join(shlex.quote(part) for part in command))

    result = subprocess.run(command, cwd=str(Path.cwd()), check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "graph_traffic failed for tick {} with exit status {}".format(
                tick, result.returncode
            )
        )

    if not snapshot_path.is_file():
        raise FileNotFoundError(
            "run for tick {} did not produce {}".format(tick, snapshot_path)
        )

    return snapshot_path


def parse_int(value, field, row_number, path):
    if value is None or value.strip() == "":
        raise ValueError(
            "{}: missing {} at row {}".format(path, field, row_number)
        )
    try:
        return int(value)
    except ValueError:
        raise ValueError(
            "{}: invalid integer {}={!r} at row {}".format(
                path, field, value, row_number
            )
        )


def parse_float(value, field, row_number, path):
    if value is None or value.strip() == "":
        raise ValueError(
            "{}: missing {} at row {}".format(path, field, row_number)
        )
    try:
        return float(value)
    except ValueError:
        raise ValueError(
            "{}: invalid float {}={!r} at row {}".format(
                path, field, value, row_number
            )
        )


def load_snapshot(path):
    """Load one rank-0 snapshot CSV into nodes and directed edges.

    Returns (nodes, edges) where
        nodes[dense_id] = {"x": float, "y": float}
        edges = list of {"src": int, "dst": int, "value": float,
                         "known": bool}
    """
    nodes = {}
    edges = []

    required_fields = {
        "record_type",
        "node_id",
        "x",
        "y",
        "edge_id",
        "src",
        "dst",
        "stamp",
        "value",
    }

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError("{} has no CSV header".format(path))

        missing = required_fields - set(reader.fieldnames)
        if missing:
            raise ValueError(
                "{} is missing fields: {}".format(
                    path, ", ".join(sorted(missing))
                )
            )

        for row_number, row in enumerate(reader, start=2):
            record_type = (row.get("record_type") or "").strip().lower()

            if record_type == "node":
                node_id = parse_int(
                    row.get("node_id"), "node_id", row_number, path
                )
                nodes[node_id] = {
                    "x": parse_float(row.get("x"), "x", row_number, path),
                    "y": parse_float(row.get("y"), "y", row_number, path),
                }

            elif record_type == "edge":
                stamp = parse_int(
                    row.get("stamp"), "stamp", row_number, path
                )
                # stamp < 0 marks an edge the replica has never observed.
                known = stamp >= 0
                value = parse_float(
                    row.get("value"), "value", row_number, path
                ) if known else 0.0

                edges.append({
                    "src": parse_int(
                        row.get("src"), "src", row_number, path
                    ),
                    "dst": parse_int(
                        row.get("dst"), "dst", row_number, path
                    ),
                    "value": value,
                    "known": known,
                })

            else:
                raise ValueError(
                    "{}: unknown record_type {!r} at row {}".format(
                        path, record_type, row_number
                    )
                )

    if not nodes:
        raise ValueError("{} contains no node records".format(path))
    if not edges:
        raise ValueError("{} contains no edge records".format(path))

    for edge in edges:
        if edge["src"] not in nodes or edge["dst"] not in nodes:
            raise ValueError(
                "{}: edge references missing node {} -> {}".format(
                    path, edge["src"], edge["dst"]
                )
            )

    return nodes, edges


def coordinates_look_geographic(nodes):
    xs = [node["x"] for node in nodes.values()]
    ys = [node["y"] for node in nodes.values()]
    return (
        bool(xs)
        and bool(ys)
        and all(-180.0 <= value <= 180.0 for value in xs)
        and all(-90.0 <= value <= 90.0 for value in ys)
    )


def project_coordinates(nodes):
    """Equirectangular projection around mean latitude for geographic data."""
    if not coordinates_look_geographic(nodes):
        return {nid: (node["x"], node["y"]) for nid, node in nodes.items()}

    ys = [node["y"] for node in nodes.values()]
    mean_latitude = sum(ys) / float(len(ys))
    longitude_scale = math.cos(math.radians(mean_latitude))

    return {
        nid: (node["x"] * longitude_scale, node["y"])
        for nid, node in nodes.items()
    }


def build_undirected_roads(edges, coordinates):
    """Collapse directed edges to undirected road segments.

    Returns (segments, road_index_per_edge) where segments[i] is a pair of
    (x, y) endpoints and road_index_per_edge[k] maps directed edge k to its
    road index. link.csv is loaded as two directed edges, so each physical
    road appears twice; it is drawn once.
    """
    road_index_by_key = {}
    segments = []
    road_index_per_edge = np.empty(len(edges), dtype=np.int64)

    for k, edge in enumerate(edges):
        u = edge["src"]
        v = edge["dst"]
        key = (min(u, v), max(u, v))

        index = road_index_by_key.get(key)
        if index is None:
            index = len(segments)
            road_index_by_key[key] = index
            segments.append((coordinates[u], coordinates[v]))

        road_index_per_edge[k] = index

    return segments, road_index_per_edge


def aggregate_roads(edges, road_index_per_edge, road_count):
    """Per-road congestion = max over the two directions of the known value."""
    values = np.full(road_count, -np.inf, dtype=np.float64)
    known = np.zeros(road_count, dtype=np.bool_)

    for k, edge in enumerate(edges):
        if not edge["known"]:
            continue
        road = road_index_per_edge[k]
        if edge["value"] > values[road]:
            values[road] = edge["value"]
        known[road] = True

    values[~known] = 0.0
    return values, known


def determine_color_max(panels, percentile_value):
    samples = []
    for panel in panels:
        known_values = panel["values"][panel["known"]]
        finite = known_values[np.isfinite(known_values)]
        if finite.size:
            samples.append(finite)

    if not samples:
        return 1.0

    combined = np.concatenate(samples)
    color_max = float(np.percentile(combined, percentile_value))

    if not math.isfinite(color_max) or color_max <= 0.0:
        color_max = float(np.max(combined))
    if not math.isfinite(color_max) or color_max <= 0.0:
        color_max = 1.0

    return color_max


def render(panels, segments, args, output_path):
    """Draw the 1x4 row and save a single PDF. No titles or subtitles."""
    road_count = len(segments)

    color_max = determine_color_max(panels, args.percentile)
    normalization = PowerNorm(
        gamma=args.gamma, vmin=0.0, vmax=color_max, clip=True
    )

    congestion_cmap = copy.copy(plt.get_cmap(args.cmap))
    congestion_cmap.set_bad("#cccccc")

    # Shared geographic extent so all four panels align spatially.
    all_x = [point[0] for segment in segments for point in segment]
    all_y = [point[1] for segment in segments for point in segment]
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    x_margin = max((xmax - xmin) * 0.01, 1e-9)
    y_margin = max((ymax - ymin) * 0.01, 1e-9)

    # Layout fractions. The panels occupy the full width edge-to-edge; the
    # colorbar lives in a reserved strip at the bottom. Panels are butted
    # together (wspace ~ 0) to kill the inter-snapshot whitespace.
    #
    # The reserved bottom strip must be tall enough in INCHES to hold the
    # colorbar plus its tick labels and axis label, otherwise the colorbar
    # runs off the canvas. We size that strip in absolute inches and convert
    # it to a figure fraction once the height is known.
    left = 0.004
    right = 0.996
    top = 0.985
    wspace = 0.02          # near-zero gap between snapshots

    colorbar_strip_in = 0.42   # inches reserved at the bottom for the bar

    data_aspect = (ymax - ymin) / (xmax - xmin) if xmax > xmin else 1.0

    # Effective per-panel drawing width in inches, after removing side
    # margins and the tiny inter-panel gaps. Panel height follows from the
    # data aspect; total height is panels + the fixed colorbar strip, so the
    # strip never eats panel area and never gets clipped.
    usable_width = args.figwidth * (right - left)
    panel_width = usable_width / (4.0 + 3.0 * wspace)
    panel_height = panel_width * data_aspect
    figure_height = panel_height / (top - 0.0) + colorbar_strip_in

    # Convert the fixed-inch strip into the bottom fraction the panels start
    # above.
    bottom = colorbar_strip_in / figure_height

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(args.figwidth, figure_height),
        sharex=True,
        sharey=True,
    )
    figure.subplots_adjust(
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        wspace=wspace,
    )

    for index, panel in enumerate(panels):
        axis = axes[index]

        values = panel["values"]
        known = panel["known"]
        masked = np.ma.array(values, mask=~known)

        # Faint background of all roads so unobserved roads remain visible.
        background = LineCollection(
            segments,
            colors="#dddddd",
            linewidths=args.edge_width,
            zorder=1,
        )
        axis.add_collection(background)

        collection = LineCollection(
            segments,
            cmap=congestion_cmap,
            norm=normalization,
            linewidths=args.edge_width,
            zorder=2,
        )
        collection.set_array(masked)
        axis.add_collection(collection)

        axis.set_xlim(xmin - x_margin, xmax + x_margin)
        axis.set_ylim(ymin - y_margin, ymax + y_margin)
        axis.set_aspect("equal", adjustable="box")
        axis.set_axis_off()

        # Small inset step label. Placed just inside the top-left corner so it
        # consumes no extra layout space. Delete this block for caption-only.
        axis.text(
            0.02,
            0.98,
            "t = {}".format(panel["tick"]),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=6,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.7,
            },
        )

    scalar_mappable = ScalarMappable(norm=normalization, cmap=congestion_cmap)
    scalar_mappable.set_array(np.asarray([0.0, color_max]))

    # Explicit slim colorbar axes in the reserved bottom strip: [l, b, w, h]
    # in figure fractions. Position it so the bar sits in the UPPER part of
    # the strip and its tick labels fall in the remaining space below, all
    # safely above the canvas edge. The axis label is placed ABOVE the bar so
    # it can never run off the bottom of the figure.
    bar_height_frac = 0.20 * bottom          # thin bar within the strip
    bar_bottom_frac = 0.45 * bottom          # leave room below for tick text
    colorbar_axis = figure.add_axes(
        [0.30, bar_bottom_frac, 0.40, bar_height_frac]
    )
    colorbar = figure.colorbar(
        scalar_mappable,
        cax=colorbar_axis,
        orientation="horizontal",
    )
    colorbar.set_label("Edge congestion (EWMA vehicles/tick)", fontsize=7)
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.ax.tick_params(labelsize=6, length=2, pad=1)
    colorbar.outline.set_linewidth(0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        str(output_path),
        format="pdf",
        dpi=args.dpi,
    )
    plt.close(figure)

    print()
    print("Wrote {}".format(output_path))
    print("  panels:        {}".format(len(panels)))
    print("  roads/panel:   {:,}".format(road_count))
    print("  color maximum: {:.6g}".format(color_max))
    for panel in panels:
        known_count = int(np.count_nonzero(panel["known"]))
        print(
            "  t={:<5d} known roads {:,}/{:,}".format(
                panel["tick"], known_count, road_count
            )
        )

def main():
    args = parse_arguments()

    generated = []

    try:
        require_data_directory(args.data)

        snapshot_paths = []
        for tick in args.ticks:
            if args.skip_run:
                path = snapshot_path_for(args.mode, tick)
                if not path.is_file():
                    raise FileNotFoundError(
                        "--skip-run set but snapshot is missing: {}".format(
                            path
                        )
                    )
            else:
                path = run_single_snapshot(args, tick)
                generated.append(path)
            snapshot_paths.append(path)

        # The topology is identical across ticks; take it from the first and
        # verify the others match so edges line up on shared segments.
        reference_nodes = None
        reference_edge_count = None
        panels = []
        segments = None
        road_index_per_edge = None

        for tick, path in zip(args.ticks, snapshot_paths):
            nodes, edges = load_snapshot(path)

            if reference_nodes is None:
                reference_nodes = nodes
                reference_edge_count = len(edges)
                coordinates = project_coordinates(nodes)
                segments, road_index_per_edge = build_undirected_roads(
                    edges, coordinates
                )
            else:
                if set(nodes) != set(reference_nodes):
                    raise ValueError(
                        "{} has a different node set".format(path)
                    )
                if len(edges) != reference_edge_count:
                    raise ValueError(
                        "{} has a different edge count".format(path)
                    )

            values, known = aggregate_roads(
                edges, road_index_per_edge, len(segments)
            )
            panels.append({"tick": tick, "values": values, "known": known})

        render(panels, segments, args, args.output.expanduser().resolve())

        if generated and not args.keep_csv:
            for path in generated:
                if path.exists():
                    path.unlink()
            print()
            print("Removed generated CSV files; use --keep-csv to retain.")

    except (
        FileNotFoundError,
        PermissionError,
        ValueError,
        RuntimeError,
        OSError,
    ) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
