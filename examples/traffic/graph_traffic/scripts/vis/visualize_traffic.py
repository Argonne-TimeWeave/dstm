#!/usr/bin/env python3
"""
Run graph_traffic and visualize four geographic corner perspectives of
perceived congestion.

Expected layout:

    graph_traffic/
    ├── traffic
    ├── data/
    │   └── bloomington/
    │       ├── node.csv
    │       └── link.csv
    └── scripts/
        └── vis/
            └── visualize_traffic.py

The traffic executable must support:

    --traffic-vis
    --traffic-vis-warmup N

and produce four files named similarly to:

    traffic-view-southwest-node0-rank0-stale-d2.csv
    traffic-view-northwest-node880-rank8-stale-d2.csv
    traffic-view-southeast-node1566-rank15-stale-d2.csv
    traffic-view-northeast-node3132-rank31-stale-d2.csv

Each panel shows one corner node's owner-rank perspective. The underlying
replica is rank-local, so a corner node represents the congestion information
available to the rank owning that node.

By default, this script creates:

    traffic-congestion-stale-d2.gif
    traffic-congestion-stale-d2-final.png

The animated GIF contains all timestamps common to the four view files. The
PNG contains the final available warmup timestamp.

This script is compatible with Python 3.6.
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

# Support headless compute nodes.
matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import PowerNorm
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
GRAPH_TRAFFIC_DIR = SCRIPT_DIR.parent.parent

DEFAULT_EXECUTABLE = GRAPH_TRAFFIC_DIR / "traffic"
DEFAULT_DATA_DIR = GRAPH_TRAFFIC_DIR / "data" / "bloomington"

CORNER_ORDER = (
    "northwest",
    "northeast",
    "southwest",
    "southeast",
)

METRIC_LABELS = {
    "volume_capacity_ratio": (
        "Perceived effective congestion / edge capacity"
    ),
    "effective_congestion": (
        "Perceived age-decayed congestion (vehicles/tick)"
    ),
    "congestion_ewma": (
        "Last observed congestion EWMA (vehicles/tick)"
    ),
    "route_penalty": (
        "Perceived additive congestion route penalty (ticks)"
    ),
    "perceived_route_cost": (
        "Perceived total route cost (ticks)"
    ),
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run graph_traffic and create a four-panel animated congestion "
            "visualization from geographic corner perspectives."
        )
    )

    parser.add_argument(
        "--executable",
        type=Path,
        default=DEFAULT_EXECUTABLE,
        help="traffic executable (default: {})".format(
            DEFAULT_EXECUTABLE
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="directory containing node.csv and link.csv (default: {})".format(
            DEFAULT_DATA_DIR
        ),
    )
    parser.add_argument(
        "--ranks",
        type=int,
        default=32,
        help="number of MPI ranks (default: 32)",
    )
    parser.add_argument(
        "--launcher",
        default="mpirun",
        help="MPI launcher executable (default: mpirun)",
    )
    parser.add_argument(
        "--launcher-args",
        default="",
        help=(
            "additional MPI launcher arguments; example: "
            "'--bind-to core'"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("stale", "oracle", "blind"),
        default="stale",
        help="replica mode (default: stale)",
    )
    parser.add_argument(
        "--stale-delay",
        type=int,
        default=2,
        help="stale replica diffusion cadence (default: 2)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help=(
            "number of timestamps to record; records 0 <= t < N "
            "(default: 20)"
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=60,
        help="last simulation timestamp (default: 60)",
    )
    parser.add_argument(
        "--cars",
        type=int,
        default=512,
        help="initial global trip count (default: 512)",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=32,
        help="global trips spawned per tick (default: 32)",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=8,
        help="maximum cars sent per edge per tick (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="simulation random seed (default: 42)",
    )
    parser.add_argument(
        "--metric",
        choices=tuple(sorted(METRIC_LABELS)),
        default="volume_capacity_ratio",
        help=(
            "edge quantity used for color (default: "
            "volume_capacity_ratio)"
        ),
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help=(
            "percentile used as the upper color limit "
            "(default: 99)"
        ),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help=(
            "power-normalization exponent; values below 1 emphasize "
            "light congestion (default: 0.5)"
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=3,
        help="animation frames per second (default: 3)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="animation and PNG resolution (default: 160)",
    )
    parser.add_argument(
        "--edge-width",
        type=float,
        default=0.55,
        help="road width (default: 0.55)",
    )
    parser.add_argument(
        "--node-size",
        type=float,
        default=0.6,
        help="background node size (default: 0.6)",
    )
    parser.add_argument(
        "--viewpoint-size",
        type=float,
        default=90.0,
        help="corner viewpoint marker size (default: 90)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "animated output path (.gif or .mp4); default is "
            "traffic-congestion-<mode>-d<D>.gif"
        ),
    )
    parser.add_argument(
        "--frame-output",
        type=Path,
        default=None,
        help=(
            "static PNG output path; default is "
            "traffic-congestion-<mode>-d<D>-final.png"
        ),
    )
    parser.add_argument(
        "--frame-tick",
        type=int,
        default=None,
        help=(
            "timestamp used for the static PNG; default is the final "
            "available timestamp"
        ),
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help=(
            "do not execute MPI; visualize existing traffic-view CSV files"
        ),
    )
    parser.add_argument(
        "--keep-csv",
        action="store_true",
        help=(
            "retain generated traffic-view CSV files after visualization"
        ),
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="create only the static PNG",
    )

    args = parser.parse_args()

    if args.ranks < 1:
        parser.error("--ranks must be at least 1")
    if args.stale_delay < 1:
        parser.error("--stale-delay must be at least 1")
    if args.warmup < 1:
        parser.error("--warmup must be at least 1")
    if args.steps < 0:
        parser.error("--steps cannot be negative")
    if args.steps + 1 < args.warmup:
        parser.error(
            "--steps must provide at least --warmup timestamps; "
            "require steps + 1 >= warmup"
        )
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
    if args.fps < 1:
        parser.error("--fps must be at least 1")
    if args.dpi < 1:
        parser.error("--dpi must be at least 1")
    if args.edge_width <= 0.0:
        parser.error("--edge-width must be positive")
    if args.node_size < 0.0:
        parser.error("--node-size cannot be negative")
    if args.viewpoint_size <= 0.0:
        parser.error("--viewpoint-size must be positive")

    return args


def require_file(path, description):
    resolved = path.expanduser().resolve()

    if not resolved.is_file():
        raise FileNotFoundError(
            "{} not found: {}".format(description, resolved)
        )

    return resolved


def require_data_directory(path):
    resolved = path.expanduser().resolve()

    if not resolved.is_dir():
        raise FileNotFoundError(
            "data directory not found: {}".format(resolved)
        )

    node_path = resolved / "node.csv"
    link_path = resolved / "link.csv"

    if not node_path.is_file():
        raise FileNotFoundError(
            "node.csv not found: {}".format(node_path)
        )

    if not link_path.is_file():
        raise FileNotFoundError(
            "link.csv not found: {}".format(link_path)
        )

    return resolved


def parse_int(value, field, row_number, path):
    if value is None or value.strip() == "":
        raise ValueError(
            "{}: missing {} at CSV row {}".format(
                path, field, row_number
            )
        )

    try:
        return int(value)
    except ValueError:
        raise ValueError(
            "{}: invalid integer {}={!r} at CSV row {}".format(
                path, field, value, row_number
            )
        )


def parse_float(value, field, row_number, path):
    if value is None or value.strip() == "":
        raise ValueError(
            "{}: missing {} at CSV row {}".format(
                path, field, row_number
            )
        )

    try:
        return float(value)
    except ValueError:
        raise ValueError(
            "{}: invalid floating-point {}={!r} at CSV row {}".format(
                path, field, value, row_number
            )
        )


def traffic_file_pattern(args):
    return "traffic-view-*-{}-d{}.csv".format(
        args.mode,
        args.stale_delay,
    )


def discover_traffic_files(args):
    pattern = traffic_file_pattern(args)
    files = sorted(Path.cwd().glob(pattern))

    if len(files) != 4:
        raise FileNotFoundError(
            "expected exactly four traffic-view files matching {!r}, "
            "but found {}: {}".format(
                pattern,
                len(files),
                ", ".join(str(path) for path in files) or "(none)",
            )
        )

    return files


def remove_old_traffic_files(args):
    pattern = traffic_file_pattern(args)

    for path in Path.cwd().glob(pattern):
        path.unlink()


def run_simulation(args):
    executable = require_file(
        args.executable,
        "traffic executable",
    )
    data_dir = require_data_directory(args.data)

    remove_old_traffic_files(args)

    command = [
        args.launcher,
    ]

    command.extend(shlex.split(args.launcher_args))

    command.extend([
        "-n",
        str(args.ranks),
        str(executable),
        "--data",
        str(data_dir),
        "--steps",
        str(args.steps),
        "--cars",
        str(args.cars),
        "--rate",
        str(args.rate),
        "--cap",
        str(args.cap),
        "--seed",
        str(args.seed),
        "--stale-delay",
        str(args.stale_delay),
        "--replica-mode",
        args.mode,
        "--traffic-vis",
        "--traffic-vis-warmup",
        str(args.warmup),
        "--quiet",
    ])

    print("Running simulation:")
    print(
        "  "
        + " ".join(shlex.quote(part) for part in command)
    )
    print()

    result = subprocess.run(
        command,
        cwd=str(Path.cwd()),
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "graph_traffic failed with exit status {}".format(
                result.returncode
            )
        )

    return discover_traffic_files(args)


def finalize_frame(
    frames,
    known_frames,
    tick,
    values,
    known,
    path,
):
    if tick is None:
        return

    frame = np.asarray(values, dtype=np.float64)
    known_frame = np.asarray(known, dtype=np.bool_)

    if frame.size == 0:
        raise ValueError(
            "{}: timestamp {} contains no edges".format(path, tick)
        )

    frames[tick] = frame
    known_frames[tick] = known_frame


def load_traffic_view(path, metric, reference_topology=None):
    nodes = {}
    frames = {}
    known_frames = {}

    src_by_edge = []
    dst_by_edge = []

    metadata = {
        "corner": None,
        "view_node": None,
        "view_original_id": None,
        "view_rank": None,
    }

    current_tick = None
    current_values = []
    current_known = []
    expected_edge_id = 0
    topology_tick = None

    required_fields = {
        "record_type",
        "tick",
        "corner",
        "view_node",
        "view_original_id",
        "view_rank",
        "node_id",
        "original_node_id",
        "x",
        "y",
        "node_owner",
        "edge_id",
        "src",
        "dst",
        "stamp",
        "age",
        "known",
        "congestion_ewma",
        "effective_congestion",
        "capacity",
        "volume_capacity_ratio",
        "free_flow_ticks",
        "route_penalty",
        "perceived_route_cost",
    }

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as input_file:
        reader = csv.DictReader(input_file)

        if reader.fieldnames is None:
            raise ValueError(
                "{} has no CSV header".format(path)
            )

        missing = required_fields - set(reader.fieldnames)

        if missing:
            raise ValueError(
                "{} is missing fields: {}".format(
                    path,
                    ", ".join(sorted(missing)),
                )
            )

        for row_number, row in enumerate(reader, start=2):
            record_type = (
                row.get("record_type") or ""
            ).strip().lower()

            if metadata["corner"] is None:
                metadata["corner"] = (
                    row.get("corner") or ""
                ).strip().lower()
                metadata["view_node"] = parse_int(
                    row.get("view_node"),
                    "view_node",
                    row_number,
                    path,
                )
                metadata["view_original_id"] = parse_int(
                    row.get("view_original_id"),
                    "view_original_id",
                    row_number,
                    path,
                )
                metadata["view_rank"] = parse_int(
                    row.get("view_rank"),
                    "view_rank",
                    row_number,
                    path,
                )

            if record_type == "node":
                node_id = parse_int(
                    row.get("node_id"),
                    "node_id",
                    row_number,
                    path,
                )

                nodes[node_id] = {
                    "original_id": parse_int(
                        row.get("original_node_id"),
                        "original_node_id",
                        row_number,
                        path,
                    ),
                    "x": parse_float(
                        row.get("x"),
                        "x",
                        row_number,
                        path,
                    ),
                    "y": parse_float(
                        row.get("y"),
                        "y",
                        row_number,
                        path,
                    ),
                    "owner": parse_int(
                        row.get("node_owner"),
                        "node_owner",
                        row_number,
                        path,
                    ),
                }

            elif record_type == "edge":
                tick = parse_int(
                    row.get("tick"),
                    "tick",
                    row_number,
                    path,
                )

                if current_tick is None:
                    current_tick = tick
                    topology_tick = tick
                    expected_edge_id = 0

                elif tick != current_tick:
                    finalize_frame(
                        frames,
                        known_frames,
                        current_tick,
                        current_values,
                        current_known,
                        path,
                    )

                    current_tick = tick
                    current_values = []
                    current_known = []
                    expected_edge_id = 0

                edge_id = parse_int(
                    row.get("edge_id"),
                    "edge_id",
                    row_number,
                    path,
                )

                if edge_id != expected_edge_id:
                    raise ValueError(
                        "{}: expected edge_id {} but found {} "
                        "at CSV row {}".format(
                            path,
                            expected_edge_id,
                            edge_id,
                            row_number,
                        )
                    )

                src = parse_int(
                    row.get("src"),
                    "src",
                    row_number,
                    path,
                )
                dst = parse_int(
                    row.get("dst"),
                    "dst",
                    row_number,
                    path,
                )

                if tick == topology_tick:
                    src_by_edge.append(src)
                    dst_by_edge.append(dst)

                value = parse_float(
                    row.get(metric),
                    metric,
                    row_number,
                    path,
                )
                known = parse_int(
                    row.get("known"),
                    "known",
                    row_number,
                    path,
                )

                current_values.append(value)
                current_known.append(bool(known))
                expected_edge_id += 1

            else:
                raise ValueError(
                    "{}: unknown record_type {!r} at CSV row {}".format(
                        path,
                        record_type,
                        row_number,
                    )
                )

    finalize_frame(
        frames,
        known_frames,
        current_tick,
        current_values,
        current_known,
        path,
    )

    if not nodes:
        raise ValueError(
            "{} contains no node records".format(path)
        )

    if not frames:
        raise ValueError(
            "{} contains no edge frames".format(path)
        )

    edge_count = len(src_by_edge)

    for tick in frames:
        if len(frames[tick]) != edge_count:
            raise ValueError(
                "{}: timestamp {} has {} edges; expected {}".format(
                    path,
                    tick,
                    len(frames[tick]),
                    edge_count,
                )
            )

    topology = {
        "src": np.asarray(src_by_edge, dtype=np.int64),
        "dst": np.asarray(dst_by_edge, dtype=np.int64),
    }

    if reference_topology is not None:
        if not np.array_equal(
            topology["src"],
            reference_topology["src"],
        ):
            raise ValueError(
                "{} has a different source-edge topology".format(path)
            )

        if not np.array_equal(
            topology["dst"],
            reference_topology["dst"],
        ):
            raise ValueError(
                "{} has a different destination-edge topology".format(path)
            )

    return {
        "path": path,
        "nodes": nodes,
        "frames": frames,
        "known_frames": known_frames,
        "metadata": metadata,
        "topology": topology,
    }


def load_all_views(paths, metric):
    views_by_corner = {}
    reference_topology = None
    reference_nodes = None

    for path in paths:
        view = load_traffic_view(
            path,
            metric,
            reference_topology,
        )

        corner = view["metadata"]["corner"]

        if corner not in CORNER_ORDER:
            raise ValueError(
                "{} has unknown corner {!r}".format(path, corner)
            )

        if corner in views_by_corner:
            raise ValueError(
                "duplicate traffic view for corner {!r}".format(corner)
            )

        if reference_topology is None:
            reference_topology = view["topology"]
            reference_nodes = view["nodes"]
        else:
            if set(view["nodes"]) != set(reference_nodes):
                raise ValueError(
                    "{} contains a different node set".format(path)
                )

        views_by_corner[corner] = view

    missing = set(CORNER_ORDER) - set(views_by_corner)

    if missing:
        raise ValueError(
            "missing corner views: {}".format(
                ", ".join(sorted(missing))
            )
        )

    return [
        views_by_corner[corner]
        for corner in CORNER_ORDER
    ]


def coordinates_look_geographic(nodes):
    x_values = [node["x"] for node in nodes.values()]
    y_values = [node["y"] for node in nodes.values()]

    return (
        x_values
        and y_values
        and all(-180.0 <= value <= 180.0 for value in x_values)
        and all(-90.0 <= value <= 90.0 for value in y_values)
    )


def project_coordinates(nodes):
    geographic = coordinates_look_geographic(nodes)

    if not geographic:
        return {
            node_id: (node["x"], node["y"])
            for node_id, node in nodes.items()
        }

    y_values = [node["y"] for node in nodes.values()]
    mean_latitude = sum(y_values) / float(len(y_values))
    longitude_scale = math.cos(math.radians(mean_latitude))

    return {
        node_id: (
            node["x"] * longitude_scale,
            node["y"],
        )
        for node_id, node in nodes.items()
    }


def build_undirected_roads(topology, coordinates):
    src = topology["src"]
    dst = topology["dst"]
    edge_count = len(src)

    road_index_by_key = {}
    road_index_by_edge = np.empty(edge_count, dtype=np.int64)
    segments = []

    for edge_id in range(edge_count):
        u = int(src[edge_id])
        v = int(dst[edge_id])
        key = (min(u, v), max(u, v))

        road_index = road_index_by_key.get(key)

        if road_index is None:
            road_index = len(segments)
            road_index_by_key[key] = road_index

            segments.append((
                coordinates[u],
                coordinates[v],
            ))

        road_index_by_edge[edge_id] = road_index

    return (
        segments,
        road_index_by_edge,
    )


def aggregate_directed_edges(
    directed_values,
    directed_known,
    road_index_by_edge,
    road_count,
):
    road_values = np.full(
        road_count,
        -np.inf,
        dtype=np.float64,
    )
    road_known = np.zeros(
        road_count,
        dtype=np.bool_,
    )

    valid = (
        directed_known
        & np.isfinite(directed_values)
    )

    if np.any(valid):
        valid_roads = road_index_by_edge[valid]
        valid_values = directed_values[valid]

        # A physical road can have two directed congestion values. Show
        # the more congested direction.
        np.maximum.at(
            road_values,
            valid_roads,
            valid_values,
        )
        np.logical_or.at(
            road_known,
            valid_roads,
            True,
        )

    road_values[~road_known] = 0.0

    return road_values, road_known


def common_ticks(views):
    tick_set = set(views[0]["frames"])

    for view in views[1:]:
        tick_set &= set(view["frames"])

    ticks = sorted(tick_set)

    if not ticks:
        raise ValueError(
            "the four view files have no common timestamps"
        )

    return ticks


def determine_color_max(
    views,
    ticks,
    road_index_by_edge,
    road_count,
    percentile_value,
):
    samples = []

    for view in views:
        for tick in ticks:
            road_values, road_known = aggregate_directed_edges(
                view["frames"][tick],
                view["known_frames"][tick],
                road_index_by_edge,
                road_count,
            )

            valid_values = road_values[
                road_known & np.isfinite(road_values)
            ]

            if valid_values.size:
                samples.append(valid_values)

    if not samples:
        return 1.0

    combined = np.concatenate(samples)
    color_max = float(
        np.percentile(combined, percentile_value)
    )

    if not math.isfinite(color_max) or color_max <= 0.0:
        color_max = float(np.max(combined))

    if not math.isfinite(color_max) or color_max <= 0.0:
        color_max = 1.0

    return color_max


def create_visualization(
    views,
    args,
    output_path,
    frame_output_path,
):
    ticks = common_ticks(views)

    nodes = views[0]["nodes"]
    topology = views[0]["topology"]
    coordinates = project_coordinates(nodes)

    segments, road_index_by_edge = build_undirected_roads(
        topology,
        coordinates,
    )
    road_count = len(segments)

    color_max = determine_color_max(
        views,
        ticks,
        road_index_by_edge,
        road_count,
        args.percentile,
    )

    normalization = PowerNorm(
        gamma=args.gamma,
        vmin=0.0,
        vmax=color_max,
        clip=True,
    )

    congestion_cmap = copy.copy(plt.cm.inferno)
    congestion_cmap.set_bad("#b8b8b8")

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(17, 13),
        sharex=True,
        sharey=True,
    )

    figure.patch.set_facecolor("#f3f1ec")

    all_x = [position[0] for position in coordinates.values()]
    all_y = [position[1] for position in coordinates.values()]

    xmin = min(all_x)
    xmax = max(all_x)
    ymin = min(all_y)
    ymax = max(all_y)

    x_margin = max((xmax - xmin) * 0.015, 1.0)
    y_margin = max((ymax - ymin) * 0.015, 1.0)

    line_collections = []
    status_texts = []

    for axes_index, view in enumerate(views):
        axis = axes.flat[axes_index]
        metadata = view["metadata"]
        corner = metadata["corner"]
        viewpoint = metadata["view_node"]
        view_rank = metadata["view_rank"]

        axis.set_facecolor("#f3f1ec")

        collection = LineCollection(
            segments,
            cmap=congestion_cmap,
            norm=normalization,
            linewidths=args.edge_width,
            alpha=0.95,
            zorder=2,
        )
        axis.add_collection(collection)
        line_collections.append(collection)

        if args.node_size > 0.0:
            axis.scatter(
                all_x,
                all_y,
                s=args.node_size,
                color="#505050",
                linewidths=0.0,
                alpha=0.16,
                zorder=1,
            )

        viewpoint_x, viewpoint_y = coordinates[viewpoint]

        axis.scatter(
            [viewpoint_x],
            [viewpoint_y],
            marker="*",
            s=args.viewpoint_size,
            color="#00e5ff",
            edgecolors="#101010",
            linewidths=0.75,
            zorder=5,
        )

        axis.set_xlim(xmin - x_margin, xmax + x_margin)
        axis.set_ylim(ymin - y_margin, ymax + y_margin)
        axis.set_aspect("equal", adjustable="box")
        axis.set_axis_off()

        axis.set_title(
            "{} viewpoint\n"
            "dense node {}, original {}, owner rank {}".format(
                corner.capitalize(),
                viewpoint,
                metadata["view_original_id"],
                view_rank,
            ),
            fontsize=11,
        )

        status = axis.text(
            0.015,
            0.015,
            "",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="#202020",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#999999",
                "alpha": 0.88,
            },
            zorder=10,
        )
        status_texts.append(status)

    scalar_mappable = ScalarMappable(
        norm=normalization,
        cmap=congestion_cmap,
    )
    scalar_mappable.set_array(np.asarray([0.0, color_max]))

    colorbar = figure.colorbar(
        scalar_mappable,
        ax=list(axes.flat),
        fraction=0.025,
        pad=0.018,
    )
    colorbar.set_label(
        METRIC_LABELS[args.metric]
        + "\n"
        + "color scale clipped at percentile {:.1f}".format(
            args.percentile
        ),
        fontsize=10,
    )

    title = figure.suptitle(
        "",
        fontsize=15,
    )

    def render_tick(tick):
        total_known = 0
        total_roads = 0

        for index, view in enumerate(views):
            road_values, road_known = aggregate_directed_edges(
                view["frames"][tick],
                view["known_frames"][tick],
                road_index_by_edge,
                road_count,
            )

            masked_values = np.ma.array(
                road_values,
                mask=~road_known,
            )

            line_collections[index].set_array(masked_values)

            known_count = int(np.count_nonzero(road_known))
            unknown_count = road_count - known_count

            known_values = road_values[road_known]

            if known_values.size:
                mean_value = float(np.mean(known_values))
                maximum_value = float(np.max(known_values))
            else:
                mean_value = float("nan")
                maximum_value = float("nan")

            status_texts[index].set_text(
                "known roads: {:,} / {:,}\n"
                "unknown roads: {:,}\n"
                "mean: {:.4g}\n"
                "maximum: {:.4g}".format(
                    known_count,
                    road_count,
                    unknown_count,
                    mean_value,
                    maximum_value,
                )
            )

            total_known += known_count
            total_roads += road_count

        title.set_text(
            "Perceived traffic congestion at tick {}\n"
            "mode={}, MPI ranks={}, stale delay={}, "
            "metric={}".format(
                tick,
                args.mode,
                args.ranks,
                args.stale_delay,
                args.metric,
            )
        )

        artists = list(line_collections)
        artists.extend(status_texts)
        artists.append(title)
        return artists

    if args.frame_tick is None:
        frame_tick = ticks[-1]
    else:
        frame_tick = args.frame_tick

        if frame_tick not in ticks:
            raise ValueError(
                "requested --frame-tick {} is not present in all four "
                "view files; available common ticks are {} through {}".format(
                    frame_tick,
                    ticks[0],
                    ticks[-1],
                )
            )

    render_tick(frame_tick)

    frame_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        str(frame_output_path),
        dpi=args.dpi,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )

    print(
        "Static frame:  {} (tick {})".format(
            frame_output_path,
            frame_tick,
        )
    )

    if not args.no_animation:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        movie = animation.FuncAnimation(
            figure,
            render_tick,
            frames=ticks,
            interval=1000.0 / float(args.fps),
            blit=False,
            repeat=True,
        )

        suffix = output_path.suffix.lower()

        if suffix == ".gif":
            try:
                writer = animation.PillowWriter(
                    fps=args.fps,
                )
            except AttributeError:
                raise RuntimeError(
                    "this Matplotlib installation does not provide "
                    "PillowWriter; install Pillow or use an MP4 output "
                    "with ffmpeg"
                )

        elif suffix == ".mp4":
            writer = animation.FFMpegWriter(
                fps=args.fps,
                bitrate=2400,
            )

        else:
            raise ValueError(
                "--output must end in .gif or .mp4"
            )

        print(
            "Writing animation with {} frames: {}".format(
                len(ticks),
                output_path,
            )
        )

        movie.save(
            str(output_path),
            writer=writer,
            dpi=args.dpi,
        )

    plt.close(figure)

    print()
    print("Traffic visualization summary:")
    print("  viewpoints:       {}".format(len(views)))
    print("  nodes:            {:,}".format(len(nodes)))
    print("  directed edges:   {:,}".format(len(topology["src"])))
    print("  physical roads:   {:,}".format(road_count))
    print("  first tick:       {}".format(ticks[0]))
    print("  final tick:       {}".format(ticks[-1]))
    print("  animation frames: {}".format(len(ticks)))
    print("  metric:           {}".format(args.metric))
    print("  color maximum:    {:.6g}".format(color_max))

    for view in views:
        metadata = view["metadata"]
        print(
            "  {:9s}: node {:4d}, rank {:2d}, {}".format(
                metadata["corner"],
                metadata["view_node"],
                metadata["view_rank"],
                view["path"],
            )
        )


def main():
    args = parse_arguments()

    generated_files = []

    try:
        require_data_directory(args.data)

        if args.skip_run:
            traffic_files = discover_traffic_files(args)
        else:
            traffic_files = run_simulation(args)
            generated_files = list(traffic_files)

        views = load_all_views(
            traffic_files,
            args.metric,
        )

        if args.output is None:
            output_path = (
                Path.cwd()
                / "traffic-congestion-{}-d{}.gif".format(
                    args.mode,
                    args.stale_delay,
                )
            )
        else:
            output_path = args.output.expanduser().resolve()

        if args.frame_output is None:
            frame_output_path = (
                Path.cwd()
                / "traffic-congestion-{}-d{}-final.png".format(
                    args.mode,
                    args.stale_delay,
                )
            )
        else:
            frame_output_path = (
                args.frame_output.expanduser().resolve()
            )

        create_visualization(
            views,
            args,
            output_path,
            frame_output_path,
        )

        if (
            generated_files
            and not args.keep_csv
        ):
            for path in generated_files:
                if path.exists():
                    path.unlink()

            print()
            print(
                "Removed generated CSV files. Use --keep-csv to retain them."
            )

    except (
        FileNotFoundError,
        PermissionError,
        ValueError,
        RuntimeError,
        OSError,
    ) as error:
        print(
            "error: {}".format(error),
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
