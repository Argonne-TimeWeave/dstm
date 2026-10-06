#!/usr/bin/env python3

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


SYSTEM_ORDER = [
    "dstm",
    "opt_dstm",
    "sst",
    "opt_sst",
]

SYSTEM_STYLE = {
    "dstm": {
        "label": "DSTM",
        "color": "tab:blue",
        "linestyle": "--",
        "marker": "o",
    },
    "opt_dstm": {
        "label": "Optimized DSTM",
        "color": "tab:blue",
        "linestyle": "-",
        "marker": "s",
    },
    "sst": {
        "label": "SST",
        "color": "tab:red",
        "linestyle": "--",
        "marker": "^",
    },
    "opt_sst": {
        "label": "Optimized SST",
        "color": "tab:red",
        "linestyle": "-",
        "marker": "D",
    },
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Plot strong- and weak-scaling runtimes from scaling.csv."
        )
    )

    parser.add_argument(
        "csv_file",
        nargs="?",
        default="results/scaling.csv",
        help="Input scaling CSV (default: results/scaling.csv)",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="results/scaling.png",
        help="Output image (default: results/scaling.png)",
    )

    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="Use a linear runtime axis instead of logarithmic",
    )

    parser.add_argument(
        "--title",
        default="Datacenter simulation scaling",
        help="Overall figure title",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output resolution (default: 200 DPI)",
    )

    return parser.parse_args()


def load_results(csv_path):
    """
    Return:

        data[mode][system][ranks] = [runtime_1, runtime_2, ...]

    Only rows with status=ok and finite positive runtimes are retained.
    """
    data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(list)
        )
    )

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as input_file:
        reader = csv.DictReader(input_file)

        required_columns = {
            "mode",
            "system",
            "mpi_ranks",
            "runtime_seconds",
            "status",
        }

        missing = required_columns - set(
            reader.fieldnames or []
        )

        if missing:
            raise ValueError(
                "CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )

        if "ranks" in (reader.fieldnames or []):
            rank_column = "ranks"
        elif "simulation_ranks" in (reader.fieldnames or []):
            rank_column = "simulation_ranks"
        else:
            raise ValueError(
                "CSV must contain either 'ranks' or "
                "'simulation_ranks'"
            )

        for line_number, row in enumerate(
            reader,
            start=2,
        ):
            if row["status"].strip().lower() != "ok":
                continue

            mode = row["mode"].strip()
            system = row["system"].strip()

            if mode not in {"strong", "weak"}:
                continue

            if system not in SYSTEM_STYLE:
                continue

            try:
                ranks = int(row[rank_column])
                runtime = float(row["runtime_seconds"])
            except ValueError as error:
                raise ValueError(
                    f"Invalid numeric value on CSV line "
                    f"{line_number}: {error}"
                ) from error

            if ranks <= 0:
                continue

            if (
                not math.isfinite(runtime)
                or runtime <= 0.0
            ):
                continue

            data[mode][system][ranks].append(
                runtime
            )

    return data


def aggregate_system(system_data):
    """
    Return sorted rank, mean, minimum, and maximum arrays.
    """
    ranks = sorted(system_data)

    means = []
    minimums = []
    maximums = []

    for rank in ranks:
        samples = system_data[rank]

        means.append(
            sum(samples) / len(samples)
        )

        minimums.append(min(samples))
        maximums.append(max(samples))

    return ranks, means, minimums, maximums


def configure_rank_axis(axis, all_ranks):
    """
    Scaling ranks are powers of two, so a base-2 logarithmic x-axis gives
    equal spacing while retaining the actual rank values as labels.
    """
    if not all_ranks:
        return

    ranks = sorted(set(all_ranks))

    axis.set_xscale("log", base=2)
    axis.set_xticks(ranks)
    axis.set_xticklabels(
        [str(rank) for rank in ranks]
    )

    axis.set_xlim(
        ranks[0] / math.sqrt(2.0),
        ranks[-1] * math.sqrt(2.0),
    )


def plot_mode(
    axis,
    mode,
    mode_data,
    logarithmic_y,
):
    all_ranks = []

    for system in SYSTEM_ORDER:
        if system not in mode_data:
            continue

        ranks, means, minimums, maximums = (
            aggregate_system(mode_data[system])
        )

        if not ranks:
            continue

        all_ranks.extend(ranks)

        style = SYSTEM_STYLE[system]

        axis.plot(
            ranks,
            means,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2.2,
            markersize=6.0,
            markerfacecolor="white",
            markeredgewidth=1.5,
            zorder=3,
        )

        # Show variation across repetitions. With a single repetition,
        # min and max equal the mean and the shaded area is invisible.
        axis.fill_between(
            ranks,
            minimums,
            maximums,
            color=style["color"],
            alpha=0.12,
            linewidth=0.0,
            zorder=2,
        )

    configure_rank_axis(
        axis,
        all_ranks,
    )

    if logarithmic_y:
        axis.set_yscale("log")

    axis.set_title(
        f"{mode.capitalize()} scaling"
    )

    axis.set_xlabel(
        "Simulation / thermal ranks"
    )

    axis.set_ylabel(
        "Runtime (seconds)"
    )

    axis.grid(
        True,
        which="major",
        linestyle=":",
        linewidth=0.8,
        alpha=0.65,
    )

    axis.grid(
        True,
        which="minor",
        linestyle=":",
        linewidth=0.5,
        alpha=0.30,
    )

    axis.legend(
        loc="best",
        frameon=True,
    )


def main():
    args = parse_arguments()

    csv_path = Path(args.csv_file).expanduser()
    output_path = Path(args.output).expanduser()

    if not csv_path.is_file():
        raise SystemExit(
            f"Input CSV does not exist: {csv_path}"
        )

    data = load_results(csv_path)

    if not data:
        raise SystemExit(
            f"No successful timing rows found in {csv_path}"
        )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13.0, 5.4),
        constrained_layout=True,
    )

    logarithmic_y = not args.linear_y

    plot_mode(
        axes[0],
        "strong",
        data.get("strong", {}),
        logarithmic_y,
    )

    plot_mode(
        axes[1],
        "weak",
        data.get("weak", {}),
        logarithmic_y,
    )

    figure.suptitle(
        args.title,
        fontsize=15,
        fontweight="bold",
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=args.dpi,
        bbox_inches="tight",
    )

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
