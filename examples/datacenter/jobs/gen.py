#!/usr/bin/env python3
"""
Generate a synthetic job trace for the DSTM datacenter example.

Output columns, without a header:

    arrival,duration,nodes,cpu_per_node,gpu_per_node,cpu_util,gpu_util

Example:

    python3 generate_jobs.py \
        --output jobs.csv \
        --jobs 1000 \
        --simulation-seconds 3600 \
        --seed 42 \
        --profile mixed
"""

import argparse
import csv
import math
import random
from pathlib import Path

from typing import List, Tuple


def bounded_lognormal(
    rng: random.Random,
    median: float,
    sigma: float,
    minimum: int,
    maximum: int,
) -> int:
    """Draw a bounded integer from a log-normal distribution."""
    value = rng.lognormvariate(math.log(median), sigma)
    return max(minimum, min(maximum, int(round(value))))


def choose_nodes(rng: random.Random, maximum: int) -> int:
    """Prefer small jobs while occasionally producing large jobs."""
    choices = []
    weights = []

    size = 1
    weight = 1.0

    while size <= maximum:
        choices.append(size)
        weights.append(weight)
        size *= 2
        weight *= 0.58

    if choices[-1] != maximum and maximum > choices[-1]:
        choices.append(maximum)
        weights.append(weights[-1] * 0.5)

    return rng.choices(choices, weights=weights, k=1)[0]


def clipped_gaussian(
    rng: random.Random,
    mean: float,
    deviation: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(minimum, min(maximum, rng.gauss(mean, deviation)))


def choose_job_resources(
    rng: random.Random,
    profile: str,
    cpu_capacity: int,
    gpu_capacity: int,
) -> Tuple[int, int, float, float]:
    """Return cpu request, GPU request, CPU utilization, GPU utilization."""

    if profile == "cpu":
        kind = "cpu"
    elif profile == "gpu":
        kind = "gpu"
    else:
        kind = rng.choices(
            ["cpu", "gpu", "mixed"],
            weights=[0.50, 0.30, 0.20],
            k=1,
        )[0]

    if kind == "cpu":
        cpu_options = [
            value for value in (4, 8, 16, 32, 64)
            if value <= cpu_capacity
        ]
        cpu = rng.choice(cpu_options)
        gpu = 0
        cpu_util = clipped_gaussian(rng, 0.78, 0.15, 0.10, 1.0)
        gpu_util = 0.0

    elif kind == "gpu":
        cpu_options = [
            value for value in (4, 8, 16, 32)
            if value <= cpu_capacity
        ]
        gpu_options = [
            value for value in (1, 2, 4, 8)
            if value <= gpu_capacity
        ]

        cpu = rng.choice(cpu_options)
        gpu = rng.choice(gpu_options)
        cpu_util = clipped_gaussian(rng, 0.45, 0.15, 0.05, 1.0)
        gpu_util = clipped_gaussian(rng, 0.88, 0.10, 0.20, 1.0)

    else:
        cpu_options = [
            value for value in (8, 16, 32, 64)
            if value <= cpu_capacity
        ]
        gpu_options = [
            value for value in (1, 2, 4)
            if value <= gpu_capacity
        ]

        cpu = rng.choice(cpu_options)
        gpu = rng.choice(gpu_options)
        cpu_util = clipped_gaussian(rng, 0.72, 0.15, 0.10, 1.0)
        gpu_util = clipped_gaussian(rng, 0.72, 0.15, 0.10, 1.0)

    return cpu, gpu, cpu_util, gpu_util


def generate_arrivals(
    rng: random.Random,
    job_count: int,
    simulation_seconds: int,
    arrival_model: str,
) -> List[int]:
    if job_count == 0:
        return []

    if arrival_model == "uniform":
        arrivals = [
            rng.randrange(simulation_seconds)
            for _ in range(job_count)
        ]
        arrivals.sort()
        return arrivals

    if arrival_model == "burst":
        # Select a few burst centers and place jobs around them.
        number_of_bursts = max(1, min(12, job_count // 20))
        centers = [
            rng.randrange(simulation_seconds)
            for _ in range(number_of_bursts)
        ]

        spread = max(1.0, simulation_seconds / (number_of_bursts * 20.0))
        arrivals = []

        for _ in range(job_count):
            center = rng.choice(centers)
            arrival = int(round(rng.gauss(center, spread)))
            arrival = max(0, min(simulation_seconds - 1, arrival))
            arrivals.append(arrival)

        arrivals.sort()
        return arrivals

    # Poisson arrival process conditioned approximately on job_count.
    mean_interval = simulation_seconds / float(job_count)
    current = 0.0
    arrivals = []

    for _ in range(job_count):
        current += rng.expovariate(1.0 / mean_interval)
        arrival = min(simulation_seconds - 1, int(current))
        arrivals.append(arrival)

    return arrivals


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic DSTM datacenter job trace."
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("jobs.csv"),
        help="output CSV path; default: jobs.csv",
    )
    parser.add_argument(
        "--jobs",
        "-n",
        type=int,
        default=1000,
        help="number of jobs; default: 1000",
    )
    parser.add_argument(
        "--simulation-seconds",
        type=int,
        default=3600,
        help="arrival window in simulated seconds; default: 3600",
    )
    parser.add_argument(
        "--min-duration",
        type=int,
        default=5,
        help="minimum job duration in seconds; default: 5",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=900,
        help="maximum job duration in seconds; default: 900",
    )
    parser.add_argument(
        "--median-duration",
        type=float,
        default=120.0,
        help="median log-normal job duration; default: 120",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=16,
        help="maximum nodes requested by one job; default: 16",
    )
    parser.add_argument(
        "--cpu-capacity",
        type=int,
        default=64,
        help="CPU capacity per node; default: 64",
    )
    parser.add_argument(
        "--gpu-capacity",
        type=int,
        default=8,
        help="GPU capacity per node; default: 8",
    )
    parser.add_argument(
        "--profile",
        choices=("mixed", "cpu", "gpu"),
        default="mixed",
        help="resource profile; default: mixed",
    )
    parser.add_argument(
        "--arrival-model",
        choices=("poisson", "uniform", "burst"),
        default="poisson",
        help="arrival process; default: poisson",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed; default: 42",
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.jobs < 0:
        raise ValueError("--jobs must be non-negative")
    if args.simulation_seconds < 1:
        raise ValueError("--simulation-seconds must be at least 1")
    if args.min_duration < 1:
        raise ValueError("--min-duration must be at least 1")
    if args.max_duration < args.min_duration:
        raise ValueError("--max-duration must be >= --min-duration")
    if args.median_duration <= 0:
        raise ValueError("--median-duration must be positive")
    if not 1 <= args.max_nodes <= 64:
        raise ValueError("--max-nodes must be between 1 and 64")
    if not 1 <= args.cpu_capacity <= 64:
        raise ValueError("--cpu-capacity must be between 1 and 64")
    if not 1 <= args.gpu_capacity <= 8:
        raise ValueError("--gpu-capacity must be between 1 and 8")


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)

    rng = random.Random(args.seed)

    arrivals = generate_arrivals(
        rng,
        args.jobs,
        args.simulation_seconds,
        args.arrival_model,
    )

    rows = []

    for arrival in arrivals:
        duration = bounded_lognormal(
            rng,
            median=args.median_duration,
            sigma=0.85,
            minimum=args.min_duration,
            maximum=args.max_duration,
        )

        nodes = choose_nodes(rng, args.max_nodes)

        cpu, gpu, cpu_util, gpu_util = choose_job_resources(
            rng,
            args.profile,
            args.cpu_capacity,
            args.gpu_capacity,
        )

        rows.append(
            (
                arrival,
                duration,
                nodes,
                cpu,
                gpu,
                f"{cpu_util:.4f}",
                f"{gpu_util:.4f}",
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file, lineterminator="\n")
        writer.writerows(rows)

    print(f"Wrote {len(rows)} jobs to {args.output}")
    print(f"Arrival model: {args.arrival_model}")
    print(f"Resource profile: {args.profile}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()
