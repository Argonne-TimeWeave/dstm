import argparse
import os
import shlex

import sst


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--racks", type=int, default=16)
    parser.add_argument("--domains", type=int, default=4)
    parser.add_argument("--sst-ranks", type=int, default=0)

    parser.add_argument(
        "--nodes-per-rack",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--schedule-window",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--control-latency",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--jobs",
        default="jobs.csv",
    )

    return parser.parse_args(
        shlex.split(
            os.environ.get(
                "SST_MODEL_OPTIONS",
                "",
            )
        )
    )


args = parse_args()

if args.domains < 1:
    raise ValueError(
        "domains must be positive"
    )

if args.racks < 3:
    raise ValueError(
        "racks must be at least 3"
    )

if args.racks % args.domains != 0:
    raise ValueError(
        "racks must be divisible by domains"
    )

if args.nodes_per_rack < 1:
    raise ValueError(
        "nodes-per-rack must be positive"
    )

if args.steps < 1:
    raise ValueError(
        "steps must be positive"
    )

if args.schedule_window < 1:
    raise ValueError(
        "schedule-window must be positive"
    )

if args.control_latency < 1:
    raise ValueError(
        "control-latency must be positive"
    )

if (
    2 * args.control_latency
    > args.schedule_window
):
    raise ValueError(
        "control round trip exceeds "
        "schedule window"
    )

required_ranks = args.domains + 1

if (
    args.sst_ranks > 0
    and args.sst_ranks != required_ranks
):
    raise ValueError(
        "--sst-ranks must equal "
        "domains + 1: "
        f"expected {required_ranks}, "
        f"received {args.sst_ranks}"
    )

schedule_latency = (
    args.schedule_window
    - args.control_latency
)

common_params = {
    "domains": args.domains,
    "racks": args.racks,
    "nodes_per_rack": args.nodes_per_rack,
    "steps": args.steps,
    "schedule_window": args.schedule_window,
    "control_latency": args.control_latency,
}

scheduler = sst.Component(
    "scheduler",
    "opt_dcbench.scheduler",
)

scheduler.addParams({
    **common_params,
    "jobs": args.jobs,
})

# Match DSTM:
#
#   MPI rank 0: dedicated scheduler
#   MPI rank d+1: thermal domain d
scheduler.setRank(0, 0)

thermal = []

for domain in range(args.domains):
    component = sst.Component(
        f"thermal_{domain}",
        "opt_dcbench.thermal",
    )

    component.addParams({
        **common_params,
        "domain": domain,
    })

    component.setRank(domain + 1, 0)

    thermal.append(component)

# Thermal ring.
#
# Link d connects:
#
#   thermal d right
#   thermal (d + 1) mod domains left
#
# The intrinsic one-second latency is supplemented by
# HeatEvent::send(delay - 1), producing total latency 1..3s.
if args.domains > 1:
    for domain in range(args.domains):
        neighbor = (
            domain + 1
        ) % args.domains

        link = sst.Link(
            f"thermal_link_{domain}"
        )

        link.connect(
            (
                thermal[domain],
                "right",
                "1s",
            ),
            (
                thermal[neighbor],
                "left",
                "1s",
            ),
        )

# Dedicated feedback and schedule links are separate because
# their one-way delays need not be equal.
for domain in range(args.domains):
    feedback_link = sst.Link(
        f"feedback_link_{domain}"
    )

    feedback_link.connect(
        (
            thermal[domain],
            "feedback",
            f"{args.control_latency}s",
        ),
        (
            scheduler,
            f"feedback{domain}",
            f"{args.control_latency}s",
        ),
    )

    schedule_link = sst.Link(
        f"schedule_link_{domain}"
    )

    schedule_link.connect(
        (
            scheduler,
            f"schedule{domain}",
            f"{schedule_latency}s",
        ),
        (
            thermal[domain],
            "schedule",
            f"{schedule_latency}s",
        ),
    )
