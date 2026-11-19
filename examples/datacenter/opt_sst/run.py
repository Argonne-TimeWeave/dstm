import argparse
import os
import shlex

import sst


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--racks", type=int, default=16)
    parser.add_argument("--domains", type=int, default=4)
    parser.add_argument("--sst-ranks", type=int, default=0)
    parser.add_argument("--nodes-per-rack", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--schedule-window", type=int, default=60)
    parser.add_argument("--control-latency", type=int, default=30)
    # Fanout >= domains collapses to the old dedicated-scheduler star
    # (tree depth 1). The default mirrors opt_dstm's default fanout.
    parser.add_argument("--tree-fanout", type=int, default=8)
    parser.add_argument("--jobs", default="jobs.csv")

    return parser.parse_args(
        shlex.split(os.environ.get("SST_MODEL_OPTIONS", ""))
    )


args = parse_args()

if args.domains < 1:
    raise ValueError("domains must be positive")

if args.racks < 3:
    raise ValueError("racks must be at least 3")

if args.racks % args.domains != 0:
    raise ValueError("racks must be divisible by domains")

if args.nodes_per_rack < 1:
    raise ValueError("nodes-per-rack must be positive")

if args.steps < 1:
    raise ValueError("steps must be positive")

if args.schedule_window < 1:
    raise ValueError("schedule-window must be positive")

if args.control_latency < 1:
    raise ValueError("control-latency must be positive")

if 2 * args.control_latency > args.schedule_window:
    raise ValueError("control round trip exceeds schedule window")

fanout = args.tree_fanout
if fanout < 2:
    fanout = 2

required_ranks = args.domains + 1

if args.sst_ranks > 0 and args.sst_ranks != required_ranks:
    raise ValueError(
        "--sst-ranks must equal domains + 1: "
        f"expected {required_ranks}, received {args.sst_ranks}"
    )


# ---------------------------------------------------------------------------
# Relay tree over MPI ranks. Tree node 0 is the scheduler; tree node d+1 is
# thermal domain d. parent(t) = (t-1)//K; children are K*t+1 .. K*t+K.
# ---------------------------------------------------------------------------
node_count = args.domains + 1


def tree_parent(node):
    return -1 if node == 0 else (node - 1) // fanout


def tree_first_child(node):
    return fanout * node + 1


def tree_depth_of(node):
    depth = 0
    while node > 0:
        node = (node - 1) // fanout
        depth += 1
    return depth


depth = tree_depth_of(node_count - 1)
if depth < 1:
    depth = 1

schedule_latency_total = args.schedule_window - args.control_latency

if args.control_latency % depth != 0 or schedule_latency_total % depth != 0:
    raise ValueError(
        f"tree depth {depth} must divide control-latency "
        f"({args.control_latency}) and schedule delay "
        f"({schedule_latency_total}); choose a --tree-fanout whose depth does"
    )

hop_up = args.control_latency // depth
hop_down = schedule_latency_total // depth

common_params = {
    "domains": args.domains,
    "racks": args.racks,
    "nodes_per_rack": args.nodes_per_rack,
    "steps": args.steps,
    "schedule_window": args.schedule_window,
    "control_latency": args.control_latency,
    "tree_fanout": fanout,
}

scheduler = sst.Component("scheduler", "opt_dcbench.scheduler")
scheduler.addParams({**common_params, "jobs": args.jobs})

# Match DSTM: MPI rank 0 is the scheduler, rank d+1 is thermal domain d.
scheduler.setRank(0, 0)

thermal = []
for domain in range(args.domains):
    component = sst.Component(f"thermal_{domain}", "opt_dcbench.thermal")
    component.addParams({**common_params, "domain": domain})
    component.setRank(domain + 1, 0)
    thermal.append(component)

# Thermal ring (heat), unchanged: intrinsic 1s latency + HeatEvent::send(delay-1).
if args.domains > 1:
    for domain in range(args.domains):
        neighbor = (domain + 1) % args.domains
        link = sst.Link(f"thermal_link_{domain}")
        link.connect(
            (thermal[domain], "right", "1s"),
            (thermal[neighbor], "left", "1s"),
        )


def parent_component(parent_node):
    # Tree node 0 is the scheduler; tree node p (>0) is thermal domain p-1.
    return scheduler if parent_node == 0 else thermal[parent_node - 1]


# Control tree. Each thermal domain links up to its parent:
#   feedback (child -> parent) carries hop_up seconds,
#   schedule (parent -> child) carries hop_down seconds.
# Both endpoints use the same per-hop latency, exactly as the star did, so a
# one-way send is one hop. Child index ci = tree_node - first_child(parent)
# selects the parent's dn_fb{ci} / dn_sched{ci} port.
for domain in range(args.domains):
    node = domain + 1
    parent_node = tree_parent(node)
    ci = node - tree_first_child(parent_node)
    parent = parent_component(parent_node)

    feedback_link = sst.Link(f"feedback_link_{domain}")
    feedback_link.connect(
        (thermal[domain], "up_fb", f"{hop_up}s"),
        (parent, f"dn_fb{ci}", f"{hop_up}s"),
    )

    schedule_link = sst.Link(f"schedule_link_{domain}")
    schedule_link.connect(
        (parent, f"dn_sched{ci}", f"{hop_down}s"),
        (thermal[domain], "up_sched", f"{hop_down}s"),
    )
