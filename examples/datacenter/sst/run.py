import argparse
import os
import shlex
import sst

parser = argparse.ArgumentParser()

parser.add_argument("--racks", type=int, default=16)
parser.add_argument("--domains", type=int, default=4)
parser.add_argument("--sst-ranks", type=int, default=0)
parser.add_argument("--nodes-per-rack", type=int, default=8)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--jobs", default="jobs.csv")

args = parser.parse_args(
    shlex.split(os.environ.get("SST_MODEL_OPTIONS", ""))
)

if args.racks % args.domains != 0:
    raise ValueError("racks must be divisible by domains")

components = []

for domain in range(args.domains):
    component = sst.Component(
        f"domain_{domain}",
        "dcbench.datacenter"
    )

    component.addParams({
        "domain": domain,
        "domains": args.domains,
        "racks": args.racks,
        "nodes_per_rack": args.nodes_per_rack,
        "steps": args.steps,
        "jobs": args.jobs,
    })

    if args.sst_ranks > 0:
        component.setRank(domain % args.sst_ranks, 0)

    components.append(component)

for domain in range(args.domains):
    neighbor = (domain + 1) % args.domains

    link = sst.Link(f"domain_link_{domain}")
    link.connect(
        (components[domain], "right", "1s"),
        (components[neighbor], "left", "1s")
    )

# Central scheduling/control network.
#
# Domain 0 sends one schedule directly to every other domain. The 1s
# latency matches the DSTM scheduler's put at (t, +1).
for domain in range(1, args.domains):
    link = sst.Link(f"schedule_link_0_{domain}")

    link.connect(
        (components[0], f"schedule{domain}", "1s"),
        (components[domain], "schedule0", "1s")
    )
