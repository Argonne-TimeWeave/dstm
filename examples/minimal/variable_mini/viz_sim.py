#!/usr/bin/env python3
"""
Visualize variable_net_sim dump.log:
  - Parses topology (distance matrix used only to infer N; edges inferred from next-hop + in_neigh lines)
  - Parses per-rank per-timestep events (INJECT, FORWARD, DELIVERED, slot contents)
  - Renders an animation of packets flowing along edges over time.

Usage:
    python viz_sim.py dump.log                 # writes sim.mp4 (or sim.gif if ffmpeg missing)
    python viz_sim.py dump.log --out sim.gif
    python viz_sim.py dump.log --static t=15   # one PNG snapshot at t=15
    python viz_sim.py dump.log --summary       # just print stats, no rendering
"""

import argparse
import re
import sys
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx


# ---------- parsing ----------

RE_TOPO_N   = re.compile(r"N=(\d+),\s*(\d+)\s*edges")
RE_INNEIGH  = re.compile(r"node\s+(\d+):\s*in_deg=\d+\s+in_neigh=\[([^\]]*)\]")
RE_RANK_T   = re.compile(r"Rank\s+(\d+)\s*\[t=(\d+)\]:")
RE_INJECT   = re.compile(r"INJECT\s+(\S+)\s+dst=(\d+)\s+via\s+(\d+)")
RE_FORWARD  = re.compile(r"FORWARD\s+(\S+)\s+src=(\d+)\s+dst=(\d+)\s+via\s+(\d+)\s+\(hop\s+(\d+)\)")
RE_DELIVER  = re.compile(r"DELIVERED\s+(\S+)\s+.*?src=(\d+)\s+origin_t=(\d+)\s+hops=(\d+)\s+latency=(\d+)")
RE_SLOT_HDR = re.compile(r"slot from (\d+) has (\d+) packet")
RE_RANK_TS  = re.compile(r"Rank\s+(\d+):\s*time step\s+(\d+)")


def parse_log(path):
    N = None
    in_neigh = {}            # node -> list of upstream neighbors (directed edges into node)
    events = []              # list of dicts {t, rank, kind, ...}
    max_t = 0

    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")

            m = RE_TOPO_N.search(line)
            if m and N is None:
                N = int(m.group(1))
                continue

            m = RE_INNEIGH.search(line)
            if m:
                node = int(m.group(1))
                neigh_str = m.group(2).strip()
                neighs = [int(x) for x in neigh_str.split(",") if x.strip()] if neigh_str else []
                in_neigh[node] = neighs
                continue

            m = RE_RANK_T.search(line)
            if m:
                rank = int(m.group(1))
                t = int(m.group(2))
                max_t = max(max_t, t)

                mi = RE_INJECT.search(line)
                if mi:
                    events.append(dict(t=t, rank=rank, kind="INJECT",
                                       pid=mi.group(1), src=rank,
                                       dst=int(mi.group(2)), via=int(mi.group(3))))
                    continue
                mf = RE_FORWARD.search(line)
                if mf:
                    events.append(dict(t=t, rank=rank, kind="FORWARD",
                                       pid=mf.group(1), src=int(mf.group(2)),
                                       dst=int(mf.group(3)), via=int(mf.group(4)),
                                       hop=int(mf.group(5)), at=rank))
                    continue
                md = RE_DELIVER.search(line)
                if md:
                    events.append(dict(t=t, rank=rank, kind="DELIVER",
                                       pid=md.group(1), src=int(md.group(2)),
                                       origin_t=int(md.group(3)),
                                       hops=int(md.group(4)),
                                       latency=int(md.group(5))))
                    continue

    if N is None:
        # fall back to inferred N
        N = (max(in_neigh.keys()) + 1) if in_neigh else 0

    # Build directed edge set from in_neigh: for each node v with upstream u, edge u->v exists.
    edges = set()
    for v, ups in in_neigh.items():
        for u in ups:
            edges.add((u, v))

    return N, edges, in_neigh, events, max_t


# ---------- graph + layout ----------

def build_graph(N, edges):
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edges)
    # Undirected view for layout
    UG = G.to_undirected()
    try:
        pos = nx.kamada_kawai_layout(UG)
    except Exception:
        pos = nx.spring_layout(UG, seed=42)
    return G, pos


# ---------- frame state ----------

def build_timeline(events, max_t):
    """For each timestep t, collect:
        - injects at node (src) going via 'via'
        - forwards traversing edge (at -> via)
        - deliveries at node (rank)
    Returns dict t -> dict.
    """
    timeline = defaultdict(lambda: dict(
        injects=[],          # list of (src, via, pid, dst)
        forwards=[],         # list of (frm, to, pid, dst, hop)
        delivers=[],         # list of (node, pid, latency, hops)
    ))
    for e in events:
        t = e["t"]
        if e["kind"] == "INJECT":
            timeline[t]["injects"].append((e["src"], e["via"], e["pid"], e["dst"]))
        elif e["kind"] == "FORWARD":
            timeline[t]["forwards"].append((e["at"], e["via"], e["pid"], e["dst"], e["hop"]))
        elif e["kind"] == "DELIVER":
            timeline[t]["delivers"].append((e["rank"], e["pid"], e["latency"], e["hops"]))
    return timeline


# ---------- rendering ----------

def draw_frame(ax, G, pos, t, frame, node_load, title_extra=""):
    ax.clear()
    ax.set_axis_off()

    # Node coloring by current "load" (active packets sitting at node this tick)
    loads = np.array([node_load.get(n, 0) for n in G.nodes()])
    vmax = max(1, loads.max())
    node_colors = plt.cm.viridis(loads / vmax)

    # Base edges (grey)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#cccccc",
                           arrows=True, arrowsize=8, width=1.0, alpha=0.6)

    # Active forward edges (highlight)
    fwd_edges = [(a, b) for (a, b, *_rest) in frame["forwards"] if G.has_edge(a, b)]
    if fwd_edges:
        nx.draw_networkx_edges(G, pos, edgelist=fwd_edges, ax=ax,
                               edge_color="tab:red", width=2.5, arrows=True, arrowsize=14)

    # Inject edges (cyan)
    inj_edges = [(s, v) for (s, v, *_r) in frame["injects"] if G.has_edge(s, v)]
    if inj_edges:
        nx.draw_networkx_edges(G, pos, edgelist=inj_edges, ax=ax,
                               edge_color="tab:cyan", width=2.0, style="dashed",
                               arrows=True, arrowsize=12)

    # Nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=600, edgecolors="black", linewidths=0.8)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_color="white")

    # Delivery markers (gold ring)
    delivered_nodes = [n for (n, *_r) in frame["delivers"]]
    if delivered_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=delivered_nodes, ax=ax,
                               node_color="none", edgecolors="gold",
                               linewidths=3.0, node_size=900)

    n_inj = len(frame["injects"])
    n_fwd = len(frame["forwards"])
    n_dlv = len(frame["delivers"])
    ax.set_title(f"t={t}   inject={n_inj}  forward={n_fwd}  delivered={n_dlv}   {title_extra}",
                 fontsize=11)


def compute_node_load(frame):
    """How many packets are 'at' or 'leaving' each node this tick."""
    load = Counter()
    for (s, v, *_r) in frame["injects"]:
        load[s] += 1
    for (a, b, *_r) in frame["forwards"]:
        load[a] += 1
    for (n, *_r) in frame["delivers"]:
        load[n] += 1
    return load


def render_animation(G, pos, timeline, max_t, out_path, fps=4):
    fig, ax = plt.subplots(figsize=(10, 8))
    times = list(range(max_t + 1))

    def update(i):
        t = times[i]
        frame = timeline.get(t, dict(injects=[], forwards=[], delivers=[]))
        load = compute_node_load(frame)
        draw_frame(ax, G, pos, t, frame, load)
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(times),
                                   interval=1000 // fps, blit=False)

    # Pick writer
    if out_path.lower().endswith(".gif"):
        try:
            anim.save(out_path, writer=animation.PillowWriter(fps=fps))
        except Exception as e:
            print(f"GIF save failed: {e}", file=sys.stderr); raise
    else:
        try:
            anim.save(out_path, writer=animation.FFMpegWriter(fps=fps, bitrate=2000))
        except Exception:
            alt = out_path.rsplit(".", 1)[0] + ".gif"
            print(f"ffmpeg unavailable; falling back to {alt}", file=sys.stderr)
            anim.save(alt, writer=animation.PillowWriter(fps=fps))
            out_path = alt
    plt.close(fig)
    print(f"Wrote {out_path}")


def render_static(G, pos, timeline, t, out_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    frame = timeline.get(t, dict(injects=[], forwards=[], delivers=[]))
    load = compute_node_load(frame)
    draw_frame(ax, G, pos, t, frame, load, title_extra="(snapshot)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def print_summary(N, edges, events, max_t):
    kinds = Counter(e["kind"] for e in events)
    latencies = [e["latency"] for e in events if e["kind"] == "DELIVER"]
    hops = [e["hops"] for e in events if e["kind"] == "DELIVER"]
    print(f"N = {N} nodes, {len(edges)} directed edges")
    print(f"timesteps observed: 0..{max_t}")
    print(f"events: {dict(kinds)}")
    if latencies:
        print(f"delivered: {len(latencies)}, "
              f"latency mean={np.mean(latencies):.2f} max={max(latencies)}, "
              f"hops mean={np.mean(hops):.2f} max={max(hops)}")


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile")
    ap.add_argument("--out", default="sim.mp4",
                    help="output animation file (.mp4 or .gif). Default: sim.mp4")
    ap.add_argument("--static", type=str, default=None,
                    help="render a single timestep PNG, e.g. --static t=15")
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--summary", action="store_true",
                    help="just print stats, do not render")
    args = ap.parse_args()

    N, edges, in_neigh, events, max_t = parse_log(args.logfile)
    print(f"Parsed: N={N}, edges={len(edges)}, events={len(events)}, max_t={max_t}")

    if args.summary:
        print_summary(N, edges, events, max_t)
        return

    G, pos = build_graph(N, edges)
    timeline = build_timeline(events, max_t)

    if args.static is not None:
        t = int(args.static.replace("t=", ""))
        out = f"sim_t{t}.png"
        render_static(G, pos, timeline, t, out)
    else:
        render_animation(G, pos, timeline, max_t, args.out, fps=args.fps)


if __name__ == "__main__":
    main()
