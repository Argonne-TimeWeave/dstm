#!/usr/bin/env python3
"""
Minimal network simulation on top of the dstm Python bindings.

Each rank owns a 2-slot mailbox in a distributed array:
    slot 0 (SLOT_LEFT)  - written only by left neighbor
    slot 1 (SLOT_RIGHT) - written only by right neighbor

Each slot is a fixed-size struct holding up to SLOT_CAPACITY packets.
Every timestep each rank:
  1. reads both its slots,
  2. delivers packets addressed to itself or forwards them one hop,
  3. randomly injects a new packet toward a random other rank,
  4. PUTs its outbound slots into the neighbors' mailboxes at delta=1.

All slots are pre-seeded with an empty sentinel at delta=t (for every t)
so that GETs never block - this mirrors the C code's get_or_default usage.
"""

import argparse
import sys
import random

import numpy as np
from mpi4py import MPI

import dstm  # the module that contains the bindings shown in the prompt

# ----- layout constants (must match the C version's semantics) -----------
PACKET_SIZE   = 256
SLOT_CAPACITY = 8

MAILBOX_SLOTS = 2
SLOT_LEFT     = 0
SLOT_RIGHT    = 1

# 4 ints + 1 int64 = 4*4 + 8 = 24 header bytes, matches the C struct.
PAYLOAD_BYTES = PACKET_SIZE - 24

# numpy dtypes that exactly mirror the C structs (packed, no padding).
packet_dtype = np.dtype([
    ("valid",       np.int32),
    ("source_rank", np.int32),
    ("dest_rank",   np.int32),
    ("hops_taken",  np.int32),
    ("origin_time", np.int64),
    ("payload",     f"S{PAYLOAD_BYTES}"),
], align=False)
assert packet_dtype.itemsize == PACKET_SIZE, packet_dtype.itemsize

slot_dtype = np.dtype([
    ("n_packets", np.int32),
    ("_pad",      f"S{packet_dtype.itemsize - 4 + 0}"),  # placeholder, replaced
])
# Build the real slot dtype with proper alignment to match the C layout.
# In C: int n_packets; mailbox_entry_t packets[SLOT_CAPACITY];
# mailbox_entry_t starts with int (4-byte aligned), so n_packets + 0 pad,
# but the packets array begins at offset 4 only if the compiler doesn't pad.
# To be safe we pack it the same way the C struct is written (no alignment).
slot_dtype = np.dtype([
    ("n_packets", np.int32),
    ("packets",   packet_dtype, (SLOT_CAPACITY,)),
], align=False)

# A single empty-slot sentinel we reuse everywhere.
EMPTY_SLOT = np.zeros(1, dtype=slot_dtype)

def parse_args():
    p = argparse.ArgumentParser(description="dstm ring network simulation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--time", type=int, default=16, dest="timesteps")
    p.add_argument("--prob", type=int, default=60,
                   help="percent chance each rank injects a packet per step")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()

def log(rank, msg, quiet=False):
    if not quiet:
        print(f"[rank {rank}] {msg}", flush=True)

def main():
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

    dstm.init()
    dstm.log_setlevel(dstm.LOG_QUIET if args.quiet else dstm.LOG_NORMAL)

    # Build ring topology.
    topology = dstm.distance_create_ring(num_ranks)

    dstm.start(topology)

    # Each rank owns MAILBOX_SLOTS slots; total global size = num_ranks*MAILBOX_SLOTS.
    mailbox = dstm.darray_create(slot_dtype, MAILBOX_SLOTS)

    rng = random.Random(args.seed + rank)

    left_rank  = (rank - 1) % num_ranks
    right_rank = (rank + 1) % num_ranks

    # Per-rank metric counters.
    n_sent      = 0
    n_delivered = 0
    n_forwarded = 0

    # Pre-seed every slot for every timestep with an empty sentinel. 
    #   no get_or_default in py_bindings (at the time of writing this)
    for t in range(args.timesteps):
        for s in range(MAILBOX_SLOTS):
            idx = rank * MAILBOX_SLOTS + s
            dstm.darray_put(mailbox, idx, EMPTY_SLOT, 0, t)

    # Reusable scratch buffer for GETs.
    recv_buf = np.zeros(1, dtype=slot_dtype)

    for t in range(args.timesteps):
        log(rank, f"time step {t}", args.quiet)

        # Local outbox we'll PUT to the neighbors at the end of the step.
        outbox = np.zeros(MAILBOX_SLOTS, dtype=slot_dtype)

        for s in range(MAILBOX_SLOTS):
            idx = rank * MAILBOX_SLOTS + s
            dstm.darray_get(mailbox, idx, recv_buf, t, 0)
            slot = recv_buf[0]
            n_pkts = int(slot["n_packets"])

            if not args.quiet:
                side = "left" if s == SLOT_LEFT else "right"
                log(rank, f"[t={t}] {side} slot has {n_pkts} packet(s)", args.quiet)

            for p in range(n_pkts):
                pkt = slot["packets"][p].copy()  # mutable working copy
                dest = int(pkt["dest_rank"])

                if dest == rank:
                    n_delivered += 1
                    payload = bytes(pkt["payload"]).split(b"\x00", 1)[0].decode(
                        errors="replace")
                    log(rank,
                        f"[t={t}] DELIVERED - src={int(pkt['source_rank'])} "
                        f"origin_t={int(pkt['origin_time'])} "
                        f"hops={int(pkt['hops_taken'])} payload=\"{payload}\"", 
                        args.quiet)
                    continue

                # Decide direction: shorter way around the ring.
                dist_right = (dest - rank) % num_ranks
                go_right   = dist_right <= num_ranks - dist_right
                fwd_dir    = SLOT_RIGHT if go_right else SLOT_LEFT

                pkt["hops_taken"] = int(pkt["hops_taken"]) + 1

                ob = outbox[fwd_dir]
                n = int(ob["n_packets"])
                if n < SLOT_CAPACITY:
                    ob["packets"][n] = pkt
                    ob["n_packets"]  = n + 1
                    n_forwarded += 1
                    log(rank,
                        f"[t={t}] FORWARD src={int(pkt['source_rank'])} "
                        f"dst={dest} via {'right' if go_right else 'left'} "
                        f"(hop {int(pkt['hops_taken'])})", 
                        args.quiet)
                else:
                    log(rank,
                        f"[t={t}] DROP src={int(pkt['source_rank'])} "
                        f"dst={dest} - outbox full", 
                        args.quiet)

        # Randomly inject a new packet bound for some other rank.
        if num_ranks > 1 and rng.randrange(100) < args.prob:
            dest = rng.randrange(num_ranks - 1)
            if dest >= rank:
                dest += 1

            dist_right = (dest - rank) % num_ranks
            go_right   = dist_right <= num_ranks - dist_right
            fwd_dir    = SLOT_RIGHT if go_right else SLOT_LEFT

            ob = outbox[fwd_dir]
            n = int(ob["n_packets"])
            if n < SLOT_CAPACITY:
                pkt = np.zeros(1, dtype=packet_dtype)[0]
                pkt["valid"]       = 1
                pkt["source_rank"] = rank
                pkt["dest_rank"]   = dest
                pkt["hops_taken"]  = 0
                pkt["origin_time"] = t
                payload = f"src={rank} dst={dest} t={t}".encode()
                pkt["payload"] = payload[:PAYLOAD_BYTES]

                ob["packets"][n] = pkt
                ob["n_packets"]  = n + 1
                n_sent += 1

                log(rank,
                    f"[t={t}] INJECT dst={dest} via "
                    f"{'right' if go_right else 'left'}", 
                    args.quiet)

        # PUT outbound slots into the neighbors' inboxes at delta=1.
        if int(outbox[SLOT_RIGHT]["n_packets"]) > 0:
            idx = right_rank * MAILBOX_SLOTS + SLOT_LEFT
            dstm.darray_put(mailbox, idx, outbox[SLOT_RIGHT:SLOT_RIGHT + 1],
                            t, 1)

        if int(outbox[SLOT_LEFT]["n_packets"]) > 0:
            idx = left_rank * MAILBOX_SLOTS + SLOT_RIGHT
            dstm.darray_put(mailbox, idx, outbox[SLOT_LEFT:SLOT_LEFT + 1],
                            t, 1)

    # Drain / shut down. The highest (time + delta) we used was
    # (timesteps - 1) + 1 == timesteps when seeding the sentinels, and the
    # same for the per-step PUTs. Add a safety margin like the C code.
    last_time = args.timesteps + 1
    dstm.fini(last_time)
    dstm.distance_destroy(topology)

    # Metrics summary.
    print(f"Rank {rank} METRICS: sent={n_sent} delivered={n_delivered} "
          f"forwarded={n_forwarded}", flush=True)

    totals_local  = np.array([n_sent, n_delivered, n_forwarded], dtype=np.int64)
    totals_global = np.zeros(3, dtype=np.int64)
    comm.Reduce(totals_local, totals_global, op=MPI.SUM, root=0)

    if rank == 0:
        print("==== GLOBAL METRICS ====")
        print(f"  sent      = {int(totals_global[0])}")
        print(f"  delivered = {int(totals_global[1])}")
        print(f"  forwarded = {int(totals_global[2])}")
        print(f"  in-flight = {int(totals_global[0] - totals_global[1])}  "
              "(sent - delivered)")

    dstm.finalize()


if __name__ == "__main__":
    sys.exit(main() or 0)
