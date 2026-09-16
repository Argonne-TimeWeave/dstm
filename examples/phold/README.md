# PHOLD Benchmark

PHOLD is a classic PDES benchmark. Unfortunately, it is a poor fit for DSTM.

## Why is it a poor fit?

The ROSS PHOLD model assumes every rank can send an event to every other rank
at uniform cost. Translated to DSTM, this means a complete graph with all
edge distances equal to 1, which gives DSTM no meaningful drift to exploit —
the distance matrix degenerates and DSTM's heartbeat/versioning machinery
provides no benefit over a naive all-to-all.

The two runtimes also disagree on how time advances:

- **ROSS is event-driven.** Work is done only when there is an event to
  process, and virtual time jumps directly to the next event's timestamp.
- **DSTM is tick-stepping.** We iterate over `O(timesteps)` regardless of
  whether events exist for a given tick, and every tick pays the full
  inbox/outbox GET/PUT cost even when nothing is happening.

Our DSTM PHOLD is implemented in a mailbox-holding style, which means every
reachable neighbor gets a PUT every tick (empty sentinel or not). For
workloads with real per-event computation this overhead is negligible, but
PHOLD has essentially zero per-event work, so the sync tax dominates.

## What I tried

To give DSTM a fairer shake, I built a variant that uses a line topology and
restricts remote events to a small local neighborhood (`NEIGHBORHOOD_DIST`).
This lets DSTM's distance-matrix heartbeat actually do something useful
while still processing the same number of events per tick, and in that
configuration DSTM's throughput is comparable to ROSS.

## But it still isn't a fair comparison

Restricting the destination set changes the workload — ROSS PHOLD picks
destinations uniformly across *all* LPs, so a neighborhood-restricted DSTM
run is no longer running the same benchmark. Even setting that aside, the
two implementations differ in fundamental ways (event-driven vs.
tick-stepping, sparse event sends vs. dense mailbox sync, continuous
timestamps vs. integer ticks) that make any single "events/second" number
misleading in isolation.

The takeaway is not that DSTM is slow, but that PHOLD is not the right
workload to measure it with. DSTM's programming model is built around
structured, distance-aware communication over many timesteps of real work;
PHOLD is built around unstructured, uniform, zero-work event shuffling.
Comparing the two on this benchmark is essentially a category error.

### Throughput (events/second)

| Ranks | ROSS    | DSTM (complete) | DSTM (line, NBR=2) |
|------:|--------:|----------------:|-------------------:|
|     1 | 1.90 M  |   1.27 M        |   1.24 M           |
|     2 |  63 k   |    89 k         |    84 k            |
|     4 |  66 k   |   111 k         |    90 k            |
|     8 |  90 k   |   123 k         |   158 k            |
|    16 | 169 k   |    98 k         |   231 k            |
|    32 | 344 k   |    66 k         |   256 k            |
|    64 | 217 k   |    30 k         |   288 k            |
|   128 | 153 k   |   *(DNR)*       |   231 k            |
|   256 | 250 k   |   *(DNR)*       |   359 k            |

DNR = It runs I just didn't wait for it to complete. It scales poorly.

### Summary

| Aspect                | ROSS `--synch=2`         | DSTM                        |
|-----------------------|--------------------------|-----------------------------|
| Time model            | Continuous timestamps    | Integer ticks               |
| Sync mechanism        | LBTS allreduce           | Per-tick mailbox PUT/GET    |
| Sync cost scaling     | `O(log P)` reduction     | `O(neighbors)` per tick     |
| Idle-rank cost        | Near-zero                | Full tick cost              |
| Per-event work        | Handler + RNG + PQ insert| Handler + RNG + bucket push |

