# Traffic ABM example

A small distributed agent-based traffic simulation built on top of
DSTM.  It is intentionally tiny (~700 LOC) but exercises every piece
of the DSTM programming model:

- conservative discrete-event simulation driven by DSTM heartbeats,
- topology-aware shared state via `dstm_darray`,
- region distances used as Chandy/Misra lookahead,
- application-level virtual time mapped 1:1 onto `dstm_time_t`,
- cross-rank agent migration through versioned puts.

## What it simulates

A 16x16 grid city with two cross-city highways (row 8 and column 8)
and a 4x4 fully-connected downtown block in the centre.  Vehicles are
spawned at random intersections and routed greedily toward
destinations, biased so that ~60% of trips terminate in downtown.

The city is partitioned across exactly **5 MPI ranks**:

| rank | region    | intersections |
|------|-----------|---------------|
| 0    | downtown  | 16            |
| 1    | NW        | 60            |
| 2    | NE        | 60            |
| 3    | SW        | 60            |
| 4    | SE        | 60            |

Imbalance comes naturally from the bias toward downtown destinations
plus the fact that suburbs hold many more intersections.

## Build and run

```sh
./autogen.sh && ./configure && make
mpirun -n 5 ./examples/traffic/traffic --steps 80 --vehicles 200 --quiet
```

The example **requires exactly 5 MPI ranks**.  Other sizes will exit
with an error.

### Command-line flags

| flag                | default | meaning                                 |
|---------------------|---------|-----------------------------------------|
| `--steps N`         | 80      | number of simulation ticks              |
| `--vehicles M`      | 200     | vehicles spawned per rank               |
| `--seed S`          | 42      | RNG seed (deterministic per rank)       |
| `--quiet`           | off     | suppress DSTM logging                   |

### Expected output

The program prints the city summary, then a per-rank stats table.
Conservation of vehicles is verified by the `in_transit` column being
zero at the end of a sufficiently long run.

```
=== Per-rank simulation stats ===
rank        spawned  arrived   live   events  remote_rd  mig_out   mig_in ob_overfl
downtown        200      537      0     1791       3994      126      468         5
NW              200       93      0     2831        507      180       73         0
NE              200       94      0     2792       2352      220      131        17
SW              200      117      0     2749       1942      189      118        12
SE              200       92      0     2650       3260      213      138        33
--- totals ---
spawned=1000 arrived=933 live=0 in_transit=0 events=12813 migrations(sent=928,recv=928) outbox_overflows=67
event imbalance: rank0(downtown) processed 0.70x avg; remote_rd 0=3994 avg=2411
```

## How DSTM is used

### Virtual time

`dstm_time_t` ticks are identified one-to-one with simulation
seconds (one event-loop iteration per tick).  Each rank advances its
own simulation clock and uses it as the `time` argument to all
`dstm_put` / `dstm_get` calls; DSTM emits a heartbeat to all
neighbors automatically when the rank's tracked time advances.

### Region distance = lookahead

The application builds an explicit `dstm_distance_t` from the city's
inter-region adjacency using
`dstm_distance_create_from_matrix()`.  The matrix entries are the
minimum number of ticks an agent could possibly traverse from one
region to another (1 for adjacent, 2 for diagonally opposite
suburbs).  This is exactly the Chandy/Misra lookahead that DSTM uses
to determine when a versioned snapshot becomes readable.

### Shared state: `cong` darray

A single `dstm_darray<cong_record>` of length `NUM_INTERSECTIONS`,
distributed so that each rank owns its city partition.  Every tick
each rank publishes the queue length of every owned intersection at
`(time = sim_time, delta = 0)`.  Other ranks read with
`(time = sim_time, delta = -dist(self, owner))` so the get is
guaranteed to land on a readable historical version without ever
blocking — the conservative-DES tradeoff is that routing decisions
use slightly stale congestion data, but no rollback is ever needed.

For early ticks where `sim_time - dist < 0` no historical version
exists.  The example uses `dstm_darray_get_or_default()` to short-
circuit such reads to a zero record without contacting the owner.

### Cross-rank migration: `inbox` darray

A second `dstm_darray<inbox_cell>` whose elements are owned per
(receiver, sender) pair.  When a vehicle's chosen next hop crosses a
rank boundary, the sender appends its migration record into a local
outbox cell indexed by `(receiver, arrive_tick - sim_time)`.  At the
end of the tick `flush_outbox()` issues one `dstm_put` per non-empty
cell at `(time = sim_time, delta = travel)` so the cell lands in the
receiver's region version `arrive_tick`.

The receiver, when it reaches `sim_time = arrive_tick`, drains each
of its 4 inbox cells (one per sender) at `(time = sim_time,
delta = 0)` after publishing its own congestion (which guarantees
the local version exists).

The cell holds a fixed `RECORDS_PER_CELL` number of records; if more
migrations from the same sender to the same receiver share the same
arrive_tick than fit, the excess is dropped and counted in the
`ob_overfl` column.  This is a configuration limit of the example,
not of DSTM.

### Per-tick discipline

```c
for (sim_time = 0; sim_time <= sim_end; sim_time++) {
    publish_owned_congestion();   // creates local version, heartbeats
    drain_inbox();                // safe: local version exists
    process_local_events();       // routing reads remote cong
    flush_outbox();               // bulk migration writes
}
```

`publish` is intentionally first: it creates the local DSTM region
version at `sim_time` so the subsequent self-region inbox drain
won't deadlock waiting for a non-existent version, and it triggers
heartbeats to all neighbors so the distributed time horizon
advances.

## Caveats and possible improvements

- **Fixed 5-rank topology.**  A future revision could auto-partition
  any rank count using a generic graph partitioner.
- **Outbox overflows** are silently counted but the dropped vehicles
  are lost.  A more elaborate handoff (e.g. variable-length cells or
  multi-version chained writes) would handle bursts.
- **Greedy routing.**  Vehicles pick the neighbor that minimises
  Manhattan distance plus a congestion weight; no path planning.

## Reference reading inside the codebase

- `tests/mpi/allgatherv.c` — minimal stencil pattern using the same
  publish-with-positive-delta / read-with-negative-delta discipline.
- `src/region.c` — region-version snapshot mechanics.
- `src/ops.c` — the conservative readability rule
  (`dstm_region_status_update`).
