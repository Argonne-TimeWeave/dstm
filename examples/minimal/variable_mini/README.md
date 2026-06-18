# variable\_mini — variable-topology network simulation

A DSTM example that simulates packet routing over an arbitrary undirected weighted graph. Generalizes the `ring_mini` example from a fixed 2-neighbor ring to any topology defined by a static edge list (default: 4 nodes, 6 edges). Requires exactly `N` MPI ranks, one per node.

## Model

Each rank owns one mailbox slot per **incoming** edge, laid out ragged across a single `dstm_darray_t` (offsets in `slot_offset[]`). Slot ownership is derived deterministically by sorting each node's in-neighbors by rank, so sender and receiver agree on indices with no coordination. Routing uses a next-hop matrix from Floyd-Warshall over the edge weights, and each PUT uses `delta = edge_weight`, so one-hop link latency matches the graph.

## Per-timestep loop

1. **Warmup sentinel** — for each incoming slot from `u`, if `t < delta(u, rank)`, locally PUT an empty slot. This seeds slots before `u`'s real writes have had time to propagate, while preserving the single-writer-per-`(slot, timestamp)` invariant (the receiver writes during warmup, the sender writes thereafter — disjoint time ranges).
2. **Receive** from each incoming slot via `dstm_darray_get_or_default`.
3. **Process**: deliver packets addressed to this rank, otherwise forward by looking up `next[rank][dest]` and matching it to an out-neighbor index.
4. **Inject** a new packet to a random other rank with probability `--prob`.
5. **Send** each outbox to the corresponding neighbor with `delta = dstm_distance_get(topology, rank, neighbor)`.

## Usage

mpirun -n 4 ./variable\_mini [--seed S] [--time T] [--prob P] [--quiet]

Edit the `edges[][3]` table and `N` at the top of the file to change topology.
