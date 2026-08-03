# dgraph: Distributed Graph on top of DSTM

`dgraph` is a distributed graph data structure layered on the DSTM
(Distributed Space-Time Memory) runtime. Vertices are partitioned across
MPI ranks, and both vertex state and edge state are versioned by DSTM
so that different ranks can observe consistent views of the graph at
different logical times.

Source: `src/dgraph.c`
Public API: `include/dstm/dgraph.h`
Tests: `tests/mpi/dgraph.c`

---

## Data model

A `dstm_dgraph_t` is a static-topology directed graph with per-vertex
and per-edge user state:

- **Vertices** are globally numbered `[0, total_vertices)` and
  partitioned into contiguous slices, one per MPI rank. A rank's slice
  is `[local_vertex_offset, local_vertex_offset + local_vertices)`.
- **Out-edges** of each local vertex are declared at create time as a
  sorted-ascending list of destination global vertex ids.
- **Vertex state** is a fixed-size, per-vertex opaque byte blob.
- **Edge state** is a fixed-size, per-edge opaque byte blob attached to
  each out-edge slot.
- **Topology is immutable** after `dstm_dgraph_create` returns. Only
  vertex\_state and edge\_state can be mutated (via versioned puts).

### On-region layout

Each rank owns one DSTM region divided into three fixed-size slabs:

```
[ vertex_state_slab | edge_row_ptr_slab | edge_slot_slab ]
```

- `vertex_state_slab`: `local_vertices` entries of
  `vertex_state_size` bytes. Indexed by local id (`global_id -
  local_vertex_offset`). Mutable via `dstm_dgraph_vertex_put`.
- `edge_row_ptr_slab`: CSR row pointers, `local_vertices + 1` uint64
  entries. Written once at `(t=0, δ=0)`.
- `edge_slot_slab`: `local_edge_count` packed slots, each
  `[dst | user_state | padding]`, sorted by ascending `dst` within
  each vertex's window. `dst` is written once at `(t=0, δ=0)`;
  `user_state` is mutable via `dstm_dgraph_edge_put`.

### Replicated topology cache

Even though topology lives inside a DSTM region, it is also fully
replicated on every rank as plain host memory (via `MPI_Allgatherv`
during `dstm_dgraph_create`):

- `rank_row_ptrs[r][i]` — every rank's row pointer array
- `rank_edge_dsts[r][slot]` — every rank's edge destinations

**Rationale.** DSTM uses a sliding version window; old versions age
out. Topology was published at `(t=0, δ=0)` and needs to be readable
at arbitrary later times. Trying to serve topology reads via
`dstm_get` after t=0 has been reclaimed causes indefinite stalls.
Because topology is immutable, small (O(V+E) integers), and needed on
the hot path, full replication is standard practice and eliminates
DSTM traffic entirely for topology queries.

Topology helpers (`fetch_row_range`, `find_edge_slot`,
`dstm_dgraph_neighbors_get`, `dstm_dgraph_derive_distance`) all read
from these host-side caches and never call DSTM.

---

## API surface

### Creation / destruction

```c
int  dstm_dgraph_create(dstm_dgraph_t **ret,
                        size_t vertex_state_size,
                        size_t edge_state_size,
                        size_t local_vertices,
                        const size_t *local_out_degrees,
                        const size_t *local_edge_dsts,
                        const void   *local_vertex_states,
                        const void   *local_edge_states);

int  dstm_dgraph_create_equal(dstm_dgraph_t **ret, ...);

void dstm_dgraph_destroy(dstm_dgraph_t **dgraph);
```

dstm\_dgraph\_create\_equal splits total\_vertices as evenly as possible: the first total % world\_size ranks own one extra vertex.

Neighbor lists supplied via local\_edge\_dsts must be strictly ascending and in [0, total\_vertices).

## Introspection

```C
size_t dstm_dgraph_vertex_count(dstm_dgraph_t *);
size_t dstm_dgraph_local_vertex_count(dstm_dgraph_t *);
size_t dstm_dgraph_local_vertex_offset(dstm_dgraph_t *);
size_t dstm_dgraph_edge_count(dstm_dgraph_t *);
size_t dstm_dgraph_local_edge_count(dstm_dgraph_t *);
int    dstm_dgraph_vertex_owner(dstm_dgraph_t *, size_t vertex_id);
```

## Vertex State
```C
int dstm_dgraph_vertex_get(dstm_dgraph_t *, size_t vid, void *dst,
                           dstm_time_t time, dstm_time_t delta);
int dstm_dgraph_vertex_put(dstm_dgraph_t *, size_t vid, void *src,
                           dstm_time_t time, dstm_time_t delta);
int dstm_dgraph_vertex_get_or_default(dstm_dgraph_t *, size_t vid,
                                      void *dst, dstm_time_t time,
                                      dstm_time_t delta,
                                      const void *default_value);
```

The (time, delta) semantics match DSTM's dstm\_get / dstm\_put:

get: delta <= 0; effective time is time + delta
put: delta >= 0; effective time is time + delta
delta == 0 requires the owning rank to be the local rank
get\_or\_default short-circuits when time + delta < 0 and returns the caller-supplied default without contacting the owning rank

## Edge State

```C
int dstm_dgraph_edge_get(dstm_dgraph_t *, size_t u, size_t v, void *dst,
                         dstm_time_t time, dstm_time_t delta);
int dstm_dgraph_edge_put(dstm_dgraph_t *, size_t u, size_t v, void *src,
                         dstm_time_t time, dstm_time_t delta);
```

Internally, resolving the (u, v) slot is a purely-local binary-search into the replicated topology cache. The actual state transfer is a single dstm\_get/dstm\_put with the caller's (time, delta). Returns -DSTM\_ENOTFOUND if the edge was never declared.

## Neighborhood queries

```C
int dstm_dgraph_degree_get(dstm_dgraph_t *, size_t u, size_t *out_degree);
int dstm_dgraph_neighbors_get(dstm_dgraph_t *, size_t u,
                              size_t *out_neighbors, size_t *io_count);
```

Both are topology-only and never call DSTM. neighbors\_get supports a two-call capacity-query pattern: pass out\_neighbors == NULL (or an undersized buffer with \*io\_count < deg) and it returns -DSTM\_ENOMEM after writing the required capacity into \*io\_count.

## Distance-matrix derivation

```C
int dstm_dgraph_derive_distance(dstm_dgraph_t *, dstm_distance_t **ret,
                                int default_delta);
```

Builds a world\_size × world\_size symmetric reachability matrix from the graph's rank-to-rank edges and materializes a dstm\_distance\_t suitable for future dstm\_start\_custom calls: 0 on the diagonal, default\_delta on reachable off-diagonal pairs, -1 elsewhere.

## Interaction with DSTM's time model
Two DSTM contract rules directly shape how dgraph must be used:

Put-before-get phase rule. Within one dstm session, once a get at (t, δ=0) has been issued to a region, no more puts at (t, δ=0) to that region are permitted.
Sliding version window. Old versions may be reclaimed as logical time advances; a get targeting a version that has been reclaimed will stall.
dstm\_dgraph\_create publishes initial topology and (optional) initial state via puts at (t=0, δ=0). Two consequences:

One dgraph per session. Creating a second dgraph in the same DSTM session may collide at (t=0, δ=0) with reads issued by the first, corrupting state. Callers should open a fresh dstm\_start / dstm\_fini session per dgraph instance for independent tests or phases. The test harness enforces this via a run\_isolated() wrapper (see below).
Topology cache pays for immutability. Because topology reads can be needed at arbitrary later times, and DSTM cannot serve a version that has been reclaimed, topology is replicated in host memory rather than served from the region. State reads/writes continue to use DSTM as designed.

# Known constraints

One dgraph per DSTM session. As described above, multiple concurrent dgraphs within a single dstm\_start/dstm\_fini session will collide at (t=0, δ=0). The test harness enforces isolation via run\_isolated.
Topology is immutable. There is no API for adding, removing, or reordering edges after create. If you need dynamic topology, destroy and rebuild the graph in a fresh session.
Neighbor lists must be sorted ascending and unique at create time. Validation happens in dstm\_dgraph\_create and rejects violating inputs with -DSTM\_EINVAL.
Vertex/edge state sizes of 0 are legal. get/put calls short-circuit to DSTM\_SUCCESS without touching DSTM. This is useful for topology-only graphs.
