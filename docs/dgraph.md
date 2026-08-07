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

**Region ids are not rank ids.** A `dstm_ptr_t` names a
`(region, offset)` pair, and `dstm_get`/`dstm_put` take the owning rank
and the region id as separate arguments. Nothing guarantees rank `r`'s
region id equals `r`, so every rank's region id is gathered at create
time and resolved through the single accessor `dgraph_region_of()`. If
`dstm_get`/`dstm_put` ever change to expect a rank in that position,
that one function is the only thing to edit.

### Replicated topology cache

Topology also lives in plain host memory, replicated **only across
neighbor ranks** — self, plus any rank sharing at least one edge with
self in either direction:

- `rank_row_ptrs[r][i]` — rank `r`'s CSR row pointers
- `rank_edge_dsts[r][slot]` — rank `r`'s edge destinations

Both are indexed by global rank id and are `NULL` for non-neighbors.
Population happens during `dstm_dgraph_create`: membership comes from a
single `MPI_Allgather` of a 1-byte-per-rank "touched" row, OR'd with its
transpose to symmetrize; the payload is then exchanged with
point-to-point `Isend`/`Irecv` between neighbor pairs only.

**Rationale.** DSTM uses a sliding version window; old versions age out.
Topology is published at effective time 0 and must stay readable at
arbitrary later times, so serving it through `dstm_get` would stall once
version 0 is reclaimed. Topology is immutable and small (O(V+E)
integers), so replication removes DSTM traffic from every topology
query. Restricting replication to neighbors keeps cost proportional to
the local neighborhood rather than O(total\_edges) per rank — which is
the whole point of the sparse communication structure DSTM assumes.

Topology helpers (`fetch_row_range`, `find_edge_slot`,
`dstm_dgraph_degree_get`, `dstm_dgraph_neighbors_get`) read these caches
and never call DSTM.

**Consequence: topology queries are neighborhood-scoped.** Any call that
must resolve topology for a vertex owned by a non-neighbor rank returns
`-DSTM_ENOTLOCAL`. This affects `degree_get`, `neighbors_get`,
`edge_get`, and `edge_put`. Vertex state operations are unaffected: they
need no topology and work against any rank. Use
`dstm_dgraph_is_neighbor_rank` to test before querying.

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

### Introspection

```C
size_t dstm_dgraph_vertex_count(dstm_dgraph_t *);
size_t dstm_dgraph_local_vertex_count(dstm_dgraph_t *);
size_t dstm_dgraph_local_vertex_offset(dstm_dgraph_t *);
size_t dstm_dgraph_edge_count(dstm_dgraph_t *);
size_t dstm_dgraph_local_edge_count(dstm_dgraph_t *);
int    dstm_dgraph_vertex_owner(dstm_dgraph_t *, size_t vertex_id);
```c
int dstm_dgraph_is_neighbor_rank(dstm_dgraph_t *, int rank);
int dstm_dgraph_neighbor_ranks_get(dstm_dgraph_t *, int *out_ranks,
                                   size_t *io_count);
```

`is_neighbor_rank` returns 1 if the local rank holds a topology replica
for `rank` (always true for the local rank itself), 0 if not, and
`-DSTM_EINVAL` for an out-of-range rank. `neighbor_ranks_get` fills a
sorted list including self, using the same two-call capacity pattern as
`neighbors_get`.

`vertex_owner` returns `-DSTM_EINVAL` (a negative value) for
`vertex_id >= total_vertices`; check the sign before using the result as
a rank.

### Vertex State
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

### Edge State

```C
int dstm_dgraph_edge_get(dstm_dgraph_t *, size_t u, size_t v, void *dst,
                         dstm_time_t time, dstm_time_t delta);
int dstm_dgraph_edge_put(dstm_dgraph_t *, size_t u, size_t v, void *src,
                         dstm_time_t time, dstm_time_t delta);
```

Internally, resolving the (u, v) slot is a purely-local binary-search into the replicated topology cache. The actual state transfer is a single dstm\_get/dstm\_put with the caller's (time, delta). Returns -DSTM\_ENOTFOUND if the edge was never declared.

### Neighborhood queries

```C
int dstm_dgraph_degree_get(dstm_dgraph_t *, size_t u, size_t *out_degree);
int dstm_dgraph_neighbors_get(dstm_dgraph_t *, size_t u,
                              size_t *out_neighbors, size_t *io_count);
```

Both are topology-only and never call DSTM. neighbors\_get supports a two-call capacity-query pattern: pass out\_neighbors == NULL (or an undersized buffer with \*io\_count < deg) and it returns -DSTM\_ENOMEM after writing the required capacity into \*io\_count.

### Distance-matrix derivation

```c
/* Pre-session: operates on raw CSR arrays, no graph required. */
int dstm_dgraph_distance_from_topology(dstm_distance_t **ret,
                                       size_t local_vertices,
                                       const size_t *local_out_degrees,
                                       const size_t *local_edge_dsts,
                                       int default_delta);

/* Post-hoc: operates on a built graph. */
int dstm_dgraph_derive_distance(dstm_dgraph_t *, dstm_distance_t **ret,
                                int default_delta);
```

Both build a symmetric `world_size × world_size` rank reachability
matrix: `0` on the diagonal, `default_delta` on reachable off-diagonal
pairs, `-1` elsewhere. Both are collective.

**Use `distance_from_topology` for the session that will host the
graph.** `dstm_dgraph_create` issues puts and therefore requires an
already-started session, so a matrix derived from a finished graph can
never be the matrix that graph's own session was started with.
`derive_distance` is for inspection or for a subsequent session.

> **BROKEN — do not use either function yet.** Both delegate to
> `dstm_distance_create_from_matrix`, which currently returns
> `-DSTM_EINVAL` for every input tested (verified on 4 ranks). Suspected
> cause: an element-type mismatch on the matrix parameter (`int *`
> passed where `dstm_time_t *` is expected), which C accepts with only a
> warning and which makes the callee read interleaved garbage. Until
> that is resolved, build the matrix directly — see
> `build_torus_distance()` in `tests/mpi/mini_airport.c` and
> `build_rank_adjacency()` in `examples/airport/main.c`, both of which
> start from a working constructor and overwrite `distances[]` through
> `internal/distance.h`.

## Interaction with DSTM's time model

### Section should be removed. Information is generally useful for claude since it makes this mistake over and over again. Will remove soon.
Two DSTM contract rules directly shape how dgraph must be used:

**Put-before-get phase rule.** A version is identified by its *effective
time* (`time + delta`), not by the literal `(time, delta)` pair. Once
any get on effective time `T` has been issued against a rank's region,
that version is READABLE and no further put to effective time `T` is
accepted — the runtime aborts with `dstm_put contract violation: target
version is no longer WRITABLE (status=1)`.

Versions are tracked per region, not per index, so a get on one vertex
seals effective time `T` against puts to *every* vertex and edge on that
rank. Structure each tick so all puts to a given effective time precede
any get on it. The usual shape is strict alternation — tick `t` reads
effective `t` and writes effective `t+1`:

```
create:  put 0
tick 0:  get 0   put 1
tick 1:  get 1   put 2
...
```

A put with `delta = 0` issued after a get at the same `time` in the same
tick violates this, even though the two calls target different objects.

dstm\_dgraph\_create publishes initial topology and (optional) initial state via puts at (t=0, δ=0). Two consequences:

One dgraph per session. Creating a second dgraph in the same DSTM session may collide at (t=0, δ=0) with reads issued by the first, corrupting state. Callers should open a fresh dstm\_start / dstm\_fini session per dgraph instance for independent tests or phases. The test harness enforces this via a run\_isolated() wrapper (see below).
Topology cache pays for immutability. Because topology reads can be needed at arbitrary later times, and DSTM cannot serve a version that has been reclaimed, topology is replicated in host memory rather than served from the region. State reads/writes continue to use DSTM as designed.

# Known constraints

Topology is immutable. There is no API for adding, removing, or reordering edges after create. If you need dynamic topology, destroy and rebuild the graph in a fresh session.
Neighbor lists must be sorted ascending and unique at create time. Validation happens in dstm\_dgraph\_create and rejects violating inputs with -DSTM\_EINVAL.
Vertex/edge state sizes of 0 are legal. get/put calls short-circuit to DSTM\_SUCCESS without touching DSTM. This is useful for topology-only graphs.

- **Topology queries are neighborhood-scoped.** `degree_get`,
  `neighbors_get`, `edge_get`, and `edge_put` return `-DSTM_ENOTLOCAL`
  for vertices owned by non-neighbor ranks. Vertex state operations have
  no such restriction.
- **Every rank must own at least one vertex.** `local_vertices == 0`
  returns `-DSTM_EINVAL`. Note that `create_equal` produces empty ranks
  when `total_vertices < world_size`.
- **Edge existence is validated even when `edge_state_size == 0`.**
  `edge_get`/`edge_put` resolve `(u, v)` first and can still return
  `-DSTM_ENOTFOUND` or `-DSTM_ENOTLOCAL`; only the state transfer is
  skipped. Vertex operations short-circuit after bounds checking.
