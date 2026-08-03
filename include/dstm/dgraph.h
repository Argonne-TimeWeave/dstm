/**
 * DSTM distributed versioned graph (static topology).
 *
 * Distributed graph layered on DSTM versioned regions. Vertices carry
 * fixed-size state; edges are directed with fixed-size state. Both are
 * versioned with (time, delta) semantics matching dstm_get/dstm_put
 * and dstm_darray_get/dstm_darray_put.
 *
 * STATIC TOPOLOGY: edges are fixed at construction (no add/remove);
 * only state evolves over logical time. Enables tightly packed CSR
 * with no tombstones and O(log deg) edge lookup via sorted neighbors.
 *
 * Ownership: each vertex is owned by one rank via contiguous
 * partitioning of the global id space (like dstm_darray_create_equal).
 * Edge (u, v) is owned by u's owner, keeping all edge ops local to
 * the source rank.
 */

#ifndef DSTM_DGRAPH_H
#define DSTM_DGRAPH_H 1

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/* Forward decls / typedefs needed by this header. These mirror the
 * definitions in dstm.h; keeping them here (instead of #include "dstm.h")
 * avoids a circular include since dstm.h itself includes this file. */
#ifndef DSTM_TIME_T_DEFINED
#define DSTM_TIME_T_DEFINED 1
typedef int64_t dstm_time_t;
#endif

struct dstm_distance_s;
typedef struct dstm_distance_s dstm_distance_t;

/*******************************************************************************
 * Types
 ******************************************************************************/

/* Opaque handle to a distributed versioned graph. */
struct dstm_dgraph_s;
typedef struct dstm_dgraph_s dstm_dgraph_t;

/*******************************************************************************
 * Lifecycle
 *
 * Two-phase collective construction:
 *
 *   1. dstm_dgraph_create -- each rank declares its local vertex count,
 *      element sizes, per-vertex out-degrees, and packed CSR edge
 *      destinations plus optional initial state. The region is
 *      allocated with an exactly-sized edge payload slab.
 *
 *   2. (implicit) -- topology is frozen. Vertex and edge *state* may
 *      be written via *_put; the (u, v) set is immutable.
 *
 * Because out-degrees are exact at create time, there is no capacity
 * parameter, no tombstone bookkeeping, and no -DSTM_ENOMEM from mutation.
 ******************************************************************************/

/**
 * Create a distributed versioned graph with static topology.
 *
 * Collective; total vertex count is the sum of @p local_vertices across
 * ranks (via MPI_Allgather).
 *
 * Global ids are assigned by contiguous rank-order partitioning: rank 0
 * owns [0, N_0), rank 1 owns [N_0, N_0 + N_1), etc. Matches
 * dstm_darray_create.
 *
 * Local outgoing topology, in packed CSR form:
 *
 *   - @p local_out_degrees: length @p local_vertices; entry i is the
 *     out-degree of local vertex i (global id local_vertex_offset + i).
 *
 *   - local_edge_count = sum(local_out_degrees).
 *
 *   - @p local_edge_dsts: length local_edge_count, packed by source
 *     vertex in local order. Destinations within each vertex's slice
 *     MUST be in strictly ascending global-id order (duplicates
 *     rejected with -DSTM_EINVAL; self-loops permitted iff caller
 *     excludes them).
 *
 *   - @p local_edge_states (optional): packed initial edge state,
 *     local_edge_count * @p edge_state_size bytes, same order as
 *     @p local_edge_dsts. NULL means zero-init. Ignored if
 *     @p edge_state_size == 0.
 *
 *   - @p local_vertex_states (optional): packed initial vertex state,
 *     @p local_vertices * @p vertex_state_size bytes, in local order.
 *     NULL means zero-init. Ignored if @p vertex_state_size == 0.
 *
 * Initial state becomes visible at logical time 0.
 *
 * @param[out] ret                  handle to populate on success (non-NULL)
 * @param[in]  vertex_state_size    bytes of state per vertex (>= 0)
 * @param[in]  edge_state_size      bytes of state per edge (>= 0)
 * @param[in]  local_vertices       vertices owned by this rank (> 0)
 * @param[in]  local_out_degrees    per-local-vertex out-degrees
 * @param[in]  local_edge_dsts      packed CSR destination ids
 * @param[in]  local_vertex_states  optional initial vertex state, or NULL
 * @param[in]  local_edge_states    optional initial edge state, or NULL
 * @return DSTM_SUCCESS, -DSTM_EINVAL (NULL required pointers, unsorted
 *         or duplicate neighbors, dst outside [0, total_vertices)),
 *         -DSTM_ENOMEM, or -DSTM_FAILURE on MPI errors.
 */
int dstm_dgraph_create(dstm_dgraph_t **ret,
                       size_t vertex_state_size,
                       size_t edge_state_size,
                       size_t local_vertices,
                       const size_t *local_out_degrees,
                       const size_t *local_edge_dsts,
                       const void *local_vertex_states,
                       const void *local_edge_states);

/**
 * Convenience wrapper around dstm_dgraph_create that splits
 * @p total_vertices evenly across ranks (first total_vertices %
 * world_size ranks get one extra, matching dstm_darray_create_equal).
 * Caller still supplies local topology arrays sized to whatever this
 * rank owns; use dstm_dgraph_local_vertex_count/offset on a probe run,
 * or precompute with the same formula.
 *
 * @return DSTM_SUCCESS on success, error code otherwise.
 */
int dstm_dgraph_create_equal(dstm_dgraph_t **ret,
                             size_t vertex_state_size,
                             size_t edge_state_size,
                             size_t total_vertices,
                             const size_t *local_out_degrees,
                             const size_t *local_edge_dsts,
                             const void *local_vertex_states,
                             const void *local_edge_states);

/**
 * Destroy a dgraph and free all resources. Safe on NULL pointer or a
 * handle pointing to NULL. Sets *dgraph to NULL.
 */
void dstm_dgraph_destroy(dstm_dgraph_t **dgraph);

/*******************************************************************************
 * Introspection
 ******************************************************************************/

/** Total number of vertices across all ranks. */
size_t dstm_dgraph_vertex_count(dstm_dgraph_t *dgraph);

/** Number of vertices owned by the local rank. */
size_t dstm_dgraph_local_vertex_count(dstm_dgraph_t *dgraph);

/** Global id of the first vertex owned by the local rank. */
size_t dstm_dgraph_local_vertex_offset(dstm_dgraph_t *dgraph);

/** Total edges across all ranks (fixed for graph lifetime). */
size_t dstm_dgraph_edge_count(dstm_dgraph_t *dgraph);

/** Outgoing edges hosted by the local rank (fixed). */
size_t dstm_dgraph_local_edge_count(dstm_dgraph_t *dgraph);

/** MPI rank that owns @p vertex_id. */
int dstm_dgraph_vertex_owner(dstm_dgraph_t *dgraph, size_t vertex_id);

/*******************************************************************************
 * Vertex state operations
 ******************************************************************************/

/**
 * Read state of vertex @p vertex_id at effective time (time + delta)
 * into @p dst.
 *
 * Same (time, delta) semantics as dstm_darray_get: @p delta must be
 * <= 0; if @p delta == 0, vertex must be locally owned (else -DSTM_EDOM).
 *
 * @param[in]  dgraph     the graph
 * @param[in]  vertex_id  global vertex id
 * @param[out] dst        buffer of vertex_state_size bytes (non-NULL)
 * @param[in]  time       base logical timestamp
 * @param[in]  delta      non-positive offset added to @p time
 * @return DSTM_SUCCESS, -DSTM_EINVAL (NULL @p dst or positive @p delta),
 *         -DSTM_EDOM (delta == 0 on remote vertex).
 */
int dstm_dgraph_vertex_get(dstm_dgraph_t *dgraph,
                           size_t vertex_id,
                           void *dst,
                           dstm_time_t time,
                           dstm_time_t delta);

/**
 * Write @p src as state of vertex @p vertex_id, effective at
 * (time + delta).
 *
 * Same (time, delta) semantics as dstm_darray_put: @p delta must be
 * >= 0; if @p delta == 0, vertex must be locally owned.
 */
int dstm_dgraph_vertex_put(dstm_dgraph_t *dgraph,
                           size_t vertex_id,
                           void *src,
                           dstm_time_t time,
                           dstm_time_t delta);

/**
 * Variant of dstm_dgraph_vertex_get that falls back to @p default_value
 * when (time + delta) is strictly negative. Matches
 * dstm_darray_get_or_default.
 */
int dstm_dgraph_vertex_get_or_default(dstm_dgraph_t *dgraph,
                                      size_t vertex_id,
                                      void *dst,
                                      dstm_time_t time,
                                      dstm_time_t delta,
                                      const void *default_value);

/*******************************************************************************
 * Edge state operations
 *
 * The (u, v) set is fixed at construction: an edge either exists for
 * the graph's lifetime or never. These calls only read/write versioned
 * state on already-declared edges; an undeclared (u, v) returns
 * -DSTM_ENOTFOUND.
 ******************************************************************************/

/**
 * Read state of edge (@p u, @p v) at effective time (time + delta)
 * into @p dst.
 *
 * Same (time, delta) semantics as dstm_dgraph_vertex_get. Edge owner
 * is the owner of source @p u.
 *
 * @return DSTM_SUCCESS, -DSTM_ENOTFOUND if (u, v) undeclared,
 *         -DSTM_EINVAL / -DSTM_EDOM per standard rules.
 */
int dstm_dgraph_edge_get(dstm_dgraph_t *dgraph,
                         size_t u,
                         size_t v,
                         void *dst,
                         dstm_time_t time,
                         dstm_time_t delta);

/**
 * Write @p src as state of existing edge (@p u, @p v), effective at
 * (time + delta).
 *
 * @return DSTM_SUCCESS, -DSTM_ENOTFOUND if (u, v) undeclared,
 *         -DSTM_EINVAL / -DSTM_EDOM per standard rules.
 */
int dstm_dgraph_edge_put(dstm_dgraph_t *dgraph,
                         size_t u,
                         size_t v,
                         void *src,
                         dstm_time_t time,
                         dstm_time_t delta);

/*******************************************************************************
 * Neighborhood queries
 *
 * Neighbor lists are time-invariant (static topology); no (time, delta)
 * argument is needed.
 ******************************************************************************/

/**
 * Return out-degree of vertex @p u (topological, time-invariant).
 *
 * @param[out] out_degree  number of outgoing neighbors of @p u
 * @return DSTM_SUCCESS, -DSTM_EINVAL for invalid parameters.
 */
int dstm_dgraph_degree_get(dstm_dgraph_t *dgraph,
                           size_t u,
                           size_t *out_degree);

/**
 * Read outgoing-neighbor list of vertex @p u into @p out_neighbors,
 * in strictly ascending global-id order (construction order).
 *
 * On entry *@p io_count is the capacity (in size_t entries). On
 * success it is updated to the number written (== degree_get). If
 * capacity is too small, returns -DSTM_ENOMEM with *@p io_count set
 * to the required capacity for a retry.
 *
 * Two-call pattern: pass @p out_neighbors == NULL and *@p io_count == 0
 * to query required capacity; returns -DSTM_ENOMEM with *@p io_count
 * set. Use dstm_dgraph_degree_get if only the count is needed.
 *
 * Results are stable across calls (static topology).
 *
 * @return DSTM_SUCCESS, -DSTM_ENOMEM if buffer too small (with
 *         *@p io_count set to required capacity), -DSTM_EINVAL for
 *         invalid parameters.
 */
int dstm_dgraph_neighbors_get(dstm_dgraph_t *dgraph,
                              size_t u,
                              size_t *out_neighbors,
                              size_t *io_count);

/*******************************************************************************
 * Distance matrix integration
 ******************************************************************************/

/**
 * Derive a DSTM distance matrix from the graph's static topology.
 *
 * Builds a rank-level matrix where two ranks are reachable iff any
 * edge connects a vertex on one to a vertex on the other. Reachable
 * pairs get @p default_delta; unreachable pairs get -1; diagonal is 0.
 *
 * Because topology is fixed, the matrix is a pure function of the
 * graph and valid for its entire lifetime.
 *
 * Collective; all ranks must call.
 *
 * @param[in]  dgraph         the graph
 * @param[out] ret            distance matrix handle (non-NULL)
 * @param[in]  default_delta  distance for reachable rank pairs (> 0)
 * @return DSTM_SUCCESS, -DSTM_EINVAL, -DSTM_ENOMEM.
 */
int dstm_dgraph_derive_distance(dstm_dgraph_t *dgraph,
                                dstm_distance_t **ret,
                                int default_delta);

#ifdef __cplusplus
}
#endif
#endif /* DSTM_DGRAPH_H */
