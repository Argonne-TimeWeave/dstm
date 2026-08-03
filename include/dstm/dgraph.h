/**
 * DSTM distributed versioned graph.
 *
 * A dgraph is a distributed graph datastructure layered on top of DSTM's
 * versioned memory regions.  Vertices carry fixed-size state; edges are
 * directed and carry fixed-size edge state.  Every vertex and every edge
 * is versioned: reads and writes are tagged with a (time, delta) pair
 * using the same semantics as dstm_get / dstm_put and dstm_darray_get /
 * dstm_darray_put.
 *
 * Ownership: each vertex is owned by exactly one rank, assigned at
 * creation time by contiguous partitioning of the global vertex id space
 * (identical to dstm_darray_create_equal).  An edge (u, v) is owned by
 * the rank that owns its source vertex u.  This keeps all edge lookups
 * and mutations local to the source vertex's rank and requires no extra
 * resolution primitives beyond those already used by dstm_darray.
 *
 * Topology is dynamic: edges may be added and removed at runtime, and
 * these mutations are themselves timestamped operations that become
 * visible at (time + delta) just like any other put.  Vertex count is
 * fixed at creation.
 */

#ifndef DSTM_DGRAPH_H
#define DSTM_DGRAPH_H 1

#ifdef __cplusplus
extern "C" {
#endif

#include "dstm.h"

/*******************************************************************************
 * Types
 ******************************************************************************/

/* Opaque handle to a distributed versioned graph. */
struct dstm_dgraph_s;
typedef struct dstm_dgraph_s dstm_dgraph_t;

/*******************************************************************************
 * Lifecycle
 ******************************************************************************/

/**
 * Create a distributed versioned graph with @p local_vertices vertices
 * owned by the calling rank.  Every rank must call this collectively;
 * the total vertex count is the sum of @p local_vertices across all
 * ranks and is determined via MPI_Allgather during creation.
 *
 * Global vertex ids are assigned by contiguous partitioning in rank
 * order: rank 0 owns [0, rank_vertex_counts[0]), rank 1 owns
 * [rank_vertex_counts[0], rank_vertex_counts[0] + rank_vertex_counts[1]),
 * and so on.  This matches the addressing used by dstm_darray_create.
 *
 * Each vertex stores @p vertex_state_size bytes of user-defined state
 * (may be 0 for a graph that only carries edge state).  Each edge
 * stores @p edge_state_size bytes of user-defined state (may be 0 for
 * an unweighted graph).
 *
 * @p local_edge_capacity is the maximum number of outgoing edges the
 * local rank may host across all of its vertices.  It is used to
 * pre-allocate the edge index and payload slabs in the local region;
 * exceeding this capacity via subsequent dstm_dgraph_edge_add calls
 * returns -DSTM_ENOMEM.
 *
 * @param[out] ret                  handle to populate on success (non-NULL)
 * @param[in]  vertex_state_size    bytes of state per vertex (>= 0)
 * @param[in]  edge_state_size      bytes of state per edge (>= 0)
 * @param[in]  local_vertices       number of vertices owned by this rank (> 0)
 * @param[in]  local_edge_capacity  max outgoing edges hosted locally (>= 0)
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL for invalid parameters,
 *         -DSTM_ENOMEM for allocation failures, -DSTM_FAILURE on MPI errors.
 */
int dstm_dgraph_create(dstm_dgraph_t **ret,
                       size_t vertex_state_size,
                       size_t edge_state_size,
                       size_t local_vertices,
                       size_t local_edge_capacity);

/**
 * Create a distributed versioned graph with @p total_vertices vertices
 * split as evenly as possible across all ranks.  Convenience wrapper
 * around dstm_dgraph_create; the first (total_vertices % world_size)
 * ranks receive one extra vertex, matching dstm_darray_create_equal.
 *
 * @return DSTM_SUCCESS on success, error code otherwise.
 */
int dstm_dgraph_create_equal(dstm_dgraph_t **ret,
                             size_t vertex_state_size,
                             size_t edge_state_size,
                             size_t total_vertices,
                             size_t local_edge_capacity);

/**
 * Destroy a dgraph and free all associated resources.  Safe to call
 * with a NULL pointer or a handle pointing to NULL.  After this call,
 * *dgraph is set to NULL.
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

/** MPI rank that owns @p vertex_id. */
int dstm_dgraph_vertex_owner(dstm_dgraph_t *dgraph, size_t vertex_id);

/*******************************************************************************
 * Vertex operations
 ******************************************************************************/

/**
 * Read the state of vertex @p vertex_id at effective time
 * (time + delta) into @p dst.
 *
 * Same (time, delta) semantics as dstm_darray_get: @p delta must be
 * <= 0, and if @p delta == 0 the vertex must be owned by the local
 * rank (otherwise -DSTM_EDOM is returned).
 *
 * @param[in]  dgraph     the graph
 * @param[in]  vertex_id  global vertex id
 * @param[out] dst        buffer of vertex_state_size bytes (non-NULL)
 * @param[in]  time       base logical timestamp
 * @param[in]  delta      non-positive offset added to @p time
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL if @p dst is NULL or
 *         @p delta is positive, -DSTM_EDOM if @p delta == 0 and the
 *         owning rank is not the local rank.
 */
int dstm_dgraph_vertex_get(dstm_dgraph_t *dgraph,
                           size_t vertex_id,
                           void *dst,
                           dstm_time_t time,
                           dstm_time_t delta);

/**
 * Write @p src as the state of vertex @p vertex_id, taking effect at
 * time (time + delta).
 *
 * Same (time, delta) semantics as dstm_darray_put: @p delta must be
 * >= 0, and if @p delta == 0 the vertex must be owned by the local
 * rank.
 *
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL if @p src is NULL or
 *         @p delta is negative, -DSTM_EDOM if @p delta == 0 and the
 *         owning rank is not the local rank.
 */
int dstm_dgraph_vertex_put(dstm_dgraph_t *dgraph,
                           size_t vertex_id,
                           void *src,
                           dstm_time_t time,
                           dstm_time_t delta);

/**
 * Variant of dstm_dgraph_vertex_get that falls back to @p default_value
 * when the effective target time (time + delta) is strictly negative,
 * matching dstm_darray_get_or_default.
 *
 * @param default_value  pointer to vertex_state_size bytes (non-NULL)
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL if @p default_value is
 *         NULL or other arguments are invalid.
 */
int dstm_dgraph_vertex_get_or_default(dstm_dgraph_t *dgraph,
                                      size_t vertex_id,
                                      void *dst,
                                      dstm_time_t time,
                                      dstm_time_t delta,
                                      const void *default_value);

/*******************************************************************************
 * Edge operations
 ******************************************************************************/

/**
 * Read the state of the edge (@p u, @p v) at effective time
 * (time + delta) into @p dst.
 *
 * The edge must exist at the observed version, otherwise -DSTM_ENOTFOUND
 * is returned.  Same (time, delta) semantics as dstm_dgraph_vertex_get;
 * the owning rank of the edge is the owning rank of source vertex @p u.
 *
 * @param[in]  dgraph  the graph
 * @param[in]  u       source vertex id
 * @param[in]  v       destination vertex id
 * @param[out] dst     buffer of edge_state_size bytes (non-NULL)
 * @param[in]  time    base logical timestamp
 * @param[in]  delta   non-positive offset added to @p time
 * @return DSTM_SUCCESS on success, -DSTM_ENOTFOUND if the edge does not
 *         exist at the observed version, -DSTM_EINVAL / -DSTM_EDOM per
 *         the standard rules.
 */
int dstm_dgraph_edge_get(dstm_dgraph_t *dgraph,
                         size_t u,
                         size_t v,
                         void *dst,
                         dstm_time_t time,
                         dstm_time_t delta);

/**
 * Write @p src as the state of the existing edge (@p u, @p v), taking
 * effect at time (time + delta).  This updates an already-inserted
 * edge; use dstm_dgraph_edge_add for insertion.
 *
 * @return DSTM_SUCCESS on success, -DSTM_ENOTFOUND if the edge does not
 *         exist at the observed version, -DSTM_EINVAL / -DSTM_EDOM per
 *         the standard rules.
 */
int dstm_dgraph_edge_put(dstm_dgraph_t *dgraph,
                         size_t u,
                         size_t v,
                         void *src,
                         dstm_time_t time,
                         dstm_time_t delta);

/**
 * Insert a directed edge from @p u to @p v with initial state
 * @p edge_state, taking effect at time (time + delta).
 *
 * The edge becomes visible in neighbor-list queries and edge_get calls
 * whose effective time is >= (time + delta).  Standard (time, delta)
 * rules apply for a put: @p delta must be >= 0.
 *
 * Adding an edge that already exists at the observed version returns
 * -DSTM_EEXIST; use dstm_dgraph_edge_put to modify existing edge state.
 *
 * @return DSTM_SUCCESS on success, -DSTM_EEXIST if the edge already
 *         exists at (time + delta), -DSTM_ENOMEM if the local edge
 *         capacity is exhausted, -DSTM_EINVAL / -DSTM_EDOM per the
 *         standard rules.
 */
int dstm_dgraph_edge_add(dstm_dgraph_t *dgraph,
                         size_t u,
                         size_t v,
                         const void *edge_state,
                         dstm_time_t time,
                         dstm_time_t delta);

/**
 * Remove the directed edge (@p u, @p v), taking effect at time
 * (time + delta).  The edge remains visible to queries at earlier
 * versions per DSTM's versioning rules.
 *
 * Removing an edge that does not exist at the observed version returns
 * -DSTM_ENOTFOUND.
 *
 * @return DSTM_SUCCESS on success, -DSTM_ENOENT if the edge does not
 *         exist at (time + delta), -DSTM_EINVAL / -DSTM_EDOM per the
 *         standard rules.
 */
int dstm_dgraph_edge_remove(dstm_dgraph_t *dgraph,
                            size_t u,
                            size_t v,
                            dstm_time_t time,
                            dstm_time_t delta);

/*******************************************************************************
 * Neighborhood queries
 ******************************************************************************/

/**
 * Read the outgoing-neighbor list of vertex @p u at effective time
 * (time + delta) into the caller-supplied buffer @p out_neighbors.
 *
 * On entry, *@p io_count is the capacity of @p out_neighbors in
 * number of size_t entries.  On success, *@p io_count is updated to
 * the actual number of neighbors written.  If the capacity is too
 * small, -DSTM_ENOMEM is returned and *@p io_count is set to the
 * required capacity so the caller can retry.
 *
 * A two-call pattern is supported: pass @p out_neighbors == NULL and
 * *@p io_count == 0 to query the required capacity without reading
 * any ids; -DSTM_ENOMEM is returned and *@p io_count contains the
 * required capacity.
 *
 * Standard (time, delta) rules apply as a get.
 *
 * @param[in]     dgraph         the graph
 * @param[in]     u              source vertex id
 * @param[out]    out_neighbors  buffer of *io_count entries (may be NULL
 *                               only if *io_count == 0)
 * @param[in,out] io_count       capacity on entry, actual count on exit
 * @param[in]     time           base logical timestamp
 * @param[in]     delta          non-positive offset added to @p time
 * @return DSTM_SUCCESS on success, -DSTM_ENOMEM if @p out_neighbors is
 *         too small (with *@p io_count set to required capacity),
 *         -DSTM_EINVAL / -DSTM_EDOM per the standard rules.
 */
int dstm_dgraph_neighbors_get(dstm_dgraph_t *dgraph,
                              size_t u,
                              size_t *out_neighbors,
                              size_t *io_count,
                              dstm_time_t time,
                              dstm_time_t delta);

/**
 * Count the outgoing neighbors of vertex @p u at effective time
 * (time + delta) without materializing the id list.  Cheaper than
 * dstm_dgraph_neighbors_get when only the degree is needed.
 *
 * @param[out] out_degree  number of outgoing neighbors at (time + delta)
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL / -DSTM_EDOM per the
 *         standard rules.
 */
int dstm_dgraph_degree_get(dstm_dgraph_t *dgraph,
                           size_t u,
                           size_t *out_degree,
                           dstm_time_t time,
                           dstm_time_t delta);

/*******************************************************************************
 * Distance matrix integration
 ******************************************************************************/

/**
 * Derive a DSTM distance matrix from the current dgraph topology.
 *
 * Walks the graph as observed at effective time (time + delta) and
 * builds a rank-level distance matrix in which two ranks are reachable
 * iff at least one edge connects a vertex on one rank to a vertex on
 * the other.  Reachable pairs are assigned distance @p default_delta;
 * unreachable pairs are marked -1.  The diagonal is 0.
 *
 * This lets applications whose communication structure is defined by
 * an evolving graph rebuild the distance matrix from the graph itself
 * rather than maintaining a separate hand-authored topology.  The
 * caller is responsible for feeding the resulting matrix to DSTM (via
 * a future reconfiguration API); this function only produces the
 * matrix.
 *
 * All ranks must call this collectively at the same effective time.
 *
 * @param[in]  dgraph         the graph
 * @param[out] ret            distance matrix handle to populate (non-NULL)
 * @param[in]  default_delta  distance to assign to every reachable
 *                            rank pair (> 0)
 * @param[in]  time           base logical timestamp
 * @param[in]  delta          non-positive offset added to @p time
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL / -DSTM_EDOM per the
 *         standard rules, -DSTM_ENOMEM on allocation failure.
 */
int dstm_dgraph_derive_distance(dstm_dgraph_t *dgraph,
                                dstm_distance_t **ret,
                                int default_delta,
                                dstm_time_t time,
                                dstm_time_t delta);

#ifdef __cplusplus
}
#endif
#endif /* DSTM_DGRAPH_H */
