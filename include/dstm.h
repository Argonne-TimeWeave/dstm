/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

/**
 * Main DSTM header, contains most of the public API
 **/

#ifndef DSTM_H
#define DSTM_H 1

#ifdef __cplusplus
extern "C" {
#endif

#include <assert.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

// clang-format off
#include "dstm/error.h"
#include "dstm/version.h"
// clang-format on

/*******************************************************************************
 * Library Init/Fini
 ******************************************************************************/

/** Initializes the library with a user-supplied MPI communicator.
 *
 * This is the primary initialization function for DSTM. The library will
 * use the provided communicator instead of MPI_COMM_WORLD, allowing DSTM
 * to operate on a subset of ranks or to avoid collisions with other MPI
 * traffic in the application.
 *
 * The communicator is duplicated internally; the caller retains ownership
 * of @p comm and may free it after this call returns.
 *
 * @param[in,out] argc: a pointer to the number of arguments in *argv
 * @param[in,out] argv: an array of command line options.
 * @param[in]     comm: the MPI communicator DSTM should use.
 *
 * If the library detects command line options it understands, it will consume
 * those arguments and modify the two variables accordingly. Other arguments
 * will be ignored. It is safe to give a program's main arguments to this
 * library.
 *
 * @return 0 if successful; an error code otherwise.
 **/
int dstm_init_comm(int *argc, char **argv[], MPI_Comm comm);

/** Initializes the library using MPI_COMM_WORLD.
 *
 * Convenience wrapper around dstm_init_comm() that passes MPI_COMM_WORLD
 * as the communicator. Equivalent to:
 * @code
 *   dstm_init_comm(argc, argv, MPI_COMM_WORLD);
 * @endcode
 *
 * @param[in,out] argc: a pointer to the number of arguments in *argv
 * @param[in,out] argv: an array of command line options.
 *
 * If the library detects command line options it understands, it will consume
 * those arguments and modify the two variables accordingly. Other arguments
 * will be ignored. It is safe to give a program's main arguments to this
 * library.
 *
 * @return 0 if successful; an error code otherwise.
 **/
int dstm_init(int *argc, char **argv[]);

/**
 * Terminates the library. Do this before an instrumented program exits.
 * @return 0 if successful; an error code otherwise.
 **/
int dstm_finalize(void);

/*******************************************************************************
 * Logging
 ******************************************************************************/

#define DSTM_LOG_QUIET 0
#define DSTM_LOG_ERROR 1
#define DSTM_LOG_WARNING 2
#define DSTM_LOG_NORMAL 3
#define DSTM_LOG_INFO 4
#define DSTM_LOG_DEBUG 5

/**
 * Set to one by the library if at least one dstm_log_init has
 * been called
 */
extern int dstm_log_initialized;

/**
 * Initializes dstm logging
 *
 * @param f: file descriptor
 * @param nm: logging source namespace
 * @return 0 if successful, an error code otherwise
 */
int dstm_log_init(FILE *f, const char *nm);

/**
 * Prints an dstm log message at a log level, labeled with source file and line
 * number.
 *
 * @param level: log level constant
 * @param file: source file label typically ``__FILE__``
 * @param line: source line number label. typically ``__LINE__``
 * @param format: printf formatted string
 */
void dstm_log_printf(int level,
                     const char *file,
                     const char *func,
                     unsigned int line,
                     const char *format,
                     ...);

int dstm_log_setlevel(int level);

/**
 * Current log level.  Checked by logging macros before formatting
 * to avoid unnecessary work at suppressed levels.
 */
extern int dstm_log_level;

#define dstm_log_error(...)                                                    \
	dstm_log_printf(DSTM_LOG_ERROR, __FILE__, __func__, __LINE__,          \
	                __VA_ARGS__)
#define dstm_log_warning(...)                                                  \
	dstm_log_printf(DSTM_LOG_WARNING, __FILE__, __func__, __LINE__,        \
	                __VA_ARGS__)
#define dstm_log_normal(...)                                                   \
	dstm_log_printf(DSTM_LOG_NORMAL, __FILE__, __func__, __LINE__,         \
	                __VA_ARGS__)
#define dstm_log_info(...)                                                     \
	dstm_log_printf(DSTM_LOG_INFO, __FILE__, __func__, __LINE__,           \
	                __VA_ARGS__)
#define dstm_log_debug(...)                                                    \
	dstm_log_printf(DSTM_LOG_DEBUG, __FILE__, __func__, __LINE__,          \
	                __VA_ARGS__)
#define dstm_log_perror(...)                                                   \
	do {                                                                   \
		char *__dstm_errstr = strerror(errno);                         \
		dstm_log_printf(DSTM_LOG_ERROR, __FILE__, __func__, __LINE__,  \
		                __VA_ARGS__);                                  \
		dstm_log_printf(DSTM_LOG_ERROR, __FILE__, __func__, __LINE__,  \
		                "perror: %s\n", __dstm_errstr);                \
	} while (0)

/*******************************************************************************
 * Distances
 ******************************************************************************/

/**
 * Maximum supported size for distance matrices.
 * This limit prevents excessive memory allocation.
 */
#define DSTM_DISTANCE_MAX_SIZE 65536

/* distance between ranks, expressed as a 1D array of length size *size; */
struct dstm_distance_s;
typedef struct dstm_distance_s dstm_distance_t;

int dstm_distance_create_copy(dstm_distance_t **dst, dstm_distance_t *src);
int dstm_distance_get(dstm_distance_t *dst, int i, int j);
int dstm_distance_get_discard(const dstm_distance_t *dst, int i, int j);
size_t dstm_distance_size(dstm_distance_t *dst);

/**
 * Create ring topology distance matrix.
 *
 * In a ring topology, each node is connected to exactly two neighbors
 * in a circular pattern. The distance between any two nodes is the
 * minimum number of hops around the ring.
 *
 * @param[out] ret: pointer to distance structure pointer (must be non-NULL)
 * @param[in] size: number of nodes (must be >= 3 for a valid ring)
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL for invalid parameters,
 *         -DSTM_ENOMEM for allocation failures
 */
int dstm_distance_create_ring(dstm_distance_t **ret, size_t size);

/**
 * Create fully connected (complete graph) topology distance matrix.
 *
 * In a complete graph, every node has a direct connection to every other
 * node. All non-diagonal distances are 1, representing the lowest-latency
 * topology.
 *
 * @param[out] ret: pointer to distance structure pointer (must be non-NULL)
 * @param[in] size: number of nodes (must be >= 1)
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL for invalid parameters,
 *         -DSTM_ENOMEM for allocation failures
 */
int dstm_distance_create_complete(dstm_distance_t **ret, size_t size);

/**
 * Create star topology distance matrix.
 *
 * In a star topology, node 0 is the central hub connected to all other
 * nodes. Non-central nodes communicate through the center (2 hops).
 *
 * @param[out] ret: pointer to distance structure pointer (must be non-NULL)
 * @param[in] size: number of nodes (must be >= 2 for a valid star)
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL for invalid parameters,
 *         -DSTM_ENOMEM for allocation failures
 */
int dstm_distance_create_star(dstm_distance_t **ret, size_t size);

/**
 * Create line/chain topology distance matrix.
 *
 * In a line topology, nodes are connected in a linear sequence.
 * Each node (except endpoints) has exactly two neighbors.
 * Distance is the absolute difference of node indices.
 *
 * @param[out] ret: pointer to distance structure pointer (must be non-NULL)
 * @param[in] size: number of nodes (must be >= 1)
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL for invalid parameters,
 *         -DSTM_ENOMEM for allocation failures
 */
int dstm_distance_create_line(dstm_distance_t **ret, size_t size);
int dstm_distance_create_mesh(dstm_distance_t **ret, size_t size);
int dstm_distance_create_torus(dstm_distance_t **ret, size_t size);
int dstm_distance_create_tree(dstm_distance_t **ret, size_t size);
int dstm_distance_create_hypercube(dstm_distance_t **ret, size_t size);
int dstm_distance_create_dragonfly(dstm_distance_t **ret, size_t size);

/**
 * Create a distance matrix from a user-provided matrix.
 *
 * Allows applications to define arbitrary topologies (e.g. derived from
 * a partitioned application graph) instead of using the built-in
 * topology constructors.
 *
 * The matrix must be a flattened size*size array of non-negative
 * integers stored in row-major order, with all diagonal entries equal
 * to zero (distance from a node to itself is 0).  Entries do not need
 * to be symmetric, but they must satisfy the triangle inequality if
 * they are to be used as DSTM lookahead values; this function does NOT
 * enforce the triangle inequality.  Callers can use other distance
 * helpers (e.g. Floyd-Warshall on an adjacency matrix) before passing
 * the result here.
 *
 * The contents of @p matrix are copied; the caller retains ownership of
 * the input buffer.
 *
 * @param[out] ret    pointer to distance structure pointer (must be non-NULL)
 * @param[in]  size   number of nodes (must be >= 1 and <= DSTM_DISTANCE_MAX_SIZE)
 * @param[in]  matrix flattened size*size matrix of non-negative ints
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL for invalid parameters
 *         (NULL pointers, out-of-range size, negative entries, non-zero
 *         diagonal), -DSTM_ENOMEM for allocation failures
 */
int dstm_distance_create_from_matrix(dstm_distance_t **ret, size_t size,
                                     const int *matrix);

/**
 * Combine multiple rank-indexed distance matrices into one session-global
 * distance object covering the union of their communication.
 *
 * DSTM installs exactly one distance object for a session. When several
 * data structures coexist, that object must account for every rank pair
 * used by any of those structures.
 *
 * Two conservative bounds are retained internally for each rank pair:
 *
 *   - the minimum reachable distance is used by WRITABLE -> READABLE,
 *     because the smallest delta determines when no more PUTs can arrive;
 *
 *   - the maximum reachable distance is used by READABLE -> DISCARDABLE,
 *     because the largest look-back determines when no more GETs can
 *     arrive.
 *
 * dstm_distance_get() returns the minimum/readability bound. The maximum
 * discard bound is internal to the version state machine.
 *
 * The reduction is performed per cell (i, j) over the @p n inputs:
 *
 *   - diagonal entries are always 0;
 *   - a pair is reachable if any input marks it reachable;
 *   - the readable bound is the minimum non-negative reachable value;
 *   - the discardable bound is the maximum non-negative reachable value;
 *   - the pair is -1 (unreachable) only when every input is -1.
 *
 * Combining an already-combined matrix is supported and preserves both
 * bounds.
 *
 * All inputs must share the same size. The inputs are only read; the caller
 * retains ownership of @p mats and each matrix.
 *
 * @param[out] ret   distance matrix handle to populate (must be non-NULL)
 * @param[in]  mats  array of @p n input matrix pointers (non-NULL, each
 *                   entry non-NULL)
 * @param[in]  n     number of input matrices (must be >= 1)
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL for NULL pointers,
 *         n == 0, or size mismatch between inputs, -DSTM_ENOMEM for
 *         allocation failures
 */
int dstm_distance_combine(dstm_distance_t **ret,
                          const dstm_distance_t *const *mats,
                          size_t n);

/**
 * Destroy a distance matrix and free all associated resources.
 *
 * This function safely deallocates a distance matrix created by any of
 * the dstm_distance_create_* functions. Safe to call with NULL pointer.
 * After this call, *dist will be set to NULL.
 *
 * @param[in,out] dist: pointer to distance structure pointer
 */
void dstm_distance_destroy(dstm_distance_t **dist);

/*******************************************************************************
 * DSTM TIME
 ******************************************************************************/

/* type alias to a timestamp, this is logical time (within the simulation), not
 * real time. */
typedef int64_t dstm_time_t;

/*******************************************************************************
 * DSTM START/FINISH
 ******************************************************************************/

/* Options for the dstm
 * @param region_size  Size in bytes for the local region. Will be
 *                     rounded up to the next power of 2. Must be at
 *                     least DSTM_REGION_MIN_BLOCK_SIZE (64 bytes).
 */
struct dstm_opts_s {
	size_t region_size;
};
typedef struct dstm_opts_s dstm_opts_t;

extern dstm_opts_t dstm_default_options;

/**
 * Initialize the DSTM.
 *
 * Like dstm_start(), but allows specifying the size of the local memory
 * region instead of using the default DSTM_DEFAULT_REGION_SIZE.
 *
 * @param dist         Distance matrix defining the network topology.
 *                     Must have size equal to world_size.
 * @param opts         See dstm_opts_t.
 *
 * @return 0 if successful; an error code otherwise.
 */
int dstm_start_custom(dstm_distance_t *dist, dstm_opts_t opts);

/**
 * Initialize the DSTM, with default options.
 *
 * @param dist         Distance matrix defining the network topology.
 *                     Must have size equal to world_size.
 *
 *                     The matrix is session-global and singleton: it is
 *                     the ONLY matrix consulted by the region version
 *                     state machine for every region/data structure in
 *                     the session.  It MUST therefore cover the union of
 *                     all traffic of every data structure alive during
 *                     the session.  If any structure communicates across
 *                     a rank pair that the matrix marks unreachable (-1)
 *                     — or uses a smaller delta than the matrix records —
 *                     the version machine can conclude "no more PUTs/GETs
 *                     can arrive" and prematurely evict a still-needed
 *                     version.
 *
 *                     Recommended workflow when multiple structures
 *                     coexist: derive each structure's matrix (e.g.
 *                     dstm_dgraph_derive_distance /
 *                     dstm_dgraph_distance_from_topology) and merge them
 *                     with dstm_distance_combine() before passing the
 *                     result here.
 *
 * @return 0 if successful; an error code otherwise.
 */
int dstm_start(dstm_distance_t *dist);

/**
 * Flush all in-flight requests and shut down the local region.
 *
 * Spins the progress loop until every pending request (IN_PUT, LOCAL_PUT,
 * IN_GET and its GET_REP send, OUT_PUT MPI_Isend) has completed, then issues
 * an MPI_Barrier so that no rank tears down its region while a peer may still
 * be sending messages to it.
 *
 * @param last_time  The highest effective timestep submitted by this rank,
 *                   i.e. max(time + delta) across all dstm_put / dstm_get
 *                   calls.  Must be large enough to allow the region state
 *                   machine to complete the final timestep's WRITABLE and
 *                   READABLE phases.  Passing a value that is too small will
 *                   cause the drain loop to stall indefinitely.
 *
 * @return 0 if successful; an error code otherwise.
 */
int dstm_fini(dstm_time_t last_time);

/**
 * Distributed barrier over all ranks using a two-pass ring token algorithm.
 *
 * Ensures that all in-flight requests across all ranks have completed before
 * returning.  The algorithm:
 *   1. Drains all local OUT_PUT MPI_Isend operations.
 *   2. Passes a token around the ring (rank 0 -> 1 -> ... -> N-1 -> 0).
 *      Each rank receiving the token knows its predecessor has finished all
 *      sends, drains its remaining requests, then forwards the token.
 *   3. Rank 0 re-injects the token for a second pass to signal global
 *      completion to all ranks.
 *
 * @param last_time  The highest effective timestep submitted by this rank,
 *                   i.e. max(time + delta) across all dstm_put / dstm_get
 *                   calls.  All ranks must pass the same value.
 *
 * @return 0 if successful; an error code otherwise.
 */
int dstm_barrier(dstm_time_t last_time);

/*******************************************************************************
 * DSTM MEMORY MANAGEMENT
 ******************************************************************************/

/* a memory region: core object that get updated by the simulation using the
 * DSTM model.
 *
 * Has a unique ID within the DSTM, and a current time. Internally maintains a
 * list of pending requests to R/W to it.
 */

struct dstm_region_s;
typedef struct dstm_region_s dstm_region_t;

/* distributed array over regions */
struct dstm_darray_s;
typedef struct dstm_darray_s dstm_darray_t;

int dstm_darray_create(dstm_darray_t **ret,
                       size_t element_size,
                       size_t local_elements);

/* Like dstm_darray_create, but back this rank's partition on local region
 * @p region_id (as returned by dstm_region_register) instead of the default
 * region 0.  Collective over dstm_global_context.comm.  Each rank may pass a
 * different region_id; the mapping is Allgathered internally so remote
 * accesses address the correct region on each owner. */
int dstm_darray_create_on_region(dstm_darray_t **ret,
                                 size_t element_size,
                                 size_t local_elements,
                                 int region_id);

int dstm_darray_create_equal(dstm_darray_t **ret,
                             size_t element_size,
                             size_t total_elements);

int dstm_darray_create_equal_on_region(dstm_darray_t **ret,
                                       size_t element_size,
                                       size_t total_elements,
                                       int region_id);

void dstm_darray_destroy(dstm_darray_t **darray);

void *dstm_darray_get_local_ptr(dstm_darray_t *darray);
size_t dstm_darray_local_count(dstm_darray_t *darray);
size_t dstm_darray_global_offset(dstm_darray_t *darray);

/**
 * Read the element at @p global_index from the distributed array into
 * @p dst at effective time (time + delta).
 *
 * The owning rank for @p global_index is computed internally from the
 * darray partitioning and forwarded to dstm_get.  The same delta/rank
 * invariants as dstm_get apply: @p delta must be <= 0, and if
 * @p delta == 0 the element must be owned by the local rank
 * (otherwise -DSTM_EDOM is returned by the underlying dstm_get).
 *
 * @param[in]  darray        the distributed array
 * @param[in]  global_index  global element index to read
 * @param[out] dst           destination buffer of element_size bytes
 * @param[in]  time          base logical timestamp
 * @param[in]  delta         non-positive offset added to @p time
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL if @p dst is NULL or
 *         @p delta is positive, -DSTM_EDOM if @p delta == 0 and the
 *         owning rank is not the local rank.
 */
int dstm_darray_get(dstm_darray_t *darray,
                    size_t global_index,
                    void *dst,
                    dstm_time_t time,
                    dstm_time_t delta);

/**
 * Write @p src into the element at @p global_index of the distributed
 * array, taking effect at time (time + delta).
 *
 * The owning rank for @p global_index is computed internally from the
 * darray partitioning and forwarded to dstm_put.  The same delta/rank
 * invariants as dstm_put apply: @p delta must be >= 0, and if
 * @p delta == 0 the element must be owned by the local rank
 * (otherwise -DSTM_EDOM is returned by the underlying dstm_put).
 *
 * @param[in] darray        the distributed array
 * @param[in] global_index  global element index to write
 * @param[in] src           source buffer of element_size bytes
 * @param[in] time          base logical timestamp
 * @param[in] delta         non-negative offset added to @p time
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL if @p src is NULL or
 *         @p delta is negative, -DSTM_EDOM if @p delta == 0 and the
 *         owning rank is not the local rank.
 */
int dstm_darray_put(dstm_darray_t *darray,
                    size_t global_index,
                    void *src,
                    dstm_time_t time,
                    dstm_time_t delta);

/**
 * Read @p count elements from the distributed array at the global
 * indices given in @p indices, into the contiguous destination buffer
 * @p dst.  Element i is written to dst + i * element_size in the order
 * the indices appear.
 *
 * Indices may be in any order and target any combination of owning
 * ranks, including fully discontinuous ranges.  Internally each index
 * is resolved to its owning rank and a single dstm_get is issued for
 * it; there is no batching at the wire level.
 *
 * The same delta/rank invariants as dstm_darray_get apply to every
 * index in the call.  Iteration stops at the first underlying error and
 * that error code is returned; previously-issued gets are not rolled
 * back.
 *
 * @p count == 0 is a successful no-op; in that case @p indices and
 * @p dst may be NULL.
 *
 * @param[in]  darray   the distributed array
 * @param[in]  indices  array of @p count global indices
 * @param[in]  count    number of indices (and elements in @p dst)
 * @param[out] dst      destination buffer of count * element_size bytes
 * @param[in]  time     base logical timestamp
 * @param[in]  delta    non-positive offset added to @p time
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL if @p darray is NULL
 *         or if @p count > 0 and @p indices or @p dst is NULL, or the
 *         first underlying dstm_get error code encountered.
 */
int dstm_darray_get_range(dstm_darray_t *darray,
                          const size_t *indices,
                          size_t count,
                          void *dst,
                          dstm_time_t time,
                          dstm_time_t delta);

/**
 * Write @p count elements into the distributed array at the global
 * indices given in @p indices, taken from the contiguous source buffer
 * @p src.  Element i is read from src + i * element_size.
 *
 * Semantics, ordering, partial-failure behavior, and delta/rank
 * invariants mirror dstm_darray_get_range, but for dstm_darray_put.
 *
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL if @p darray is NULL
 *         or if @p count > 0 and @p indices or @p src is NULL, or the
 *         first underlying dstm_put error code encountered.
 */
int dstm_darray_put_range(dstm_darray_t *darray,
                          const size_t *indices,
                          size_t count,
                          const void *src,
                          dstm_time_t time,
                          dstm_time_t delta);

/**
 * Scattered-destination variant of dstm_darray_get_range: each element
 * i is written to the per-element user buffer @p dsts[i] (each pointing
 * to element_size bytes) instead of a contiguous output buffer.
 *
 * Same semantics, ordering, partial-failure behavior, and delta/rank
 * invariants as dstm_darray_get_range.
 *
 * @p count == 0 is a successful no-op; in that case @p indices and
 * @p dsts may be NULL.
 *
 * @param[in]  darray   the distributed array
 * @param[in]  indices  array of @p count global indices
 * @param[in]  count    number of indices
 * @param[out] dsts     array of @p count destination pointers, each to
 *                      element_size bytes of writable memory
 * @param[in]  time     base logical timestamp
 * @param[in]  delta    non-positive offset added to @p time
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL if @p darray is NULL
 *         or if @p count > 0 and @p indices or @p dsts is NULL, or the
 *         first underlying dstm_get error code encountered.  A NULL
 *         entry in @p dsts is forwarded to dstm_get and surfaces as
 *         -DSTM_EINVAL from that call.
 */
int dstm_darray_get_rangev(dstm_darray_t *darray,
                           const size_t *indices,
                           size_t count,
                           void *const *dsts,
                           dstm_time_t time,
                           dstm_time_t delta);

/**
 * Scattered-source variant of dstm_darray_put_range: each element i is
 * read from the per-element user buffer @p srcs[i] (each pointing to
 * element_size bytes) instead of a contiguous input buffer.
 *
 * Same semantics, ordering, partial-failure behavior, and delta/rank
 * invariants as dstm_darray_put_range.
 *
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL if @p darray is NULL
 *         or if @p count > 0 and @p indices or @p srcs is NULL, or the
 *         first underlying dstm_put error code encountered.  A NULL
 *         entry in @p srcs is forwarded to dstm_put and surfaces as
 *         -DSTM_EINVAL from that call.
 */
int dstm_darray_put_rangev(dstm_darray_t *darray,
                           const size_t *indices,
                           size_t count,
                           const void *const *srcs,
                           dstm_time_t time,
                           dstm_time_t delta);

/**
 * All-to-all gather: every rank publishes its full local slice via a
 * put at time (time + deltas[self]) and reads every other rank's full
 * slice via a get at time (time + deltas[r]).
 *
 * The @p deltas array must contain exactly world_size entries.  Each
 * entry must be strictly signed according to its role:
 *   - deltas[self] > 0 : the local put strictly advances logical time;
 *   - deltas[r]    < 0 : every remote get strictly looks into the past.
 * Zero is not allowed in any slot of @p deltas; this preserves the
 * invariant that delta == 0 is only meaningful for the local rank, but
 * the local slot is itself a put that requires a strictly positive
 * delta.
 *
 * @param[in]  darray  the distributed array
 * @param[in]  src     local source buffer (local_count * element_size bytes)
 * @param[out] dst     destination buffer for the gathered global array
 *                     (total_count * element_size bytes)
 * @param[in]  time    base logical timestamp shared by all participants
 * @param[in]  deltas  array of world_size signed offsets; deltas[self]
 *                     must be > 0 and deltas[r] must be < 0 for r != self
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL if any of @p darray,
 *         @p src, @p dst or @p deltas is NULL, -DSTM_EDOM if any
 *         @p deltas entry violates the sign rule above.
 */
int dstm_darray_allgatherv(dstm_darray_t *darray,
                           void *src,
                           void *dst,
                           dstm_time_t time,
                           const dstm_time_t *deltas);

/**
 * Like dstm_darray_get, but if the effective target time
 * (time + delta) is strictly negative, copies element_size bytes from
 * @p default_value into @p dst and returns DSTM_SUCCESS without
 * contacting the owning rank.
 *
 * Useful for early ticks of a simulation where the target version
 * predates the start of history and would otherwise block forever.
 *
 * @param default_value  pointer to element_size bytes used as the
 *                       fallback value.  MUST be non-NULL.
 *
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL if @p default_value is
 *         NULL or other arguments are invalid.  For non-negative target
 *         times, behaves exactly like dstm_darray_get and may return
 *         -DSTM_EDOM if @p delta == 0 and the owning rank is not the
 *         local rank.
 */
int dstm_darray_get_or_default(dstm_darray_t *darray,
                               size_t global_index,
                               void *dst,
                               dstm_time_t time,
                               dstm_time_t delta,
                               const void *default_value);

/************************
* DGRAPH
***********************/
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
 *   1. dstm_dgraph_create -- each rank declares its local vertex count,
 *      element sizes, per-vertex out-degrees, and packed CSR edge
 *      destinations plus optional initial state. The region is
 *      allocated with an exactly-sized edge payload slab.
 *   2. (implicit) -- topology is frozen. Vertex and edge *state* may
 *      be written via *_put; the (u, v) set is immutable.
 ******************************************************************************/

/**
 * Create a distributed versioned graph with static topology.
 *
 * Collective; total vertex count is the sum of @p local_vertices across
 * ranks (via MPI_Allgather).
 *
 * Global ids are assigned by contiguous rank-order partitioning: rank 0
 * owns [0, N_0), rank 1 owns [N_0, N_0 + N_1), etc. 
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
 *     MUST be in strictly ascending global-id order.
 *
 *   - @p local_edge_states (optional): packed initial edge state,
 *     local_edge_count * @p edge_state_size bytes, same order as
 *     @p local_edge_dsts.
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

/* Like dstm_dgraph_create, but back this rank's slabs on local region
 * @p region_id (as returned by dstm_region_register) instead of the
 * default region 0.  Collective over dstm_global_context.comm.  Each rank
 * may pass a different region_id; the mapping is Allgathered internally so
 * remote vertex/edge state accesses address the correct region on each
 * owner. */
int dstm_dgraph_create_on_region(dstm_dgraph_t **ret,
                                 size_t vertex_state_size,
                                 size_t edge_state_size,
                                 size_t local_vertices,
                                 const size_t *local_out_degrees,
                                 const size_t *local_edge_dsts,
                                 const void *local_vertex_states,
                                 const void *local_edge_states,
                                 int region_id);

// matching dstm_darray_create_equal
int dstm_dgraph_create_equal(dstm_dgraph_t **ret,
                             size_t vertex_state_size,
                             size_t edge_state_size,
                             size_t total_vertices,
                             const size_t *local_out_degrees,
                             const size_t *local_edge_dsts,
                             const void *local_vertex_states,
                             const void *local_edge_states);

int dstm_dgraph_create_equal_on_region(dstm_dgraph_t **ret,
                                       size_t vertex_state_size,
                                       size_t edge_state_size,
                                       size_t total_vertices,
                                       const size_t *local_out_degrees,
                                       const size_t *local_edge_dsts,
                                       const void *local_vertex_states,
                                       const void *local_edge_states,
                                       int region_id);

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

/**
 * Return 1 if @p rank is a topology-neighbor of the local rank (i.e.,
 * the local rank holds a replica of @p rank's topology and can perform
 * edge lookups against vertices owned by @p rank without additional
 * MPI traffic). Returns 1 for the local rank itself, 0 otherwise.
 * Returns -DSTM_EINVAL on invalid input.
 */
int dstm_dgraph_is_neighbor_rank(dstm_dgraph_t *dgraph, int rank);

/**
 * Fill @p out_ranks with the sorted list of neighbor ranks (including
 * the local rank). Same two-call capacity pattern as neighbors_get.
 */
int dstm_dgraph_neighbor_ranks_get(dstm_dgraph_t *dgraph,
                                   int *out_ranks,
                                   size_t *io_count);

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
 * Neighbor lists are time-invariant (static topology); no (time, delta).
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
 * @return DSTM_SUCCESS, -DSTM_ENOMEM if buffer too small , -DSTM_EINVAL for
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


/**
 * Build a DSTM distance matrix directly from local CSR topology.
 *
 * Takes the same local_vertices / local_out_degrees / local_edge_dsts
 * arrays that will later be handed to dstm_dgraph_create, and derives
 * the rank-level distance matrix from them without a dstm_dgraph_t.
 *
 * Requires only that dstm_init / dstm_init_comm has completed. The
 * DSTM communicator is live from that point on, including between
 * sessions, so this may be called before or after dstm_start
 *
 * No graph state exists at this point, so every edge crossing a rank
 * boundary weighs @p default_delta. Use
 * dstm_dgraph_derive_distance_weighted on a built graph when per-edge
 * weights must come from vertex or edge state.
 *
 * Collective; all ranks must call.
 *
 * @return DSTM_SUCCESS, -DSTM_EINVAL, -DSTM_ENOMEM, -DSTM_FAILURE.
 */
int dstm_dgraph_distance_from_topology(dstm_distance_t **ret,
                                       size_t local_vertices,
                                       const size_t *local_out_degrees,
                                       const size_t *local_edge_dsts,
                                       int default_delta);

typedef int (*dstm_dgraph_edge_weight_fn)(
        size_t src_vid, size_t dst_vid,
        const void *src_vertex_state,
        const void *dst_vertex_state,
        const void *edge_state,
        void *user);

int dstm_dgraph_derive_distance_weighted(
        dstm_dgraph_t *g, dstm_distance_t **ret, int default_delta,
        dstm_dgraph_edge_weight_fn weight_fn, void *user);

/************************
 * DQUEUE
 *
 * A distributed queue built on dstm_region_register + dstm_append /
 * dstm_get.  Each rank owns ONE region holding ONE slot; each
 * (rank, timestamp) is a distinct version of that slot.  All producers
 * APPEND into the same slot (dstm_append accumulates in MPI arrival
 * order), so there are no per-neighbor blocks and no wholesale-overwrite
 * collisions.  "Neighbor" (reachability) still comes from the distance
 * matrix handed at create time; it bounds who can ever push to whom.
 *
 * CAPACITY SIZING RULE.  Because every producer appends into ONE slot,
 * `capacity` must be the SUM over all producers of their maximum per-tick
 * contribution to a single owner -- NOT one producer's worth.  push() and
 * the append handler bound-check and return -DSTM_ENOMEM / abort on
 * overflow.
 *
 * APPEND-ORDERING DECISION.  Concatenation order is MPI arrival order and
 * is NONDETERMINISTIC across runs.  This dqueue does NOT sort: queued
 * elements must be order-independent, or must carry their own key so the
 * consumer can sort after pop.
 *
 * push == append (delta >= 0): publish one element to the destination
 *   owner's slot immediately with a single dstm_append at (time + delta).
 *   No staging, no flush: appends from all producers accumulate on the
 *   owner in MPI arrival order.
 *
 * pop_incoming == get (delta <= 0): one local get of this rank's slot at a
 *   single (time, delta) version; read all live elements into one buffer --
 *   the whole incoming queue, drained at once.
 *
 * Elements are fixed-size POD (no embedded pointers).
 ***********************/

struct dstm_dqueue_s;
typedef struct dstm_dqueue_s dstm_dqueue_t;

/**
 * Create a sparse distributed queue over the neighbors implied by @p dist.
 *
 * Collective over dstm_global_context.comm. @p dist must have size equal to
 * world_size (typically the same matrix handed to dstm_start). Only
 * reachable (dest, src) blocks are allocated.
 *
 * @param[out] ret           handle to populate on success (non-NULL)
 * @param[in]  element_size   bytes per queued element (> 0)
 * @param[in]  capacity       max elements per neighbor block (> 0)
 * @param[in]  dist           distance matrix defining neighbor reachability
 * @return DSTM_SUCCESS, -DSTM_EINVAL, -DSTM_ENOMEM, -DSTM_FAILURE.
 */
int dstm_dqueue_create(dstm_dqueue_t **ret,
                       size_t element_size,
                       size_t capacity,
                       dstm_distance_t *dist);

/* Like dstm_dqueue_create, but back this rank's blocks on local region
 * @p region_id instead of the default region 0. Collective. */
int dstm_dqueue_create_on_region(dstm_dqueue_t **ret,
                                 size_t element_size,
                                 size_t capacity,
                                 dstm_distance_t *dist,
                                 int region_id);

/**
 * Destroy a dqueue and free all resources. Safe on NULL or a handle
 * pointing to NULL. Sets *dqueue to NULL.
 */
void dstm_dqueue_destroy(dstm_dqueue_t **dqueue);

/*******************************************************************************
 * Introspection
 ******************************************************************************/

/** Bytes per element. */
size_t dstm_dqueue_element_size(dstm_dqueue_t *dqueue);

/** Max elements per neighbor block. */
size_t dstm_dqueue_capacity(dstm_dqueue_t *dqueue);

/**
 * Upper bound on the number of elements a single dstm_dqueue_pop_incoming
 * can return.  With the single append-based queue this equals `capacity`:
 * the whole incoming queue is ONE slot holding at most `capacity`
 * elements (down from the old n_neighbors * capacity).  Size a pop buffer
 * to this once at setup to avoid the two-call pattern.
 */
size_t dstm_dqueue_max_incoming(dstm_dqueue_t *dqueue);

/*******************************************************************************
 * Push  (producer side, == append, issued immediately)
 ******************************************************************************/

/**
 * Publish one element (element_size bytes) to @p dest_rank's incoming queue
 * at effective time (time + delta), with a single dstm_append. 
 *
 * All producers targeting the same (owner, time + delta) accumulate into the
 * same version slot in MPI arrival order; the order of distinct producers'
 * elements is therefore nondeterministic (see the APPEND-ORDERING DECISION
 * in the banner above).
 *
 * CAPACITY: the owner's single slot must hold the SUM over all producers of
 * their per-tick contributions.  Overflow is enforced on the owning rank by
 * the append handler.
 *
 * @return DSTM_SUCCESS, -DSTM_EINVAL (NULL args or @p dest_rank out of
 *         range), -DSTM_EDOM (remote push with @p delta == 0), or the
 *         underlying dstm_append error.
 */
int dstm_dqueue_push(dstm_dqueue_t *dqueue,
                     int dest_rank,
                     const void *element,
                     dstm_time_t time,
                     dstm_time_t delta);

/*******************************************************************************
 * Pop  (consumer side, == get) : the single-queue surface
 ******************************************************************************/

/**
 * Drain this rank's ENTIRE incoming queue at the version (time + delta):
 * read every incoming neighbor block and concatenate all live elements
 * contiguously into @p out_elements, writing the total element count to
 * *@p io_count. Producers (sources) are not visible to the caller.
 *
 * Mirrors dstm_darray_get delta rules: @p delta <= 0; a pop that reads any
 * remote block requires @p delta < 0.
 *
 * Buffer sizing, two supported styles:
 *   - size @p out_elements to dstm_dqueue_max_incoming() and it never
 *     overflows; or
 *   - two-call pattern: on entry *@p io_count is the buffer capacity in
 *     elements; if too small, returns -DSTM_ENOMEM with *@p io_count set to
 *     the exact total required. Passing out_elements == NULL with
 *     *io_count == 0 queries the count only.
 *
 * Early-tick fallback: if (time + delta) < 0, yields *@p io_count == 0 and
 * DSTM_SUCCESS without any MPI traffic.
 *
 * @return DSTM_SUCCESS, -DSTM_EINVAL, -DSTM_ENOMEM (buffer too small), or
 *         the underlying dstm_get error.
 */
int dstm_dqueue_pop_incoming(dstm_dqueue_t *dqueue,
                             void *out_elements,
                             size_t *io_count,
                             dstm_time_t time,
                             dstm_time_t delta);

/* opaque type for pointers into regions */
#define DSTM_REGIONID_INVALID -1
#define DSTM_REGIONID_LOCAL -2
struct dstm_ptr_s {
	int region;
	intptr_t offset;
};
typedef struct dstm_ptr_s dstm_ptr_t;

dstm_ptr_t dstm_region_ptr_fromvoid(dstm_region_t *r, void *ptr);
void *dstm_region_ptr_tovoid(dstm_region_t *r, dstm_ptr_t ptr);
int dstm_ptr_isnull(dstm_ptr_t ptr);

/* Return the default local region id (always 0).  Region ids are an
 * intra-rank namespace and are NOT related to the MPI rank. */
int dstm_get_local_region_id(void);

/* Look up a local region by its id (0 == default). NULL if none. */
dstm_region_t *dstm_get_local_region(int region_id);

/* Register an additional local backing store; returns its region id
 * (>= 0) via *region_id, or an error. Collective is NOT required.
 *
 * PRECONDITION: all regions a rank intends to use must be registered
 * before that rank (or any neighbor) issues operations on them.
 * Neighbor heartbeats only populate the neighbor-clock table for regions
 * that already exist when the heartbeat is received; a region registered
 * after a neighbor has advanced its clock would start with a stale
 * (time 0) neighbor entry for that region. */
int dstm_region_register(size_t size, int *region_id);

/* Unregister/destroy a local region previously registered. */
int dstm_region_unregister(int region_id);

/* Construct a dstm_ptr_t referring to byte @p offset within region
 * @p region (which may be a rank id, DSTM_REGIONID_LOCAL, or
 * DSTM_REGIONID_INVALID). Performs no validation. */
static inline dstm_ptr_t dstm_ptr_make(int region, intptr_t offset)
{
        dstm_ptr_t p;
        p.region = region;
        p.offset = offset;
        return p;
}

/*******************************************************************************
 * DSTM BASIC OPERATIONS
 ******************************************************************************/

/**
 * Write @p size bytes from @p src into the region identified by
 * @p region_id, hosted on @p rank, at the location @p dst, taking
 * effect at logical time (time + delta).
 *
 * The triple (rank, region_id, dst.region) must be consistent:
 * @p region_id must equal @p dst.region (the region argument and the
 * pointer's region must agree).  @p delta must be >= 0 (a put cannot
 * target the past).
 *
 * Local-rank invariant for delta == 0: when @p delta is zero, the
 * effective target time equals @p time and only the local rank can
 * meaningfully observe its own current state.  Therefore @p rank MUST
 * equal the local rank when @p delta == 0; remote puts with
 * @p delta == 0 are rejected with
 * -DSTM_EDOM.  Non-zero @p delta permits any @p rank.
 *
 * Put-before-get phase rule: local rank access with delta == 0 requires us to
 * enforce additional restrictions. Once a get at same time and delta == 0 has
 * been issued, a put at same time and delta == 0 is not allowed. The dstm will
 * enforce this and abort the program with a contract-violation message logged
 * via dstm_log_error.
 *
 * @param[in] rank       target MPI rank
 * @param[in] region_id  region identifier (must be == @p dst.region)
 * @param[in] time       base logical timestamp
 * @param[in] delta      non-negative offset added to @p time
 * @param[in] dst        opaque pointer into the target region
 * @param[in] src        source buffer of @p size bytes (must be non-NULL)
 * @param[in] size       number of bytes to copy
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL if @p src is NULL,
 *         @p delta is negative, @p region_id is out of range, or
 *         @p region_id != @p dst.region; -DSTM_EDOM if @p delta == 0
 *         and @p rank is not the local rank.  Aborts the program if
 *         the put violates the put-before-get phase rule above.
 */
int dstm_put(int rank,
             int region_id,
             dstm_time_t time,
             dstm_time_t delta,
             dstm_ptr_t dst,
             void *src,
             size_t size);

/**
 * Read @p size bytes from the region identified by @p region_id,
 * hosted on @p rank, at the location @p src, into the local buffer
 * @p dst, observing the version at logical time (time + delta).
 *
 * The triple (rank, region_id, src.region) must be consistent:
 * @p region_id must equal @p src.region.  @p delta must be <= 0 (a
 * get cannot target the future).
 *
 * Local-rank invariant for delta == 0: when @p delta is zero, the
 * effective target time equals @p time and only the local rank can
 * meaningfully observe its own current state.  Therefore @p rank MUST
 * equal the local rank id (see dstm_get_local_region_id) when
 * @p delta == 0; remote gets with @p delta == 0 are rejected with
 * -DSTM_EDOM.  Non-zero @p delta permits any @p rank.
 *
 * Put-before-get phase rule: local rank access with delta == 0 requires
 * us to enforce additional restrictions. A local get with delta == 0 
 * is only allowed after all local put with delta == 0 with the same time.
 * 
 * This call is blocking: it drives the progress loop until the
 * requested version is available locally.
 *
 * @param[in]  rank       target MPI rank
 * @param[in]  region_id  region identifier (must be == @p src.region)
 * @param[in]  time       base logical timestamp
 * @param[in]  delta      non-positive offset added to @p time
 * @param[out] dst        destination buffer of @p size bytes (non-NULL)
 * @param[in]  src        opaque pointer into the source region
 * @param[in]  size       number of bytes to copy
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL if @p dst is NULL,
 *         @p delta is positive, @p region_id is out of range, or
 *         @p region_id != @p src.region; -DSTM_EDOM if @p delta == 0
 *         and @p rank is not the local rank.
 */
int dstm_get(int rank,
             int region_id,
             dstm_time_t time,
             dstm_time_t delta,
             void *dst,
             dstm_ptr_t src,
             size_t size);

/**
 * Like dstm_get, but if the effective target time (time + delta) is
 * strictly negative, copies @p size bytes from @p default_value into
 * @p dst and returns DSTM_SUCCESS without contacting the owning rank.
 *
 * Useful for early ticks of a simulation where the target version
 * predates the start of history and would otherwise block forever.
 *
 * @param default_value  pointer to @p size bytes used as the fallback
 *                       value.  MUST be non-NULL.
 *
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL if @p default_value
 *         is NULL or other arguments are invalid.  For non-negative
 *         target times, behaves exactly like dstm_get and may return
 *         -DSTM_EDOM if @p delta == 0 and @p rank is not the local
 *         rank.
 */
int dstm_get_or_default(int rank,
                        int region_id,
                        dstm_time_t time,
                        dstm_time_t delta,
                        void *dst,
                        dstm_ptr_t src,
                        size_t size,
                        const void *default_value);

/**
 * Append @p size bytes from @p src into an accumulating slot in the
 * region identified by @p region_id, hosted on @p rank, taking effect
 * at logical time (time + delta). 
 *
 * SLOT LAYOUT.  @p dst must point at the first byte of an accumulating
 * slot shaped as:
 *
 *     [ uint64_t byte_count | <accumulated bytes> ]
 *
 * The byte_count field lives at dst.offset and is maintained by the
 * append handler.  Each append copies its @p size payload bytes to
 * dst.offset + sizeof(uint64_t) + byte_count, then adds @p size to
 * byte_count.  The caller is responsible for ensuring the slot's region
 * is large enough to hold sizeof(uint64_t) plus the SUM of all
 * producers' per-tick payloads; an append that would overflow the slot
 * is a programming error (see the data-structure layer, e.g. dqueue,
 * which bound-checks and returns -DSTM_ENOMEM before issuing the op).
 *
 * Order is MPI arrival order and is therefore nondeterministic across runs.   
 *
 * All other rules mirror dstm_put exactly:
 *   - @p region_id must equal @p dst.region.
 *   - @p delta must be >= 0 (an append cannot target the past).
 *   - @p delta == 0 requires @p rank to be the local rank; a remote
 *     append with @p delta == 0 is rejected with -DSTM_EDOM.
 *   - the target version must still be WRITABLE when the append applies
 *     (put/append-before-get phase rule); a violation aborts the program.
 *
 * @param[in] rank       target MPI rank
 * @param[in] region_id  region identifier (must be == @p dst.region)
 * @param[in] time       base logical timestamp
 * @param[in] delta      non-negative offset added to @p time
 * @param[in] dst        opaque pointer to the first byte of the slot
 * @param[in] src        source buffer of @p size bytes (must be non-NULL)
 * @param[in] size       number of bytes to append
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL if @p src is NULL,
 *         @p delta is negative, @p region_id is out of range, or
 *         @p region_id != @p dst.region; -DSTM_EDOM if @p delta == 0
 *         and @p rank is not the local rank.
 */
int dstm_append(int rank,
                int region_id,
                dstm_time_t time,
                dstm_time_t delta,
                dstm_ptr_t dst,
                void *src,
                size_t size);

#ifdef __cplusplus
}
#endif

#endif
