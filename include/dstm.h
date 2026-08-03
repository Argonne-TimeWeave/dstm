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
#include "dstm/dgraph.h"
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
#ifndef DSTM_TIME_T_DEFINED
#define DSTM_TIME_T_DEFINED 1
typedef int64_t dstm_time_t;
#endif

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
int dstm_darray_create_equal(dstm_darray_t **ret,
                             size_t element_size,
                             size_t total_elements);
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

int dstm_get_local_region_id(void);
dstm_region_t *dstm_get_local_region(void);

/*******************************************************************************
 * DSTM BASIC OPERATIONS
 ******************************************************************************/

/**
 * Write @p size bytes from @p src into the region identified by
 * @p region_id, hosted on @p rank, at the location @p dst, taking
 * effect at logical time (time + delta).
 *
 * The triple (rank, region_id, dst.region) must be consistent: each
 * rank's region id is equal to its rank.  @p delta must be >= 0 (a
 * put cannot target the past).
 *
 * Local-rank invariant for delta == 0: when @p delta is zero, the
 * effective target time equals @p time and only the local rank can
 * meaningfully observe its own current state.  Therefore @p rank MUST
 * equal the local rank id (see dstm_get_local_region_id) when
 * @p delta == 0; remote puts with @p delta == 0 are rejected with
 * -DSTM_EDOM.  Non-zero @p delta permits any @p rank.
 *
 * Put-before-get phase rule: local rank access with delta == 0 requires us to
 * enforce additional restrictions. Once a get at same time and delta == 0 has
 * been issued, a put at same time and delta == 0 is not allowed. The dstm will
 * enforce this and abort the program with a contract-violation message logged
 * via dstm_log_error.
 *
 * @param[in] rank       target MPI rank (must be == @p region_id)
 * @param[in] region_id  region identifier (must be == @p rank)
 * @param[in] time       base logical timestamp
 * @param[in] delta      non-negative offset added to @p time
 * @param[in] dst        opaque pointer into the target region
 * @param[in] src        source buffer of @p size bytes (must be non-NULL)
 * @param[in] size       number of bytes to copy
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL if @p src is NULL or
 *         @p delta is negative, -DSTM_EDOM if @p delta == 0 and @p rank
 *         is not the local rank.  Aborts the program if the put
 *         violates the put-before-get phase rule above.
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
 * The triple (rank, region_id, src.region) must be consistent: each
 * rank's region id is equal to its rank.  @p delta must be <= 0 (a
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
 * @param[in]  rank       target MPI rank (must be == @p region_id)
 * @param[in]  region_id  region identifier (must be == @p rank)
 * @param[in]  time       base logical timestamp
 * @param[in]  delta      non-positive offset added to @p time
 * @param[out] dst        destination buffer of @p size bytes (non-NULL)
 * @param[in]  src        opaque pointer into the source region
 * @param[in]  size       number of bytes to copy
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL if @p dst is NULL or
 *         @p delta is positive, -DSTM_EDOM if @p delta == 0 and @p rank
 *         is not the local rank.
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

#ifdef __cplusplus
}
#endif

#endif
