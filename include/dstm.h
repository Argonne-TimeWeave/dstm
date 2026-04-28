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

// clang-format off
#include "dstm/error.h"
#include "dstm/version.h"
// clang-format on

/*******************************************************************************
 * Library Init/Fini
 ******************************************************************************/

/** initializes the library, in particular using environment variables to figure
 * out if the library should emit messages at all and if a ratelimit is in
 * place.
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

/* basic configuration of dstm:
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

int dstm_darray_get(dstm_darray_t *darray,
                    size_t global_index,
                    void *dst,
                    dstm_time_t time,
                    dstm_time_t delta);
int dstm_darray_put(dstm_darray_t *darray,
                    size_t global_index,
                    void *src,
                    dstm_time_t time,
                    dstm_time_t delta);
int dstm_darray_allgatherv(dstm_darray_t *darray,
                           void *src,
                           void *dst,
                           dstm_time_t time,
                           const dstm_time_t *deltas);

/* Format: { element_size: N, total_elements: N, local_elements: N } */
#define DSTM_DARRAY_SNPRINTF_SIZE 128
int dstm_darray_snprintf(char *buf, size_t n, dstm_darray_t *a);

#define dstm_log_darray(level, label, a)                                       \
	do {                                                                   \
		if ((level) <= dstm_log_level) {                               \
			char __buf[DSTM_DARRAY_SNPRINTF_SIZE];                 \
			dstm_darray_snprintf(__buf, sizeof(__buf), (a));       \
			dstm_log_printf((level), __FILE__, __func__, __LINE__, \
			                "%s: %s\n", (label), __buf);           \
		}                                                              \
	} while (0)

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

/* Maximum buffer sizes guaranteed to hold the formatted representation. */
#define DSTM_PTR_SNPRINTF_SIZE 64
#define DSTM_REGION_SNPRINTF_SIZE 128

int dstm_ptr_snprintf(char *buf, size_t n, dstm_ptr_t p);
int dstm_region_snprintf(char *buf, size_t n, dstm_region_t *r);

/* Level-parameterised logging helpers.  The level check is performed before
 * any formatting work so suppressed messages have near-zero overhead. */
#define dstm_log_ptr(level, label, ptr)                                        \
	do {                                                                   \
		if ((level) <= dstm_log_level) {                               \
			char __buf[DSTM_PTR_SNPRINTF_SIZE];                    \
			dstm_ptr_snprintf(__buf, sizeof(__buf), (ptr));        \
			dstm_log_printf((level), __FILE__, __func__, __LINE__, \
			                "%s: %s\n", (label), __buf);           \
		}                                                              \
	} while (0)

#define dstm_log_region(level, label, r)                                       \
	do {                                                                   \
		if ((level) <= dstm_log_level) {                               \
			char __buf[DSTM_REGION_SNPRINTF_SIZE];                 \
			dstm_region_snprintf(__buf, sizeof(__buf), (r));       \
			dstm_log_printf((level), __FILE__, __func__, __LINE__, \
			                "%s: %s\n", (label), __buf);           \
		}                                                              \
	} while (0)

/*******************************************************************************
 * DSTM BASIC OPERATIONS
 ******************************************************************************/

int dstm_put(int rank,
             int region_id,
             dstm_time_t time,
             dstm_time_t delta,
             dstm_ptr_t dst,
             void *src,
             size_t size);
int dstm_get(int rank,
             int region_id,
             dstm_time_t time,
             dstm_time_t delta,
             void *dst,
             dstm_ptr_t src,
             size_t size);

#ifdef __cplusplus
}
#endif

#endif
