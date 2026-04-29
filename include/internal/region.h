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
 * Region management, internal
 **/

#ifndef DSTM_INTERNAL_REGION_H
#define DSTM_INTERNAL_REGION_H 1

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "internal/uthash.h"
#include "internal/utlist.h"

/*******************************************************************************
 * CONSTANTS
 ******************************************************************************/

#define DSTM_DEFAULT_REGION_SIZE (1 << 20)
#define DSTM_REGION_MIN_BLOCK_SIZE 64 /* 64 bytes minimum allocation */
#define DSTM_REGION_MAX_ORDER 30 /* Supports up to 1GB regions */

/*******************************************************************************
 * REGION STATUS VALUES
 *
 * Used by the per-version state machine and the messaging layer (heartbeats).
 ******************************************************************************/

#define DSTM_REGION_STATUS_WRITABLE 0
#define DSTM_REGION_STATUS_READABLE 1
#define DSTM_REGION_STATUS_DISCARDABLE 2

/*******************************************************************************
 * BUDDY ALLOCATOR STRUCTURES
 ******************************************************************************/

/* Free block header - stored in free blocks themselves */
struct dstm_region_block_s {
	struct dstm_region_block_s *next; /* Next free block in list */
};
typedef struct dstm_region_block_s dstm_region_block_t;

/**
 * Allocation block header - prepended to each allocated block
 *
 * Layout in memory:
 *   [header: 16 bytes][user data: block_size - 16 bytes]
 *
 * The header stores metadata needed to free the block correctly.
 * Using a 16-byte header ensures user data starts at a 16-byte aligned
 * address (since blocks are always power-of-2 sized and >= 64 bytes),
 * which is beneficial for SIMD operations and cache alignment.
 *
 * When a block is allocated:
 *   - The header is written with the allocation order
 *   - A pointer to the user data (after header) is returned
 *
 * When a block is freed:
 *   - The header is read to determine the block's order
 *   - The block is merged with buddies if possible
 *   - The header space is reused for free list linkage
 */
struct dstm_region_block_header_s {
	uint8_t order; /* Allocation order (0 to max_order) */
	uint8_t _padding[15]; /* Padding to 16 bytes for alignment */
};
typedef struct dstm_region_block_header_s dstm_region_block_header_t;

#define DSTM_REGION_HEADER_SIZE sizeof(dstm_region_block_header_t)

/* Neighbor info for distributed memory tracking */
struct dstm_region_ninfo_s {
	int rank;
	dstm_time_t time;
	int state; /* DSTM_REGION_STATUS_WRITABLE or DSTM_REGION_STATUS_READABLE
	            */
	UT_hash_handle hh;
};
typedef struct dstm_region_ninfo_s dstm_region_ninfo_t;

/*******************************************************************************
 * VERSION STRUCTURE
 *
 * Each version represents a full snapshot of the region's data buffer at a
 * specific logical timestamp.  Versions are maintained in a doubly-linked list
 * sorted by time.  Each version has its own state machine:
 *   WRITABLE -> READABLE -> DISCARDABLE
 *
 * PUTs create/update the target version immediately.  GETs wait for the
 * version to become READABLE.  Old versions are evicted (DISCARDABLE) once
 * all GETs are done and no neighbor can still send a GET for that timestamp
 * (distance-based check).
 ******************************************************************************/

struct dstm_region_version_s {
	dstm_time_t time; /* Logical timestamp this version represents */
	int status; /* WRITABLE, READABLE, or DISCARDABLE */
	void *buffer; /* Full copy of region data (malloc'd, region->size) */
	struct dstm_region_version_s *prev, *next; /* utlist DL linkage */
};
typedef struct dstm_region_version_s dstm_region_version_t;

/* Region structure with buddy allocator */
struct dstm_region_s {
	/* Memory buffer */
	void *buffer; /* Base address of region */
	size_t size; /* Total size (power of 2) */
	int max_order; /* log2(size/MIN_BLOCK_SIZE) */
	int owns_buffer; /* 1 if we allocated, 0 if user-provided */

	/* Buddy allocator state */
	dstm_region_block_t **free_lists; /* Array of free lists, one per order
	                                   */
	uint8_t *bitmap; /* Split bitmap for buddy tracking */
	size_t bitmap_size; /* Size of bitmap in bytes */

	dstm_time_t time; /* Minimum active version timestamp */
	dstm_region_ninfo_t *ninfos;

	/* DL list of active versions, sorted by time (ascending).
	 * Each version has its own buffer and state machine. */
	dstm_region_version_t *versions;

	/* Number of barrier tokens received but not yet consumed by the
	 * barrier DFA.  The BARRIER message handler increments this on
	 * every token arrival; the barrier DFA decrements it when it
	 * consumes a token to advance to the next state. */
	int barrier_token_count;

	/* Set to 1 after the first barrier round completes (all ranks
	 * have drained their outgoing requests).  When set, the version
	 * transition checks (WRITABLE->READABLE, READABLE->DISCARDABLE)
	 * skip ALL neighbor distance checks, since no rank can issue new
	 * PUT/GET messages.  Reset to 0 when the barrier exits. */
	int barrier_all_drained;
};

/*******************************************************************************
 * Create/Destroy
 ******************************************************************************/

int dstm_region_create(dstm_region_t **region, size_t size);
int dstm_region_create_with_buffer(dstm_region_t **region,
                                   void *buffer,
                                   size_t size);
void dstm_region_destroy(dstm_region_t **region);

/*******************************************************************************
 * Methods
 ******************************************************************************/

int dstm_region_set_ninfo(dstm_region_t *region,
                          int rank,
                          dstm_time_t time,
                          int state);
int dstm_region_get_ninfo(dstm_region_t *region, int rank, dstm_time_t *time,
			  int *state);

dstm_ptr_t dstm_region_malloc(dstm_region_t *r, size_t size);
void dstm_region_free(dstm_region_t *r, dstm_ptr_t ptr);

/*******************************************************************************
 * Version management
 ******************************************************************************/

/* Find an existing version for the given timestamp, or NULL. */
dstm_region_version_t *dstm_region_version_find(dstm_region_t *region,
                                                dstm_time_t time);

/* Find or create a version for the given timestamp.  New versions are
 * created with an uninitialized buffer and WRITABLE status, inserted in
 * sorted order.  Returns NULL on allocation failure. */
dstm_region_version_t *
dstm_region_version_find_or_create(dstm_region_t *region, dstm_time_t time);

/* Destroy a single version: free its buffer and remove it from the
 * region's version list. */
void dstm_region_version_destroy(dstm_region_t *region,
                                 dstm_region_version_t *version);

/*******************************************************************************
 * Formatting helpers
 *
 * Each *_snprintf function follows the standard snprintf contract: writes at
 * most n bytes (including the NUL terminator) into buf, and returns the number
 * of bytes that would have been written had n been large enough (not counting
 * the NUL).  Use the accompanying SIZE macro to allocate a buffer that is
 * always sufficient.
 ******************************************************************************/

#define DSTM_PTR_SNPRINTF_SIZE 64
int dstm_ptr_snprintf(char *buf, size_t n, dstm_ptr_t p);

#define DSTM_REGION_NINFO_SNPRINTF_SIZE 96
int dstm_region_ninfo_snprintf(char *buf, size_t n, dstm_region_ninfo_t *ni);

#define DSTM_REGION_VERSION_SNPRINTF_SIZE 128
int dstm_region_version_snprintf(char *buf, size_t n,
                                 dstm_region_version_t *v);

#define DSTM_REGION_SNPRINTF_SIZE 128
int dstm_region_snprintf(char *buf, size_t n, dstm_region_t *r);

/* Emit a log message at the given level with label: <formatted value>.
 * The buffer is allocated on the stack and the level is checked before any
 * formatting work is done. */
#define dstm_log_ptr(level, label, ptr)                                        \
	do {                                                                   \
		if ((level) <= dstm_log_level) {                               \
			char __buf[DSTM_PTR_SNPRINTF_SIZE];                    \
			dstm_ptr_snprintf(__buf, sizeof(__buf), (ptr));        \
			dstm_log_printf((level), __FILE__, __func__, __LINE__, \
			                "%s: %s\n", (label), __buf);           \
		}                                                              \
	} while (0)

#define dstm_log_region_ninfo(level, label, ni)                                \
	do {                                                                   \
		if ((level) <= dstm_log_level) {                               \
			char __buf[DSTM_REGION_NINFO_SNPRINTF_SIZE];           \
			dstm_region_ninfo_snprintf(__buf, sizeof(__buf),       \
			                           (ni));                      \
			dstm_log_printf((level), __FILE__, __func__, __LINE__, \
			                "%s: %s\n", (label), __buf);           \
		}                                                              \
	} while (0)

#define dstm_log_region_version(level, label, v)                               \
	do {                                                                   \
		if ((level) <= dstm_log_level) {                               \
			char __buf[DSTM_REGION_VERSION_SNPRINTF_SIZE];         \
			dstm_region_version_snprintf(__buf, sizeof(__buf),     \
			                             (v));                     \
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

#ifdef __cplusplus
}
#endif

#endif
