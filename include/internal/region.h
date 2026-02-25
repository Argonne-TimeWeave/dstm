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

#define DSTM_DEFAULT_REGION_SIZE (1 << 30)
#define DSTM_REGION_MIN_BLOCK_SIZE 64 /* 64 bytes minimum allocation */
#define DSTM_REGION_MAX_ORDER 30 /* Supports up to 1GB regions */

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
	UT_hash_handle hh;
};
typedef struct dstm_region_ninfo_s dstm_region_ninfo_t;

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

	dstm_time_t time;
	dstm_region_ninfo_t *ninfos;
};

/*******************************************************************************
 * Create/Destroy
 ******************************************************************************/

int dstm_region_create(dstm_region_t **region, size_t size);
int dstm_region_create_with_buffer(dstm_region_t **region,
                                   void *buffer,
                                   size_t size);
void dstm_region_destroy(dstm_region_t **region);

int dstm_region_set_ninfo(dstm_region_t *region, int rank, dstm_time_t time);
int dstm_region_get_ntime(dstm_region_t *region, int rank, dstm_time_t *time);

dstm_ptr_t dstm_region_malloc(dstm_region_t *r, size_t size);
void dstm_region_free(dstm_region_t *r, dstm_ptr_t ptr);

#ifdef __cplusplus
}
#endif

#endif
