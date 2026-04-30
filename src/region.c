/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "config.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "dstm.h"

#include "internal/globals.h"
#include "internal/msg.h"
#include "internal/region.h"

/*******************************************************************************
 * Helper Functions - dstm_ptr_t
 ******************************************************************************/

int dstm_ptr_isnull(dstm_ptr_t ptr)
{
	if (ptr.region == DSTM_REGIONID_INVALID)
		return 1;

	if (ptr.region == DSTM_REGIONID_LOCAL)
		return (ptr.offset == 0);

	return 0;
}

dstm_ptr_t dstm_region_ptr_fromvoid(dstm_region_t *r, void *ptr)
{
	dstm_ptr_t p = {DSTM_REGIONID_INVALID, 0};
	if (r == NULL || ptr == NULL)
		return p;

	/* Validate pointer is within region */
	if (ptr < r->buffer || ptr >= (void *)((char *)r->buffer + r->size)) {
		return p;
	}

	/* convert to offset */
	p.region = dstm_global_context.rank;
	p.offset = (intptr_t)((char *)ptr - (char *)r->buffer);
	return p;
}

void *dstm_region_ptr_tovoid(dstm_region_t *r, dstm_ptr_t ptr)
{
	if (ptr.region == DSTM_REGIONID_INVALID || r == NULL)
		return NULL;

	if (ptr.region == DSTM_REGIONID_LOCAL)
		return (void *)ptr.offset;

	if (ptr.region != dstm_global_context.rank ||
	    ptr.offset > (intptr_t)r->size)
		return NULL;

	return (void *)((char *)r->buffer + ptr.offset);
}

/*******************************************************************************
 * Helper Functions - Math and Utilities
 ******************************************************************************/

/* Check if a number is a power of 2 */
static inline int is_power_of_2(size_t x)
{
	return x != 0 && (x & (x - 1)) == 0;
}

/* Round up to next power of 2 */
static size_t next_power_of_2(size_t x)
{
	if (x == 0)
		return 1;
	x--;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	x |= x >> 32;
	x++;
	return x;
}

/* Calculate log2 of a power of 2 */
static int log2_pow2(size_t x)
{
	int log = 0;
	while (x > 1) {
		x >>= 1;
		log++;
	}
	return log;
}

/*******************************************************************************
 * Helper Functions - Order Calculations
 ******************************************************************************/

/* Convert size to order (size = MIN_BLOCK_SIZE * 2^order) */
static int dstm_region_size_to_order(size_t size)
{
	/* Account for header overhead in allocated blocks */
	size += DSTM_REGION_HEADER_SIZE;

	if (size < DSTM_REGION_MIN_BLOCK_SIZE)
		size = DSTM_REGION_MIN_BLOCK_SIZE;

	size_t rounded = next_power_of_2(size);
	return log2_pow2(rounded) - log2_pow2(DSTM_REGION_MIN_BLOCK_SIZE);
}

/* Convert order to size */
static inline size_t dstm_region_order_to_size(int order)
{
	return DSTM_REGION_MIN_BLOCK_SIZE << order;
}

/*******************************************************************************
 * Helper Functions - Bitmap Operations
 ******************************************************************************/

/* Set a bit in the bitmap (block is split) */
static inline void dstm_region_set_split(dstm_region_t *r, size_t index)
{
	r->bitmap[index / 8] |= (1 << (index % 8));
}

/* Clear a bit in the bitmap (block is merged) */
static inline void dstm_region_clear_split(dstm_region_t *r, size_t index)
{
	r->bitmap[index / 8] &= ~(1 << (index % 8));
}

/* Check if a bit is set in the bitmap */
static inline int dstm_region_is_split(dstm_region_t *r, size_t index)
{
	return (r->bitmap[index / 8] >> (index % 8)) & 1;
}

/*******************************************************************************
 * Helper Functions - Address Calculations
 ******************************************************************************/

/* Get block index from pointer */
static inline size_t dstm_region_ptr_to_index(dstm_region_t *r, void *ptr)
{
	return ((char *)ptr - (char *)r->buffer) / DSTM_REGION_MIN_BLOCK_SIZE;
}

/* Get pointer from block index */
static inline void *dstm_region_index_to_ptr(dstm_region_t *r, size_t index)
{
	return (char *)r->buffer + (index * DSTM_REGION_MIN_BLOCK_SIZE);
}

/* Get buddy index for a block at given order */
static inline size_t dstm_region_get_buddy_index(size_t index, int order)
{
	return index ^ (1UL << order);
}

/* Get buddy pointer for a block */
static inline void *
dstm_region_get_buddy(dstm_region_t *r, void *block, int order)
{
	size_t index = dstm_region_ptr_to_index(r, block);
	size_t buddy_index = dstm_region_get_buddy_index(index, order);
	return dstm_region_index_to_ptr(r, buddy_index);
}

/*******************************************************************************
 * Helper Functions - Free List Management
 ******************************************************************************/

/* Add block to free list */
static void
dstm_region_add_to_free_list(dstm_region_t *r, void *block, int order)
{
	dstm_region_block_t *b = (dstm_region_block_t *)block;
	b->next = r->free_lists[order];
	r->free_lists[order] = b;
}

/* Remove block from free list */
static void
dstm_region_remove_from_free_list(dstm_region_t *r, void *block, int order)
{
	dstm_region_block_t **curr = &r->free_lists[order];
	while (*curr != NULL) {
		if (*curr == (dstm_region_block_t *)block) {
			*curr = (*curr)->next;
			return;
		}
		curr = &(*curr)->next;
	}
}

/* Check if a block is in the free list */
static int dstm_region_is_in_free_list(dstm_region_t *r, void *block, int order)
{
	dstm_region_block_t *curr = r->free_lists[order];
	while (curr != NULL) {
		if (curr == (dstm_region_block_t *)block)
			return 1;
		curr = curr->next;
	}
	return 0;
}

/*******************************************************************************
 * Helper Functions - Block Splitting
 ******************************************************************************/

/* Split a block from from_order down to to_order */
static void dstm_region_split_block(dstm_region_t *r,
                                    void *block,
                                    int from_order,
                                    int to_order)
{
	while (from_order > to_order) {
		from_order--;
		size_t index = dstm_region_ptr_to_index(r, block);

		/* Mark parent as split */
		dstm_region_set_split(r, index >> (from_order + 1));

		/* Add right buddy to free list */
		void *buddy = dstm_region_get_buddy(r, block, from_order);
		dstm_region_add_to_free_list(r, buddy, from_order);
	}
}

/*******************************************************************************
 * Helper Functions - Block Merging
 ******************************************************************************/

/* Try to merge block with its buddy, recursively */
static void *dstm_region_merge_block(dstm_region_t *r, void *block, int *order)
{
	while (*order < r->max_order) {
		void *buddy = dstm_region_get_buddy(r, block, *order);

		/* Check if buddy is free */
		if (!dstm_region_is_in_free_list(r, buddy, *order))
			break;

		/* Remove buddy from free list */
		dstm_region_remove_from_free_list(r, buddy, *order);

		/* Merge: use the lower address as the merged block */
		if (buddy < block)
			block = buddy;

		/* Clear split bit for parent */
		size_t index = dstm_region_ptr_to_index(r, block);
		dstm_region_clear_split(r, index >> (*order + 1));

		(*order)++;
	}

	return block;
}

/*******************************************************************************
 * Init functions
 ******************************************************************************/

/* Common initialization logic for region */
static int dstm_region_init_allocator(dstm_region_t *r)
{
	/* Validate that size is power of 2 */
	if (!is_power_of_2(r->size))
		return -DSTM_EINVAL;

	/* Validate minimum size */
	if (r->size < DSTM_REGION_MIN_BLOCK_SIZE)
		return -DSTM_EINVAL;

	/* Ensure minimum block size can accommodate both:
	 * - Allocation header (when allocated)
	 * - Free list pointer (when free)
	 */
	assert(DSTM_REGION_MIN_BLOCK_SIZE >=
	       sizeof(dstm_region_block_header_t));
	assert(DSTM_REGION_MIN_BLOCK_SIZE >= sizeof(dstm_region_block_t));

	/* Calculate max order */
	r->max_order =
	        log2_pow2(r->size) - log2_pow2(DSTM_REGION_MIN_BLOCK_SIZE);

	if (r->max_order > DSTM_REGION_MAX_ORDER)
		return -DSTM_EINVAL;

	/* Allocate free lists array */
	r->free_lists = calloc((size_t)(r->max_order + 1),
	                       sizeof(dstm_region_block_t *));
	if (r->free_lists == NULL)
		return -DSTM_ENOMEM;

	/* Calculate bitmap size */
	size_t num_blocks = r->size / DSTM_REGION_MIN_BLOCK_SIZE;
	r->bitmap_size = (num_blocks + 7) / 8;
	r->bitmap = calloc(r->bitmap_size, 1);
	if (r->bitmap == NULL)
		goto err_free_lists;

	/* Initialize with entire region as one large free block */
	dstm_region_add_to_free_list(r, r->buffer, r->max_order);

	/* Initialize metadata */
	r->time = 0;
	r->ninfos = NULL;
	r->versions = NULL;
	r->barrier_token_count = 0;
	r->barrier_all_drained = 0;

	return DSTM_SUCCESS;

err_free_lists:
	free(r->free_lists);
	r->free_lists = NULL;
	return -DSTM_ENOMEM;
}

int dstm_region_create(dstm_region_t **region, size_t size)
{
	if (region == NULL)
		return -DSTM_EINVAL;

	dstm_region_t *r = calloc(1, sizeof(dstm_region_t));
	if (r == NULL)
		return -DSTM_ENOMEM;

	/* Round size up to next power of 2 */
	r->size = next_power_of_2(size);

	/* Allocate buffer */
	r->buffer = malloc(r->size);
	if (r->buffer == NULL)
		goto err_free_region;

	r->owns_buffer = 1;

	/* Initialize buddy allocator */
	int err = dstm_region_init_allocator(r);
	if (err != DSTM_SUCCESS)
		goto err_free_buffer;

	*region = r;
	return DSTM_SUCCESS;

err_free_buffer:
	free(r->buffer);
err_free_region:
	free(r);
	return -DSTM_ENOMEM;
}

int dstm_region_create_with_buffer(dstm_region_t **region,
                                   void *buffer,
                                   size_t size)
{
	if (region == NULL || buffer == NULL)
		return -DSTM_EINVAL;

	/* Validate that size is power of 2 */
	if (!is_power_of_2(size))
		return -DSTM_EINVAL;

	dstm_region_t *r = calloc(1, sizeof(dstm_region_t));
	if (r == NULL)
		return -DSTM_ENOMEM;

	r->buffer = buffer;
	r->size = size;
	r->owns_buffer = 0;

	/* Initialize buddy allocator */
	int err = dstm_region_init_allocator(r);
	if (err != DSTM_SUCCESS)
		goto err_free_region;

	*region = r;
	return DSTM_SUCCESS;

err_free_region:
	free(r);
	return err;
}

void dstm_region_destroy(dstm_region_t **region)
{
	if (region == NULL)
		return;

	dstm_region_t *r = *region;
	if (r == NULL)
		return;

	/* Free all active versions */
	dstm_region_version_t *v, *vtmp;
	DL_FOREACH_SAFE(r->versions, v, vtmp)
	{
		DL_DELETE(r->versions, v);
		free(v->buffer);
		free(v);
	}

	/* Free neighbor info hash table */
	dstm_region_ninfo_t *ni, *tmp;
	HASH_ITER(hh, r->ninfos, ni, tmp)
	{
		HASH_DEL(r->ninfos, ni);
		free(ni);
	}

	/* Free buddy allocator structures */
	if (r->free_lists != NULL)
		free(r->free_lists);

	if (r->bitmap != NULL)
		free(r->bitmap);

	/* Free buffer if we own it */
	if (r->owns_buffer && r->buffer != NULL)
		free(r->buffer);

	free(r);
	*region = NULL;
}

/*******************************************************************************
 * Neighbor Info Functions
 ******************************************************************************/

int dstm_region_set_ninfo(dstm_region_t *region, int rank, dstm_time_t time)
{
	if (region == NULL)
		return -DSTM_EINVAL;

	dstm_region_ninfo_t *ni;
	HASH_FIND_INT(region->ninfos, &rank, ni);
	if (ni == NULL) {
		ni = calloc(1, sizeof(*ni));
		assert(ni != NULL);
		ni->rank = rank;
		ni->time = time;
		HASH_ADD_INT(region->ninfos, rank, ni);
	} else {
		if (time > ni->time)
			ni->time = time;
	}
	return DSTM_SUCCESS;
}

int dstm_region_get_ninfo(dstm_region_t *region, int rank, dstm_time_t *time)
{
	if (region == NULL)
		return -DSTM_EINVAL;

	dstm_region_ninfo_t *ni;
	HASH_FIND_INT(region->ninfos, &rank, ni);
	if (ni == NULL) {
		*time = 0;
		return -DSTM_EINVAL;
	}

	*time = ni->time;
	return DSTM_SUCCESS;
}

/*******************************************************************************
 * Version Management Functions
 ******************************************************************************/

/* Comparison function for sorted insertion into the version DL list. */
static int dstm_region_version_cmp(dstm_region_version_t *a,
                                   dstm_region_version_t *b)
{
	if (a->time < b->time)
		return -1;
	if (a->time > b->time)
		return 1;
	return 0;
}

dstm_region_version_t *dstm_region_version_find(dstm_region_t *region,
                                                dstm_time_t time)
{
	if (region == NULL)
		return NULL;

	dstm_region_version_t *it;
	DL_FOREACH(region->versions, it)
	{
		if (it->time == time)
			return it;
		/* List is sorted ascending; stop early if past target. */
		if (it->time > time)
			return NULL;
	}
	return NULL;
}

dstm_region_version_t *
dstm_region_version_find_or_create(dstm_region_t *region, dstm_time_t time)
{
	if (region == NULL)
		return NULL;

	dstm_region_version_t *existing = dstm_region_version_find(region,
	                                                           time);
	if (existing != NULL)
		return existing;

	dstm_region_version_t *v = calloc(1, sizeof(*v));
	if (v == NULL)
		return NULL;

	v->time = time;
	v->status = DSTM_REGION_STATUS_WRITABLE;
	v->buffer = malloc(region->size);
	if (v->buffer == NULL) {
		free(v);
		return NULL;
	}

	DL_INSERT_INORDER(region->versions, v, dstm_region_version_cmp);
	return v;
}

void dstm_region_version_destroy(dstm_region_t *region,
                                 dstm_region_version_t *version)
{
	if (region == NULL || version == NULL)
		return;

	DL_DELETE(region->versions, version);
	free(version->buffer);
	free(version);
}

/*******************************************************************************
 * Allocation Functions
 ******************************************************************************/

dstm_ptr_t dstm_region_malloc(dstm_region_t *r, size_t size)
{
	dstm_ptr_t ret = {DSTM_REGIONID_INVALID, 0};
	if (r == NULL || size == 0)
		return ret;

	/* Calculate required order */
	int order = dstm_region_size_to_order(size);

	/* Check if order is valid */
	if (order > r->max_order)
		return ret;

	/* Find smallest available block >= required order */
	int current_order = order;
	while (current_order <= r->max_order &&
	       r->free_lists[current_order] == NULL) {
		current_order++;
	}

	/* No block available */
	if (current_order > r->max_order)
		return ret;

	/* Remove block from free list */
	dstm_region_block_t *block = r->free_lists[current_order];
	r->free_lists[current_order] = block->next;

	/* Split block if necessary */
	if (current_order > order) {
		dstm_region_split_block(r, block, current_order, order);
	}

	/* Store allocation order in header */
	dstm_region_block_header_t *header =
	        (dstm_region_block_header_t *)block;
	header->order = (uint8_t)order;

	/* Return pointer to user data (after header) */
	void *user_ptr = (char *)block + DSTM_REGION_HEADER_SIZE;
	return dstm_region_ptr_fromvoid(r, user_ptr);
}

void dstm_region_free(dstm_region_t *r, dstm_ptr_t p)
{
	void *user_ptr = dstm_region_ptr_tovoid(r, p);

	if (r == NULL || user_ptr == NULL)
		return;

	/* Get actual block pointer (before header) */
	void *block_ptr = (char *)user_ptr - DSTM_REGION_HEADER_SIZE;

	/* Validate block pointer is within region */
	if (block_ptr < r->buffer ||
	    block_ptr >= (void *)((char *)r->buffer + r->size)) {
		return;
	}

	/* Read allocation order from header */
	dstm_region_block_header_t *header =
	        (dstm_region_block_header_t *)block_ptr;
	int order = header->order;

	/* Try to merge with buddy (updates order to final merged order) */
	void *merged_ptr = dstm_region_merge_block(r, block_ptr, &order);

	/* Add to free list at final order */
	dstm_region_add_to_free_list(r, merged_ptr, order);
}

/*******************************************************************************
 * Formatting helpers
 ******************************************************************************/

json_t *dstm_ptr_to_json(dstm_ptr_t p)
{
	if (dstm_ptr_isnull(p))
		return json_null();
	if (p.region == DSTM_REGIONID_LOCAL)
		return json_pack("{s:s, s:I}", "region", "LOCAL", "offset",
		                 (json_int_t)p.offset);
	if (p.region == DSTM_REGIONID_INVALID)
		return json_pack("{s:s, s:I}", "region", "INVALID", "offset",
		                 (json_int_t)p.offset);
	return json_pack("{s:i, s:I}", "region", p.region, "offset",
	                 (json_int_t)p.offset);
}

json_t *dstm_region_ninfo_to_json(dstm_region_ninfo_t *ni)
{
	if (ni == NULL)
		return json_null();
	return json_pack("{s:i, s:I}", "rank", ni->rank, "time",
	                 (json_int_t)ni->time);
}

json_t *dstm_region_version_to_json(dstm_region_version_t *v)
{
	if (v == NULL)
		return json_null();
	const char *status_str;
	switch (v->status) {
	case DSTM_REGION_STATUS_WRITABLE:
		status_str = "writable";
		break;
	case DSTM_REGION_STATUS_READABLE:
		status_str = "readable";
		break;
	case DSTM_REGION_STATUS_DISCARDABLE:
		status_str = "discardable";
		break;
	default:
		status_str = "unknown";
		break;
	}
	json_t *buf = json_sprintf("%p", v->buffer);
	return json_pack("{s:s, s:s, s:I}", "buffer", buf, "status", status_str,
	                 "time", (json_int_t)v->time);
}

json_t *dstm_region_to_json(dstm_region_t *r)
{
	if (r == NULL)
		return json_null();
	json_t *buf = json_sprintf("%p", r->buffer);
	return json_pack("{s:s, s:I, s:I}", "buffer", buf, "size",
	                 (json_int_t)r->size, "time", (json_int_t)r->time);
}
