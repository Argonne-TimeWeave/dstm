/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <check.h>
#include <stdlib.h>
#include <string.h>

#include "dstm.h"

#include "internal/region.h"

/*******************************************************************************
 * Test Helpers
 ******************************************************************************/

/* Helper to verify memory is writable and readable */
static int verify_memory_accessible(void *ptr, size_t size)
{
	if (ptr == NULL)
		return 0;

	/* Write pattern to memory */
	memset(ptr, 0xAA, size);

	/* Read back and verify */
	unsigned char *bytes = (unsigned char *)ptr;
	for (size_t i = 0; i < size; i++) {
		if (bytes[i] != 0xAA)
			return 0;
	}

	return 1;
}

/* Helper to check if size is power of 2 */
static int is_power_of_2(size_t size)
{
	return size != 0 && (size & (size - 1)) == 0;
}

/* Helper to round up to next power of 2 */
static size_t next_power_of_2(size_t size)
{
	if (size == 0)
		return 1;
	size--;
	size |= size >> 1;
	size |= size >> 2;
	size |= size >> 4;
	size |= size >> 8;
	size |= size >> 16;
	size |= size >> 32;
	size++;
	return size;
}

/*******************************************************************************
 * Create/Destroy Tests
 ******************************************************************************/

START_TEST(test_region_create_null)
{
	int rc = dstm_region_create(NULL, 4096);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
}
END_TEST

START_TEST(test_region_create_destroy)
{
	dstm_region_t *region = NULL;
	int rc = dstm_region_create(&region, 4096);
	ck_assert_int_eq(rc, DSTM_SUCCESS);
	ck_assert_ptr_nonnull(region);

	dstm_region_destroy(&region);
	ck_assert_ptr_null(region);
}
END_TEST

START_TEST(test_region_create_various_sizes)
{
	size_t sizes[] = {4096, 65536, 1048576};

	for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
		dstm_region_t *region = NULL;
		int rc = dstm_region_create(&region, sizes[i]);
		ck_assert_int_eq(rc, DSTM_SUCCESS);
		ck_assert_ptr_nonnull(region);

		/* Verify region has expected size (rounded up to power of 2) */
		size_t expected_size = next_power_of_2(sizes[i]);
		ck_assert_uint_eq(region->size, expected_size);
		ck_assert(is_power_of_2(region->size));

		dstm_region_destroy(&region);
	}
}
END_TEST

START_TEST(test_region_create_with_buffer)
{
	size_t size = 4096;
	void *buffer = aligned_alloc(64, size);
	ck_assert_ptr_nonnull(buffer);

	dstm_region_t *region = NULL;
	int rc = dstm_region_create_with_buffer(&region, buffer, size);
	ck_assert_int_eq(rc, DSTM_SUCCESS);
	ck_assert_ptr_nonnull(region);

	/* Verify buffer is used */
	ck_assert_ptr_eq(region->buffer, buffer);
	ck_assert_int_eq(region->owns_buffer, 0);

	/* Use the region */
	dstm_ptr_t ptr = dstm_region_malloc(region, 128);
	ck_assert(!dstm_ptr_isnull(ptr));

	dstm_region_destroy(&region);
	free(buffer);
}
END_TEST

START_TEST(test_region_create_with_buffer_invalid_size)
{
	size_t size = 4097; /* Not power of 2 */
	void *buffer = malloc(size);
	ck_assert_ptr_nonnull(buffer);

	dstm_region_t *region = NULL;
	int rc = dstm_region_create_with_buffer(&region, buffer, size);
	ck_assert_int_eq(rc, -DSTM_EINVAL);

	free(buffer);
}
END_TEST

START_TEST(test_region_create_with_buffer_null)
{
	dstm_region_t *region = NULL;
	int rc = dstm_region_create_with_buffer(&region, NULL, 4096);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
}
END_TEST

/*******************************************************************************
 * Basic Allocation Tests
 ******************************************************************************/

START_TEST(test_region_malloc_null_region)
{
	dstm_ptr_t ptr = dstm_region_malloc(NULL, 128);
	ck_assert(dstm_ptr_isnull(ptr));
}
END_TEST

START_TEST(test_region_malloc_zero_size)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 4096);

	dstm_ptr_t ptr = dstm_region_malloc(region, 0);
	ck_assert(dstm_ptr_isnull(ptr));

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_malloc_simple)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 4096);

	dstm_ptr_t ptr = dstm_region_malloc(region, 128);
	ck_assert(!dstm_ptr_isnull(ptr));
	ck_assert(verify_memory_accessible(dstm_region_ptr_tovoid(region, ptr),
	                                   128));

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_malloc_multiple)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 4096);

	void *ptrs[10];
	for (int i = 0; i < 10; i++) {
		ptrs[i] = dstm_region_ptr_tovoid(
		        region, dstm_region_malloc(region, 64));
		ck_assert_ptr_nonnull(ptrs[i]);
		ck_assert(verify_memory_accessible(ptrs[i], 64));
	}

	/* Verify all pointers are distinct */
	for (int i = 0; i < 10; i++) {
		for (int j = i + 1; j < 10; j++) {
			ck_assert_ptr_ne(ptrs[i], ptrs[j]);
		}
	}

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_malloc_write_read)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 4096);

	int *int_ptr = dstm_region_ptr_tovoid(
	        region, dstm_region_malloc(region, sizeof(int)));
	ck_assert_ptr_nonnull(int_ptr);

	*int_ptr = 42;
	ck_assert_int_eq(*int_ptr, 42);

	dstm_region_destroy(&region);
}
END_TEST

/*******************************************************************************
 * Basic Deallocation Tests
 ******************************************************************************/

START_TEST(test_region_free_null_region)
{
	dstm_ptr_t ptr = {DSTM_REGIONID_LOCAL, (intptr_t)0x1000};
	dstm_region_free(NULL, ptr);
	/* Should not crash */
}
END_TEST

START_TEST(test_region_free_null_ptr)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 4096);

	dstm_ptr_t ptr = {DSTM_REGIONID_LOCAL, 0};
	dstm_region_free(region, ptr);
	/* Should not crash */

	ptr = (dstm_ptr_t){DSTM_REGIONID_INVALID, 0};
	dstm_region_free(region, ptr);
	/* Should not crash */

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_free_simple)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 4096);

	dstm_ptr_t ptr = dstm_region_malloc(region, 256);
	ck_assert(!dstm_ptr_isnull(ptr));

	dstm_region_free(region, ptr);
	/* Should not crash */

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_free_multiple)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 4096);

	dstm_ptr_t ptrs[5];
	for (int i = 0; i < 5; i++) {
		ptrs[i] = dstm_region_malloc(region, 128);
		ck_assert(!dstm_ptr_isnull(ptrs[i]));
	}

	for (int i = 0; i < 5; i++) {
		dstm_region_free(region, ptrs[i]);
	}

	dstm_region_destroy(&region);
}
END_TEST

/*******************************************************************************
 * Buddy Allocator Specific Tests
 ******************************************************************************/

START_TEST(test_region_malloc_exact_sizes)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 8192);

	/* Test power-of-2 sizes from min to large */
	size_t sizes[] = {64, 128, 256, 512, 1024, 2048, 4096};

	for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
		dstm_ptr_t p = dstm_region_malloc(region, sizes[i]);
		void *ptr = dstm_region_ptr_tovoid(region, p);
		ck_assert_ptr_nonnull(ptr);
		ck_assert(verify_memory_accessible(ptr, sizes[i]));
		dstm_region_free(region, p);
	}

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_malloc_round_up)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 4096);

	/* Non-power-of-2 sizes should round up */
	void *ptr1 = dstm_region_ptr_tovoid(
	        region, dstm_region_malloc(region, 100)); /* Should get 128 */
	ck_assert_ptr_nonnull(ptr1);
	ck_assert(verify_memory_accessible(ptr1, 128));

	void *ptr2 = dstm_region_ptr_tovoid(
	        region, dstm_region_malloc(region, 200)); /* Should get 256 */
	ck_assert_ptr_nonnull(ptr2);
	ck_assert(verify_memory_accessible(ptr2, 256));

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_malloc_max_size)
{
	size_t region_size = 8192;
	dstm_region_t *region = NULL;
	dstm_region_create(&region, region_size);

	/* Should be able to allocate half the region */
	void *ptr = dstm_region_ptr_tovoid(
	        region, dstm_region_malloc(region, (region_size / 2) - 16));
	ck_assert_ptr_nonnull(ptr);

	/* Second allocation of the other half should succeed */
	void *ptr2 = dstm_region_ptr_tovoid(
	        region, dstm_region_malloc(region, (region_size / 2) - 16));
	ck_assert_ptr_nonnull(ptr2);

	/* Third allocation should fail */
	void *ptr3 =
	        dstm_region_ptr_tovoid(region, dstm_region_malloc(region, 64));
	ck_assert_ptr_null(ptr3);

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_malloc_too_large)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 4096);

	/* Try to allocate more than region size */
	dstm_ptr_t ptr = dstm_region_malloc(region, 8192);
	ck_assert(dstm_ptr_isnull(ptr));

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_coalesce_simple)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 2048);

	/* Allocate two buddy blocks */
	dstm_ptr_t ptr1 = dstm_region_malloc(region, 1000);
	dstm_ptr_t ptr2 = dstm_region_malloc(region, 1000);
	ck_assert(!dstm_ptr_isnull(ptr1));
	ck_assert(!dstm_ptr_isnull(ptr2));

	/* Free both - they should coalesce */
	dstm_region_free(region, ptr1);
	dstm_region_free(region, ptr2);

	/* Should now be able to allocate 2048-16 after coalescing */
	dstm_ptr_t ptr3 = dstm_region_malloc(region, 2000);
	ck_assert(!dstm_ptr_isnull(ptr3));

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_coalesce_complex)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 2048);

	/* Allocate 4 blocks of 512 each */
	dstm_ptr_t ptrs[4];
	for (int i = 0; i < 4; i++) {
		ptrs[i] = dstm_region_malloc(region, 480);
		ck_assert(!dstm_ptr_isnull(ptrs[i]));
	}

	/* Free all - should coalesce back up */
	for (int i = 0; i < 4; i++) {
		dstm_region_free(region, ptrs[i]);
	}

	/* Should be able to allocate larger block now */
	void *large = dstm_region_ptr_tovoid(
	        region, dstm_region_malloc(region, 2000));
	ck_assert_ptr_nonnull(large);

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_minimum_block)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 4096);

	/* Allocate minimum block size */
	dstm_ptr_t p = dstm_region_malloc(region, DSTM_REGION_MIN_BLOCK_SIZE);
	void *ptr = dstm_region_ptr_tovoid(region, p);
	ck_assert_ptr_nonnull(ptr);
	ck_assert(verify_memory_accessible(ptr, DSTM_REGION_MIN_BLOCK_SIZE));

	dstm_region_free(region, p);
	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_below_minimum)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 4096);

	/* Allocate less than minimum - should round up to minimum */
	void *ptr =
	        dstm_region_ptr_tovoid(region, dstm_region_malloc(region, 1));
	ck_assert_ptr_nonnull(ptr);
	ck_assert(verify_memory_accessible(ptr, DSTM_REGION_MIN_BLOCK_SIZE));

	dstm_region_destroy(&region);
}
END_TEST

/*******************************************************************************
 * Fragmentation Tests
 ******************************************************************************/

START_TEST(test_region_reuse_freed_blocks)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 256);

	/* Allocate and free */
	dstm_ptr_t ptr1 = dstm_region_malloc(region, 200);
	void *p1 = dstm_region_ptr_tovoid(region, ptr1);
	ck_assert_ptr_nonnull(p1);
	dstm_region_free(region, ptr1);

	/* Allocate again - should reuse freed block */
	dstm_ptr_t ptr2 = dstm_region_malloc(region, 200);
	void *p2 = dstm_region_ptr_tovoid(region, ptr2);
	ck_assert_ptr_nonnull(p2);
	ck_assert_ptr_eq(p1, p2); /* Should be same address */

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_fragmentation_interleaved)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 4096);

	dstm_ptr_t ptrs[8];

	/* Allocate 8 blocks */
	for (int i = 0; i < 8; i++) {
		ptrs[i] = dstm_region_malloc(region, 480);
		ck_assert(!dstm_ptr_isnull(ptrs[i]));
	}

	/* Free every other block */
	for (int i = 0; i < 8; i += 2) {
		dstm_region_free(region, ptrs[i]);
	}

	/* Should be able to allocate 4 more blocks of same size */
	for (int i = 0; i < 4; i++) {
		dstm_ptr_t p = dstm_region_malloc(region, 480);
		ck_assert(!dstm_ptr_isnull(p));
	}

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_fragmentation_pattern)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 2048);

	/* Create a specific fragmentation pattern */
	dstm_ptr_t p1 = dstm_region_malloc(region, 400);
	dstm_ptr_t p2 = dstm_region_malloc(region, 400);
	dstm_ptr_t p3 = dstm_region_malloc(region, 400);
	dstm_ptr_t p4 = dstm_region_malloc(region, 400);

	ck_assert(!dstm_ptr_isnull(p1));
	ck_assert(!dstm_ptr_isnull(p2));
	ck_assert(!dstm_ptr_isnull(p3));
	ck_assert(!dstm_ptr_isnull(p4));

	/* Free middle block */
	dstm_region_free(region, p2);

	/* Allocate smaller block - should fit in freed space */
	void *p5 =
	        dstm_region_ptr_tovoid(region, dstm_region_malloc(region, 256));
	ck_assert_ptr_nonnull(p5);

	dstm_region_destroy(&region);
}
END_TEST

/*******************************************************************************
 * Exhaustion Tests
 ******************************************************************************/

START_TEST(test_region_exhaustion)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 4096);

	dstm_ptr_t ptrs[64];
	int count = 0;

	/* Allocate until exhausted */
	for (int i = 0; i < 64; i++) {
		ptrs[i] = dstm_region_malloc(region, 48);
		if (!dstm_ptr_isnull(ptrs[i]))
			count++;
		else
			break;
	}

	ck_assert_int_eq(count, 64);

	/* Next allocation should fail */
	dstm_ptr_t ptr = dstm_region_malloc(region, 64);
	ck_assert(dstm_ptr_isnull(ptr));

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_exhaustion_recovery)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 8192);

	/* Fill region completely */
	dstm_ptr_t ptr1 = dstm_region_malloc(region, 8192 - 32);
	ck_assert(!dstm_ptr_isnull(ptr1));

	/* Second allocation should fail - region is full */
	dstm_ptr_t ptr2 = dstm_region_malloc(region, 64);
	ck_assert(dstm_ptr_isnull(ptr2));

	/* Free first allocation and try again - should succeed */
	dstm_region_free(region, ptr1);
	dstm_ptr_t ptr3 = dstm_region_malloc(region, 64);
	ck_assert(!dstm_ptr_isnull(ptr3));

	dstm_region_destroy(&region);
}
END_TEST

/*******************************************************************************
 * Stress Tests
 ******************************************************************************/

START_TEST(test_region_stress_mixed_sizes)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 65536);

	dstm_ptr_t ptrs[100];
	size_t sizes[] = {48, 112, 240, 498, 1008};
	int allocated = 0;

	/* Allocate mix of sizes */
	for (int i = 0; i < 100; i++) {
		size_t size = sizes[i % 5];
		ptrs[i] = dstm_region_malloc(region, size);
		if (!dstm_ptr_isnull(ptrs[i])) {
			void *p = dstm_region_ptr_tovoid(region, ptrs[i]);
			ck_assert(verify_memory_accessible(p, size));
			allocated++;
		}
	}

	ck_assert_int_eq(allocated, 100);

	/* Free half */
	for (int i = 0; i < allocated; i += 2) {
		dstm_region_free(region, ptrs[i]);
	}

	/* Allocate more */
	for (int i = 0; i < 20; i++) {
		dstm_ptr_t ptr = dstm_region_malloc(region, 112);
		ck_assert(!dstm_ptr_isnull(ptr));
	}

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_stress_many_small)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 16384);

	/* Allocate many small blocks */
	dstm_ptr_t ptrs[200];
	int count = 0;
	for (int i = 0; i < 200; i++) {
		ptrs[i] = dstm_region_malloc(region, 48);
		if (!dstm_ptr_isnull(ptrs[i]))
			count++;
	}

	/* Should allocate at least some blocks */
	ck_assert_int_eq(count, 200);

	dstm_region_destroy(&region);
}
END_TEST

/*******************************************************************************
 * Version Management Tests
 ******************************************************************************/

START_TEST(test_version_find_nonexistent)
{
	dstm_region_t *r = NULL;
	int err = dstm_region_create(&r, 4096);
	ck_assert_int_eq(err, DSTM_SUCCESS);
	ck_assert_ptr_nonnull(r);

	/* No versions exist yet */
	dstm_region_version_t *v = dstm_region_version_find(r, 0);
	ck_assert_ptr_null(v);

	v = dstm_region_version_find(r, 42);
	ck_assert_ptr_null(v);

	/* NULL region */
	v = dstm_region_version_find(NULL, 0);
	ck_assert_ptr_null(v);

	dstm_region_destroy(&r);
}
END_TEST

START_TEST(test_version_find_or_create)
{
	dstm_region_t *r = NULL;
	int err = dstm_region_create(&r, 4096);
	ck_assert_int_eq(err, DSTM_SUCCESS);

	/* Create version 0 */
	dstm_region_version_t *v0 = dstm_region_version_find_or_create(r, 0);
	ck_assert_ptr_nonnull(v0);
	ck_assert_int_eq(v0->time, 0);
	ck_assert_int_eq(v0->status, DSTM_REGION_STATUS_WRITABLE);
	ck_assert_ptr_nonnull(v0->buffer);

	/* Find it again — should return the same pointer */
	dstm_region_version_t *v0b = dstm_region_version_find_or_create(r, 0);
	ck_assert_ptr_eq(v0, v0b);

	/* Also findable via find */
	dstm_region_version_t *v0c = dstm_region_version_find(r, 0);
	ck_assert_ptr_eq(v0, v0c);

	dstm_region_destroy(&r);
}
END_TEST

START_TEST(test_version_buffer_zero_initialised)
{
	/* Newly-created versions must have all-zero buffers so that
	 * applications which only put to specific offsets can rely on
	 * unwritten offsets reading as zero rather than garbage. */
	dstm_region_t *r = NULL;
	const size_t region_size = 8192;
	int err = dstm_region_create(&r, region_size);
	ck_assert_int_eq(err, DSTM_SUCCESS);

	dstm_region_version_t *v = dstm_region_version_find_or_create(r, 0);
	ck_assert_ptr_nonnull(v);
	ck_assert_ptr_nonnull(v->buffer);

	const unsigned char *bytes = (const unsigned char *)v->buffer;
	for (size_t i = 0; i < region_size; i++)
		ck_assert_int_eq(bytes[i], 0);

	/* Same guarantee for additional versions */
	dstm_region_version_t *v2 = dstm_region_version_find_or_create(r, 17);
	ck_assert_ptr_nonnull(v2);
	const unsigned char *bytes2 = (const unsigned char *)v2->buffer;
	for (size_t i = 0; i < region_size; i++)
		ck_assert_int_eq(bytes2[i], 0);

	dstm_region_destroy(&r);
}
END_TEST

START_TEST(test_version_multiple)
{
	dstm_region_t *r = NULL;
	int err = dstm_region_create(&r, 4096);
	ck_assert_int_eq(err, DSTM_SUCCESS);

	/* Create versions out of order */
	dstm_region_version_t *v5 = dstm_region_version_find_or_create(r, 5);
	dstm_region_version_t *v2 = dstm_region_version_find_or_create(r, 2);
	dstm_region_version_t *v8 = dstm_region_version_find_or_create(r, 8);
	dstm_region_version_t *v0 = dstm_region_version_find_or_create(r, 0);

	ck_assert_ptr_nonnull(v0);
	ck_assert_ptr_nonnull(v2);
	ck_assert_ptr_nonnull(v5);
	ck_assert_ptr_nonnull(v8);

	/* Verify sorted order (ascending by time) */
	ck_assert_ptr_eq(r->versions, v0);
	ck_assert_ptr_eq(v0->next, v2);
	ck_assert_ptr_eq(v2->next, v5);
	ck_assert_ptr_eq(v5->next, v8);

	/* Each has its own buffer */
	ck_assert_ptr_ne(v0->buffer, v2->buffer);
	ck_assert_ptr_ne(v2->buffer, v5->buffer);
	ck_assert_ptr_ne(v5->buffer, v8->buffer);

	/* All start WRITABLE */
	ck_assert_int_eq(v0->status, DSTM_REGION_STATUS_WRITABLE);
	ck_assert_int_eq(v2->status, DSTM_REGION_STATUS_WRITABLE);
	ck_assert_int_eq(v5->status, DSTM_REGION_STATUS_WRITABLE);
	ck_assert_int_eq(v8->status, DSTM_REGION_STATUS_WRITABLE);

	/* Find each one */
	ck_assert_ptr_eq(dstm_region_version_find(r, 0), v0);
	ck_assert_ptr_eq(dstm_region_version_find(r, 2), v2);
	ck_assert_ptr_eq(dstm_region_version_find(r, 5), v5);
	ck_assert_ptr_eq(dstm_region_version_find(r, 8), v8);

	/* Non-existent timestamps return NULL */
	ck_assert_ptr_null(dstm_region_version_find(r, 1));
	ck_assert_ptr_null(dstm_region_version_find(r, 3));
	ck_assert_ptr_null(dstm_region_version_find(r, 99));

	dstm_region_destroy(&r);
}
END_TEST

START_TEST(test_version_destroy)
{
	dstm_region_t *r = NULL;
	int err = dstm_region_create(&r, 4096);
	ck_assert_int_eq(err, DSTM_SUCCESS);

	dstm_region_version_t *v0 = dstm_region_version_find_or_create(r, 0);
	dstm_region_version_t *v1 = dstm_region_version_find_or_create(r, 1);
	dstm_region_version_t *v2 = dstm_region_version_find_or_create(r, 2);
	ck_assert_ptr_nonnull(v0);
	ck_assert_ptr_nonnull(v1);
	ck_assert_ptr_nonnull(v2);

	/* Destroy the middle version */
	dstm_region_version_destroy(r, v1);

	/* v1 is gone */
	ck_assert_ptr_null(dstm_region_version_find(r, 1));

	/* v0 and v2 are still there */
	ck_assert_ptr_eq(dstm_region_version_find(r, 0), v0);
	ck_assert_ptr_eq(dstm_region_version_find(r, 2), v2);

	/* List integrity: v0 -> v2 */
	ck_assert_ptr_eq(r->versions, v0);
	ck_assert_ptr_eq(v0->next, v2);

	/* Destroy the head */
	dstm_region_version_destroy(r, v0);
	ck_assert_ptr_eq(r->versions, v2);
	ck_assert_ptr_null(dstm_region_version_find(r, 0));

	/* Destroy the last one */
	dstm_region_version_destroy(r, v2);
	ck_assert_ptr_null(r->versions);

	dstm_region_destroy(&r);
}
END_TEST

START_TEST(test_version_buffer_independence)
{
	dstm_region_t *r = NULL;
	int err = dstm_region_create(&r, 4096);
	ck_assert_int_eq(err, DSTM_SUCCESS);

	dstm_region_version_t *v0 = dstm_region_version_find_or_create(r, 0);
	dstm_region_version_t *v1 = dstm_region_version_find_or_create(r, 1);
	ck_assert_ptr_nonnull(v0);
	ck_assert_ptr_nonnull(v1);

	/* Write different patterns to each version's buffer */
	memset(v0->buffer, 0xAA, r->size);
	memset(v1->buffer, 0xBB, r->size);

	/* Verify they are independent */
	unsigned char *b0 = (unsigned char *)v0->buffer;
	unsigned char *b1 = (unsigned char *)v1->buffer;
	for (size_t i = 0; i < r->size; i++) {
		ck_assert_uint_eq(b0[i], 0xAA);
		ck_assert_uint_eq(b1[i], 0xBB);
	}

	dstm_region_destroy(&r);
}
END_TEST

START_TEST(test_version_status_transitions)
{
	dstm_region_t *r = NULL;
	int err = dstm_region_create(&r, 4096);
	ck_assert_int_eq(err, DSTM_SUCCESS);

	dstm_region_version_t *v = dstm_region_version_find_or_create(r, 0);
	ck_assert_ptr_nonnull(v);

	/* Starts WRITABLE */
	ck_assert_int_eq(v->status, DSTM_REGION_STATUS_WRITABLE);

	/* Manual transition to READABLE */
	v->status = DSTM_REGION_STATUS_READABLE;
	ck_assert_int_eq(v->status, DSTM_REGION_STATUS_READABLE);

	/* Manual transition to DISCARDABLE */
	v->status = DSTM_REGION_STATUS_DISCARDABLE;
	ck_assert_int_eq(v->status, DSTM_REGION_STATUS_DISCARDABLE);

	dstm_region_destroy(&r);
}
END_TEST

/*******************************************************************************
 * Test Suite Setup
 ******************************************************************************/

Suite *region_suite(void)
{
	Suite *s;
	TCase *tc_create;
	TCase *tc_alloc;
	TCase *tc_dealloc;
	TCase *tc_buddy;
	TCase *tc_frag;
	TCase *tc_exhaust;
	TCase *tc_stress;
	TCase *tc_version;

	s = suite_create("region");

	/* Create/Destroy tests */
	tc_create = tcase_create("create_destroy");
	tcase_add_test(tc_create, test_region_create_null);
	tcase_add_test(tc_create, test_region_create_destroy);
	tcase_add_test(tc_create, test_region_create_various_sizes);
	tcase_add_test(tc_create, test_region_create_with_buffer);
	tcase_add_test(tc_create, test_region_create_with_buffer_invalid_size);
	tcase_add_test(tc_create, test_region_create_with_buffer_null);
	suite_add_tcase(s, tc_create);

	/* Basic allocation tests */
	tc_alloc = tcase_create("allocation");
	tcase_add_test(tc_alloc, test_region_malloc_null_region);
	tcase_add_test(tc_alloc, test_region_malloc_zero_size);
	tcase_add_test(tc_alloc, test_region_malloc_simple);
	tcase_add_test(tc_alloc, test_region_malloc_multiple);
	tcase_add_test(tc_alloc, test_region_malloc_write_read);
	suite_add_tcase(s, tc_alloc);

	/* Basic deallocation tests */
	tc_dealloc = tcase_create("deallocation");
	tcase_add_test(tc_dealloc, test_region_free_null_region);
	tcase_add_test(tc_dealloc, test_region_free_null_ptr);
	tcase_add_test(tc_dealloc, test_region_free_simple);
	tcase_add_test(tc_dealloc, test_region_free_multiple);
	suite_add_tcase(s, tc_dealloc);

	/* Buddy allocator tests */
	tc_buddy = tcase_create("buddy_allocator");
	tcase_add_test(tc_buddy, test_region_malloc_exact_sizes);
	tcase_add_test(tc_buddy, test_region_malloc_round_up);
	tcase_add_test(tc_buddy, test_region_malloc_max_size);
	tcase_add_test(tc_buddy, test_region_malloc_too_large);
	tcase_add_test(tc_buddy, test_region_coalesce_simple);
	tcase_add_test(tc_buddy, test_region_coalesce_complex);
	tcase_add_test(tc_buddy, test_region_minimum_block);
	tcase_add_test(tc_buddy, test_region_below_minimum);
	suite_add_tcase(s, tc_buddy);

	/* Fragmentation tests */
	tc_frag = tcase_create("fragmentation");
	tcase_add_test(tc_frag, test_region_reuse_freed_blocks);
	tcase_add_test(tc_frag, test_region_fragmentation_interleaved);
	tcase_add_test(tc_frag, test_region_fragmentation_pattern);
	suite_add_tcase(s, tc_frag);

	/* Exhaustion tests */
	tc_exhaust = tcase_create("exhaustion");
	tcase_add_test(tc_exhaust, test_region_exhaustion);
	tcase_add_test(tc_exhaust, test_region_exhaustion_recovery);
	suite_add_tcase(s, tc_exhaust);

	/* Stress tests */
	tc_stress = tcase_create("stress");
	tcase_add_test(tc_stress, test_region_stress_mixed_sizes);
	tcase_add_test(tc_stress, test_region_stress_many_small);
	suite_add_tcase(s, tc_stress);

	/* Version management tests */
	tc_version = tcase_create("version");
	tcase_add_test(tc_version, test_version_find_nonexistent);
	tcase_add_test(tc_version, test_version_find_or_create);
	tcase_add_test(tc_version, test_version_buffer_zero_initialised);
	tcase_add_test(tc_version, test_version_multiple);
	tcase_add_test(tc_version, test_version_destroy);
	tcase_add_test(tc_version, test_version_buffer_independence);
	tcase_add_test(tc_version, test_version_status_transitions);
	suite_add_tcase(s, tc_version);

	return s;
}

int main(void)
{
	int failed;
	Suite *s;
	SRunner *sr;

	s = region_suite();
	sr = srunner_create(s);

	srunner_run_all(sr, CK_VERBOSE);
	failed = srunner_ntests_failed(sr);
	srunner_free(sr);
	return (failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
