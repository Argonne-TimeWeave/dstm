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
	void *ptr = dstm_region_malloc(region, 128);
	ck_assert_ptr_nonnull(ptr);

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
	void *ptr = dstm_region_malloc(NULL, 128);
	ck_assert_ptr_null(ptr);
}
END_TEST

START_TEST(test_region_malloc_zero_size)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 4096);

	void *ptr = dstm_region_malloc(region, 0);
	ck_assert_ptr_null(ptr);

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_malloc_simple)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 4096);

	void *ptr = dstm_region_malloc(region, 128);
	ck_assert_ptr_nonnull(ptr);
	ck_assert(verify_memory_accessible(ptr, 128));

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_malloc_multiple)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 4096);

	void *ptrs[10];
	for (int i = 0; i < 10; i++) {
		ptrs[i] = dstm_region_malloc(region, 64);
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

	int *int_ptr = dstm_region_malloc(region, sizeof(int));
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
	dstm_region_free(NULL, (void *)0x1000);
	/* Should not crash */
}
END_TEST

START_TEST(test_region_free_null_ptr)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 4096);

	dstm_region_free(region, NULL);
	/* Should not crash */

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_free_simple)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 4096);

	void *ptr = dstm_region_malloc(region, 256);
	ck_assert_ptr_nonnull(ptr);

	dstm_region_free(region, ptr);
	/* Should not crash */

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_free_multiple)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 4096);

	void *ptrs[5];
	for (int i = 0; i < 5; i++) {
		ptrs[i] = dstm_region_malloc(region, 128);
		ck_assert_ptr_nonnull(ptrs[i]);
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
	dstm_region_create(&region, 65536);

	/* Test power-of-2 sizes from min to large */
	size_t sizes[] = {64, 128, 256, 512, 1024, 2048, 4096};

	for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
		void *ptr = dstm_region_malloc(region, sizes[i]);
		ck_assert_ptr_nonnull(ptr);
		ck_assert(verify_memory_accessible(ptr, sizes[i]));
		dstm_region_free(region, ptr);
	}

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_malloc_round_up)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 4096);

	/* Non-power-of-2 sizes should round up */
	void *ptr1 = dstm_region_malloc(region, 100); /* Should get 128 */
	ck_assert_ptr_nonnull(ptr1);
	ck_assert(verify_memory_accessible(ptr1, 100));

	void *ptr2 = dstm_region_malloc(region, 200); /* Should get 256 */
	ck_assert_ptr_nonnull(ptr2);
	ck_assert(verify_memory_accessible(ptr2, 200));

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_malloc_max_size)
{
	size_t region_size = 4096;
	dstm_region_t *region = NULL;
	dstm_region_create(&region, region_size);

	/* Should be able to allocate entire region */
	void *ptr = dstm_region_malloc(region, region_size);
	ck_assert_ptr_nonnull(ptr);

	/* Second allocation should fail */
	void *ptr2 = dstm_region_malloc(region, 64);
	ck_assert_ptr_null(ptr2);

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_malloc_too_large)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 4096);

	/* Try to allocate more than region size */
	void *ptr = dstm_region_malloc(region, 8192);
	ck_assert_ptr_null(ptr);

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_coalesce_simple)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 4096);

	/* Allocate two buddy blocks */
	void *ptr1 = dstm_region_malloc(region, 1024);
	void *ptr2 = dstm_region_malloc(region, 1024);
	ck_assert_ptr_nonnull(ptr1);
	ck_assert_ptr_nonnull(ptr2);

	/* Free both - they should coalesce */
	dstm_region_free(region, ptr1);
	dstm_region_free(region, ptr2);

	/* Should now be able to allocate 2048 (or larger) */
	void *ptr3 = dstm_region_malloc(region, 2048);
	ck_assert_ptr_nonnull(ptr3);

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_coalesce_complex)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 8192);

	/* Allocate 4 blocks of 512 each */
	void *ptrs[4];
	for (int i = 0; i < 4; i++) {
		ptrs[i] = dstm_region_malloc(region, 512);
		ck_assert_ptr_nonnull(ptrs[i]);
	}

	/* Free all - should coalesce back up */
	for (int i = 0; i < 4; i++) {
		dstm_region_free(region, ptrs[i]);
	}

	/* Should be able to allocate larger block now */
	void *large = dstm_region_malloc(region, 2048);
	ck_assert_ptr_nonnull(large);

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_minimum_block)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 4096);

	/* Allocate minimum block size */
	void *ptr = dstm_region_malloc(region, DSTM_REGION_MIN_BLOCK_SIZE);
	ck_assert_ptr_nonnull(ptr);
	ck_assert(verify_memory_accessible(ptr, DSTM_REGION_MIN_BLOCK_SIZE));

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_below_minimum)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 4096);

	/* Allocate less than minimum - should round up to minimum */
	void *ptr = dstm_region_malloc(region, 1);
	ck_assert_ptr_nonnull(ptr);
	ck_assert(verify_memory_accessible(ptr, 1));

	dstm_region_destroy(&region);
}
END_TEST

/*******************************************************************************
 * Fragmentation Tests
 ******************************************************************************/

START_TEST(test_region_reuse_freed_blocks)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 4096);

	/* Allocate and free */
	void *ptr1 = dstm_region_malloc(region, 256);
	ck_assert_ptr_nonnull(ptr1);
	dstm_region_free(region, ptr1);

	/* Allocate again - should reuse freed block */
	void *ptr2 = dstm_region_malloc(region, 256);
	ck_assert_ptr_nonnull(ptr2);
	ck_assert_ptr_eq(ptr1, ptr2); /* Should be same address */

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_fragmentation_interleaved)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 8192);

	void *ptrs[8];

	/* Allocate 8 blocks */
	for (int i = 0; i < 8; i++) {
		ptrs[i] = dstm_region_malloc(region, 512);
		ck_assert_ptr_nonnull(ptrs[i]);
	}

	/* Free every other block */
	for (int i = 0; i < 8; i += 2) {
		dstm_region_free(region, ptrs[i]);
	}

	/* Should be able to allocate 4 more blocks of same size */
	for (int i = 0; i < 4; i++) {
		void *ptr = dstm_region_malloc(region, 512);
		ck_assert_ptr_nonnull(ptr);
	}

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_fragmentation_pattern)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 4096);

	/* Create a specific fragmentation pattern */
	void *p1 = dstm_region_malloc(region, 512);
	void *p2 = dstm_region_malloc(region, 512);
	void *p3 = dstm_region_malloc(region, 512);

	ck_assert_ptr_nonnull(p1);
	ck_assert_ptr_nonnull(p2);
	ck_assert_ptr_nonnull(p3);

	/* Free middle block */
	dstm_region_free(region, p2);

	/* Allocate smaller block - should fit in freed space */
	void *p4 = dstm_region_malloc(region, 256);
	ck_assert_ptr_nonnull(p4);

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

	void *ptrs[64];
	int count = 0;

	/* Allocate until exhausted */
	for (int i = 0; i < 64; i++) {
		ptrs[i] = dstm_region_malloc(region, 64);
		if (ptrs[i] != NULL)
			count++;
		else
			break;
	}

	ck_assert_int_eq(count, 64); /* 4096 / 64 = 64 blocks */

	/* Next allocation should fail */
	void *ptr = dstm_region_malloc(region, 64);
	ck_assert_ptr_null(ptr);

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_exhaustion_recovery)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 4096);

	/* Fill region */
	void *ptr1 = dstm_region_malloc(region, 4096);
	ck_assert_ptr_nonnull(ptr1);

	/* Should fail */
	void *ptr2 = dstm_region_malloc(region, 64);
	ck_assert_ptr_null(ptr2);

	/* Free and try again */
	dstm_region_free(region, ptr1);
	ptr2 = dstm_region_malloc(region, 64);
	ck_assert_ptr_nonnull(ptr2);

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

	void *ptrs[100];
	size_t sizes[] = {64, 128, 256, 512, 1024};
	int allocated = 0;

	/* Allocate mix of sizes */
	for (int i = 0; i < 100; i++) {
		size_t size = sizes[i % 5];
		ptrs[i] = dstm_region_malloc(region, size);
		if (ptrs[i] != NULL) {
			ck_assert(verify_memory_accessible(ptrs[i], size));
			allocated++;
		}
	}

	ck_assert_int_gt(allocated, 0);

	/* Free half */
	for (int i = 0; i < allocated; i += 2) {
		if (ptrs[i] != NULL)
			dstm_region_free(region, ptrs[i]);
	}

	/* Allocate more */
	for (int i = 0; i < 20; i++) {
		void *ptr = dstm_region_malloc(region, 128);
		/* May succeed or fail depending on fragmentation */
		(void)ptr;
	}

	dstm_region_destroy(&region);
}
END_TEST

START_TEST(test_region_stress_many_small)
{
	dstm_region_t *region = NULL;
	dstm_region_create(&region, 16384);

	/* Allocate many small blocks */
	void *ptrs[200];
	int count = 0;
	for (int i = 0; i < 200; i++) {
		ptrs[i] = dstm_region_malloc(region, 64);
		if (ptrs[i] != NULL)
			count++;
	}

	/* Should allocate at least some blocks */
	ck_assert_int_gt(count, 0);

	dstm_region_destroy(&region);
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
