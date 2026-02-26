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

#include "dstm.h"
#include "internal/distance.h"

/*******************************************************************************
 * Helper Functions
 ******************************************************************************/

/* Verify distance matrix is symmetric */
static int is_symmetric(dstm_distance_t *dist)
{
	for (size_t i = 0; i < dist->size; i++)
		for (size_t j = 0; j < dist->size; j++) {
			int d_ij = dstm_distance_get(dist, i, j);
			int d_ji = dstm_distance_get(dist, j, i);
			if (d_ij != d_ji)
				return 0;
		}
	return 1;
}

/* Verify diagonal is all zeros */
static int diagonal_is_zero(dstm_distance_t *dist)
{
	for (size_t i = 0; i < dist->size; i++)
		if (dstm_distance_get(dist, i, i) != 0)
			return 0;
	return 1;
}

/* Verify all distances are non-negative */
static int all_nonnegative(dstm_distance_t *dist)
{
	for (size_t i = 0; i < dist->size; i++)
		for (size_t j = 0; j < dist->size; j++)
			if (dstm_distance_get(dist, i, j) < 0)
				return 0;
	return 1;
}

/*******************************************************************************
 * Ring Topology Tests
 ******************************************************************************/

START_TEST(test_ring_null_pointer)
{
	int rc = dstm_distance_create_ring(NULL, 4);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_int_eq(dstm_errno, DSTM_EINVAL);
}
END_TEST

START_TEST(test_ring_invalid_size_too_small)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_ring(&dist, 2);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_ptr_null(dist);
	ck_assert_int_eq(dstm_errno, DSTM_EINVAL);
}
END_TEST

START_TEST(test_ring_invalid_size_too_large)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_ring(&dist, DSTM_DISTANCE_MAX_SIZE + 1);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_ptr_null(dist);
	ck_assert_int_eq(dstm_errno, DSTM_EINVAL);
}
END_TEST

START_TEST(test_ring_create_destroy)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_ring(&dist, 4);
	ck_assert_int_eq(rc, DSTM_SUCCESS);
	ck_assert_ptr_nonnull(dist);
	ck_assert_ptr_nonnull(dist->distances);
	ck_assert_uint_eq(dist->size, 4);

	dstm_distance_destroy(&dist);
	ck_assert_ptr_null(dist);
}
END_TEST

START_TEST(test_ring_destroy_null)
{
	/* Should not crash with NULL */
	dstm_distance_destroy(NULL);

	dstm_distance_t *dist = NULL;
	dstm_distance_destroy(&dist);
}
END_TEST

START_TEST(test_ring_properties)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_ring(&dist, 8);
	ck_assert_int_eq(rc, DSTM_SUCCESS);

	/* Check symmetric */
	ck_assert(is_symmetric(dist));

	/* Check diagonal is zero */
	ck_assert(diagonal_is_zero(dist));

	/* Check all non-negative */
	ck_assert(all_nonnegative(dist));

	/* Check neighbors are distance 1 */
	ck_assert_int_eq(dstm_distance_get(dist, 0, 1), 1);
	ck_assert_int_eq(dstm_distance_get(dist, 0, 7), 1);
	ck_assert_int_eq(dstm_distance_get(dist, 3, 4), 1);
	ck_assert_int_eq(dstm_distance_get(dist, 7, 0), 1);

	/* Check opposite nodes in ring of 8 (diameter = 4) */
	ck_assert_int_eq(dstm_distance_get(dist, 0, 4), 4);
	ck_assert_int_eq(dstm_distance_get(dist, 1, 5), 4);

	/* Check intermediate distances */
	ck_assert_int_eq(dstm_distance_get(dist, 0, 2), 2);
	ck_assert_int_eq(dstm_distance_get(dist, 0, 3), 3);

	dstm_distance_destroy(&dist);
}
END_TEST

START_TEST(test_ring_small_size)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_ring(&dist, 3);
	ck_assert_int_eq(rc, DSTM_SUCCESS);
	ck_assert_uint_eq(dist->size, 3);

	/* In a ring of 3, max distance is 1 (triangle) */
	ck_assert_int_eq(dstm_distance_get(dist, 0, 1), 1);
	ck_assert_int_eq(dstm_distance_get(dist, 1, 2), 1);
	ck_assert_int_eq(dstm_distance_get(dist, 2, 0), 1);

	dstm_distance_destroy(&dist);
}
END_TEST

/*******************************************************************************
 * Complete Graph Tests
 ******************************************************************************/

START_TEST(test_complete_null_pointer)
{
	int rc = dstm_distance_create_complete(NULL, 5);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_int_eq(dstm_errno, DSTM_EINVAL);
}
END_TEST

START_TEST(test_complete_invalid_size_zero)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_complete(&dist, 0);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_ptr_null(dist);
}
END_TEST

START_TEST(test_complete_invalid_size_too_large)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_complete(&dist,
	                                       DSTM_DISTANCE_MAX_SIZE + 1);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_ptr_null(dist);
}
END_TEST

START_TEST(test_complete_create_destroy)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_complete(&dist, 5);
	ck_assert_int_eq(rc, DSTM_SUCCESS);
	ck_assert_ptr_nonnull(dist);
	ck_assert_ptr_nonnull(dist->distances);
	ck_assert_uint_eq(dist->size, 5);

	dstm_distance_destroy(&dist);
	ck_assert_ptr_null(dist);
}
END_TEST

START_TEST(test_complete_all_distance_one)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_complete(&dist, 5);
	ck_assert_int_eq(rc, DSTM_SUCCESS);

	/* Check symmetric */
	ck_assert(is_symmetric(dist));

	/* Check diagonal is zero */
	ck_assert(diagonal_is_zero(dist));

	/* All non-diagonal entries should be 1 */
	for (size_t i = 0; i < dist->size; i++)
		for (size_t j = 0; j < dist->size; j++)
			if (i != j)
				ck_assert_int_eq(dstm_distance_get(dist, i, j),
				                 1);

	dstm_distance_destroy(&dist);
}
END_TEST

START_TEST(test_complete_single_node)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_complete(&dist, 1);
	ck_assert_int_eq(rc, DSTM_SUCCESS);
	ck_assert_uint_eq(dist->size, 1);

	/* Single node has distance 0 to itself */
	ck_assert_int_eq(dstm_distance_get(dist, 0, 0), 0);

	dstm_distance_destroy(&dist);
}
END_TEST

/*******************************************************************************
 * Star Topology Tests
 ******************************************************************************/

START_TEST(test_star_null_pointer)
{
	int rc = dstm_distance_create_star(NULL, 6);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_int_eq(dstm_errno, DSTM_EINVAL);
}
END_TEST

START_TEST(test_star_invalid_size_too_small)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_star(&dist, 1);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_ptr_null(dist);
}
END_TEST

START_TEST(test_star_invalid_size_too_large)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_star(&dist, DSTM_DISTANCE_MAX_SIZE + 1);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_ptr_null(dist);
}
END_TEST

START_TEST(test_star_create_destroy)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_star(&dist, 6);
	ck_assert_int_eq(rc, DSTM_SUCCESS);
	ck_assert_ptr_nonnull(dist);
	ck_assert_ptr_nonnull(dist->distances);
	ck_assert_uint_eq(dist->size, 6);

	dstm_distance_destroy(&dist);
	ck_assert_ptr_null(dist);
}
END_TEST

START_TEST(test_star_center_distances)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_star(&dist, 6);
	ck_assert_int_eq(rc, DSTM_SUCCESS);

	/* Check symmetric */
	ck_assert(is_symmetric(dist));

	/* Check diagonal is zero */
	ck_assert(diagonal_is_zero(dist));

	/* Center (node 0) to all others: distance 1 */
	for (size_t i = 1; i < dist->size; i++)
		ck_assert_int_eq(dstm_distance_get(dist, 0, i), 1);

	/* Non-center to non-center: distance 2 (via center) */
	ck_assert_int_eq(dstm_distance_get(dist, 1, 2), 2);
	ck_assert_int_eq(dstm_distance_get(dist, 3, 5), 2);
	ck_assert_int_eq(dstm_distance_get(dist, 2, 4), 2);

	dstm_distance_destroy(&dist);
}
END_TEST

START_TEST(test_star_minimal_size)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_star(&dist, 2);
	ck_assert_int_eq(rc, DSTM_SUCCESS);
	ck_assert_uint_eq(dist->size, 2);

	/* With only 2 nodes, it's just a direct connection */
	ck_assert_int_eq(dstm_distance_get(dist, 0, 1), 1);
	ck_assert_int_eq(dstm_distance_get(dist, 1, 0), 1);

	dstm_distance_destroy(&dist);
}
END_TEST

/*******************************************************************************
 * Line Topology Tests
 ******************************************************************************/

START_TEST(test_line_null_pointer)
{
	int rc = dstm_distance_create_line(NULL, 5);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_int_eq(dstm_errno, DSTM_EINVAL);
}
END_TEST

START_TEST(test_line_invalid_size_zero)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_line(&dist, 0);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_ptr_null(dist);
}
END_TEST

START_TEST(test_line_invalid_size_too_large)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_line(&dist, DSTM_DISTANCE_MAX_SIZE + 1);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_ptr_null(dist);
}
END_TEST

START_TEST(test_line_create_destroy)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_line(&dist, 5);
	ck_assert_int_eq(rc, DSTM_SUCCESS);
	ck_assert_ptr_nonnull(dist);
	ck_assert_ptr_nonnull(dist->distances);
	ck_assert_uint_eq(dist->size, 5);

	dstm_distance_destroy(&dist);
	ck_assert_ptr_null(dist);
}
END_TEST

START_TEST(test_line_distances)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_line(&dist, 5);
	ck_assert_int_eq(rc, DSTM_SUCCESS);

	/* Check symmetric */
	ck_assert(is_symmetric(dist));

	/* Check diagonal is zero */
	ck_assert(diagonal_is_zero(dist));

	/* Check all non-negative */
	ck_assert(all_nonnegative(dist));

	/* Distance should be |i - j| */
	ck_assert_int_eq(dstm_distance_get(dist, 0, 1), 1);
	ck_assert_int_eq(dstm_distance_get(dist, 0, 2), 2);
	ck_assert_int_eq(dstm_distance_get(dist, 0, 4), 4);
	ck_assert_int_eq(dstm_distance_get(dist, 1, 3), 2);
	ck_assert_int_eq(dstm_distance_get(dist, 2, 4), 2);

	dstm_distance_destroy(&dist);
}
END_TEST

START_TEST(test_line_single_node)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_line(&dist, 1);
	ck_assert_int_eq(rc, DSTM_SUCCESS);
	ck_assert_uint_eq(dist->size, 1);

	/* Single node has distance 0 to itself */
	ck_assert_int_eq(dstm_distance_get(dist, 0, 0), 0);

	dstm_distance_destroy(&dist);
}
END_TEST

START_TEST(test_line_two_nodes)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_line(&dist, 2);
	ck_assert_int_eq(rc, DSTM_SUCCESS);
	ck_assert_uint_eq(dist->size, 2);

	/* Two nodes, distance 1 apart */
	ck_assert_int_eq(dstm_distance_get(dist, 0, 1), 1);
	ck_assert_int_eq(dstm_distance_get(dist, 1, 0), 1);

	dstm_distance_destroy(&dist);
}
END_TEST

/*******************************************************************************
 * Test Suite Setup
 ******************************************************************************/

Suite *distances_suite(void)
{
	Suite *s;
	TCase *tc_ring, *tc_complete, *tc_star, *tc_line;

	s = suite_create("distances");

	/* Ring tests */
	tc_ring = tcase_create("ring");
	tcase_add_test(tc_ring, test_ring_null_pointer);
	tcase_add_test(tc_ring, test_ring_invalid_size_too_small);
	tcase_add_test(tc_ring, test_ring_invalid_size_too_large);
	tcase_add_test(tc_ring, test_ring_create_destroy);
	tcase_add_test(tc_ring, test_ring_destroy_null);
	tcase_add_test(tc_ring, test_ring_properties);
	tcase_add_test(tc_ring, test_ring_small_size);
	suite_add_tcase(s, tc_ring);

	/* Complete graph tests */
	tc_complete = tcase_create("complete");
	tcase_add_test(tc_complete, test_complete_null_pointer);
	tcase_add_test(tc_complete, test_complete_invalid_size_zero);
	tcase_add_test(tc_complete, test_complete_invalid_size_too_large);
	tcase_add_test(tc_complete, test_complete_create_destroy);
	tcase_add_test(tc_complete, test_complete_all_distance_one);
	tcase_add_test(tc_complete, test_complete_single_node);
	suite_add_tcase(s, tc_complete);

	/* Star tests */
	tc_star = tcase_create("star");
	tcase_add_test(tc_star, test_star_null_pointer);
	tcase_add_test(tc_star, test_star_invalid_size_too_small);
	tcase_add_test(tc_star, test_star_invalid_size_too_large);
	tcase_add_test(tc_star, test_star_create_destroy);
	tcase_add_test(tc_star, test_star_center_distances);
	tcase_add_test(tc_star, test_star_minimal_size);
	suite_add_tcase(s, tc_star);

	/* Line tests */
	tc_line = tcase_create("line");
	tcase_add_test(tc_line, test_line_null_pointer);
	tcase_add_test(tc_line, test_line_invalid_size_zero);
	tcase_add_test(tc_line, test_line_invalid_size_too_large);
	tcase_add_test(tc_line, test_line_create_destroy);
	tcase_add_test(tc_line, test_line_distances);
	tcase_add_test(tc_line, test_line_single_node);
	tcase_add_test(tc_line, test_line_two_nodes);
	suite_add_tcase(s, tc_line);

	return s;
}

int main(void)
{
	int failed;
	Suite *s;
	SRunner *sr;

	s = distances_suite();
	sr = srunner_create(s);

	srunner_run_all(sr, CK_VERBOSE);
	failed = srunner_ntests_failed(sr);
	srunner_free(sr);
	return (failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
