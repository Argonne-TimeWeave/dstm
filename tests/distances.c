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
 * Mesh Topology Tests
 ******************************************************************************/

START_TEST(test_mesh_null_pointer)
{
	int rc = dstm_distance_create_mesh(NULL, 9);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_int_eq(dstm_errno, DSTM_EINVAL);
}
END_TEST

START_TEST(test_mesh_invalid_size_zero)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_mesh(&dist, 0);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_ptr_null(dist);
}
END_TEST

START_TEST(test_mesh_invalid_size_too_large)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_mesh(&dist, DSTM_DISTANCE_MAX_SIZE + 1);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_ptr_null(dist);
}
END_TEST

START_TEST(test_mesh_create_destroy)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_mesh(&dist, 9);
	ck_assert_int_eq(rc, DSTM_SUCCESS);
	ck_assert_ptr_nonnull(dist);
	ck_assert_ptr_nonnull(dist->distances);
	ck_assert_uint_eq(dist->size, 9);

	dstm_distance_destroy(&dist);
	ck_assert_ptr_null(dist);
}
END_TEST

START_TEST(test_mesh_properties)
{
	/* 3x4 grid: 12 nodes */
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_mesh(&dist, 12);
	ck_assert_int_eq(rc, DSTM_SUCCESS);

	ck_assert(is_symmetric(dist));
	ck_assert(diagonal_is_zero(dist));
	ck_assert(all_nonnegative(dist));

	/* 3x4 grid, cols=4: node i at (i/4, i%4)
	 * node 0=(0,0), node 3=(0,3), node 8=(2,0), node 11=(2,3)
	 * dist(0,3) = |0-0|+|0-3| = 3
	 * dist(0,8) = |0-2|+|0-0| = 2
	 * dist(0,11)= |0-2|+|0-3| = 5
	 * dist(1,6) = |0-1|+|1-2| = 2
	 */
	ck_assert_int_eq(dstm_distance_get(dist, 0, 1), 1);
	ck_assert_int_eq(dstm_distance_get(dist, 0, 4), 1);
	ck_assert_int_eq(dstm_distance_get(dist, 0, 3), 3);
	ck_assert_int_eq(dstm_distance_get(dist, 0, 8), 2);
	ck_assert_int_eq(dstm_distance_get(dist, 0, 11), 5);
	ck_assert_int_eq(dstm_distance_get(dist, 1, 6), 2);

	dstm_distance_destroy(&dist);
}
END_TEST

START_TEST(test_mesh_square)
{
	/* 4x4 square grid */
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_mesh(&dist, 16);
	ck_assert_int_eq(rc, DSTM_SUCCESS);

	ck_assert(is_symmetric(dist));
	ck_assert(diagonal_is_zero(dist));

	/* corners: 0=(0,0) and 15=(3,3), dist=6 */
	ck_assert_int_eq(dstm_distance_get(dist, 0, 15), 6);
	/* adjacent */
	ck_assert_int_eq(dstm_distance_get(dist, 5, 6), 1);
	ck_assert_int_eq(dstm_distance_get(dist, 5, 9), 1);

	dstm_distance_destroy(&dist);
}
END_TEST

/*******************************************************************************
 * Torus Topology Tests
 ******************************************************************************/

START_TEST(test_torus_null_pointer)
{
	int rc = dstm_distance_create_torus(NULL, 9);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_int_eq(dstm_errno, DSTM_EINVAL);
}
END_TEST

START_TEST(test_torus_invalid_size_too_small)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_torus(&dist, 3);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_ptr_null(dist);
}
END_TEST

START_TEST(test_torus_invalid_size_too_large)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_torus(&dist, DSTM_DISTANCE_MAX_SIZE + 1);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_ptr_null(dist);
}
END_TEST

START_TEST(test_torus_create_destroy)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_torus(&dist, 9);
	ck_assert_int_eq(rc, DSTM_SUCCESS);
	ck_assert_ptr_nonnull(dist);
	ck_assert_ptr_nonnull(dist->distances);
	ck_assert_uint_eq(dist->size, 9);

	dstm_distance_destroy(&dist);
	ck_assert_ptr_null(dist);
}
END_TEST

START_TEST(test_torus_shorter_than_mesh)
{
	/* On a torus, wraparound paths make some distances shorter than mesh */
	dstm_distance_t *mesh = NULL, *torus = NULL;
	int rc;

	rc = dstm_distance_create_mesh(&mesh, 16);
	ck_assert_int_eq(rc, DSTM_SUCCESS);

	rc = dstm_distance_create_torus(&torus, 16);
	ck_assert_int_eq(rc, DSTM_SUCCESS);

	ck_assert(is_symmetric(torus));
	ck_assert(diagonal_is_zero(torus));
	ck_assert(all_nonnegative(torus));

	/* Corner-to-corner on 4x4: mesh=6, torus=2 (wrap 1 in each dim) */
	ck_assert_int_eq(dstm_distance_get(mesh, 0, 15), 6);
	ck_assert_int_eq(dstm_distance_get(torus, 0, 15), 2);

	/* Direct neighbors unchanged */
	ck_assert_int_eq(dstm_distance_get(torus, 0, 1), 1);
	ck_assert_int_eq(dstm_distance_get(torus, 0, 4), 1);

	/* Wraparound neighbors: node 0 and node 3 (right edge wraps, dist=1) */
	ck_assert_int_eq(dstm_distance_get(torus, 0, 3), 1);
	/* Wraparound: node 0 and node 12 (bottom edge wraps, dist=1) */
	ck_assert_int_eq(dstm_distance_get(torus, 0, 12), 1);

	dstm_distance_destroy(&mesh);
	dstm_distance_destroy(&torus);
}
END_TEST

/*******************************************************************************
 * Tree Topology Tests
 ******************************************************************************/

START_TEST(test_tree_null_pointer)
{
	int rc = dstm_distance_create_tree(NULL, 7);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_int_eq(dstm_errno, DSTM_EINVAL);
}
END_TEST

START_TEST(test_tree_invalid_size_zero)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_tree(&dist, 0);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_ptr_null(dist);
}
END_TEST

START_TEST(test_tree_invalid_size_too_large)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_tree(&dist, DSTM_DISTANCE_MAX_SIZE + 1);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_ptr_null(dist);
}
END_TEST

START_TEST(test_tree_create_destroy)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_tree(&dist, 7);
	ck_assert_int_eq(rc, DSTM_SUCCESS);
	ck_assert_ptr_nonnull(dist);
	ck_assert_ptr_nonnull(dist->distances);
	ck_assert_uint_eq(dist->size, 7);

	dstm_distance_destroy(&dist);
	ck_assert_ptr_null(dist);
}
END_TEST

START_TEST(test_tree_properties)
{
	/* 7-node complete binary tree:
	 *         0
	 *        / \
	 *       1   2
	 *      / \ / \
	 *     3 4 5  6
	 */
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_tree(&dist, 7);
	ck_assert_int_eq(rc, DSTM_SUCCESS);

	ck_assert(is_symmetric(dist));
	ck_assert(diagonal_is_zero(dist));
	ck_assert(all_nonnegative(dist));

	/* root to children: 1 hop */
	ck_assert_int_eq(dstm_distance_get(dist, 0, 1), 1);
	ck_assert_int_eq(dstm_distance_get(dist, 0, 2), 1);

	/* root to leaves: 2 hops */
	ck_assert_int_eq(dstm_distance_get(dist, 0, 3), 2);
	ck_assert_int_eq(dstm_distance_get(dist, 0, 6), 2);

	/* sibling leaves: 2 hops (up to parent, down to sibling) */
	ck_assert_int_eq(dstm_distance_get(dist, 3, 4), 2);
	ck_assert_int_eq(dstm_distance_get(dist, 5, 6), 2);

	/* leaves in different subtrees: 4 hops (up 2, down 2) */
	ck_assert_int_eq(dstm_distance_get(dist, 3, 5), 4);
	ck_assert_int_eq(dstm_distance_get(dist, 4, 6), 4);

	dstm_distance_destroy(&dist);
}
END_TEST

START_TEST(test_tree_single_node)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_tree(&dist, 1);
	ck_assert_int_eq(rc, DSTM_SUCCESS);
	ck_assert_int_eq(dstm_distance_get(dist, 0, 0), 0);
	dstm_distance_destroy(&dist);
}
END_TEST

/*******************************************************************************
 * Hypercube Topology Tests
 ******************************************************************************/

START_TEST(test_hypercube_null_pointer)
{
	int rc = dstm_distance_create_hypercube(NULL, 8);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_int_eq(dstm_errno, DSTM_EINVAL);
}
END_TEST

START_TEST(test_hypercube_invalid_size_one)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_hypercube(&dist, 1);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_ptr_null(dist);
}
END_TEST

START_TEST(test_hypercube_invalid_size_not_power_of_two)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_hypercube(&dist, 6);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_ptr_null(dist);
	rc = dstm_distance_create_hypercube(&dist, 12);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_ptr_null(dist);
}
END_TEST

START_TEST(test_hypercube_invalid_size_too_large)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_hypercube(&dist,
	                                        DSTM_DISTANCE_MAX_SIZE + 1);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_ptr_null(dist);
}
END_TEST

START_TEST(test_hypercube_create_destroy)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_hypercube(&dist, 8);
	ck_assert_int_eq(rc, DSTM_SUCCESS);
	ck_assert_ptr_nonnull(dist);
	ck_assert_ptr_nonnull(dist->distances);
	ck_assert_uint_eq(dist->size, 8);

	dstm_distance_destroy(&dist);
	ck_assert_ptr_null(dist);
}
END_TEST

START_TEST(test_hypercube_properties)
{
	/* 3D hypercube: 8 nodes labeled 0..7 (binary 000..111) */
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_hypercube(&dist, 8);
	ck_assert_int_eq(rc, DSTM_SUCCESS);

	ck_assert(is_symmetric(dist));
	ck_assert(diagonal_is_zero(dist));
	ck_assert(all_nonnegative(dist));

	/* neighbors differ by 1 bit */
	ck_assert_int_eq(dstm_distance_get(dist, 0, 1), 1); /* 000 vs 001 */
	ck_assert_int_eq(dstm_distance_get(dist, 0, 2), 1); /* 000 vs 010 */
	ck_assert_int_eq(dstm_distance_get(dist, 0, 4), 1); /* 000 vs 100 */

	/* 2 bit differences */
	ck_assert_int_eq(dstm_distance_get(dist, 0, 3), 2); /* 000 vs 011 */
	ck_assert_int_eq(dstm_distance_get(dist, 0, 5), 2); /* 000 vs 101 */
	ck_assert_int_eq(dstm_distance_get(dist, 0, 6), 2); /* 000 vs 110 */

	/* antipodal: 3 bits differ */
	ck_assert_int_eq(dstm_distance_get(dist, 0, 7), 3); /* 000 vs 111 */
	ck_assert_int_eq(dstm_distance_get(dist, 1, 6), 3); /* 001 vs 110 */

	dstm_distance_destroy(&dist);
}
END_TEST

START_TEST(test_hypercube_2d)
{
	/* 2D hypercube (square): 4 nodes */
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_hypercube(&dist, 4);
	ck_assert_int_eq(rc, DSTM_SUCCESS);
	ck_assert_uint_eq(dist->size, 4);

	/* 00 vs 01: 1, 00 vs 10: 1, 00 vs 11: 2 */
	ck_assert_int_eq(dstm_distance_get(dist, 0, 1), 1);
	ck_assert_int_eq(dstm_distance_get(dist, 0, 2), 1);
	ck_assert_int_eq(dstm_distance_get(dist, 0, 3), 2);

	dstm_distance_destroy(&dist);
}
END_TEST

/*******************************************************************************
 * Dragonfly Topology Tests
 ******************************************************************************/

START_TEST(test_dragonfly_null_pointer)
{
	int rc = dstm_distance_create_dragonfly(NULL, 9);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_int_eq(dstm_errno, DSTM_EINVAL);
}
END_TEST

START_TEST(test_dragonfly_invalid_size_too_small)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_dragonfly(&dist, 3);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_ptr_null(dist);
}
END_TEST

START_TEST(test_dragonfly_invalid_size_too_large)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_dragonfly(&dist,
	                                        DSTM_DISTANCE_MAX_SIZE + 1);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_ptr_null(dist);
}
END_TEST

START_TEST(test_dragonfly_create_destroy)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_dragonfly(&dist, 9);
	ck_assert_int_eq(rc, DSTM_SUCCESS);
	ck_assert_ptr_nonnull(dist);
	ck_assert_ptr_nonnull(dist->distances);
	ck_assert_uint_eq(dist->size, 9);

	dstm_distance_destroy(&dist);
	ck_assert_ptr_null(dist);
}
END_TEST

START_TEST(test_dragonfly_properties)
{
	/* 9 nodes: 3 groups of 3 */
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_dragonfly(&dist, 9);
	ck_assert_int_eq(rc, DSTM_SUCCESS);

	ck_assert(is_symmetric(dist));
	ck_assert(diagonal_is_zero(dist));
	ck_assert(all_nonnegative(dist));

	/* Intra-group: nodes 0,1,2 are in group 0 -> 1 hop */
	ck_assert_int_eq(dstm_distance_get(dist, 0, 1), 1);
	ck_assert_int_eq(dstm_distance_get(dist, 0, 2), 1);
	ck_assert_int_eq(dstm_distance_get(dist, 1, 2), 1);

	/* Inter-group: group 0 to group 1 (nodes 3,4,5) -> 3 hops */
	ck_assert_int_eq(dstm_distance_get(dist, 0, 3), 3);
	ck_assert_int_eq(dstm_distance_get(dist, 0, 5), 3);

	/* Inter-group: group 0 to group 2 (nodes 6,7,8) -> 3 hops */
	ck_assert_int_eq(dstm_distance_get(dist, 0, 6), 3);
	ck_assert_int_eq(dstm_distance_get(dist, 2, 8), 3);

	dstm_distance_destroy(&dist);
}
END_TEST

START_TEST(test_dragonfly_minimal)
{
	/* 4 nodes: 2 groups of 2 */
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_dragonfly(&dist, 4);
	ck_assert_int_eq(rc, DSTM_SUCCESS);

	/* Intra-group */
	ck_assert_int_eq(dstm_distance_get(dist, 0, 1), 1);
	ck_assert_int_eq(dstm_distance_get(dist, 2, 3), 1);

	/* Inter-group */
	ck_assert_int_eq(dstm_distance_get(dist, 0, 2), 3);
	ck_assert_int_eq(dstm_distance_get(dist, 1, 3), 3);

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
	TCase *tc_mesh, *tc_torus, *tc_tree, *tc_hypercube, *tc_dragonfly;

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

	/* Mesh tests */
	tc_mesh = tcase_create("mesh");
	tcase_add_test(tc_mesh, test_mesh_null_pointer);
	tcase_add_test(tc_mesh, test_mesh_invalid_size_zero);
	tcase_add_test(tc_mesh, test_mesh_invalid_size_too_large);
	tcase_add_test(tc_mesh, test_mesh_create_destroy);
	tcase_add_test(tc_mesh, test_mesh_properties);
	tcase_add_test(tc_mesh, test_mesh_square);
	suite_add_tcase(s, tc_mesh);

	/* Torus tests */
	tc_torus = tcase_create("torus");
	tcase_add_test(tc_torus, test_torus_null_pointer);
	tcase_add_test(tc_torus, test_torus_invalid_size_too_small);
	tcase_add_test(tc_torus, test_torus_invalid_size_too_large);
	tcase_add_test(tc_torus, test_torus_create_destroy);
	tcase_add_test(tc_torus, test_torus_shorter_than_mesh);
	suite_add_tcase(s, tc_torus);

	/* Tree tests */
	tc_tree = tcase_create("tree");
	tcase_add_test(tc_tree, test_tree_null_pointer);
	tcase_add_test(tc_tree, test_tree_invalid_size_zero);
	tcase_add_test(tc_tree, test_tree_invalid_size_too_large);
	tcase_add_test(tc_tree, test_tree_create_destroy);
	tcase_add_test(tc_tree, test_tree_properties);
	tcase_add_test(tc_tree, test_tree_single_node);
	suite_add_tcase(s, tc_tree);

	/* Hypercube tests */
	tc_hypercube = tcase_create("hypercube");
	tcase_add_test(tc_hypercube, test_hypercube_null_pointer);
	tcase_add_test(tc_hypercube, test_hypercube_invalid_size_one);
	tcase_add_test(tc_hypercube,
	               test_hypercube_invalid_size_not_power_of_two);
	tcase_add_test(tc_hypercube, test_hypercube_invalid_size_too_large);
	tcase_add_test(tc_hypercube, test_hypercube_create_destroy);
	tcase_add_test(tc_hypercube, test_hypercube_properties);
	tcase_add_test(tc_hypercube, test_hypercube_2d);
	suite_add_tcase(s, tc_hypercube);

	/* Dragonfly tests */
	tc_dragonfly = tcase_create("dragonfly");
	tcase_add_test(tc_dragonfly, test_dragonfly_null_pointer);
	tcase_add_test(tc_dragonfly, test_dragonfly_invalid_size_too_small);
	tcase_add_test(tc_dragonfly, test_dragonfly_invalid_size_too_large);
	tcase_add_test(tc_dragonfly, test_dragonfly_create_destroy);
	tcase_add_test(tc_dragonfly, test_dragonfly_properties);
	tcase_add_test(tc_dragonfly, test_dragonfly_minimal);
	suite_add_tcase(s, tc_dragonfly);

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
