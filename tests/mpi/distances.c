/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "dstm.h"
#include "internal/distance.h"

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);
	dstm_init(&argc, &argv);

	int world_size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	/*
	 * Test 1: Ring topology with actual world_size
	 */
	if (world_size >= 3) {
		dstm_distance_t *ring_dist = NULL;
		int rc = dstm_distance_create_ring(&ring_dist, world_size);
		assert(rc == DSTM_SUCCESS);
		assert(ring_dist != NULL);
		assert(ring_dist->distances != NULL);
		assert(ring_dist->size == (size_t)world_size);

		/* Verify the ring was created correctly */
		int d_self = dstm_distance_get(ring_dist, rank, rank);
		assert(d_self == 0);

		/* Check neighbors */
		int next = (rank + 1) % world_size;
		int prev = (rank + world_size - 1) % world_size;
		assert(dstm_distance_get(ring_dist, rank, next) == 1);
		assert(dstm_distance_get(ring_dist, rank, prev) == 1);

		/* Test with dstm_start */
		rc = dstm_start(ring_dist);
		assert(rc == DSTM_SUCCESS);

		dstm_fini();
		dstm_distance_destroy(&ring_dist);
		assert(ring_dist == NULL);

		if (rank == 0)
			printf("Ring topology test passed (size=%d)\n",
			       world_size);
	} else {
		if (rank == 0)
			printf("Skipping ring test (need >= 3 ranks, have %d)\n",
			       world_size);
	}

	/*
	 * Test 2: Complete graph topology
	 */
	{
		dstm_distance_t *complete_dist = NULL;
		int rc = dstm_distance_create_complete(&complete_dist,
		                                       world_size);
		assert(rc == DSTM_SUCCESS);
		assert(complete_dist != NULL);
		assert(complete_dist->size == (size_t)world_size);

		/* Verify all non-diagonal distances are 1 */
		for (int i = 0; i < world_size; i++) {
			int d = dstm_distance_get(complete_dist, rank, i);
			if (i == rank)
				assert(d == 0);
			else
				assert(d == 1);
		}

		/* Test with dstm_start */
		rc = dstm_start(complete_dist);
		assert(rc == DSTM_SUCCESS);

		dstm_fini();
		dstm_distance_destroy(&complete_dist);

		if (rank == 0)
			printf("Complete graph topology test passed (size=%d)\n",
			       world_size);
	}

	/*
	 * Test 3: Star topology (if we have at least 2 nodes)
	 */
	if (world_size >= 2) {
		dstm_distance_t *star_dist = NULL;
		int rc = dstm_distance_create_star(&star_dist, world_size);
		assert(rc == DSTM_SUCCESS);
		assert(star_dist != NULL);
		assert(star_dist->size == (size_t)world_size);

		/* Verify distances from center (rank 0) are all 1 */
		if (rank == 0) {
			for (int i = 1; i < world_size; i++) {
				int d = dstm_distance_get(star_dist, 0, i);
				assert(d == 1);
			}
		} else {
			/* Non-center nodes: distance to center is 1 */
			int d = dstm_distance_get(star_dist, rank, 0);
			assert(d == 1);

			/* Distance to other non-center nodes is 2 */
			for (int i = 1; i < world_size; i++) {
				if (i != rank) {
					int d2 = dstm_distance_get(star_dist,
					                           rank, i);
					assert(d2 == 2);
				}
			}
		}

		/* Test with dstm_start */
		rc = dstm_start(star_dist);
		assert(rc == DSTM_SUCCESS);

		dstm_fini();
		dstm_distance_destroy(&star_dist);

		if (rank == 0)
			printf("Star topology test passed (size=%d)\n",
			       world_size);
	}

	/*
	 * Test 4: Line topology
	 */
	{
		dstm_distance_t *line_dist = NULL;
		int rc = dstm_distance_create_line(&line_dist, world_size);
		assert(rc == DSTM_SUCCESS);
		assert(line_dist != NULL);
		assert(line_dist->size == (size_t)world_size);

		/* Verify distances are |i - j| */
		for (int i = 0; i < world_size; i++) {
			int expected = (i > rank) ? (i - rank) : (rank - i);
			int d = dstm_distance_get(line_dist, rank, i);
			assert(d == expected);
		}

		/* Test with dstm_start */
		rc = dstm_start(line_dist);
		assert(rc == DSTM_SUCCESS);

		dstm_fini();
		dstm_distance_destroy(&line_dist);

		if (rank == 0)
			printf("Line topology test passed (size=%d)\n",
			       world_size);
	}

	if (rank == 0)
		printf("All distance matrix tests passed!\n");

	dstm_finalize();
	MPI_Finalize();
	return 0;
}
