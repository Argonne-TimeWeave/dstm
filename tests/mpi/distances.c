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
	dstm_log_setlevel(DSTM_LOG_NORMAL);

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

		dstm_fini(0);
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

		dstm_fini(0);
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

		dstm_fini(0);
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

		dstm_fini(0);
		dstm_distance_destroy(&line_dist);

		if (rank == 0)
			printf("Line topology test passed (size=%d)\n",
			       world_size);
	}

	/*
        * Test 5: Mesh/2D grid topology
        */
        {
		dstm_distance_t *mesh_dist = NULL;
		int rc = dstm_distance_create_mesh(&mesh_dist, world_size);
		assert(rc == DSTM_SUCCESS);
		assert(mesh_dist != NULL);
		assert(mesh_dist->size == (size_t)world_size);

		/* Compute grid dimensions (as square as possible) */
		size_t cols = world_size;
		for (size_t r = 1; r * r <= (size_t)world_size; r++) {
		    if ((size_t)world_size % r == 0)
			cols = (size_t)world_size / r;
		}
		size_t rows = world_size / cols;
		
		/* Get this rank's position */
		size_t rank_row = (size_t)rank / cols;
		size_t rank_col = (size_t)rank % cols;

		/* Verify distances are Manhattan distances */
		for (int j = 0; j < world_size; j++) {
		    size_t row_j = (size_t)j / cols;
		    size_t col_j = (size_t)j % cols;
		    int row_dist = (int)rank_row - (int)row_j;
		    int col_dist = (int)rank_col - (int)col_j;
		    int expected = (row_dist < 0 ? -row_dist : row_dist) +
			       (col_dist < 0 ? -col_dist : col_dist);
		    int d = dstm_distance_get(mesh_dist, rank, j);
		    assert(d == expected);
		}

		/* Test with dstm_start */
		rc = dstm_start(mesh_dist);
		assert(rc == DSTM_SUCCESS);

		dstm_fini(0);
		dstm_distance_destroy(&mesh_dist);

		if (rank == 0)
		    printf("Mesh topology test passed (size=%d, %ld x %ld grid)\n",
			   world_size, rows, cols);
        }

        /*
        * Test 6: Torus/2D toroidal topology
        */
        {
		dstm_distance_t *torus_dist = NULL;
		int rc = dstm_distance_create_torus(&torus_dist, world_size);
		assert(rc == DSTM_SUCCESS);
		assert(torus_dist != NULL);
		assert(torus_dist->size == (size_t)world_size);

		/* Compute grid dimensions (as square as possible) */
		size_t rows = 1, cols = world_size;
		for (size_t r = 1; r * r <= (size_t)world_size; r++) {
		    if ((size_t)world_size % r == 0) {
			rows = r;
			cols = (size_t)world_size / r;
		    }
		}

		/* Get this rank's position */
		size_t rank_row = (size_t)rank / cols;
		size_t rank_col = (size_t)rank % cols;

		/* Verify distances are toroidal (with wraparound) */
		for (int j = 0; j < world_size; j++) {
		    size_t row_j = (size_t)j / cols;
		    size_t col_j = (size_t)j % cols;

		    /* Compute distance with wraparound */
		    int row_diff = (int)rank_row - (int)row_j;
		    int col_diff = (int)rank_col - (int)col_j;
		    if (row_diff < 0)
			row_diff = -row_diff;
		    if (col_diff < 0)
			col_diff = -col_diff;

		    /* Take shorter path (direct or wraparound) */
		    int row_dist = row_diff < (int)rows - row_diff ?
				   row_diff :
				   (int)rows - row_diff;
		    int col_dist = col_diff < (int)cols - col_diff ?
				   col_diff :
				   (int)cols - col_diff;
		    int expected = row_dist + col_dist;

		    int d = dstm_distance_get(torus_dist, rank, j);
		    assert(d == expected);
		}

		/* Test with dstm_start */
		rc = dstm_start(torus_dist);
		assert(rc == DSTM_SUCCESS);

		dstm_fini(0);
		dstm_distance_destroy(&torus_dist);

		if (rank == 0)
		    printf("Torus topology test passed (size=%d, %ld x %ld torus)\n",
			   world_size, rows, cols);
	}

	/*
	 * Test 7: Hypercube topology
	 */
	        if ((world_size & (world_size - 1)) == 0) {  /* Check if power of 2 */
			dstm_distance_t *hypercube_dist = NULL;
			int rc = dstm_distance_create_hypercube(&hypercube_dist, world_size);
			assert(rc == DSTM_SUCCESS);
			assert(hypercube_dist != NULL);
			assert(hypercube_dist->size == (size_t)world_size);

			/* Verify distances are Hamming distances */
			for (int j = 0; j < world_size; j++) {
			    /* Compute Hamming distance = popcount(rank XOR j) */
			    size_t xor_val = (size_t)rank ^ (size_t)j;
			    int expected = 0;
			    size_t temp = xor_val;
			    while (temp) {
				expected += temp & 1;
				temp >>= 1;
			    }

			    int d = dstm_distance_get(hypercube_dist, rank, j);
			    assert(d == expected);
			}

			/* Test with dstm_start */
			rc = dstm_start(hypercube_dist);
			assert(rc == DSTM_SUCCESS);

			dstm_fini(0);
			dstm_distance_destroy(&hypercube_dist);

			if (rank == 0) {
			    /* Count dimensions: log2(world_size) */
			    int dims = 0;
			    int temp_size = world_size;
			    while (temp_size > 1) {
				dims++;
				temp_size >>= 1;
			    }
			    printf("Hypercube topology test passed (size=%d, %dD hypercube)\n",
				   world_size, dims);
			}
	        } 
		else {
		    if (rank == 0)
		        printf("Skipping hypercube test (need power of 2 ranks, have %d)\n",
			   world_size);
	}


	/*
	* Test 8: Dragonfly topology
	*/
	{
		dstm_distance_t *dragonfly_dist = NULL;
		int rc = dstm_distance_create_dragonfly(&dragonfly_dist, world_size);
		assert(rc == DSTM_SUCCESS);
		assert(dragonfly_dist != NULL);
		assert(dragonfly_dist->size == (size_t)world_size);

		/* Determine number of groups (aim for sqrt(size) nodes per group) */
		size_t group_size = 1;
		for (size_t gs = 2; gs * gs <= (size_t)world_size; gs++) {
		    if ((size_t)world_size % gs == 0)
			group_size = gs;
		}
		size_t num_groups = world_size / group_size;

		/* Get this rank's group */
		size_t rank_group = (size_t)rank / group_size;

		/* Verify distances */
		for (int j = 0; j < world_size; j++) {
		    size_t group_j = (size_t)j / group_size;
		    int expected;

		    if (rank == j) {
			expected = 0;
		    } else if (rank_group == group_j) {
			/* Same group: 1 hop */
			expected = 1;
		    } else {
			/* Different groups: 3 hops */
			expected = 3;
		    }

		    int d = dstm_distance_get(dragonfly_dist, rank, j);
		    assert(d == expected);
		}

		// This barrier saves us from a stall
		// 	It seems like dstm_start has some sync expectations
		// 	that other topologies do not expose. 
		// TODO if we want to fix this possibility
		MPI_Barrier(MPI_COMM_WORLD); 

		/* Test with dstm_start */ 
		rc = dstm_start(dragonfly_dist);
		assert(rc == DSTM_SUCCESS);

		dstm_fini(0);
		dstm_distance_destroy(&dragonfly_dist);

		if (rank == 0)
		    printf("Dragonfly topology test passed (size=%d, %ld groups of %ld nodes)\n",
			   world_size, num_groups, group_size);
	}


	if (rank == 0)
		printf("All distance matrix tests passed!\n");

	dstm_finalize();
	MPI_Finalize();
	return 0;
}
