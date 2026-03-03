/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

/*
 * Stencil test for dstm_darray_allgatherv.
 *
 * Each rank owns one int element in a distributed array of world_size
 * elements.  At each timestep t, every rank:
 *
 *   - puts its local value (rank + t) at time (t + 1) via the local put
 *     (put_delta = +1)
 *   - gets every other rank's slice at time (t - distance) via the remote
 *     gets (get_delta[r] = -distance(self, r))
 *
 * The stencil loop starts at t = max_dist so that all get targets have
 * already been written by a previous step's put.
 *
 * Topology: ring for world_size >= 3, complete graph for world_size == 2.
 */

#include <assert.h>
#include <mpi.h>
#include <stdlib.h>

#include "dstm.h"

/*******************************************************************************
 * Stencil test
 ******************************************************************************/

static void
test_allgatherv_stencil(int rank, int world_size, dstm_distance_t *dist)
{
	/* One int element per rank. */
	dstm_darray_t *arr = NULL;
	int rc = dstm_darray_create(&arr, sizeof(int), 1);
	assert(rc == DSTM_SUCCESS);

	/* Output buffer: one int slot per rank. */
	int *dst_buf = malloc((size_t)world_size * sizeof(int));
	assert(dst_buf != NULL);

	/* Per-rank deltas array: deltas[self] >= 0 (put), rest <= 0 (get). */
	dstm_time_t *deltas = malloc((size_t)world_size * sizeof(dstm_time_t));
	assert(deltas != NULL);

	/* Determine the maximum topological distance from this rank to any
	 * other, so we know the earliest safe starting timestep. */
	int max_dist = 0;
	for (int r = 0; r < world_size; r++) {
		int d = dstm_distance_get(dist, rank, r);
		if (d > max_dist)
			max_dist = d;
	}

	/* Run 3 timesteps beginning at max_dist, ensuring every get target
	 * time (t - distance) has already been written by a prior put. */
	for (int t = max_dist; t < max_dist + 3; t++) {
		/* This rank's contribution at step t. */
		int local_val = rank + t;

		/* Build deltas: local put goes one step into the future;
		 * remote gets reach back by the topological distance. */
		deltas[rank] = 1;
		for (int r = 0; r < world_size; r++) {
			if (r != rank)
				deltas[r] = -dstm_distance_get(dist, rank, r);
		}

		rc = dstm_darray_allgatherv(arr, &local_val, dst_buf,
		                            (dstm_time_t)t, deltas);
		assert(rc == DSTM_SUCCESS);

		/* Verify: each remote slot holds what rank r put at the
		 * effective read time (t - distance(rank, r)), i.e.
		 * r + (t - distance(rank, r)). */
		for (int r = 0; r < world_size; r++) {
			if (r == rank)
				continue;
			int dist_r = dstm_distance_get(dist, rank, r);
			int expected = r + (t - dist_r);
			assert(dst_buf[r] == expected);
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}

	free(deltas);
	free(dst_buf);
	dstm_darray_destroy(&arr);
}

/*******************************************************************************
 * main
 ******************************************************************************/

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);
	dstm_init(&argc, &argv);

	int rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	/* Choose topology: ring needs >= 3 ranks. */
	dstm_distance_t *dist = NULL;
	if (world_size >= 3)
		dstm_distance_create_ring(&dist, world_size);
	else
		dstm_distance_create_complete(&dist, world_size);

	dstm_start(dist);

	test_allgatherv_stencil(rank, world_size, dist);

	if (rank == 0)
		printf("PASS: allgatherv stencil (world_size=%d)\n",
		       world_size);

	dstm_fini();
	dstm_distance_destroy(&dist);
	dstm_finalize();
	MPI_Finalize();
	return EXIT_SUCCESS;
}
