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
 * Tests for dstm_get_or_default and dstm_darray_get_or_default.
 *
 *  - Negative effective target time short-circuits to the default
 *    value without contacting the owning rank (and so cannot block).
 *  - NULL default_value is rejected with -DSTM_EINVAL.
 *  - Non-negative target time delegates to the underlying get
 *    (verified by reading a value that was actually published).
 */

#include <assert.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "dstm.h"

static void test_negative_target_returns_default(int rank, int world_size)
{
	(void)world_size;
	dstm_darray_t *arr = NULL;
	int rc = dstm_darray_create(&arr, sizeof(int), 1);
	assert(rc == DSTM_SUCCESS);

	/* Read another rank's slot at a negative target time.  This MUST
	 * return immediately with the default value and not block. */
	int peer = (rank + 1) % world_size;
	int sentinel = 0xDEADBEEF;
	int dst = 0;
	rc = dstm_darray_get_or_default(arr, (size_t)peer, &dst,
	                                /*time=*/0, /*delta=*/-5,
	                                &sentinel);
	assert(rc == DSTM_SUCCESS);
	assert(dst == sentinel);

	dstm_darray_destroy(&arr);
}

static void test_null_default_is_einval(int rank, int world_size)
{
	(void)rank;
	(void)world_size;
	dstm_darray_t *arr = NULL;
	int rc = dstm_darray_create(&arr, sizeof(int), 1);
	assert(rc == DSTM_SUCCESS);

	int dst = 0;
	rc = dstm_darray_get_or_default(arr, 0, &dst, 0, -1,
	                                /*default_value=*/NULL);
	assert(rc == -DSTM_EINVAL);

	/* Same check on the lower-level dstm_get_or_default API. */
	dstm_ptr_t fake_src = {.region = rank, .offset = 0};
	int dst2 = 0;
	rc = dstm_get_or_default(rank, rank, 0, -1, &dst2, fake_src,
	                         sizeof(int), NULL);
	assert(rc == -DSTM_EINVAL);

	dstm_darray_destroy(&arr);
}

static void test_nonnegative_falls_through(int rank, int world_size,
                                           dstm_distance_t *dist)
{
	dstm_darray_t *arr = NULL;
	int rc = dstm_darray_create(&arr, sizeof(int), 1);
	assert(rc == DSTM_SUCCESS);

	int max_dist = 0;
	for (int r = 0; r < world_size; r++) {
		int dd = dstm_distance_get(dist, rank, r);
		if (dd > max_dist)
			max_dist = dd;
	}

	/* Publish a known value into our own slot at every tick from 0 to
	 * max_dist*2 (inclusive) so that any peer reading us at any
	 * timestamp in that window finds a version. */
	int payload = 1000 + rank;
	for (int t = 0; t <= max_dist * 2; t++) {
		rc = dstm_darray_put(arr, (size_t)rank, &payload,
		                     (dstm_time_t)t, /*delta=*/0);
		assert(rc == DSTM_SUCCESS);
	}

	/* Read peer at time = max_dist with delta = -dist(rank, peer);
	 * effective target is non-negative so this must perform a real
	 * remote get and return the published value. */
	int peer = (rank + 1) % world_size;
	int d = dstm_distance_get(dist, rank, peer);

	int dst = 0;
	int sentinel = -1;
	rc = dstm_darray_get_or_default(arr, (size_t)peer, &dst,
	                                /*time=*/(dstm_time_t)max_dist*2,
	                                /*delta=*/-(dstm_time_t)d,
	                                &sentinel);
	assert(rc == DSTM_SUCCESS);
	assert(dst == 1000 + peer);
	assert(dst != sentinel);

	dstm_darray_destroy(&arr);
}

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);
	dstm_init(&argc, &argv);
	dstm_log_setlevel(DSTM_LOG_DEBUG);

	int rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	dstm_distance_t *dist = NULL;
	if (world_size >= 3)
		dstm_distance_create_ring(&dist, world_size);
	else
		dstm_distance_create_complete(&dist, world_size);

	dstm_start(dist);

	test_negative_target_returns_default(rank, world_size);
	test_null_default_is_einval(rank, world_size);
	test_nonnegative_falls_through(rank, world_size, dist);

	if (rank == 0)
		printf("PASS: get_or_default (world_size=%d)\n", world_size);

	int max_dist = 0;
	for (int r = 0; r < world_size; r++) {
		int d = dstm_distance_get(dist, rank, r);
		if (d > max_dist)
			max_dist = d;
	}
	dstm_fini((dstm_time_t)(max_dist * 2 + 2));
	dstm_distance_destroy(&dist);
	dstm_finalize();
	MPI_Finalize();
	return EXIT_SUCCESS;
}
