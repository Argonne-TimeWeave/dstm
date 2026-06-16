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
 * Negative test for the put-before-get phase rule.
 *
 * Sequence on a single rank's region at effective time 0:
 *   1. dstm_darray_put(...)  -- creates the version, applies the value;
 *   2. dstm_darray_get(...)  -- drives the version to READABLE;
 *   3. dstm_darray_put(...)  -- second put on the same effective time;
 *      this is a contract violation because the version has already
 *      been observed.  Its progress MUST trigger the abort() in
 *      dstm_request_progress_local_put.
 *
 * The bats driver expects this program to exit with a non-zero status
 * (specifically, terminated by SIGABRT); a clean exit is a regression.
 */

#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "dstm.h"

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

	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create(&a, sizeof(int), 1);
	assert(rc == DSTM_SUCCESS);

	size_t idx = dstm_darray_global_offset(a);

	/* 1. Initial put: legal, creates the version at time 0. */
	int val = 42;
	rc = dstm_darray_put(a, idx, &val, /*time=*/0, /*delta=*/0);
	assert(rc == DSTM_SUCCESS);

	/* 2. Get on the same effective time: drives the version to
	 *    READABLE within dstm_get's drive loop.  Legal under the
	 *    put-before-get rule (the put was issued first). */
	int out = 0;
	rc = dstm_darray_get(a, idx, &out, /*time=*/0, /*delta=*/0);
	assert(rc == DSTM_SUCCESS);

	/* 3. Second put on the same effective time: ILLEGAL.  The
	 *    version is no longer WRITABLE.  This put's enqueue itself
	 *    succeeds; the violation is detected when the request is
	 *    progressed.  We force progress via a subsequent get on
	 *    the same effective time (which spins dstm_do_progress
	 *    and visits the violating put). */
	int val2 = 99;
	rc = dstm_darray_put(a, idx, &val2, /*time=*/0, /*delta=*/0);
	assert(rc == DSTM_SUCCESS);

	/* This get should never return: the violating put will abort
	 * the program inside dstm_request_progress_local_put. */
	int out2 = 0;
	rc = dstm_darray_get(a, idx, &out2, /*time=*/0, /*delta=*/0);

	/* If we reach here the contract enforcement failed. */
	fprintf(stderr,
	        "FAIL: put_after_get reached end of main without aborting "
	        "(rc=%d)\n",
	        rc);

	dstm_darray_destroy(&a);
	dstm_fini(0);
	dstm_distance_destroy(&dist);
	dstm_finalize();
	MPI_Finalize();
	return EXIT_SUCCESS;
}
