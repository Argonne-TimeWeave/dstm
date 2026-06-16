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
 * Regression test: multiple dstm_start / dstm_fini cycles within a
 * single dstm_init / dstm_finalize session must all complete.
 *
 * Before the fix, the file-scope `dstm_local_heartbeat_time` static in
 * src/ops.c survived across sessions.  Once session 1 advanced it to
 * `T`, session 2's puts/gets at time <= T silently skipped heartbeat
 * emission.  Peers therefore never learned of the new session's
 * clock, version transitions never fired, and the barrier in session
 * 2's dstm_fini blocked forever.
 *
 * This test does three iterations of:
 *   dstm_start -> one put + one get at time=1 -> dstm_fini(1)
 *
 * Each iteration involves real put/get traffic that exercises the
 * heartbeat path.  Without the fix, the second iteration's dstm_fini
 * hangs and the surrounding `timeout` in the bats harness kills the
 * run.  With the fix, all three iterations complete and the process
 * exits 0.
 *
 * Requires world_size >= 2 (the put/get target is a neighbor on a
 * ring).
 */

#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "dstm.h"

#define NUM_ITERATIONS 3

int main(int argc, char *argv[])
{
	int rank, world_size;

	MPI_Init(&argc, &argv);
	dstm_init(&argc, &argv);
	dstm_log_setlevel(DSTM_LOG_DEBUG);

	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (world_size < 2) {
		if (rank == 0)
			fprintf(stderr,
			        "multi_session requires world_size >= 2 "
			        "(got %d)\n",
			        world_size);
		dstm_finalize();
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
		fprintf(stderr, "rank %d starting iter %d\n", rank, iter);

		dstm_distance_t *dist = NULL;
		int rc;
		if (world_size >= 3)
			rc = dstm_distance_create_ring(&dist,
			                               (size_t)world_size);
		else
			rc = dstm_distance_create_complete(&dist,
			                                   (size_t)world_size);
		assert(rc == DSTM_SUCCESS);

		rc = dstm_start(dist);
		assert(rc == DSTM_SUCCESS);

		/* Real work that bumps the logical clock to time=1 on
		 * every rank, so the heartbeat path is exercised on
		 * each iteration. */
		dstm_darray_t *a = NULL;
		rc = dstm_darray_create(&a, sizeof(int), 1);
		assert(rc == DSTM_SUCCESS);

		/* Publish our slot at time=0 with delta=+1: the value
		 * becomes readable at logical time 1. */
		int payload = 1000 * (iter + 1) + rank;
		rc = dstm_darray_put(a, (size_t)rank, &payload, /*time=*/0,
		                     /*delta=*/1);
		assert(rc == DSTM_SUCCESS);

		/* Read the neighbor's slot at effective time 1 (the
		 * matching version published above).  Remote reads
		 * require delta != 0, so we issue (time=2, delta=-1). */
		int peer = (rank + 1) % world_size;
		int out = 0;
		rc = dstm_darray_get(a, (size_t)peer, &out, /*time=*/2,
		                     /*delta=*/-1);
		assert(rc == DSTM_SUCCESS);
		assert(out == 1000 * (iter + 1) + peer);

		dstm_darray_destroy(&a);

		/* Tear the session down.  last_time must be >= the
		 * highest logical time submitted (here, 2 from the
		 * remote get).  Pre-fix, this call blocks forever on
		 * iter >= 1 because the heartbeat counter from the
		 * previous session suppresses heartbeat emission this
		 * session. */
		rc = dstm_fini((dstm_time_t)2);
		assert(rc == DSTM_SUCCESS);

		dstm_distance_destroy(&dist);

		fprintf(stderr, "rank %d completed iter %d\n", rank, iter);
	}

	if (rank == 0)
		printf("PASS: multi_session (world_size=%d, iterations=%d)\n",
		       world_size, NUM_ITERATIONS);

	dstm_finalize();
	MPI_Finalize();
	return EXIT_SUCCESS;
}
