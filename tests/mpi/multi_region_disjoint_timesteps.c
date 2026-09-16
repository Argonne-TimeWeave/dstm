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
 * multi_region_disjoint_timesteps.c
 *
 * Verify that multiple regions work when their operations never share an
 * originating/source timestep.
 *
 * Region 0 is accessed only at even source times:
 *
 *     0, 2, 4, 6, ...
 *
 * Region 1 is accessed only at odd source times:
 *
 *     1, 3, 5, 7, ...
 *
 * At each source timestep, the selected region receives:
 *
 *   1. a PUT of this rank's current value at delta +1; and
 *   2. after the first visit to that region, a GET of the right-hand
 *      neighbor's value from the region's previous visit at delta -1.
 *
 * For example, region 0:
 *
 *   PUT at source time 0, delta +1 -> effective time 1
 *   GET at source time 2, delta -1 -> effective time 1
 *
 * Region 1:
 *
 *   PUT at source time 1, delta +1 -> effective time 2
 *   GET at source time 3, delta -1 -> effective time 2
 *
 * Thus each GET reads the value written on the previous visit to the same
 * region, while source timestamps remain globally increasing and operations
 * on the two regions never overlap at a source timestep.
 *
 * This specifically exercises the rank-wide heartbeat clock together with
 * per-(rank, region_id) neighbor clocks. Heartbeats generated while operating
 * on one region must not make the other region's still-needed version
 * transition or disappear prematurely.
 *
 * Run with:
 *
 *     mpirun -n 2 ./multi_region_disjoint_timesteps
 *     mpirun -n 4 ./multi_region_disjoint_timesteps
 */

#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "dstm.h"

#define REGION0_MARK 0x0A000000
#define REGION1_MARK 0x0B000000

/*
 * Encode the region, producing rank, and originating timestep into the value.
 * The masks are sufficient for the small MPI test sizes and timestep count.
 */
#define TAG(mark, rank, time)                                                  \
	((int)((mark) | (((rank) & 0xff) << 8) | ((time) & 0xff)))

static int g_checks = 0;
static int g_failures = 0;

#define CHECK_EQ(got, expected, description)                                  \
	do {                                                                   \
		int check_got = (int)(got);                                    \
		int check_expected = (int)(expected);                           \
		g_checks++;                                                     \
		if (check_got != check_expected) {                              \
			g_failures++;                                           \
			fprintf(stderr,                                        \
			        "rank %d: FAIL [%-30s] got 0x%08x, "            \
			        "expected 0x%08x (%s:%d)\n",                    \
			        rank, (description),                           \
			        (unsigned)check_got,                           \
			        (unsigned)check_expected,                      \
			        __FILE__, __LINE__);                           \
		}                                                              \
	} while (0)

int main(int argc, char **argv)
{
	int rc;
	int rank;
	int world_size;

	MPI_Init(&argc, &argv);

	rc = dstm_init(&argc, &argv);
	if (rc != DSTM_SUCCESS) {
		fprintf(stderr, "dstm_init failed: %d\n", rc);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	if (world_size < 2) {
		if (rank == 0) {
			fprintf(stderr,
			        "multi_region_disjoint_timesteps requires "
			        "at least 2 ranks\n");
		}

		dstm_finalize();
		MPI_Finalize();
		return 77;
	}

	/*
	 * A complete distance matrix is appropriate because each rank reads
	 * from its right-hand neighbor.
	 */
	dstm_distance_t *distance = NULL;
	rc = dstm_distance_create_complete(&distance, (size_t)world_size);
	assert(rc == DSTM_SUCCESS);
	assert(distance != NULL);

	rc = dstm_start(distance);
	assert(rc == DSTM_SUCCESS);

	/*
	 * Region 0 was created by dstm_start(). Register the second backing
	 * region before issuing any operation or receiving any heartbeat.
	 */
	int region1_id = -1;
	rc = dstm_region_register(1 << 20, &region1_id);
	assert(rc == DSTM_SUCCESS);
	assert(region1_id == 1);

	dstm_darray_t *array0 = NULL;
	dstm_darray_t *array1 = NULL;

	/*
	 * Each rank owns one int in each distributed array.
	 */
	rc = dstm_darray_create_on_region(&array0, sizeof(int), 1, 0);
	assert(rc == DSTM_SUCCESS);
	assert(array0 != NULL);

	rc = dstm_darray_create_on_region(&array1, sizeof(int), 1, region1_id);
	assert(rc == DSTM_SUCCESS);
	assert(array1 != NULL);

	const int peer = (rank + 1) % world_size;

	/*
	 * Four visits to each region:
	 *
	 *   region 0: source times 0, 2, 4, 6
	 *   region 1: source times 1, 3, 5, 7
	 *
	 * At no source time does the test issue an operation on both regions.
	 */
	const int visits_per_region = 4;
	const int source_timesteps = visits_per_region * 2;

	for (int time = 0; time < source_timesteps; time++) {
		const int region_index = time & 1;
		const int mark =
		        region_index == 0 ? REGION0_MARK : REGION1_MARK;
		dstm_darray_t *array =
		        region_index == 0 ? array0 : array1;

		/*
		 * Publish this rank's value into the selected region. It becomes
		 * effective at time + 1.
		 */
		int value = TAG(mark, rank, time);

		rc = dstm_darray_put(array, (size_t)rank, &value,
		                     (dstm_time_t)time, 1);
		CHECK_EQ(rc, DSTM_SUCCESS,
		         region_index == 0 ? "region0 put"
		                           : "region1 put");

		/*
		 * The previous visit to this region occurred at source time
		 * time - 2 and wrote a version effective at time - 1.
		 *
		 * Reading now with delta -1 requests that same effective time:
		 *
		 *     time + (-1) == time - 1
		 */
		if (time >= 2) {
			int received = 0;
			int expected = TAG(mark, peer, time - 2);

			rc = dstm_darray_get(array, (size_t)peer, &received,
			                     (dstm_time_t)time, -1);
			CHECK_EQ(rc, DSTM_SUCCESS,
			         region_index == 0 ? "region0 remote get"
			                           : "region1 remote get");

			CHECK_EQ(received, expected,
			         region_index == 0 ? "region0 remote value"
			                           : "region1 remote value");
		}
	}

	/*
	 * Drain all outstanding requests and make the final versions safe to
	 * discard before destroying the arrays.
	 */
	const dstm_time_t final_time =
	        (dstm_time_t)source_timesteps + 1;

	rc = dstm_barrier(final_time);
	CHECK_EQ(rc, DSTM_SUCCESS, "final barrier");

	dstm_darray_destroy(&array1);
	dstm_darray_destroy(&array0);

	rc = dstm_fini(final_time);
	CHECK_EQ(rc, DSTM_SUCCESS, "dstm_fini");

	dstm_distance_destroy(&distance);
	dstm_finalize();

	/*
	 * Aggregate results while MPI is still active.
	 */
	int total_checks = 0;
	int total_failures = 0;

	MPI_Reduce(&g_checks, &total_checks, 1, MPI_INT, MPI_SUM, 0,
	           MPI_COMM_WORLD);
	MPI_Reduce(&g_failures, &total_failures, 1, MPI_INT, MPI_SUM, 0,
	           MPI_COMM_WORLD);

	if (rank == 0) {
		fprintf(stderr,
		        "=== multi_region_disjoint_timesteps: "
		        "%d checks, %d failures (world_size=%d) ===\n",
		        total_checks, total_failures, world_size);

		if (total_failures == 0) {
			printf("PASS: multi_region_disjoint_timesteps "
			       "(world_size=%d)\n",
			       world_size);
		} else {
			printf("FAIL: multi_region_disjoint_timesteps "
			       "(%d failures)\n",
			       total_failures);
		}
	}

	MPI_Finalize();

	return total_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
