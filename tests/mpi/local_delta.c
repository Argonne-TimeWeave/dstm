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
 * Tests for the (delta == 0 => rank == local) invariant on dstm_put,
 * dstm_get, and the tightened deltas[] sign rule on
 * dstm_darray_allgatherv.
 *
 * All "remote with delta == 0" calls must return -DSTM_EDOM.  Local
 * calls with delta == 0 and remote calls with non-zero delta must
 * continue to behave exactly as before.
 *
 * Tests that need a remote peer skip themselves cleanly when run with
 * world_size < 2.
 */

#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "dstm.h"

static int rank;
static int world_size;

/*******************************************************************************
 * Low-level dstm_put / dstm_get tests
 ******************************************************************************/

/* Remote rank with delta == 0 must be rejected with -DSTM_EDOM by
 * dstm_put.  We use a fake dstm_ptr_t pointed at the peer's region
 * because the rank check fires before the pointer is dereferenced. */
static void test_put_delta_zero_remote_rejected(void)
{
	if (world_size < 2)
		return;

	int peer = (rank + 1) % world_size;
	int val = 42;
	dstm_ptr_t dst = {.region = peer, .offset = 0};
	int rc = dstm_put(peer, peer, /*time=*/0, /*delta=*/0, dst, &val,
	                  sizeof(val));
	assert(rc == -DSTM_EDOM);
}

/* Local rank with delta == 0 must succeed: this is the regression
 * guard for the legitimate use of the API. */
static void test_put_delta_zero_local_accepted(void)
{
	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create(&a, sizeof(int), 1);
	assert(rc == DSTM_SUCCESS);

	int val = 7;
	size_t idx = dstm_darray_global_offset(a);
	rc = dstm_darray_put(a, idx, &val, /*time=*/0, /*delta=*/0);
	assert(rc == DSTM_SUCCESS);

	dstm_darray_destroy(&a);
}

/* Remote rank with non-zero (positive) delta must still be accepted by
 * dstm_put.  This guards against accidentally over-tightening the
 * check.  Routed through the darray wrapper so we get a valid
 * dstm_ptr_t for the peer's region. */
static void test_put_nonzero_delta_remote_still_works(void)
{
	if (world_size < 2)
		return;

	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create(&a, sizeof(int), 1);
	assert(rc == DSTM_SUCCESS);

	int peer = (rank + 1) % world_size;
	int val = 99;
	rc = dstm_darray_put(a, (size_t)peer, &val, /*time=*/0, /*delta=*/1);
	assert(rc == DSTM_SUCCESS);

	dstm_darray_destroy(&a);
}

/* Remote rank with delta == 0 must be rejected with -DSTM_EDOM by
 * dstm_get. */
static void test_get_delta_zero_remote_rejected(void)
{
	if (world_size < 2)
		return;

	int peer = (rank + 1) % world_size;
	int out = 0;
	dstm_ptr_t src = {.region = peer, .offset = 0};
	int rc = dstm_get(peer, peer, /*time=*/0, /*delta=*/0, &out, src,
	                  sizeof(out));
	assert(rc == -DSTM_EDOM);
}

/* Local rank with delta == 0 must succeed (i.e. must not be rejected
 * with -DSTM_EDOM).  Per the put-before-get phase rule we issue the
 * put first and the get second on the same effective time; this is
 * the legal ordering.  We don't assert on the read value here: the
 * relative ordering of a same-tick put and get within the progress
 * loop is governed by implementation details and isn't part of the
 * invariant under test. */
static void test_get_delta_zero_local_accepted(void)
{
	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create(&a, sizeof(int), 1);
	assert(rc == DSTM_SUCCESS);

	int val = 1234;
	size_t idx = dstm_darray_global_offset(a);
	rc = dstm_darray_put(a, idx, &val, /*time=*/0, /*delta=*/0);
	assert(rc == DSTM_SUCCESS);

	int out = 0;
	rc = dstm_darray_get(a, idx, &out, /*time=*/0, /*delta=*/0);
	assert(rc == DSTM_SUCCESS);

	dstm_darray_destroy(&a);
}

/* Remote rank with non-zero (negative) delta must still be accepted by
 * dstm_get.  Pair it with a put at delta=+1 so the requested version
 * exists. */
static void test_get_nonzero_delta_remote_still_works(void)
{
	if (world_size < 2)
		return;

	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create(&a, sizeof(int), 1);
	assert(rc == DSTM_SUCCESS);

	/* Each rank publishes its own slot at time 0 with delta=+1, so
	 * the value becomes readable at logical time 1. */
	int payload = 5000 + rank;
	rc = dstm_darray_put(a, (size_t)rank, &payload, /*time=*/0,
	                     /*delta=*/1);
	assert(rc == DSTM_SUCCESS);

	int peer = (rank + 1) % world_size;
	int out = 0;
	rc = dstm_darray_get(a, (size_t)peer, &out, /*time=*/2, /*delta=*/-1);
	assert(rc == DSTM_SUCCESS);
	assert(out == 5000 + peer);

	dstm_darray_destroy(&a);
}

/*******************************************************************************
 * dstm_darray_allgatherv deltas[] sign tests
 ******************************************************************************/

/* deltas[peer] == 0 (peer != self) must be rejected with -DSTM_EDOM. */
static void test_allgatherv_zero_remote_delta_rejected(void)
{
	if (world_size < 2)
		return;

	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create(&a, sizeof(int), 1);
	assert(rc == DSTM_SUCCESS);

	int src_val = rank;
	int *dst_buf = malloc((size_t)world_size * sizeof(int));
	assert(dst_buf != NULL);

	dstm_time_t *deltas = malloc((size_t)world_size * sizeof(dstm_time_t));
	assert(deltas != NULL);

	/* Build a valid deltas[] then poison one remote slot with 0. */
	deltas[rank] = 1;
	for (int r = 0; r < world_size; r++)
		if (r != rank)
			deltas[r] = -1;
	int peer = (rank + 1) % world_size;
	deltas[peer] = 0;

	rc = dstm_darray_allgatherv(a, &src_val, dst_buf, /*time=*/0, deltas);
	assert(rc == -DSTM_EDOM);

	free(deltas);
	free(dst_buf);
	dstm_darray_destroy(&a);
}

/* deltas[self] == 0 must be rejected with -DSTM_EDOM. */
static void test_allgatherv_zero_local_delta_rejected(void)
{
	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create(&a, sizeof(int), 1);
	assert(rc == DSTM_SUCCESS);

	int src_val = rank;
	int *dst_buf = malloc((size_t)world_size * sizeof(int));
	assert(dst_buf != NULL);

	dstm_time_t *deltas = malloc((size_t)world_size * sizeof(dstm_time_t));
	assert(deltas != NULL);

	deltas[rank] = 0; /* poisoned */
	for (int r = 0; r < world_size; r++)
		if (r != rank)
			deltas[r] = -1;

	rc = dstm_darray_allgatherv(a, &src_val, dst_buf, /*time=*/0, deltas);
	assert(rc == -DSTM_EDOM);

	free(deltas);
	free(dst_buf);
	dstm_darray_destroy(&a);
}

/*******************************************************************************
 * main
 ******************************************************************************/

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);
	dstm_init(&argc, &argv);
	dstm_log_setlevel(DSTM_LOG_DEBUG);

	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	dstm_distance_t *dist = NULL;
	if (world_size >= 3)
		dstm_distance_create_ring(&dist, world_size);
	else
		dstm_distance_create_complete(&dist, world_size);

	dstm_start(dist);

	/* Validation tests: these must short-circuit before any state
	 * is modified, so they're safe to run before/after live ones. */
	test_put_delta_zero_remote_rejected();
	test_get_delta_zero_remote_rejected();
	test_allgatherv_zero_remote_delta_rejected();
	test_allgatherv_zero_local_delta_rejected();

	/* Positive tests: regression guards for the unaffected paths. */
	test_put_delta_zero_local_accepted();
	test_get_delta_zero_local_accepted();
	test_put_nonzero_delta_remote_still_works();
	test_get_nonzero_delta_remote_still_works();

	if (rank == 0)
		printf("PASS: local_delta (world_size=%d)\n", world_size);

	/* Highest effective timestep submitted: time=2 + delta=-1 read,
	 * and time=0 + delta=+1 puts.  Use 3 as a safe upper bound. */
	dstm_fini((dstm_time_t)3);
	dstm_distance_destroy(&dist);
	dstm_finalize();
	MPI_Finalize();
	return EXIT_SUCCESS;
}
