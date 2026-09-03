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
 * combine_distances_insufficient.c  (NEGATIVE CONTROL)
 *
 * Companion to combine_distances.c.  Same two single-cross-edge graphs
 * (A: 0->1 on region A, B: 1->2 on region B) and the same lockstep pull
 * traffic, but the session is started with matrix A ALONE.  Matrix A
 * marks the pair (1,2) unreachable, so when graph B issues its cross-rank
 * GET from rank 1 to rank 2, the version state machine believes no GET
 * can ever arrive for that pair and evicts the version prematurely --
 * which must trip the abort() in dstm_request_progress_in_get (or, if the
 * timing differs, deadlock until the harness timeout).
 *
 * This test is EXPECTED TO FAIL (non-zero exit).  It exists to prove that
 * dstm_distance_combine() is necessary: if this ever exits 0, the version
 * machine's premature-eviction safety net has regressed.
 *
 * Requires exactly 3 ranks.
 */

#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "dstm.h"

#define REGB_MARK 0x0B000000
#define TAG(mark, who, t) ((int)((mark) | (((who) & 0xff) << 8) | ((t) & 0xff)))

static void pair_topology(int self, int lo, int hi,
                          size_t *out_deg, size_t *dst)
{
	if (self == lo) {
		*out_deg = 1;
		*dst = (size_t)hi;
	} else {
		*out_deg = 0;
		*dst = 0;
	}
	(void)hi;
}

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);
	dstm_init(&argc, &argv);

	int rank, ws;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ws);

	if (ws != 3) {
		if (rank == 0)
			fprintf(stderr, "requires exactly 3 ranks\n");
		dstm_finalize();
		MPI_Finalize();
		return 77;
	}

	const int DELTA = 1;

	/* Derive ONLY matrix A (cross edge 0->1).  Deliberately omit B. */
	size_t deg_a, dst_a;
	pair_topology(rank, 0, 1, &deg_a, &dst_a);
	dstm_distance_t *da = NULL;
	int rc = dstm_dgraph_distance_from_topology(&da, 1, &deg_a,
	                                            deg_a ? &dst_a : NULL,
	                                            DELTA);
	assert(rc == DSTM_SUCCESS);

	/* Confirm the insufficiency precondition: A marks (1,2) unreachable. */
	assert(dstm_distance_get(da, 1, 2) == -1);

	/* Start with the INSUFFICIENT matrix. */
	rc = dstm_start(da);
	assert(rc == DSTM_SUCCESS);

	int region_b = -1;
	rc = dstm_region_register(1 << 20, &region_b);
	assert(rc == DSTM_SUCCESS);

	/* Graph B: single cross edge 1 -> 2 on region B. */
	size_t out_deg, dst;
	pair_topology(rank, 1, 2, &out_deg, &dst);
	dstm_dgraph_t *gb = NULL;
	rc = dstm_dgraph_create_on_region(&gb, sizeof(int), 0, 1, &out_deg,
	                                  out_deg ? &dst : NULL, NULL, NULL,
	                                  region_b);
	assert(rc == DSTM_SUCCESS);

	/* Same lockstep pull traffic as the positive test.  Rank 1 reads
	 * rank 2's vertex across the (1,2) edge -- the pair matrix A marks
	 * unreachable.  This is where premature eviction must bite. */
	const int STEPS = 4;
	for (int t = 0; t < STEPS; t++) {
		int vb = TAG(REGB_MARK, rank, t);
		int prc = dstm_dgraph_vertex_put(gb, (size_t)rank, &vb, t, 1);
		assert(prc == DSTM_SUCCESS);

		if (t >= 1 && rank == 1) {
			int r1 = 0;
			/* This GET should hit an already-evicted version and
			 * abort inside the DSTM.  If it somehow succeeds the
			 * value would still be wrong -- either way this test
			 * must not reach a clean exit. */
			dstm_dgraph_vertex_get(gb, 2, &r1, t + 1, -1);
		}
	}

	dstm_barrier((dstm_time_t)STEPS + 1);
	dstm_dgraph_destroy(&gb);
	dstm_fini((dstm_time_t)STEPS + 1);
	dstm_region_unregister(region_b);
	dstm_distance_destroy(&da);
	dstm_finalize();

	/* If we ever get here, the safety net failed to catch the premature
	 * eviction.  Signal failure loudly. */
	if (rank == 0)
		fprintf(stderr,
		        "UNEXPECTED: insufficient matrix did NOT trigger "
		        "premature-eviction abort\n");

	MPI_Finalize();
	return EXIT_FAILURE;
}
