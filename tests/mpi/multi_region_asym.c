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
 * multi_region_asym.c
 *
 * Exercises the ONE path the other multi-region tests never hit: a darray
 * whose backing region id DIFFERS from rank to rank.  Every existing test
 * lands region 1 on every rank, so the rank_region_id Allgather in
 * dstm_darray_create_on_region always gathers a constant.  Here each rank
 * registers a rank-dependent number of extra regions and backs the darray
 * on a rank-dependent id, forcing the Allgathered mapping to actually
 * matter: a remote get must address the OWNER's local region id, not the
 * requester's.
 *
 * Layout (world_size ranks, id chosen per rank):
 *   rank r registers (r % 3) + 1 extra regions, then backs its darray
 *   slice on region id = 1 + (r % 3)  (i.e. 1, 2, or 3, cycling).
 * So neighbouring ranks generally use different region ids, and a get of
 * a peer's element must resolve the peer's id from the mapping table.
 *
 * Requires >= 2 ranks.
 */

#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "dstm.h"

#define MARK 0x0C000000
#define TAG(rank, t) ((int)(MARK | (((rank) & 0xffff) << 8) | ((t) & 0xff)))

static int g_failures = 0;
static int g_checks = 0;

#define CHECK_EQ(got, want, what)                                             \
	do {                                                                  \
		g_checks++;                                                    \
		if ((int)(got) != (int)(want)) {                              \
			g_failures++;                                         \
			fprintf(stderr,                                       \
			        "MISMATCH [%-24s] got 0x%08x want 0x%08x "    \
			        "(%s:%d)\n",                                  \
			        (what), (unsigned)(got), (unsigned)(want),    \
			        __FILE__, __LINE__);                         \
		}                                                             \
	} while (0)

#define CHECK_TRUE(cond, what)                                                \
	do {                                                                  \
		g_checks++;                                                    \
		if (!(cond)) {                                                \
			g_failures++;                                         \
			fprintf(stderr, "FAILED   [%-24s] (%s:%d)\n", (what), \
			        __FILE__, __LINE__);                         \
		}                                                             \
	} while (0)

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);
	dstm_init(&argc, &argv);

	int rank, ws;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ws);

	if (ws < 2) {
		if (rank == 0)
			fprintf(stderr, "needs >= 2 ranks\n");
		dstm_finalize();
		MPI_Finalize();
		return 77;
	}

	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_complete(&dist, (size_t)ws);
	assert(rc == DSTM_SUCCESS);
	rc = dstm_start(dist);
	assert(rc == DSTM_SUCCESS);

	/* Register a rank-dependent number of extra regions so the chosen
	 * backing id varies across ranks. */
	int nextra = (rank % 3) + 1;         /* 1..3 */
	int my_region = -1;
	for (int i = 0; i < nextra; i++) {
		int id = -1;
		CHECK_EQ(dstm_region_register(1 << 18, &id), DSTM_SUCCESS,
		         "asym.register_rc");
		my_region = id; /* last one is the backing region */
	}
	/* backing id should be exactly nextra (ids 1..nextra were handed
	 * out in order after the default 0). */
	CHECK_EQ(my_region, nextra, "asym.backing_id_matches");

	/* One element per rank, backed on this rank's (divergent) region. */
	dstm_darray_t *a = NULL;
	CHECK_EQ(dstm_darray_create_on_region(&a, sizeof(int), 1, my_region),
	         DSTM_SUCCESS, "asym.create_rc");
	assert(a != NULL);

	int peer = (rank + 1) % ws;

	/* Lockstep pull: write own element, read peer's from the PREVIOUS
	 * timestep.  The peer's element lives on the peer's region id, which
	 * differs from ours -- the get must resolve it via the Allgathered
	 * rank_region_id table. */
	const int STEPS = 4;
	for (int t = 0; t < STEPS; t++) {
		int v = TAG(rank, t);
		CHECK_EQ(dstm_darray_put(a, (size_t)rank, &v, t, 1),
		         DSTM_SUCCESS, "asym.put");

		if (t >= 1) {
			int want = TAG(peer, t - 1);
			int got = 0;
			CHECK_EQ(dstm_darray_get(a, (size_t)peer, &got, t + 1,
			                         -1),
			         DSTM_SUCCESS, "asym.get_rc");
			CHECK_EQ(got, want, "asym.get_val");
		}
	}

	dstm_barrier((dstm_time_t)STEPS + 1);
	dstm_darray_destroy(&a);
	dstm_fini((dstm_time_t)STEPS + 1);

	dstm_distance_destroy(&dist);
	dstm_finalize();

	int total_fail = 0, total_checks = 0;
	MPI_Reduce(&g_failures, &total_fail, 1, MPI_INT, MPI_SUM, 0,
	           MPI_COMM_WORLD);
	MPI_Reduce(&g_checks, &total_checks, 1, MPI_INT, MPI_SUM, 0,
	           MPI_COMM_WORLD);

	if (rank == 0) {
		fprintf(stderr,
		        "=== multi_region_asym: %d checks, %d failures "
		        "(world_size=%d) ===\n",
		        total_checks, total_fail, ws);
		if (total_fail == 0)
			printf("PASS: multi_region_asym (world_size=%d)\n", ws);
		else
			printf("FAIL: multi_region_asym (%d failures)\n",
			       total_fail);
	}

	MPI_Finalize();
	return (total_fail == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
