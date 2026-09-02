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
 * combine_distances.c
 *
 * Proves dstm_distance_combine() produces a session matrix that safely
 * covers the union of two data structures' traffic, so neither
 * structure's version is prematurely evicted.
 *
 * Requires exactly 3 ranks.
 *
 * Two directed-ring graphs, each on its own region, but drawn over
 * DIFFERENT rank subsets so their individual matrices each mark a pair
 * unreachable that the other uses:
 *
 *   Graph A (region A): edge i -> (i+1) only for i in {0,1}
 *                       => cross-rank edges 0->1, 1->2  ... (full ring here)
 *
 * To make the union matter we instead use two *disjoint-pair* line
 * graphs:
 *
 *   Graph A: single cross edge 0 -> 1  (matrix A: only (0,1) reachable)
 *   Graph B: single cross edge 1 -> 2  (matrix B: only (1,2) reachable)
 *
 * Combined matrix: (0,1) and (1,2) reachable at delta 1, (0,2) == -1.
 *
 * Traffic is driven in LOCKSTEP across all timesteps by every rank on
 * BOTH graphs (mirroring multi_dgraph.c), so every version that is read
 * has been written.  The pull pattern is: each rank writes its own
 * vertex at effective time t (put at t, delta 1 -> version t+1) and its
 * neighbor along its graph's edge reads it at (t+1, delta -1 -> t).
 *
 * If the session matrix marked (1,2) unreachable (matrix A alone), the
 * version machine would evict graph B's version on rank 1/2 before the
 * cross get arrived, aborting in dstm_request_progress_in_get.  The
 * combined matrix keeps it alive.
 */

#include <assert.h>
#include <inttypes.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dstm.h"

#define REGA_MARK 0x0A000000
#define REGB_MARK 0x0B000000
#define TAG(mark, who, t) ((int)((mark) | (((who) & 0xff) << 8) | ((t) & 0xff)))

static int g_failures = 0;
static int g_checks = 0;

#define CHECK_EQ(got, want, what)                                              \
	do {                                                                   \
		g_checks++;                                                     \
		if ((int)(got) != (int)(want)) {                               \
			g_failures++;                                          \
			fprintf(stderr,                                        \
			        "MISMATCH [%-24s] got 0x%08x want 0x%08x "     \
			        "(%s:%d)\n",                                   \
			        (what), (unsigned)(got), (unsigned)(want),     \
			        __FILE__, __LINE__);                          \
		} else {                                                       \
			fprintf(stderr, "ok       [%-24s] 0x%08x\n", (what),   \
			        (unsigned)(got));                             \
		}                                                              \
	} while (0)

#define CHECK_TRUE(cond, what)                                                 \
	do {                                                                   \
		g_checks++;                                                     \
		if (!(cond)) {                                                 \
			g_failures++;                                          \
			fprintf(stderr, "FAILED   [%-24s] (%s:%d)\n", (what),  \
			        __FILE__, __LINE__);                          \
		} else {                                                       \
			fprintf(stderr, "ok       [%-24s]\n", (what));         \
		}                                                              \
	} while (0)

/* Build a single-cross-edge graph: rank `lo` owns an edge lo -> hi.
 * Only rank `lo` has an outgoing edge; hi and everyone else have none.
 * One vertex per rank, global id == rank. */
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

static dstm_dgraph_t *make_pair_graph(int self, int lo, int hi, int region_id)
{
	size_t out_deg, dst;
	pair_topology(self, lo, hi, &out_deg, &dst);

	dstm_dgraph_t *g = NULL;
	int rc = dstm_dgraph_create_on_region(
	        &g, sizeof(int), 0, 1, &out_deg,
	        out_deg ? &dst : NULL, NULL, NULL, region_id);
	CHECK_EQ(rc, DSTM_SUCCESS, "make_pair_graph.rc");
	assert(g != NULL);
	return g;
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
			fprintf(stderr,
			        "combine_distances requires exactly 3 ranks\n");
		dstm_finalize();
		MPI_Finalize();
		return 77;
	}

	const int DELTA = 1;

	/* --- Derive each structure's matrix from topology (needs only
	 * dstm_init).  Rank-order partitioning (id == rank) matches what the
	 * in-session graphs will use. -------------------------------------- */
	size_t deg_a, dst_a, deg_b, dst_b;
	pair_topology(rank, 0, 1, &deg_a, &dst_a); /* A: cross edge 0->1 */
	pair_topology(rank, 1, 2, &deg_b, &dst_b); /* B: cross edge 1->2 */

	dstm_distance_t *da = NULL, *db = NULL, *combined = NULL;
	int rc;
	rc = dstm_dgraph_distance_from_topology(&da, 1, &deg_a,
	                                        deg_a ? &dst_a : NULL, DELTA);
	CHECK_EQ(rc, DSTM_SUCCESS, "derive_A.rc");
	rc = dstm_dgraph_distance_from_topology(&db, 1, &deg_b,
	                                        deg_b ? &dst_b : NULL, DELTA);
	CHECK_EQ(rc, DSTM_SUCCESS, "derive_B.rc");

	const dstm_distance_t *const mats[2] = {da, db};
	rc = dstm_distance_combine(&combined, mats, 2);
	CHECK_EQ(rc, DSTM_SUCCESS, "combine.rc");
	CHECK_TRUE(combined != NULL, "combine.nonnull");

	/* Prove each individual matrix marks the OTHER pair unreachable, and
	 * the union covers both. */
	CHECK_EQ(dstm_distance_get(da, 1, 2), -1, "matA_marks_1_2_unreach");
	CHECK_EQ(dstm_distance_get(db, 0, 1), -1, "matB_marks_0_1_unreach");
	CHECK_EQ(dstm_distance_get(combined, 0, 0), 0, "union_diag0");
	CHECK_EQ(dstm_distance_get(combined, 0, 1), DELTA, "union_0_1");
	CHECK_EQ(dstm_distance_get(combined, 1, 0), DELTA, "union_1_0");
	CHECK_EQ(dstm_distance_get(combined, 1, 2), DELTA, "union_1_2");
	CHECK_EQ(dstm_distance_get(combined, 2, 1), DELTA, "union_2_1");
	CHECK_EQ(dstm_distance_get(combined, 0, 2), -1, "union_0_2_unreach");

	/* --- Session with the COMBINED matrix ----------------------------- */
	rc = dstm_start(combined);
	CHECK_EQ(rc, DSTM_SUCCESS, "start.rc");

	int region_a = -1, region_b = -1;
	rc = dstm_region_register(1 << 20, &region_a);
	CHECK_EQ(rc, DSTM_SUCCESS, "register_A.rc");
	rc = dstm_region_register(1 << 20, &region_b);
	CHECK_EQ(rc, DSTM_SUCCESS, "register_B.rc");

	dstm_dgraph_t *ga = make_pair_graph(rank, 0, 1, region_a);
	dstm_dgraph_t *gb = make_pair_graph(rank, 1, 2, region_b);

	/* Lockstep pull pattern, EVERY rank writes its own vertex on BOTH
	 * graphs every timestep so every version exists.  Only the cross-pair
	 * reader actually consumes a peer version, but the uniform clock
	 * advance is what the version machine relies on. */
	const int STEPS = 4;
	for (int t = 0; t < STEPS; t++) {
		int va = TAG(REGA_MARK, rank, t);
		int vb = TAG(REGB_MARK, rank, t);

		/* put at effective time t+1 (delta=1) */
		CHECK_EQ(dstm_dgraph_vertex_put(ga, (size_t)rank, &va, t, 1),
		         DSTM_SUCCESS, "vput_A");
		CHECK_EQ(dstm_dgraph_vertex_put(gb, (size_t)rank, &vb, t, 1),
		         DSTM_SUCCESS, "vput_B");

		if (t >= 1) {
			int want_t = t - 1;

			/* Graph A cross read: rank 0 reads rank 1's vertex
			 * (the edge is 0->1, so 0's neighbor is 1). */
			if (rank == 0) {
				int r0 = 0;
				CHECK_EQ(dstm_dgraph_vertex_get(ga, 1, &r0,
				                                t + 1, -1),
				         DSTM_SUCCESS, "vget_A_rc");
				CHECK_EQ(r0, TAG(REGA_MARK, 1, want_t),
				         "vget_A_val");
			}

			/* Graph B cross read: rank 1 reads rank 2's vertex
			 * (the edge is 1->2). */
			if (rank == 1) {
				int r1 = 0;
				CHECK_EQ(dstm_dgraph_vertex_get(gb, 2, &r1,
				                                t + 1, -1),
				         DSTM_SUCCESS, "vget_B_rc");
				CHECK_EQ(r1, TAG(REGB_MARK, 2, want_t),
				         "vget_B_val");
			}
		}
	}

	dstm_barrier((dstm_time_t)STEPS + 1);

	dstm_dgraph_destroy(&ga);
	dstm_dgraph_destroy(&gb);

	rc = dstm_fini((dstm_time_t)STEPS + 1);
	CHECK_EQ(rc, DSTM_SUCCESS, "fini.rc");

	dstm_region_unregister(region_a);
	dstm_region_unregister(region_b);

	dstm_distance_destroy(&combined);
	dstm_distance_destroy(&da);
	dstm_distance_destroy(&db);

	dstm_finalize();

	int total_fail = 0, total_checks = 0;
	MPI_Reduce(&g_failures, &total_fail, 1, MPI_INT, MPI_SUM, 0,
	           MPI_COMM_WORLD);
	MPI_Reduce(&g_checks, &total_checks, 1, MPI_INT, MPI_SUM, 0,
	           MPI_COMM_WORLD);

	if (rank == 0) {
		fprintf(stderr, "=== combine_distances: %d checks, %d failures "
		                "===\n",
		        total_checks, total_fail);
		if (total_fail == 0)
			printf("PASS: combine_distances\n");
		else
			printf("FAIL: combine_distances (%d failures)\n",
			       total_fail);
	}

	MPI_Finalize();
	return (total_fail == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
