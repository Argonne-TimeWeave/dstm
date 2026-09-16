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
 * MULTI-DGRAPH EXPLORATORY TEST (public API only)
 * ===============================================
 *
 * Determines whether the multi-region refactor supports TWO distributed
 * graphs living on TWO SEPARATE local backings, with vertex/edge state
 * puts and gets to both graphs interleaved in logical time.
 *
 * Public surface only:
 *   - dstm_region_register()            -> obtain a second local region id
 *   - dstm_dgraph_create_on_region()    -> back a dgraph on a chosen region
 *   - dstm_dgraph_vertex_put/get()      -> drive vertex-state traffic
 *   - dstm_dgraph_edge_put/get()        -> drive edge-state traffic
 *   - dstm_dgraph_neighbors_get()       -> confirm topology is per-graph
 *
 * Written to FIND BUGS, not to pass:
 *   - every expected value is derived independently from its inputs,
 *   - values are namespaced per region (0x0A.. vs 0x0B..) so any
 *     cross-region contamination is immediately visible,
 *   - operations on both graphs are interleaved at the same timesteps so
 *     any rank-level-vs-region-level state confusion (version state
 *     machine, ninfos keyed by (rank, region), heartbeat clock, barrier)
 *     surfaces,
 *   - every mismatch is reported (no abort-on-first); the process exits
 *     non-zero if anything is wrong.
 *
 * Topology (per rank, identical on both graphs): each rank owns exactly
 * one vertex, with a single outgoing edge to its right neighbor's vertex,
 * forming a directed ring.  Simple, but enough to force remote vertex and
 * edge state traffic across both regions.
 *
 * Run:   mpirun -n 2 ./multi_dgraph      (also try -n 4)
 */

#include <assert.h>
#include <inttypes.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dstm.h"

/* ---- value tagging so contamination is visible --------------------- */
#define REG0_MARK 0x0A000000
#define REG1_MARK 0x0B000000
#define REG_MASK  0x0F000000
/* Encode (region, rank/vertex, kind, timestep) into one int so a value
 * read from the wrong region/vertex/kind/time is obvious on inspection.
 * kind: 0 = vertex state, 1 = edge state. */
#define TAG(mark, who, kind, t)                                               \
        ((int)((mark) | (((who) & 0xff) << 12) | (((kind) & 0xff) << 4) |     \
               ((t) & 0xf)))

/* ---- report-everything scoreboard ---------------------------------- */
static int g_failures = 0;
static int g_checks = 0;

#define CHECK_EQ(got, want, what)                                              \
        do {                                                                   \
                g_checks++;                                                     \
                if ((int)(got) != (int)(want)) {                               \
                        g_failures++;                                          \
                        fprintf(stderr,                                        \
                                "MISMATCH [%-28s] got 0x%08x want 0x%08x "     \
                                "(%s:%d)\n",                                   \
                                (what), (unsigned)(got), (unsigned)(want),     \
                                __FILE__, __LINE__);                          \
                } else {                                                       \
                        fprintf(stderr, "ok       [%-28s] 0x%08x\n", (what),   \
                                (unsigned)(got));                             \
                }                                                              \
        } while (0)

#define CHECK_TRUE(cond, what)                                                 \
        do {                                                                   \
                g_checks++;                                                     \
                if (!(cond)) {                                                 \
                        g_failures++;                                          \
                        fprintf(stderr, "FAILED   [%-28s] (%s:%d)\n", (what),  \
                                __FILE__, __LINE__);                          \
                } else {                                                       \
                        fprintf(stderr, "ok       [%-28s]\n", (what));         \
                }                                                              \
        } while (0)

/* Confirm a value belongs to the region namespace we expect. */
#define CHECK_REGION(val, mark, what)                                          \
        CHECK_EQ((val) & REG_MASK, (mark), what)

/* ---------------------------------------------------------------------
 * Phase 0: register a second local region (public API).
 * ------------------------------------------------------------------- */
static int phase0_register(int rank)
{
        (void)rank;

        CHECK_EQ(dstm_get_local_region_id(), 0, "phase0.default_region_id");
        CHECK_TRUE(dstm_get_local_region(0) != NULL, "phase0.region0_exists");

        int rid = -1;
        int rc = dstm_region_register(1 << 20 /* 1 MiB */, &rid);
        CHECK_EQ(rc, DSTM_SUCCESS, "phase0.register_rc");
        CHECK_EQ(rid, 1, "phase0.second_region_id");

        CHECK_TRUE(dstm_get_local_region(1) != NULL, "phase0.region1_exists");
        CHECK_TRUE(dstm_get_local_region(0) != dstm_get_local_region(1),
                   "phase0.regions_distinct");

        return rid;
}

/* Build the directed ring on one region and sanity-check its topology.
 * Each rank: 1 local vertex, 1 out-edge to (rank+1)%ws.  Vertex and edge
 * state are both sizeof(int). */
static dstm_dgraph_t *make_ring(int rank, int ws, int region_id)
{
        (void)rank;
        size_t local_vertices    = 1;
        size_t out_degrees[1]    = {1};
        /* single edge dst = right neighbor's global vertex id */
        size_t edge_dsts[1]      = {(size_t)((rank + 1) % ws)};

        dstm_dgraph_t *g = NULL;
        int rc = dstm_dgraph_create_on_region(&g,
                                              sizeof(int),   /* vertex state */
                                              sizeof(int),   /* edge state   */
                                              local_vertices,
                                              out_degrees,
                                              edge_dsts,
                                              NULL,          /* zero vstate  */
                                              NULL,          /* zero estate  */
                                              region_id);
        CHECK_EQ(rc, DSTM_SUCCESS, "make_ring.create_rc");
        assert(g != NULL);

        /* Topology is per-graph and time-invariant: confirm our own
         * vertex has exactly one neighbor, the right one. */
        size_t deg = 0;
        CHECK_EQ(dstm_dgraph_degree_get(g, (size_t)rank, &deg),
                 DSTM_SUCCESS, "make_ring.degree_rc");
        CHECK_EQ((int)deg, 1, "make_ring.degree_val");

        size_t nb[1] = {0};
        size_t cnt = 1;
        CHECK_EQ(dstm_dgraph_neighbors_get(g, (size_t)rank, nb, &cnt),
                 DSTM_SUCCESS, "make_ring.neighbors_rc");
        CHECK_EQ((int)cnt, 1, "make_ring.neighbors_cnt");
        CHECK_EQ((int)nb[0], (rank + 1) % ws, "make_ring.neighbors_val");

        return g;
}

/* ---------------------------------------------------------------------
 * Phase 1: TWO dgraphs on TWO backings, LOCAL vertex state interleaved.
 *
 * Each rank writes its OWN vertex state in both graphs at the same
 * timesteps, then reads its own value back.  Isolates the per-region
 * version state machine: if region_id filtering is wrong, one graph's
 * pending put will gate the other graph's version.
 * ------------------------------------------------------------------- */
static void phase1_local(int rank, int ws)
{
        dstm_dgraph_t *g0 = make_ring(rank, ws, 0);
        dstm_dgraph_t *g1 = make_ring(rank, ws, 1);
        assert(g0 != NULL && g1 != NULL);

        const int STEPS = 4;
        for (int t = 0; t < STEPS; t++) {
                int v0 = TAG(REG0_MARK, rank, 0, t);
                int v1 = TAG(REG1_MARK, rank, 0, t);
                CHECK_EQ(dstm_dgraph_vertex_put(g0, (size_t)rank, &v0, t, 1),
                         DSTM_SUCCESS, "phase1.vput_g0");
                CHECK_EQ(dstm_dgraph_vertex_put(g1, (size_t)rank, &v1, t, 1),
                         DSTM_SUCCESS, "phase1.vput_g1");

                if (t >= 1) {
                        int want_t = t - 1;
                        int r0 = 0, r1 = 0;
                        CHECK_EQ(dstm_dgraph_vertex_get(g0, (size_t)rank, &r0,
                                                        t + 1, -1),
                                 DSTM_SUCCESS, "phase1.vget_g0_rc");
                        CHECK_EQ(dstm_dgraph_vertex_get(g1, (size_t)rank, &r1,
                                                        t + 1, -1),
                                 DSTM_SUCCESS, "phase1.vget_g1_rc");
                        CHECK_EQ(r0, TAG(REG0_MARK, rank, 0, want_t),
                                 "phase1.vget_g0_val");
                        CHECK_EQ(r1, TAG(REG1_MARK, rank, 0, want_t),
                                 "phase1.vget_g1_val");
                        CHECK_REGION(r0, REG0_MARK, "phase1.g0_isolation");
                        CHECK_REGION(r1, REG1_MARK, "phase1.g1_isolation");
                }
        }

        dstm_barrier((dstm_time_t)STEPS + 1);
        dstm_dgraph_destroy(&g0);
        dstm_dgraph_destroy(&g1);

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
                fprintf(stderr, "--- phase1 (local vertex, two dgraphs) done "
                                "---\n");
}

/* ---------------------------------------------------------------------
 * Phase 2: TWO dgraphs on TWO backings, REMOTE vertex state interleaved.
 *
 * Each rank writes its own vertex state in both graphs, then reads its
 * right-neighbor's vertex state from both.  Drives OUT_GET/IN_GET across
 * the wire for two regions at once, exercising the wire header region_id
 * and per-(rank, region) neighbor-clock routing.
 * ------------------------------------------------------------------- */
static void phase2_remote(int rank, int ws)
{
        if (ws < 2) {
                if (rank == 0)
                        fprintf(stderr, "--- phase2 skipped (need >=2) ---\n");
                return;
        }

        dstm_dgraph_t *g0 = make_ring(rank, ws, 0);
        dstm_dgraph_t *g1 = make_ring(rank, ws, 1);
        assert(g0 != NULL && g1 != NULL);

        int peer = (rank + 1) % ws;

        const int STEPS = 4;
        for (int t = 0; t < STEPS; t++) {
                int v0 = TAG(REG0_MARK, rank, 0, t);
                int v1 = TAG(REG1_MARK, rank, 0, t);
                CHECK_EQ(dstm_dgraph_vertex_put(g0, (size_t)rank, &v0, t, 1),
                         DSTM_SUCCESS, "phase2.vput_g0");
                CHECK_EQ(dstm_dgraph_vertex_put(g1, (size_t)rank, &v1, t, 1),
                         DSTM_SUCCESS, "phase2.vput_g1");

                if (t >= 1) {
                        int want_t = t - 1;
                        int r0 = 0, r1 = 0;
                        CHECK_EQ(dstm_dgraph_vertex_get(g0, (size_t)peer, &r0,
                                                        t + 1, -1),
                                 DSTM_SUCCESS, "phase2.vget_g0_rc");
                        CHECK_EQ(dstm_dgraph_vertex_get(g1, (size_t)peer, &r1,
                                                        t + 1, -1),
                                 DSTM_SUCCESS, "phase2.vget_g1_rc");
                        CHECK_EQ(r0, TAG(REG0_MARK, peer, 0, want_t),
                                 "phase2.vget_g0_val");
                        CHECK_EQ(r1, TAG(REG1_MARK, peer, 0, want_t),
                                 "phase2.vget_g1_val");
                        CHECK_REGION(r0, REG0_MARK, "phase2.g0_isolation");
                        CHECK_REGION(r1, REG1_MARK, "phase2.g1_isolation");
                }
        }

        dstm_barrier((dstm_time_t)STEPS + 1);
        dstm_dgraph_destroy(&g0);
        dstm_dgraph_destroy(&g1);

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
                fprintf(stderr, "--- phase2 (remote vertex, two dgraphs) done "
                                "---\n");
}

/* ---------------------------------------------------------------------
 * Phase 3: TWO dgraphs on TWO backings, EDGE state, remote interleaved.
 *
 * The edge (rank -> peer) is OWNED by the source rank (owner of u), so a
 * remote reader is peer's left neighbor.  Each rank writes its own edge
 * state in both graphs, then reads the edge state of the edge OWNED by
 * its right neighbor (peer -> peer2), interleaved.  This stresses edge
 * slot resolution and versioned edge state across two regions at once.
 * ------------------------------------------------------------------- */
static void phase3_edges(int rank, int ws)
{
        if (ws < 2) {
                if (rank == 0)
                        fprintf(stderr, "--- phase3 skipped (need >=2) ---\n");
                return;
        }

        dstm_dgraph_t *g0 = make_ring(rank, ws, 0);
        dstm_dgraph_t *g1 = make_ring(rank, ws, 1);
        assert(g0 != NULL && g1 != NULL);

        int peer  = (rank + 1) % ws;         /* my edge target */
        int peer2 = (rank + 2) % ws;         /* peer's edge target */

        const int STEPS = 4;
        for (int t = 0; t < STEPS; t++) {
                int e0 = TAG(REG0_MARK, rank, 1, t);
                int e1 = TAG(REG1_MARK, rank, 1, t);
                /* edge (rank -> peer), owned locally */
                CHECK_EQ(dstm_dgraph_edge_put(g0, (size_t)rank, (size_t)peer,
                                              &e0, t, 1),
                         DSTM_SUCCESS, "phase3.eput_g0");
                CHECK_EQ(dstm_dgraph_edge_put(g1, (size_t)rank, (size_t)peer,
                                              &e1, t, 1),
                         DSTM_SUCCESS, "phase3.eput_g1");

                if (t >= 1) {
                        int want_t = t - 1;
                        int r0 = 0, r1 = 0;
                        /* edge (peer -> peer2), owned by peer: remote read */
                        CHECK_EQ(dstm_dgraph_edge_get(g0, (size_t)peer,
                                                      (size_t)peer2, &r0,
                                                      t + 1, -1),
                                 DSTM_SUCCESS, "phase3.eget_g0_rc");
                        CHECK_EQ(dstm_dgraph_edge_get(g1, (size_t)peer,
                                                      (size_t)peer2, &r1,
                                                      t + 1, -1),
                                 DSTM_SUCCESS, "phase3.eget_g1_rc");
                        CHECK_EQ(r0, TAG(REG0_MARK, peer, 1, want_t),
                                 "phase3.eget_g0_val");
                        CHECK_EQ(r1, TAG(REG1_MARK, peer, 1, want_t),
                                 "phase3.eget_g1_val");
                        CHECK_REGION(r0, REG0_MARK, "phase3.g0_isolation");
                        CHECK_REGION(r1, REG1_MARK, "phase3.g1_isolation");
                }
        }

        dstm_barrier((dstm_time_t)STEPS + 1);
        dstm_dgraph_destroy(&g0);
        dstm_dgraph_destroy(&g1);

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
                fprintf(stderr, "--- phase3 (remote edges, two dgraphs) done "
                                "---\n");
}

typedef void (*phase_fn)(int rank, int ws);

static void run_phase(dstm_distance_t *dist, int ws, int rank, phase_fn fn)
{
        int rc = dstm_start(dist);
        assert(rc == DSTM_SUCCESS);

        int rid = phase0_register(rank);   /* region 1 registered per session */
        assert(rid == 1);

        fn(rank, ws);

        dstm_fini((dstm_time_t)8);
        MPI_Barrier(MPI_COMM_WORLD);
}

/* ---------------------------------------------------------------------
 * main
 * ------------------------------------------------------------------- */
int main(int argc, char *argv[])
{
        MPI_Init(&argc, &argv);
        dstm_init(&argc, &argv);
        dstm_log_setlevel(DSTM_LOG_DEBUG);

        int rank, ws;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ws);

        /* Complete topology: every rank reaches every other at distance 1,
         * the simplest correct setting for arbitrary remote gets.  Note we
         * deliberately do NOT derive the distance matrix from either graph
         * here -- a shared session matrix must cover the union of all data
         * structures' traffic, and the complete matrix trivially does. */
        dstm_distance_t *dist = NULL;
        int rc = dstm_distance_create_complete(&dist, (size_t)ws);
        assert(rc == DSTM_SUCCESS);

        run_phase(dist, ws, rank, phase1_local);
        run_phase(dist, ws, rank, phase2_remote);
        run_phase(dist, ws, rank, phase3_edges);

        dstm_distance_destroy(&dist);
        dstm_finalize();

        int total_fail = 0, total_checks = 0;
        MPI_Reduce(&g_failures, &total_fail, 1, MPI_INT, MPI_SUM, 0,
                   MPI_COMM_WORLD);
        MPI_Reduce(&g_checks, &total_checks, 1, MPI_INT, MPI_SUM, 0,
                   MPI_COMM_WORLD);

        if (rank == 0) {
                fprintf(stderr,
                        "=== multi_dgraph: %d checks, %d failures "
                        "(world_size=%d) ===\n",
                        total_checks, total_fail, ws);
                if (total_fail == 0)
                        printf("PASS: multi_dgraph (world_size=%d)\n", ws);
                else
                        printf("FAIL: multi_dgraph (%d failures)\n",
                               total_fail);
        }

        MPI_Finalize();
        return (total_fail == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
