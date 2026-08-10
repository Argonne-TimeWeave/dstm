/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <assert.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "dstm.h"

/*******************************************************************************
 * Helpers
 ******************************************************************************/

static int rank;
static int world_size;

/* Open a fresh dstm session so each functional test runs on its own
 * timeline starting at logical time 0.  Mirrors tests/mpi/darray.c. */
static dstm_distance_t *session_begin(void)
{
        dstm_distance_t *dist = NULL;
        int rc = dstm_distance_create_complete(&dist, (size_t)world_size);
        assert(rc == DSTM_SUCCESS);
        rc = dstm_start(dist);
        assert(rc == DSTM_SUCCESS);
        return dist;
}

static void session_end(dstm_distance_t **dist, dstm_time_t last_time)
{
        int rc = dstm_fini(last_time);
        assert(rc == DSTM_SUCCESS);
        dstm_distance_destroy(dist);
}

/*******************************************************************************
 * Ring builder
 *
 * Each rank owns `per_rank` vertices; every vertex has exactly one
 * outgoing edge to (vid + 1) mod total.  This exercises cross-rank
 * edges: the last vertex of each rank points into the next rank
 * (except rank world_size-1's last vertex, which wraps back to 0).
 *
 * vertex_state_size = sizeof(int), edge_state_size = sizeof(int),
 * both zero-initialized by the graph (we pass NULL).
 ******************************************************************************/

static dstm_dgraph_t *build_ring(size_t per_rank)
{
        size_t total = per_rank * (size_t)world_size;
        size_t off = per_rank * (size_t)rank;

        size_t *degrees = malloc(per_rank * sizeof(size_t));
        size_t *dsts = malloc(per_rank * sizeof(size_t));
        assert(degrees != NULL && dsts != NULL);

        for (size_t i = 0; i < per_rank; i++) {
                degrees[i] = 1;
                dsts[i] = (off + i + 1) % total;
        }

        dstm_dgraph_t *g = NULL;
        int rc = dstm_dgraph_create(&g, sizeof(int), sizeof(int), per_rank,
                                    degrees, dsts, NULL, NULL);
        assert(rc == DSTM_SUCCESS);
        assert(g != NULL);

        free(degrees);
        free(dsts);
        return g;
}

/*******************************************************************************
 * Fixed-graph builder
 *
 * Each rank owns 3 vertices with a small hand-crafted local topology
 * that exercises multiple out-edges per vertex and sorted-dst lookup.
 *
 * Local vertex layout (offsets relative to local base `off`):
 *   local 0 (off+0): out-degree 2 -> {off+1, off+2}
 *   local 1 (off+1): out-degree 1 -> {off+2}
 *   local 2 (off+2): out-degree 0
 *
 * All destinations are locally owned so tests can be validated without
 * caring about cross-rank ownership.  Total local edges = 3.
 ******************************************************************************/

#define FIXED_PER_RANK ((size_t)3)
#define FIXED_LOCAL_EDGES ((size_t)3)

static dstm_dgraph_t *build_fixed(void)
{
        size_t off = FIXED_PER_RANK * (size_t)rank;

        size_t degrees[FIXED_PER_RANK] = { 2, 1, 0 };
        size_t dsts[FIXED_LOCAL_EDGES] = { off + 1, off + 2, off + 2 };

        dstm_dgraph_t *g = NULL;
        int rc = dstm_dgraph_create(&g, sizeof(int), sizeof(int),
                                    FIXED_PER_RANK, degrees, dsts, NULL, NULL);
        assert(rc == DSTM_SUCCESS);
        assert(g != NULL);
        return g;
}

/*******************************************************************************
 * create_destroy tests
 ******************************************************************************/

/* Basic create + destroy on a ring. */
static void test_create_destroy_basic(void)
{
        dstm_dgraph_t *g = build_ring(4);
        assert(dstm_dgraph_local_vertex_count(g) == 4);
        assert(dstm_dgraph_vertex_count(g) == 4 * (size_t)world_size);
        assert(dstm_dgraph_local_edge_count(g) == 4);
        assert(dstm_dgraph_edge_count(g) == 4 * (size_t)world_size);
        dstm_dgraph_destroy(&g);
        assert(g == NULL);
}

/* dstm_dgraph_destroy must not crash on NULL or &NULL. */
static void test_destroy_null_safety(void)
{
        dstm_dgraph_destroy(NULL);
        dstm_dgraph_t *g = NULL;
        dstm_dgraph_destroy(&g);
        assert(g == NULL);
}

/* create_equal: total_vertices split evenly, sum of local counts
 * matches the total passed in. */
static void test_create_equal(void)
{
        size_t total = (size_t)world_size * 5 + 2;

        /* Compute what this rank will own using the same formula
         * dstm_darray_create_equal uses: first `total % world_size`
         * ranks get one extra. */
        size_t base = total / (size_t)world_size;
        size_t rem = total % (size_t)world_size;
        size_t local = base + ((size_t)rank < rem ? 1 : 0);

        /* One self-free ring on locally-owned vertices: everyone's
         * last vertex has zero degree.  Keeps this test independent
         * of cross-rank owner formulas. */
        size_t *degrees = calloc(local, sizeof(size_t));
        assert(degrees != NULL);
        /* No edges -> degrees all zero, dsts unused. */

        dstm_dgraph_t *g = NULL;
        int rc = dstm_dgraph_create_equal(&g, sizeof(int), sizeof(int),
                                          total, degrees, NULL, NULL, NULL);
        assert(rc == DSTM_SUCCESS);
        assert(dstm_dgraph_local_vertex_count(g) == local);

        size_t sum = 0;
        MPI_Allreduce(&local, &sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                      MPI_COMM_WORLD);
        assert(sum == total);
        assert(dstm_dgraph_vertex_count(g) == total);

        free(degrees);
        dstm_dgraph_destroy(&g);
}

/*******************************************************************************
 * Introspection tests
 ******************************************************************************/

/* Rank 0's local_vertex_offset is 0; every rank's offset matches the
 * prefix sum of previous ranks' local counts. */
static void test_local_vertex_offset(void)
{
        const size_t per_rank = 4;
        dstm_dgraph_t *g = build_ring(per_rank);

        size_t local = dstm_dgraph_local_vertex_count(g);
        size_t off = dstm_dgraph_local_vertex_offset(g);

        size_t expected = 0;
        MPI_Exscan(&local, &expected, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                   MPI_COMM_WORLD);
        if (rank == 0)
                expected = 0;
        assert(off == expected);

        if (rank == 0)
                assert(off == 0);

        dstm_dgraph_destroy(&g);
}

/* vertex_owner returns the rank whose partition contains vertex_id. */
static void test_vertex_owner(void)
{
        const size_t per_rank = 4;
        dstm_dgraph_t *g = build_ring(per_rank);

        for (int r = 0; r < world_size; r++) {
                size_t vid = (size_t)r * per_rank;
                assert(dstm_dgraph_vertex_owner(g, vid) == r);
                assert(dstm_dgraph_vertex_owner(g, vid + per_rank - 1) == r);
        }

        dstm_dgraph_destroy(&g);
}

/*******************************************************************************
 * Vertex put/get invalid-arg tests
 ******************************************************************************/

/* vertex_get with NULL dst must return -DSTM_EINVAL. */
static void test_vertex_get_null_dst(void)
{
        dstm_dgraph_t *g = build_ring(4);
        size_t off = dstm_dgraph_local_vertex_offset(g);
        int rc = dstm_dgraph_vertex_get(g, off, NULL, 0, 0);
        assert(rc == -DSTM_EINVAL);
        dstm_dgraph_destroy(&g);
}

/* vertex_get with positive delta must return -DSTM_EINVAL. */
static void test_vertex_get_positive_delta(void)
{
        dstm_dgraph_t *g = build_ring(4);
        size_t off = dstm_dgraph_local_vertex_offset(g);
        int val = 0;
        int rc = dstm_dgraph_vertex_get(g, off, &val, 0, 1);
        assert(rc == -DSTM_EINVAL);
        dstm_dgraph_destroy(&g);
}

/* vertex_put with negative delta must return -DSTM_EINVAL. */
static void test_vertex_put_negative_delta(void)
{
        dstm_dgraph_t *g = build_ring(4);
        size_t off = dstm_dgraph_local_vertex_offset(g);
        int val = 42;
        int rc = dstm_dgraph_vertex_put(g, off, &val, 0, -1);
        assert(rc == -DSTM_EINVAL);
        dstm_dgraph_destroy(&g);
}

/*******************************************************************************
 * Edge put/get invalid-arg tests
 ******************************************************************************/

/* edge_get for an (u, v) pair that was never declared returns
 * -DSTM_ENOTFOUND.  Use a locally-owned u so we don't accidentally
 * exercise the remote path first. */
static void test_edge_get_not_found(void)
{
        dstm_dgraph_t *g = build_fixed();
        size_t off = FIXED_PER_RANK * (size_t)rank;

        /* Local vertex 2 has zero out-edges: (off+2, off+0) doesn't
         * exist. */
        int val = 0;
        int rc = dstm_dgraph_edge_get(g, off + 2, off + 0, &val, 0, 0);
        assert(rc == -DSTM_ENOTFOUND);

        dstm_dgraph_destroy(&g);
}

/*******************************************************************************
 * Neighborhood queries
 ******************************************************************************/

/* degree_get returns the exact out-degree we declared, and
 * neighbors_get returns them in ascending global-id order. */
static void test_neighbors_fixed(void)
{
        dstm_dgraph_t *g = build_fixed();
        size_t off = FIXED_PER_RANK * (size_t)rank;

        /* local 0 -> {off+1, off+2} (degree 2) */
        size_t deg = 0;
        int rc = dstm_dgraph_degree_get(g, off + 0, &deg);
        assert(rc == DSTM_SUCCESS);
        assert(deg == 2);

        size_t buf[4];
        size_t count = 4;
        memset(buf, 0xFF, sizeof(buf));
        rc = dstm_dgraph_neighbors_get(g, off + 0, buf, &count);
        assert(rc == DSTM_SUCCESS);
        assert(count == 2);
        assert(buf[0] == off + 1);
        assert(buf[1] == off + 2);

        /* local 1 -> {off+2} (degree 1) */
        rc = dstm_dgraph_degree_get(g, off + 1, &deg);
        assert(rc == DSTM_SUCCESS);
        assert(deg == 1);

        count = 4;
        rc = dstm_dgraph_neighbors_get(g, off + 1, buf, &count);
        assert(rc == DSTM_SUCCESS);
        assert(count == 1);
        assert(buf[0] == off + 2);

        /* local 2 -> {} (degree 0) */
        rc = dstm_dgraph_degree_get(g, off + 2, &deg);
        assert(rc == DSTM_SUCCESS);
        assert(deg == 0);

        count = 4;
        rc = dstm_dgraph_neighbors_get(g, off + 2, buf, &count);
        assert(rc == DSTM_SUCCESS);
        assert(count == 0);

        dstm_dgraph_destroy(&g);
}

/* Two-call capacity-query pattern: passing NULL buffer with *io_count
 * == 0 must return -DSTM_ENOMEM and populate *io_count with the
 * required capacity. */
static void test_neighbors_capacity_query(void)
{
        dstm_dgraph_t *g = build_fixed();
        size_t off = FIXED_PER_RANK * (size_t)rank;

        size_t need = 0;
        int rc = dstm_dgraph_neighbors_get(g, off + 0, NULL, &need);
        assert(rc == -DSTM_ENOMEM);
        assert(need == 2);

        size_t *buf = malloc(need * sizeof(size_t));
        assert(buf != NULL);
        rc = dstm_dgraph_neighbors_get(g, off + 0, buf, &need);
        assert(rc == DSTM_SUCCESS);
        assert(need == 2);
        assert(buf[0] == off + 1);
        assert(buf[1] == off + 2);

        free(buf);
        dstm_dgraph_destroy(&g);
}

/*******************************************************************************
 * Functional round-trips
 *
 * Uses the same t_put=0 / t_get=2 / delta=-1 pattern as darray.c.
 ******************************************************************************/

/* Each rank writes state to its own vertices at t=0, reads back at
 * t=2 with delta=-1, and confirms round-trip.  Uses the ring builder
 * so vertex ids span the whole graph. */
static void test_vertex_roundtrip_local(void)
{
        const size_t per_rank = 4;
        const dstm_time_t t_put = 0;
        const dstm_time_t t_get = 2;

        dstm_distance_t *dist = session_begin();
        dstm_dgraph_t *g = build_ring(per_rank);
        size_t off = dstm_dgraph_local_vertex_offset(g);

        for (size_t i = 0; i < per_rank; i++) {
                int val = (int)((off + i) * 10 + 1);
                int rc = dstm_dgraph_vertex_put(g, off + i, &val, t_put, 1);
                assert(rc == DSTM_SUCCESS);
        }

        for (size_t i = 0; i < per_rank; i++) {
                int got = -1;
                int rc = dstm_dgraph_vertex_get(g, off + i, &got, t_get, -1);
                assert(rc == DSTM_SUCCESS);
                int expected = (int)((off + i) * 10 + 1);
                assert(got == expected);
        }

        dstm_dgraph_destroy(&g);
        session_end(&dist, t_put + 1);
}

/* Same round-trip but each rank reads every OTHER rank's vertices at
 * t_get with delta=-1.  Exercises the cross-rank vertex_get path. */
static void test_vertex_roundtrip_cross_rank(void)
{
        const size_t per_rank = 4;
        const dstm_time_t t_put = 0;
        const dstm_time_t t_get = 2;
        dstm_distance_t *dist = session_begin();
        dstm_dgraph_t *g = build_ring(per_rank);

        size_t off = dstm_dgraph_local_vertex_offset(g);

        /* Every rank publishes its local slice. */
        for (size_t i = 0; i < per_rank; i++) {
                int val = (int)((off + i) * 10 + 2);
                int rc = dstm_dgraph_vertex_put(g, off + i, &val, t_put, 1);
                assert(rc == DSTM_SUCCESS);
        }

        /* Then every rank reads the FIRST vertex of every rank. */
        for (int r = 0; r < world_size; r++) {
                size_t vid = (size_t)r * per_rank;

                int got = -1;
                int rc = dstm_dgraph_vertex_get(g, vid, &got, t_get, -1);

                int expected = (int)(vid * 10 + 2);
                fprintf(stderr,
                        "[rank %d] GET  vid=%zu -> rc=%d got=%d expected=%d "
                        "%s\n",
                        rank, vid, rc, got, expected,
                        (rc == DSTM_SUCCESS && got == expected)
                                ? "MATCH"
                                : "MISMATCH");
                fflush(stderr);
                assert(rc == DSTM_SUCCESS);
                assert(got == expected);
        }
        dstm_dgraph_destroy(&g);
        session_end(&dist, t_put + 1);

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
                fprintf(stderr,
                        "===== test_vertex_roundtrip_cross_rank DONE =====\n\n");
                fflush(stderr);
        }
}

/* Edge state round-trip on the fixed graph.  Every rank writes state
 * to its 3 local edges at t=0 and reads them back at t=2. */
static void test_edge_roundtrip_local(void)
{
        const dstm_time_t t_put = 0;
        const dstm_time_t t_get = 2;

        dstm_distance_t *dist = session_begin();
        dstm_dgraph_t *g = build_fixed();
        size_t off = FIXED_PER_RANK * (size_t)rank;

        /* Local edges (from build_fixed): (off+0, off+1), (off+0, off+2),
         * (off+1, off+2). */
        struct { size_t u, v; int val; } edges[FIXED_LOCAL_EDGES] = {
                { off + 0, off + 1, (int)(off * 100 + 11) },
                { off + 0, off + 2, (int)(off * 100 + 12) },
                { off + 1, off + 2, (int)(off * 100 + 22) },
        };

        for (size_t i = 0; i < FIXED_LOCAL_EDGES; i++) {
                int rc = dstm_dgraph_edge_put(g, edges[i].u, edges[i].v,
                                              &edges[i].val, t_put, 1);
                assert(rc == DSTM_SUCCESS);
        }

        for (size_t i = 0; i < FIXED_LOCAL_EDGES; i++) {
                int got = -1;
                int rc = dstm_dgraph_edge_get(g, edges[i].u, edges[i].v,
                                              &got, t_get, -1);
                assert(rc == DSTM_SUCCESS);
                assert(got == edges[i].val);
        }

        dstm_dgraph_destroy(&g);
        session_end(&dist, t_put + 1);
}

/* vertex_get_or_default: at negative effective time (time + delta < 0)
 * the call returns the default value regardless of what was put. */
static void test_vertex_get_or_default(void)
{
        const dstm_time_t t_put = 5;
        dstm_distance_t *dist = session_begin();
        dstm_dgraph_t *g = build_ring(4);
        size_t off = dstm_dgraph_local_vertex_offset(g);

        int published = 999;
        int rc = dstm_dgraph_vertex_put(g, off, &published, t_put, 1);
        assert(rc == DSTM_SUCCESS);

        /* Effective time = 0 + (-1) = -1 -> default value. */
        int dflt = 12345;
        int got = -1;
        rc = dstm_dgraph_vertex_get_or_default(g, off, &got, 0, -1, &dflt);
        assert(rc == DSTM_SUCCESS);
        assert(got == dflt);

        dstm_dgraph_destroy(&g);
        session_end(&dist, t_put + 1);
}

/*******************************************************************************
 * Distance-matrix derivation
 *
 * Covers all three entry points:
 *   dstm_dgraph_distance_from_topology   (pre-session, raw CSR, constant)
 *   dstm_dgraph_derive_distance          (post-hoc, constant default_delta)
 *   dstm_dgraph_derive_distance_weighted (post-hoc, per-edge weight callback)
 *
 * Invariants asserted throughout:
 *   - matrix is rank x rank (size == world_size)
 *   - diagonal is 0
 *   - unreachable rank pairs are -1
 *   - reachable pairs carry the MIN weight over all edges crossing that
 *     rank boundary in either direction
 ******************************************************************************/

/* Fetch cell (i, j) from a derived distance matrix with bounds checks. */
static int dist_at(dstm_distance_t *d, int i, int j)
{
        assert(d != NULL);
        assert((size_t)dstm_distance_size(d) == (size_t)world_size);
        return dstm_distance_get(d, i, j);
}

/* Assert the universal structural invariants: square, zero diagonal. */
static void assert_distance_shape(dstm_distance_t *d)
{
        assert((size_t)dstm_distance_size(d) == (size_t)world_size);
        for (int i = 0; i < world_size; i++)
                assert(dist_at(d, i, i) == 0);
}

/* --- from_topology: pre-session, constant default_delta ------------------
 *
 * Build the SAME ring topology that build_ring() uses, but describe it
 * with the raw local CSR arrays that distance_from_topology consumes.
 * This runs with no live graph and no started session. Every rank's
 * single vertex points to the next rank's base vertex, so the induced
 * rank graph is itself a directed ring; symmetrized + min-reduced it
 * couples each rank to its two ring neighbors at exactly default_delta,
 * and leaves all other pairs unreachable (-1).
 *
 * Requires world_size >= 3 for the "far" pairs to be genuinely
 * unreachable; for world_size < 3 every pair is a neighbor and we only
 * check shape + reachability. */
static void test_distance_from_topology_ring(void)
{
        const int    default_delta = 5;
        const size_t per_rank      = 1; /* one vertex per rank */
        size_t total               = per_rank * (size_t)world_size;

        /* One vertex, out-degree 1, pointing at the next rank's vertex. */
        size_t degrees[1] = { 1 };
        size_t dsts[1]    = { (per_rank * (size_t)rank + 1) % total };

        dstm_distance_t *d = NULL;
        int rc = dstm_dgraph_distance_from_topology(&d, per_rank, degrees,
                                                    dsts, default_delta);
        assert(rc == DSTM_SUCCESS);
        assert(d != NULL);
        assert_distance_shape(d);

        for (int i = 0; i < world_size; i++) {
                for (int j = 0; j < world_size; j++) {
                        if (i == j)
                                continue;

                        int prev = (i - 1 + world_size) % world_size;
                        int next = (i + 1) % world_size;
                        int adjacent = (j == prev || j == next);

                        int cell = dist_at(d, i, j);
                        if (adjacent) {
                                assert(cell == default_delta);
                        } else if (world_size >= 3) {
                                /* Genuinely non-adjacent only exists
                                 * for a ring of 3+ ranks. */
                                assert(cell == -1);
                        }
                }
        }

        dstm_distance_destroy(&d);
}

/* --- from_topology: EINVAL surface ---------------------------------------
 * NULL ret, non-positive default_delta, and NULL degrees with
 * local_vertices > 0 must all be rejected before any collective. */
static void test_distance_from_topology_einval(void)
{
        dstm_distance_t *d = NULL;
        size_t degrees[1]  = { 0 };

        assert(dstm_dgraph_distance_from_topology(NULL, 1, degrees, NULL, 5)
               == -DSTM_EINVAL);
        assert(dstm_dgraph_distance_from_topology(&d, 1, degrees, NULL, 0)
               == -DSTM_EINVAL);
        assert(dstm_dgraph_distance_from_topology(&d, 1, NULL, NULL, 5)
               == -DSTM_EINVAL);
        assert(d == NULL);
}

/* --- derive_distance: constant default_delta over a built ring ----------
 *
 * derive_distance(g, ...) with the ring graph must produce the SAME
 * matrix as from_topology on the equivalent CSR, because a NULL-callback
 * weighted derive is exactly the constant-default_delta case. We assert
 * the ring coupling directly rather than comparing objects (there is no
 * matrix-equality API), which also confirms the two entry points agree. */
static void test_derive_distance_ring(void)
{
        const int    default_delta = 7;
        const size_t per_rank      = 1;

        /* derive_distance issues no puts, but build_ring/create does, so
         * we need a live session. */
        dstm_distance_t *sess = session_begin();
        dstm_dgraph_t   *g    = build_ring(per_rank);

        dstm_distance_t *d = NULL;
        int rc = dstm_dgraph_derive_distance(g, &d, default_delta);
        assert(rc == DSTM_SUCCESS);
        assert(d != NULL);
        assert_distance_shape(d);

        for (int i = 0; i < world_size; i++) {
                for (int j = 0; j < world_size; j++) {
                        if (i == j)
                                continue;
                        int prev = (i - 1 + world_size) % world_size;
                        int next = (i + 1) % world_size;
                        int adjacent = (j == prev || j == next);
                        int cell = dist_at(d, i, j);
                        if (adjacent)
                                assert(cell == default_delta);
                        else if (world_size >= 3)
                                assert(cell == -1);
                }
        }

        dstm_distance_destroy(&d);
        dstm_dgraph_destroy(&g);
        session_end(&sess, 0);
}

/* --- derive_distance: EINVAL surface ------------------------------------- */
static void test_derive_distance_einval(void)
{
        dstm_distance_t *sess = session_begin();
        dstm_dgraph_t   *g    = build_ring(1);
        dstm_distance_t *d    = NULL;

        assert(dstm_dgraph_derive_distance(NULL, &d, 5) == -DSTM_EINVAL);
        assert(dstm_dgraph_derive_distance(g, NULL, 5) == -DSTM_EINVAL);
        assert(dstm_dgraph_derive_distance(g, &d, 0) == -DSTM_EINVAL);
        assert(d == NULL);

        dstm_dgraph_destroy(&g);
        session_end(&sess, 0);
}

/* --- weighted callback: NULL fn reproduces the constant case ------------- */
static void test_derive_weighted_null_matches_constant(void)
{
        const int default_delta = 9;

        dstm_distance_t *sess = session_begin();
        dstm_dgraph_t   *g    = build_ring(1);

        dstm_distance_t *dc = NULL; /* via derive_distance   */
        dstm_distance_t *dw = NULL; /* via weighted, fn=NULL */

        int rc = dstm_dgraph_derive_distance(g, &dc, default_delta);
        assert(rc == DSTM_SUCCESS);
        rc = dstm_dgraph_derive_distance_weighted(g, &dw, default_delta,
                                                  NULL, NULL);
        assert(rc == DSTM_SUCCESS);

        assert((size_t)dstm_distance_size(dc) ==
               (size_t)dstm_distance_size(dw));
        for (int i = 0; i < world_size; i++)
                for (int j = 0; j < world_size; j++)
                        assert(dist_at(dc, i, j) == dist_at(dw, i, j));

        dstm_distance_destroy(&dc);
        dstm_distance_destroy(&dw);
        dstm_dgraph_destroy(&g);
        session_end(&sess, 0);
}

/* Weight callback: distance = the edge's initial (seeded) int weight.
 * Reads static edge state only -- consistent with distance_from_topology,
 * which has no session and no dynamic state to read. */
static int edge_weight_from_state(size_t src_vid, size_t dst_vid,
                                  const void *src_vs, const void *dst_vs,
                                  const void *edge_s, void *user)
{
        (void)src_vid; (void)dst_vid; (void)src_vs; (void)dst_vs; (void)user;
        assert(edge_s != NULL);
        int w = *(const int *)edge_s;
        assert(w > 0);
        return w;
}

/* Fixed graph, but with initial edge weights SEEDED at create time
 * (local_edge_states), not written via edge_put. The weight callback reads
 * this static state. All edges are intra-rank, so the induced rank matrix
 * has no cross-rank coupling: diagonal 0, everything else -1, independent
 * of the weights. This isolates "weights are read from static state" from
 * "cross-rank min-reduction". */
static void test_derive_weighted_local_only(void)
{
        size_t off = FIXED_PER_RANK * (size_t)rank;

        size_t degrees[FIXED_PER_RANK] = { 2, 1, 0 };
        size_t dsts[FIXED_LOCAL_EDGES] = { off + 1, off + 2, off + 2 };

        /* Seed distinct positive weights, in edge-slot order. Slot order
         * matches CSR: vertex0's edges {off+1, off+2} then vertex1's {off+2}. */
        int edge_states[FIXED_LOCAL_EDGES] = { 3, 8, 4 };

        dstm_distance_t *sess = session_begin();

        dstm_dgraph_t *g = NULL;
        int rc = dstm_dgraph_create(&g, sizeof(int), sizeof(int),
                                    FIXED_PER_RANK, degrees, dsts,
                                    NULL, edge_states);
        assert(rc == DSTM_SUCCESS);
        assert(g != NULL);

        dstm_distance_t *d = NULL;
        rc = dstm_dgraph_derive_distance_weighted(
                g, &d, /*default_delta*/ 1, edge_weight_from_state, NULL);
        assert(rc == DSTM_SUCCESS);
        assert(d != NULL);
        assert_distance_shape(d);

        if (world_size >= 2) {
                for (int i = 0; i < world_size; i++)
                        for (int j = 0; j < world_size; j++)
                                if (i != j)
                                        assert(dist_at(d, i, j) == -1);
        }

        dstm_distance_destroy(&d);
        dstm_dgraph_destroy(&g);
        session_end(&sess, 0);   /* no puts issued beyond create's seed */
}

/* --- weighted callback: bad weight (<= 0) is rejected -------------------- */
static int edge_weight_bad(size_t src_vid, size_t dst_vid,
                           const void *src_vs, const void *dst_vs,
                           const void *edge_s, void *user)
{
        (void)src_vid; (void)dst_vid; (void)src_vs; (void)dst_vs;
        (void)edge_s;  (void)user;
        return 0; /* invalid: weights must be > 0 */
}

/* Only meaningful when this rank actually has at least one edge to walk,
 * which the ring guarantees (one out-edge per vertex). A callback that
 * returns <= 0 must make derive_weighted fail with -DSTM_EINVAL. */
static void test_derive_weighted_bad_weight(void)
{
        dstm_distance_t *sess = session_begin();
        dstm_dgraph_t   *g    = build_ring(1);

        dstm_distance_t *d = NULL;
        int rc = dstm_dgraph_derive_distance_weighted(g, &d, 1,
                                                      edge_weight_bad, NULL);
        assert(rc == -DSTM_EINVAL);
        assert(d == NULL);

        dstm_dgraph_destroy(&g);
        session_end(&sess, 0);
}


/* Wrap a void() test in its own dstm session so that every call to
 * dstm_dgraph_create publishes its initial state at (t=0, delta=0) on
 * a fresh timeline.  Multiple graphs in a single session collide at
 * (t=0, delta=0) because dstm's put-before-get phase rule forbids
 * additional puts at a (time, delta) after any get at that same
 * (time, delta) has been issued. */
static void run_isolated(void (*test)(void))
{
        dstm_distance_t *dist = session_begin();
        test();
        session_end(&dist, 0);
}

int main(int argc, char *argv[])
{
        MPI_Init(&argc, &argv);
        dstm_init(&argc, &argv);
        dstm_log_setlevel(DSTM_LOG_ERROR);

        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        /* Every test gets its own start/fini session.  This is required
         * because dstm_dgraph_create publishes its initial state via
         * puts at (t=0, delta=0), and DSTM's put-before-get phase rule
         * makes it unsafe to issue more puts at that (time, delta)
         * after the first get at (t=0, delta=0) has fired.  Sharing a
         * session across multiple graph instances causes silent state
         * contamination. */

        /* create/destroy */
        run_isolated(test_create_destroy_basic);
        run_isolated(test_destroy_null_safety);
        run_isolated(test_create_equal);

        /* introspection */
        run_isolated(test_local_vertex_offset);
        run_isolated(test_vertex_owner);

        /* vertex invalid args */
        run_isolated(test_vertex_get_null_dst);
        run_isolated(test_vertex_get_positive_delta);
        run_isolated(test_vertex_put_negative_delta);

        /* edge invalid args */
        run_isolated(test_edge_get_not_found);

        /* neighborhood (time-invariant) */
        run_isolated(test_neighbors_fixed);
        run_isolated(test_neighbors_capacity_query);

        /* Functional round-trips already open their own sessions
         * internally, so call them directly. */
        test_vertex_roundtrip_local();
        test_vertex_roundtrip_cross_rank();
        test_edge_roundtrip_local();
        test_vertex_get_or_default();

        /* distance derivation: from_topology runs WITHOUT a session. */
        test_distance_from_topology_ring();
        test_distance_from_topology_einval();

        /* distance derivation: post-hoc paths open their own sessions. */
        test_derive_distance_ring();
        test_derive_distance_einval();
        test_derive_weighted_null_matches_constant();
        test_derive_weighted_local_only();
        test_derive_weighted_bad_weight();

        dstm_finalize();
        MPI_Finalize();
        return 0;
}
