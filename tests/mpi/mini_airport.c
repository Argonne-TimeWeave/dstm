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
 * Airport mini-simulation on a dgraph.
 *
 * A scaled-down, self-checking version of examples/airport, rebuilt on
 * dstm_dgraph instead of a dstm_darray of gate slots. Its purpose is to
 * exercise the dgraph API end to end and assert a conservation
 * invariant, not to benchmark anything.
 *
 * TOPOLOGY
 *   An N x N toroidal grid of airports, N chosen so that N*N divides
 *   evenly across world_size (see choose_grid_side). Each airport is a
 *   vertex with exactly four out-edges to its N/S/E/W neighbors.
 *   Contiguous rank-order partitioning: rank r owns global ids
 *   [r*per_rank, (r+1)*per_rank).
 *
 * EDGE ORIENTATION  --  the important part
 *   dgraph stores edge (u, v) on *u's owner*. So we read an edge slot as
 *
 *       edge (x, y) == "inbox at airport x, for planes arriving from y"
 *
 *   which puts every operation on the legal side of the ownership
 *   boundary and mirrors the reference simulation's gate indexing
 *   (gates[dest][src]):
 *
 *     consume  edge_get(u, v, t, 0)      u is local  -> delta 0 legal
 *     produce  edge_put(v, u, t, +1)     v may be remote -> delta +1 legal
 *
 *   The torus is symmetric, so edge (v, u) exists whenever (u, v) does;
 *   no extra edges are declared for the reverse direction.
 *
 * PER-TICK SCHEDULE  (matches airport_timestep)
 *   1. Absorb every local inbox edge at (t, delta=0) into a per-airport
 *      pending-landing queue.
 *   2. Land up to LANDINGS_PER_TICK planes per airport, pending -> ground.
 *   3. Depart every plane on the ground into a neighbor's inbox at
 *      (t, delta=+1), so it is visible to that neighbor at tick t+1.
 *      Every out-direction is published each tick, empty or not, so a
 *      version always exists for the consumer and no edge version ever
 *      has two writers.
 *
 * INVARIANT
 *   Total planes (on ground + pending + in flight on edges) is conserved
 *   and equals total_airports * INITIAL_PLANES.
 *
 * REQUIRES
 *   dstm_dgraph_distance_from_topology(), which derives a distance
 *   matrix from raw CSR arrays with no live session. It is needed
 *   because dstm_dgraph_create issues dstm_put calls and therefore
 *   requires an already-started session -- a matrix derived from a
 *   finished graph can never be the matrix that graph's session was
 *   started with.
 */

#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dstm.h"

#include "internal/distance.h"

/*******************************************************************************
 * Tunables
 ******************************************************************************/

/* Max planes one edge can carry in one tick. Generously oversized for
 * the traffic these settings produce. */
#define BUNDLE_CAPACITY   32

/* Planes each airport can land per tick (reference uses 3). */
#define LANDINGS_PER_TICK 2

/* Planes each airport starts with. */
#define INITIAL_PLANES    2

/* Ticks to run. */
#define NUM_TIMESTEPS     8

/* Simulated-time distance between connected ranks. Must be >= 1: with 0,
 * an edge version would become discardable at exactly the tick its
 * consumer reads it. Producers write with this delta. */
#define EDGE_DELTA        1

/* Every vertex has four out-edges on a torus. */
#define DEGREE            4

/*******************************************************************************
 * Error handling
 ******************************************************************************/

static int g_rank;
static int g_world_size;

#define CHECK(expr, what)                                                     \
        do {                                                                  \
                int _rc = (expr);                                             \
                if (_rc != DSTM_SUCCESS) {                                    \
                        fprintf(stderr,                                       \
                                "[rank %d] %s failed: rc=%d at %s:%d\n",      \
                                g_rank, (what), _rc, __FILE__, __LINE__);     \
                        fflush(stderr);                                       \
                        MPI_Abort(MPI_COMM_WORLD, 1);                         \
                }                                                             \
        } while (0)

#define REQUIRE(cond, msg)                                                    \
        do {                                                                  \
                if (!(cond)) {                                                \
                        fprintf(stderr,                                       \
                                "[rank %d] check failed: %s at %s:%d\n",      \
                                g_rank, (msg), __FILE__, __LINE__);           \
                        fflush(stderr);                                       \
                        MPI_Abort(MPI_COMM_WORLD, 1);                         \
                }                                                             \
        } while (0)

/*******************************************************************************
 * State
 ******************************************************************************/

/* Per-airport vertex state. Mirrored on the host so we can iterate over
 * all local airports cheaply; the versioned copy is written every tick
 * and read back once at the end to exercise the vertex path. */
typedef struct {
        int       on_ground;
        int       pending;
        long long landed;
        long long departed;
} airport_state_t;

/* Per-edge in-flight bundle: planes crossing this edge in one tick. */
typedef struct {
        int       n_planes;
        long long plane_ids[BUNDLE_CAPACITY];
} edge_bundle_t;

typedef struct {
        int              grid;
        size_t           per_rank;
        size_t           off;      /* local_vertex_offset */
        airport_state_t *host;     /* length per_rank */
        long long        next_plane_id;
} sim_ctx_t;

/*******************************************************************************
 * Torus helpers
 ******************************************************************************/

/* Global id of gid's neighbor in direction dir (0=N, 1=S, 2=E, 3=W) on
 * an N x N torus of side `grid`. */
static size_t torus_neighbor(size_t gid, int grid, int dir)
{
        int row = (int)(gid / (size_t)grid);
        int col = (int)(gid % (size_t)grid);

        switch (dir) {
        case 0: row = (row - 1 + grid) % grid; break;
        case 1: row = (row + 1) % grid;        break;
        case 2: col = (col + 1) % grid;        break;
        case 3: col = (col - 1 + grid) % grid; break;
        }
        return (size_t)row * (size_t)grid + (size_t)col;
}

/* Smallest N such that N >= 4 (so the four torus neighbors are always
 * distinct), N*N is divisible by world_size, and each rank gets at
 * least MIN_PER_RANK airports. */
static int choose_grid_side(int world_size)
{
        const int MIN_PER_RANK = 4;
        int min_total = world_size * MIN_PER_RANK;

        for (int n = 4; n <= 128; n++) {
                int total = n * n;
                if (total < min_total)          continue;
                if (total % world_size != 0)    continue;
                return n;
        }
        return 0;   /* caller reports; unreachable for world_size <= 32 */
}

/*******************************************************************************
 * Distance matrix
 *
 * Rank r owns airports [r*per_rank, (r+1)*per_rank). Two ranks are
 * connected iff some airport on one has a torus neighbor on the other.
 * Purely local: the topology is a closed-form function of grid and
 * per_rank, so every rank derives the same matrix with no communication.
 *
 * Mirrors build_rank_adjacency() in examples/airport/main.c, including
 * its use of internal/distance.h, because dstm_distance_create_from_matrix
 * rejects our input with -DSTM_EINVAL (see notes at the call site).
 ******************************************************************************/

static dstm_distance_t *build_torus_distance(int grid, size_t per_rank)
{
        dstm_distance_t *d = NULL;
        int              n = g_world_size;

        /* Start from a constructor known to work in this build, then
         * overwrite every entry. */
        CHECK(dstm_distance_create_complete(&d, (size_t)n),
              "distance_create_complete");
        REQUIRE(d != NULL && d->distances != NULL, "distance matrix alloc");

        for (int i = 0; i < n * n; i++)
                d->distances[i] = -1;

        /* Local ranks are distance 0, per the DSTM distance-matrix
         * encoding. NOTE: examples/airport leaves the diagonal at -1 and
         * apparently still runs; if this line causes trouble, drop it and
         * you match the reference exactly. */
        for (int i = 0; i < n; i++)
                d->distances[i * n + i] = 0;

        size_t total = (size_t)grid * (size_t)grid;
        for (size_t gid = 0; gid < total; gid++) {
                int owner = (int)(gid / per_rank);

                for (int dir = 0; dir < DEGREE; dir++) {
                        size_t ngid   = torus_neighbor(gid, grid, dir);
                        int    nowner = (int)(ngid / per_rank);

                        if (nowner == owner)
                                continue;   /* same rank; no DSTM traffic */

                        d->distances[owner  * n + nowner] = EDGE_DELTA;
                        d->distances[nowner * n + owner]  = EDGE_DELTA;
                }
        }

        if (g_rank == 0) {
                int reachable = 0;
                for (int i = 0; i < n; i++)
                        for (int j = 0; j < n; j++)
                                if (i != j && d->distances[i * n + j] > 0)
                                        reachable++;
                printf("[airport_mini] rank adjacency: %d/%d off-diagonal "
                       "pairs reachable at delta=%d\n",
                       reachable, n * (n - 1), EDGE_DELTA);
                fflush(stdout);
        }
        return d;
}

/*******************************************************************************
 * Topology construction
 *
 * Builds packed CSR arrays for this rank's local vertices. Called before
 * dstm_start (to derive the distance matrix) and reused for
 * dstm_dgraph_create, so both see byte-identical topology.
 ******************************************************************************/

static void build_local_topology(int grid, size_t per_rank, size_t off,
                                 size_t **out_degrees, size_t **out_dsts)
{
        size_t *degrees = malloc(per_rank * sizeof(size_t));
        size_t *dsts    = malloc(per_rank * DEGREE * sizeof(size_t));

        REQUIRE(degrees != NULL && dsts != NULL, "topology malloc");

        for (size_t i = 0; i < per_rank; i++) {
                size_t gid = off + i;
                size_t n[DEGREE];

                degrees[i] = DEGREE;
                for (int d = 0; d < DEGREE; d++)
                        n[d] = torus_neighbor(gid, grid, d);

                /* dgraph requires strictly ascending destinations within
                 * each vertex's slice. Insertion sort of four elements. */
                for (int a = 0; a < DEGREE; a++) {
                        for (int b = a + 1; b < DEGREE; b++) {
                                if (n[b] < n[a]) {
                                        size_t tmp = n[a];
                                        n[a] = n[b];
                                        n[b] = tmp;
                                }
                        }
                }
                for (int a = 1; a < DEGREE; a++)
                        REQUIRE(n[a] > n[a - 1],
                                "duplicate torus neighbor (grid too small?)");

                for (int a = 0; a < DEGREE; a++)
                        dsts[i * DEGREE + a] = n[a];
        }

        *out_degrees = degrees;
        *out_dsts    = dsts;
}

/*******************************************************************************
 * Introspection smoke test
 ******************************************************************************/

static void check_introspection(dstm_dgraph_t *g, const sim_ctx_t *ctx,
                                size_t total_vertices)
{
        REQUIRE(dstm_dgraph_vertex_count(g) == total_vertices,
                "vertex_count");
        REQUIRE(dstm_dgraph_local_vertex_count(g) == ctx->per_rank,
                "local_vertex_count");
        REQUIRE(dstm_dgraph_local_vertex_offset(g) == ctx->off,
                "local_vertex_offset");
        REQUIRE(dstm_dgraph_edge_count(g) == total_vertices * DEGREE,
                "edge_count");
        REQUIRE(dstm_dgraph_local_edge_count(g) == ctx->per_rank * DEGREE,
                "local_edge_count");

        /* Ownership agrees with contiguous rank-order partitioning. */
        for (size_t i = 0; i < ctx->per_rank; i++)
                REQUIRE(dstm_dgraph_vertex_owner(g, ctx->off + i) == g_rank,
                        "vertex_owner on local vertex");

        /* Self is always a neighbor rank; the neighbor list is sorted
         * and every member reports as a neighbor. */
        REQUIRE(dstm_dgraph_is_neighbor_rank(g, g_rank) == 1,
                "is_neighbor_rank(self)");

        size_t nrc = 0;
        int rc = dstm_dgraph_neighbor_ranks_get(g, NULL, &nrc);
        REQUIRE(rc == -DSTM_ENOMEM && nrc >= 1,
                "neighbor_ranks_get capacity probe");

        int *nranks = malloc(nrc * sizeof(int));
        REQUIRE(nranks != NULL, "nranks malloc");
        CHECK(dstm_dgraph_neighbor_ranks_get(g, nranks, &nrc),
              "neighbor_ranks_get");
        for (size_t i = 0; i < nrc; i++) {
                REQUIRE(dstm_dgraph_is_neighbor_rank(g, nranks[i]) == 1,
                        "neighbor list member is a neighbor");
                if (i > 0)
                        REQUIRE(nranks[i] > nranks[i - 1],
                                "neighbor list sorted");
        }
        free(nranks);

        /* Degree and the two-call neighbors_get pattern on a local
         * vertex, plus symmetry of the torus (needed for the inbox
         * orientation to be well defined). */
        size_t u = ctx->off;
        size_t deg = 0;
        CHECK(dstm_dgraph_degree_get(g, u, &deg), "degree_get");
        REQUIRE(deg == DEGREE, "degree == 4");

        size_t cap = 0;
        rc = dstm_dgraph_neighbors_get(g, u, NULL, &cap);
        REQUIRE(rc == -DSTM_ENOMEM && cap == DEGREE,
                "neighbors_get capacity probe");

        size_t nbrs[DEGREE];
        cap = DEGREE;
        CHECK(dstm_dgraph_neighbors_get(g, u, nbrs, &cap), "neighbors_get");
        REQUIRE(cap == DEGREE, "neighbors_get count");
        for (int k = 1; k < DEGREE; k++)
                REQUIRE(nbrs[k] > nbrs[k - 1], "neighbors ascending");

        /* An undeclared edge must be reported, not silently accepted.
         * A vertex is never its own torus neighbor here. */
        edge_bundle_t tmp;
        rc = dstm_dgraph_edge_get(g, u, u, &tmp, 0, 0);
        REQUIRE(rc == -DSTM_ENOTFOUND, "self-edge should not exist");

        /* get_or_default short-circuits on negative effective time
         * without touching the network. */
        airport_state_t fallback = { .on_ground = -7 };
        airport_state_t probe    = { 0 };
        CHECK(dstm_dgraph_vertex_get_or_default(g, u, &probe, 0, -1,
                                                &fallback),
              "vertex_get_or_default(negative time)");
        REQUIRE(probe.on_ground == -7, "get_or_default returned fallback");
}

/*******************************************************************************
 * One simulation tick
 ******************************************************************************/

static void airport_tick(dstm_dgraph_t *g, sim_ctx_t *ctx, dstm_time_t t)
{
        size_t nbrs[DEGREE];
        size_t nc;

        /* --- Phase 1: absorb arrivals. -----------------------------------
         *
         * Inbox edge (u, v) is owned by u's owner == self, so this is a
         * local read and delta 0 is legal. At t == 0 the versions read
         * are the empty bundles published by dstm_dgraph_create. */
        {
                edge_bundle_t buf;

                for (size_t i = 0; i < ctx->per_rank; i++) {
                        size_t u = ctx->off + i;

                        nc = DEGREE;
                        CHECK(dstm_dgraph_neighbors_get(g, u, nbrs, &nc),
                              "phase1 neighbors_get");
                        REQUIRE(nc == DEGREE, "phase1 degree");

                        for (size_t k = 0; k < nc; k++) {
                                CHECK(dstm_dgraph_edge_get(g, u, nbrs[k],
                                                           &buf, t, 0),
                                      "phase1 edge_get (inbox)");
                                REQUIRE(buf.n_planes >= 0 &&
                                        buf.n_planes <= BUNDLE_CAPACITY,
                                        "phase1 bundle size sane");
                                ctx->host[i].pending += buf.n_planes;
                        }
                }
        }

        /* --- Phase 2: land up to the per-tick budget. --------------------- */
        for (size_t i = 0; i < ctx->per_rank; i++) {
                int budget = LANDINGS_PER_TICK;
                int can = ctx->host[i].pending < budget
                          ? ctx->host[i].pending : budget;

                if (can == 0)
                        continue;
                ctx->host[i].pending   -= can;
                ctx->host[i].on_ground += can;
                ctx->host[i].landed    += can;
        }

        /* --- Phase 3: depart into neighbors' inboxes. --------------------
         *
         * Writing edge (v, u) targets v's owner, which may be remote, so
         * delta must be > 0. Every direction is published whether or not
         * planes crossed, so each edge version has exactly one writer and
         * always exists for its consumer. */
        for (size_t i = 0; i < ctx->per_rank; i++) {
                size_t        u = ctx->off + i;
                edge_bundle_t out[DEGREE];
                int           planes = ctx->host[i].on_ground;

                memset(out, 0, sizeof(out));

                nc = DEGREE;
                CHECK(dstm_dgraph_neighbors_get(g, u, nbrs, &nc),
                      "phase3 neighbors_get");
                REQUIRE(nc == DEGREE, "phase3 degree");

                for (int p = 0; p < planes; p++) {
                        /* Deterministic round-robin over directions. */
                        int d = (int)((t + (dstm_time_t)p +
                                       (dstm_time_t)i) & 3);
                        int tries = 0;

                        while (out[d].n_planes >= BUNDLE_CAPACITY &&
                               tries < DEGREE) {
                                d = (d + 1) & 3;
                                tries++;
                        }
                        if (out[d].n_planes >= BUNDLE_CAPACITY)
                                break;   /* all bundles full; stay grounded */

                        out[d].plane_ids[out[d].n_planes++] =
                                ctx->next_plane_id++;
                        ctx->host[i].on_ground--;
                        ctx->host[i].departed++;
                }

                for (int d = 0; d < DEGREE; d++)
                        CHECK(dstm_dgraph_edge_put(g, nbrs[d], u, &out[d],
                                                   t, EDGE_DELTA),
                              "phase3 edge_put (neighbor inbox)");

                /* Must target t+1, not t: phase 1 already issued gets at
                 * effective time t, which sealed that version against
                 * further writes. Every put in this tick lands on t+1,
                 * every get on t. */
                CHECK(dstm_dgraph_vertex_put(g, u, &ctx->host[i], t, EDGE_DELTA),
                      "phase3 vertex_put");
        }
}

/*******************************************************************************
 * Final accounting
 *
 * After ticks 0..T-1, edge versions 1..T have been written and versions
 * 0..T-1 absorbed, so exactly version T is outstanding on every edge.
 * All ranks issue these reads at originating time T, which is what makes
 * version T readable (its writers have all advanced past T-1).
 ******************************************************************************/

static long long drain_local_inflight(dstm_dgraph_t *g, const sim_ctx_t *ctx,
                                      dstm_time_t t_last)
{
        edge_bundle_t buf;
        size_t        nbrs[DEGREE];
        size_t        nc;
        long long     inflight = 0;

        for (size_t i = 0; i < ctx->per_rank; i++) {
                size_t u = ctx->off + i;

                nc = DEGREE;
                CHECK(dstm_dgraph_neighbors_get(g, u, nbrs, &nc),
                      "drain neighbors_get");

                for (size_t k = 0; k < nc; k++) {
                        CHECK(dstm_dgraph_edge_get(g, u, nbrs[k], &buf,
                                                   t_last, 0),
                              "drain edge_get");
                        REQUIRE(buf.n_planes >= 0 &&
                                buf.n_planes <= BUNDLE_CAPACITY,
                                "drain bundle size sane");
                        inflight += buf.n_planes;
                }
        }
        return inflight;
}

/* Read back the versioned vertex state written during the last tick and
 * confirm it matches the host mirror. Tick T-1 wrote at effective time
 * T, so read at (T, delta 0) -- local, current version, same effective
 * time as the edge drain. */
static void verify_vertex_state(dstm_dgraph_t *g, const sim_ctx_t *ctx,
                                dstm_time_t t_last)
{
        for (size_t i = 0; i < ctx->per_rank; i++) {
                airport_state_t vs;

                CHECK(dstm_dgraph_vertex_get(g, ctx->off + i, &vs,
                                             t_last, 0),
                      "vertex_get (readback)");

                REQUIRE(vs.on_ground == ctx->host[i].on_ground &&
                        vs.pending   == ctx->host[i].pending   &&
                        vs.landed    == ctx->host[i].landed    &&
                        vs.departed  == ctx->host[i].departed,
                        "versioned vertex state matches host mirror");
        }
}

/*******************************************************************************
 * The test
 ******************************************************************************/

static void test_airport_mini(void)
{
        int grid = choose_grid_side(g_world_size);
        REQUIRE(grid > 0, "no usable grid side for this world_size");

        size_t total_vertices = (size_t)grid * (size_t)grid;
        size_t per_rank       = total_vertices / (size_t)g_world_size;
        size_t off            = per_rank * (size_t)g_rank;

        if (g_rank == 0) {
                printf("[airport_mini] world_size=%d grid=%dx%d "
                       "airports=%zu per_rank=%zu ticks=%d\n",
                       g_world_size, grid, grid, total_vertices, per_rank,
                       NUM_TIMESTEPS);
                fflush(stdout);
        }

        /* --- Topology first: the distance matrix must exist before
         *     dstm_start, and dstm_dgraph_create needs a live session. */
        size_t *degrees = NULL;
        size_t *dsts    = NULL;
        build_local_topology(grid, per_rank, off, &degrees, &dsts);

        dstm_distance_t *dist = build_torus_distance(grid, per_rank);

        CHECK(dstm_start(dist), "dstm_start");

        /* --- Graph. Initial vertex state matches the host mirror;
         *     edges start as empty bundles (NULL => zero-init). */
        airport_state_t *vstates = calloc(per_rank, sizeof(airport_state_t));
        REQUIRE(vstates != NULL, "vstates calloc");
        for (size_t i = 0; i < per_rank; i++)
                vstates[i].on_ground = INITIAL_PLANES;

        dstm_dgraph_t *g = NULL;
        CHECK(dstm_dgraph_create(&g,
                                 sizeof(airport_state_t),
                                 sizeof(edge_bundle_t),
                                 per_rank,
                                 degrees, dsts,
                                 vstates, NULL),
              "dgraph_create");
        REQUIRE(g != NULL, "dgraph_create returned NULL");

        free(degrees);
        free(dsts);
        free(vstates);

        /* --- Host-side context. */
        sim_ctx_t ctx = {
                .grid          = grid,
                .per_rank      = per_rank,
                .off           = dstm_dgraph_local_vertex_offset(g),
                .host          = calloc(per_rank, sizeof(airport_state_t)),
                .next_plane_id = (long long)g_rank * 1000000LL,
        };
        REQUIRE(ctx.host != NULL, "host mirror calloc");
        REQUIRE(ctx.off == off, "local_vertex_offset matches assumption");
        for (size_t i = 0; i < per_rank; i++)
                ctx.host[i].on_ground = INITIAL_PLANES;

        check_introspection(g, &ctx, total_vertices);

        long long global_initial = 0;
        long long local_initial  =
                (long long)per_rank * (long long)INITIAL_PLANES;
        MPI_Allreduce(&local_initial, &global_initial, 1, MPI_LONG_LONG,
                      MPI_SUM, MPI_COMM_WORLD);
        REQUIRE(global_initial ==
                (long long)total_vertices * (long long)INITIAL_PLANES,
                "initial plane count");

        /* --- Run. No per-tick MPI_Barrier: ranks are allowed to drift,
         *     and dstm's own visibility rules order the data. */
        for (dstm_time_t t = 0; t < NUM_TIMESTEPS; t++)
                airport_tick(g, &ctx, t);

        /* --- Account. */
        dstm_time_t t_last = (dstm_time_t)NUM_TIMESTEPS;

        verify_vertex_state(g, &ctx, t_last);

        long long local_ground = 0;
        long long local_landed = 0;
        long long local_departed = 0;
        for (size_t i = 0; i < per_rank; i++) {
                local_ground   += ctx.host[i].on_ground + ctx.host[i].pending;
                local_landed   += ctx.host[i].landed;
                local_departed += ctx.host[i].departed;
        }
        long long local_inflight = drain_local_inflight(g, &ctx, t_last);
        long long local_total    = local_ground + local_inflight;

        long long global_total = 0, global_inflight = 0;
        long long global_landed = 0, global_departed = 0;
        MPI_Allreduce(&local_total,    &global_total,    1, MPI_LONG_LONG,
                      MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_inflight, &global_inflight, 1, MPI_LONG_LONG,
                      MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_landed,   &global_landed,   1, MPI_LONG_LONG,
                      MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_departed, &global_departed, 1, MPI_LONG_LONG,
                      MPI_SUM, MPI_COMM_WORLD);

        if (g_rank == 0) {
                printf("[airport_mini] after %d ticks: total=%lld "
                       "(initial=%lld) departed=%lld landed=%lld "
                       "inflight=%lld\n",
                       NUM_TIMESTEPS, global_total, global_initial,
                       global_departed, global_landed, global_inflight);
                fflush(stdout);
        }

        /* CONSERVATION: every departure moves one plane from the ground
         * into an edge bundle; every landing moves one from pending to
         * the ground. Nothing is created or destroyed. */
        REQUIRE(global_total == global_initial, "planes conserved");
        REQUIRE(global_departed >= global_landed,
                "departures >= landings");
        REQUIRE(global_landed  >  0, "some planes landed");
        REQUIRE(global_inflight > 0, "some planes still in flight");

        /* The Allreduces above guarantee every rank finished its reads,
         * so tearing the graph down now cannot strand a remote get. */
        free(ctx.host);
        dstm_dgraph_destroy(&g);

        CHECK(dstm_fini(t_last + 1), "dstm_fini");
        dstm_distance_destroy(&dist);

        if (g_rank == 0) {
                printf("[airport_mini] PASS\n");
                fflush(stdout);
        }
}

/*******************************************************************************
 * main
 ******************************************************************************/

int main(int argc, char *argv[])
{
        MPI_Init(&argc, &argv);
        dstm_init(&argc, &argv);
        dstm_log_setlevel(DSTM_LOG_ERROR);

        MPI_Comm_size(MPI_COMM_WORLD, &g_world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);

        /* The grid-side search covers up to 32 ranks; fail loudly rather
         * than silently picking a degenerate grid. */
        if (g_world_size > 32) {
                if (g_rank == 0)
                        fprintf(stderr,
                                "airport_mini: targets world_size <= 32 "
                                "(got %d)\n", g_world_size);
                dstm_finalize();
                MPI_Finalize();
                return 1;
        }

        test_airport_mini();

        dstm_finalize();
        MPI_Finalize();
        return 0;
}
