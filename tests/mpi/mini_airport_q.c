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
 * Airport mini-simulation on a dqueue.
 *
 * A dqueue-based twin of mini_airport.c (which uses dstm_dgraph). Same
 * toroidal airport grid, same conservation invariant, but planes travel
 * through a single sparse distributed queue instead of per-edge dgraph
 * bundles.
 *
 * TOPOLOGY
 *   An N x N toroidal grid of airports, N chosen so N*N divides evenly
 *   across world_size. Each airport has four torus neighbors (N/S/E/W).
 *   Contiguous rank-order partitioning: rank r owns global airport ids
 *   [r*per_rank, (r+1)*per_rank).
 *
 * TRANSPORT  --  the difference from the dgraph version
 *   dgraph routes by edge (airport->airport). dqueue routes by RANK. A
 *   departing plane is addressed to the OWNING RANK of its destination
 *   airport, and carries the destination airport id in its payload so the
 *   consumer can deliver it to the correct local airport.
 *
 *   A plane whose destination airport is owned by the SAME rank never
 *   enters the queue: it is delivered locally in-process. Only cross-rank
 *   planes ride the dqueue. This keeps the queue purely for inter-rank
 *   traffic, exactly the regime dqueue targets.
 *
 * NEIGHBOR SET
 *   The dqueue's neighbors come from the distance matrix: rank s neighbors
 *   self iff dstm_distance_get(dist, self, s) != -1. We build the SAME
 *   torus rank-adjacency matrix as examples/airport (delta=1 for ranks
 *   sharing a torus edge, -1 otherwise), so the queue's neighbor set is
 *   exactly the set of ranks a plane could ever be pushed to.
 *
 * PER-TICK SCHEDULE
 *   1. Drain the incoming queue at (t, 0): every arrived plane is added to
 *      its destination airport's pending-landing count. (At t == 0 the
 *      queue is empty.)
 *   2. Land up to LANDINGS_PER_TICK planes per airport: pending -> ground.
 *   3. Depart every grounded plane to a torus neighbor. Same-rank
 *      destinations are delivered immediately into local pending; cross-
 *      rank destinations are pushed to the owning rank. Then flush at
 *      (t, +1) so all cross-rank planes are visible at tick t+1. Every
 *      outgoing neighbor block is published (flush publishes all), so a
 *      version always exists for the consumer.
 *
 * INVARIANT
 *   Total planes (ground + pending + in-flight in the queue) is conserved
 *   and equals total_airports * INITIAL_PLANES.
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

/* Max planes one neighbor block can carry in one tick. Oversized for the
 * traffic these settings produce. */
#define QUEUE_CAPACITY    64

/* Planes each airport can land per tick. */
#define LANDINGS_PER_TICK 2

/* Planes each airport starts with. */
#define INITIAL_PLANES    2

/* Ticks to run. */
#define NUM_TIMESTEPS     8

/* Simulated-time distance between connected ranks. */
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

/* Per-airport state, host-side only (not versioned: the queue carries all
 * the distributed state in this variant). */
typedef struct {
        int       on_ground;
        int       pending;
        long long landed;
        long long departed;
} airport_state_t;

/* A single plane in flight through the queue: it knows which airport it is
 * headed to.  Fixed-size POD, no pointers (dqueue requirement). */
typedef struct {
        long long plane_id;
        size_t    dst_airport;   /* global airport id */
} plane_msg_t;

typedef struct {
        int              grid;
        size_t           per_rank;
        size_t           off;        /* first local airport global id */
        airport_state_t *host;       /* length per_rank */
        long long        next_plane_id;
} sim_ctx_t;

/*******************************************************************************
 * Torus helpers
 ******************************************************************************/

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

static int choose_grid_side(int world_size)
{
        const int MIN_PER_RANK = 4;
        int min_total = world_size * MIN_PER_RANK;

        for (int n = 4; n <= 128; n++) {
                int total = n * n;
                if (total < min_total)       continue;
                if (total % world_size != 0) continue;
                return n;
        }
        return 0;
}

/* Owning rank of a global airport id under contiguous partitioning. */
static int airport_owner(size_t gid, size_t per_rank)
{
        return (int)(gid / per_rank);
}

/*******************************************************************************
 * Distance matrix
 *
 * Rank-level torus adjacency: two ranks are reachable (delta 1) iff some
 * airport on one has a torus neighbor on the other; otherwise -1. Mirrors
 * build_rank_adjacency() in examples/airport. Purely local -- a closed-form
 * function of grid/per_rank -- so every rank derives the same matrix.
 ******************************************************************************/

static dstm_distance_t *build_torus_distance(int grid, size_t per_rank)
{
        dstm_distance_t *d = NULL;
        int              n = g_world_size;

        CHECK(dstm_distance_create_complete(&d, (size_t)n),
              "distance_create_complete");
        REQUIRE(d != NULL && d->distances != NULL, "distance matrix alloc");

        for (int i = 0; i < n * n; i++)
                d->distances[i] = -1;
        for (int i = 0; i < n; i++)
                d->distances[i * n + i] = 0;

        size_t total = (size_t)grid * (size_t)grid;
        for (size_t gid = 0; gid < total; gid++) {
                int owner = airport_owner(gid, per_rank);
                for (int dir = 0; dir < DEGREE; dir++) {
                        size_t ngid   = torus_neighbor(gid, grid, dir);
                        int    nowner = airport_owner(ngid, per_rank);
                        if (nowner == owner)
                                continue;
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
                printf("[airport_q] rank adjacency: %d/%d off-diagonal "
                       "pairs reachable at delta=%d\n",
                       reachable, n * (n - 1), EDGE_DELTA);
                fflush(stdout);
        }
        return d;
}

/*******************************************************************************
 * Local delivery
 *
 * Deposit a plane bound for airport `dst` (which must be local) into that
 * airport's pending-landing count.
 ******************************************************************************/

static void deliver_local(sim_ctx_t *ctx, size_t dst_airport)
{
        REQUIRE(dst_airport >= ctx->off &&
                dst_airport < ctx->off + ctx->per_rank,
                "deliver_local: airport not owned by this rank");
        size_t li = dst_airport - ctx->off;
        ctx->host[li].pending++;
}

/*******************************************************************************
 * One simulation tick
 ******************************************************************************/

static void airport_tick(dstm_dqueue_t *q, sim_ctx_t *ctx, dstm_time_t t)
{
        /* --- Phase 1: drain arrivals from the queue. ---------------------
         *
         * Local read of the current version: produce was at (t-1, +1), so
         * the planes for tick t are in version t, read at (t, 0). At t == 0
         * the queue has nothing published yet; the early-tick fallback in
         * pop_incoming (effective time < 0 would be empty) does not apply
         * here since t >= 0, but version 0 WILL exist because every rank
         * flushes every tick -- including the empty flush at the end of
         * tick 0's phase 3 which publishes version 1, not 0. So at t == 0
         * we must NOT pop (no version 0 was ever written).
         *
         * We therefore pop only for t >= 1, reading version t written by
         * the flush at the end of tick t-1. */
        if (t >= 1) {
                size_t cap = dstm_dqueue_max_incoming(q);
                plane_msg_t *in = malloc(cap * sizeof(plane_msg_t));
                REQUIRE(in != NULL, "phase1 inbox malloc");

                size_t count = cap;
                CHECK(dstm_dqueue_pop_incoming(q, in, &count, t, 0),
                      "phase1 pop_incoming");
                REQUIRE(count <= cap, "phase1 count within capacity");

                for (size_t k = 0; k < count; k++) {
                        REQUIRE(airport_owner(in[k].dst_airport,
                                              ctx->per_rank) == g_rank,
                                "arrived plane not addressed to this rank");
                        deliver_local(ctx, in[k].dst_airport);
                }
                free(in);
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

        /* --- Phase 3: depart grounded planes to torus neighbors. ---------
         *
         * Each grounded plane picks a torus-neighbor airport (round-robin
         * over directions). Same-rank destinations are delivered locally in
         * place; cross-rank destinations are staged into the queue for the
         * owning rank. Then one flush publishes version t+1 for EVERY
         * outgoing neighbor block (empty or not), so each consumer always
         * has a version to read and no block has two writers. */
        for (size_t i = 0; i < ctx->per_rank; i++) {
                size_t u      = ctx->off + i;
                int    planes = ctx->host[i].on_ground;

                for (int p = 0; p < planes; p++) {
                        int    dir = (int)((t + (dstm_time_t)p +
                                            (dstm_time_t)i) & 3);
                        size_t dst = torus_neighbor(u, ctx->grid, dir);
                        int    owner = airport_owner(dst, ctx->per_rank);

                        plane_msg_t msg = {
                                .plane_id    = ctx->next_plane_id++,
                                .dst_airport = dst,
                        };

                        ctx->host[i].on_ground--;
                        ctx->host[i].departed++;

                        if (owner == g_rank) {
                                /* Stays on this rank: deliver immediately,
                                 * it never enters the queue. */
                                deliver_local(ctx, dst);
                        } else {
                                /* Cross-rank: stage for the owning rank. */
                                int rc = dstm_dqueue_push(q, owner, &msg);
                                REQUIRE(rc == DSTM_SUCCESS,
                                        "phase3 dqueue_push (queue full?)");
                        }
                }
        }

        /* Publish all staged cross-rank planes for version t+1. Flush
         * publishes every outgoing block, so unreachable-to-us peers that
         * got nothing still see a well-formed (empty) version. */
        CHECK(dstm_dqueue_flush(q, t, 1), "phase3 flush");
}

/*******************************************************************************
 * Final drain
 *
 * After ticks 0..T-1, the last flush published version T. One final pop at
 * (T, 0) drains everything still in flight so it can be counted. We do NOT
 * re-land these planes (the sim is over); we only tally them as pending so
 * conservation holds.
 ******************************************************************************/

static long long drain_inflight(dstm_dqueue_t *q, sim_ctx_t *ctx,
                                dstm_time_t t_last)
{
        size_t cap = dstm_dqueue_max_incoming(q);
        plane_msg_t *in = malloc(cap * sizeof(plane_msg_t));
        REQUIRE(in != NULL, "drain inbox malloc");

        size_t count = cap;
        CHECK(dstm_dqueue_pop_incoming(q, in, &count, t_last, 0),
              "drain pop_incoming");
        REQUIRE(count <= cap, "drain count within capacity");

        for (size_t k = 0; k < count; k++) {
                REQUIRE(airport_owner(in[k].dst_airport, ctx->per_rank) ==
                        g_rank,
                        "drained plane not addressed to this rank");
                deliver_local(ctx, in[k].dst_airport);
        }
        free(in);
        return (long long)count;
}

/*******************************************************************************
 * The test
 ******************************************************************************/

static void test_airport_queue(void)
{
        int grid = choose_grid_side(g_world_size);
        REQUIRE(grid > 0, "no usable grid side for this world_size");

        size_t total_vertices = (size_t)grid * (size_t)grid;
        size_t per_rank       = total_vertices / (size_t)g_world_size;
        size_t off            = per_rank * (size_t)g_rank;

        if (g_rank == 0) {
                printf("[airport_q] world_size=%d grid=%dx%d airports=%zu "
                       "per_rank=%zu ticks=%d\n",
                       g_world_size, grid, grid, total_vertices, per_rank,
                       NUM_TIMESTEPS);
                fflush(stdout);
        }

        dstm_distance_t *dist = build_torus_distance(grid, per_rank);
        CHECK(dstm_start(dist), "dstm_start");

        /* The queue needs capacity for the worst-case cross-rank traffic a
         * single neighbor can receive in one tick. With INITIAL_PLANES
         * small and LANDINGS_PER_TICK bounding ground growth, QUEUE_CAPACITY
         * is comfortably oversized. */
        dstm_dqueue_t *q = NULL;
        CHECK(dstm_dqueue_create(&q, sizeof(plane_msg_t), QUEUE_CAPACITY,
                                 dist),
              "dqueue_create");
        REQUIRE(q != NULL, "dqueue_create returned NULL");
        REQUIRE(dstm_dqueue_element_size(q) == sizeof(plane_msg_t),
                "element_size echo");
        REQUIRE(dstm_dqueue_capacity(q) == QUEUE_CAPACITY,
                "capacity echo");

        sim_ctx_t ctx = {
                .grid          = grid,
                .per_rank      = per_rank,
                .off           = off,
                .host          = calloc(per_rank, sizeof(airport_state_t)),
                .next_plane_id = (long long)g_rank * 1000000LL,
        };
        REQUIRE(ctx.host != NULL, "host mirror calloc");
        for (size_t i = 0; i < per_rank; i++)
                ctx.host[i].on_ground = INITIAL_PLANES;

        long long global_initial = 0;
        long long local_initial  =
                (long long)per_rank * (long long)INITIAL_PLANES;
        MPI_Allreduce(&local_initial, &global_initial, 1, MPI_LONG_LONG,
                      MPI_SUM, MPI_COMM_WORLD);
        REQUIRE(global_initial ==
                (long long)total_vertices * (long long)INITIAL_PLANES,
                "initial plane count");

        /* Run. No per-tick MPI_Barrier: ranks drift and dqueue's version
         * visibility orders the transport. */
        for (dstm_time_t t = 0; t < NUM_TIMESTEPS; t++)
                airport_tick(q, &ctx, t);

        /* Drain whatever the final flush (version NUM_TIMESTEPS) left in
         * flight so it can be counted. */
        dstm_time_t t_last = (dstm_time_t)NUM_TIMESTEPS;
        long long local_inflight = drain_inflight(q, &ctx, t_last);

        /* Account: every plane is either on the ground, pending (including
         * just-drained in-flight), or was already counted via the host
         * mirror. Ground + pending covers all planes this rank holds. */
        long long local_ground = 0;
        long long local_landed = 0;
        long long local_departed = 0;
        for (size_t i = 0; i < per_rank; i++) {
                local_ground   += ctx.host[i].on_ground + ctx.host[i].pending;
                local_landed   += ctx.host[i].landed;
                local_departed += ctx.host[i].departed;
        }

        long long global_total = 0, global_inflight = 0;
        long long global_landed = 0, global_departed = 0;
        MPI_Allreduce(&local_ground,   &global_total,    1, MPI_LONG_LONG,
                      MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_inflight, &global_inflight, 1, MPI_LONG_LONG,
                      MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_landed,   &global_landed,   1, MPI_LONG_LONG,
                      MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_departed, &global_departed, 1, MPI_LONG_LONG,
                      MPI_SUM, MPI_COMM_WORLD);

        if (g_rank == 0) {
                printf("[airport_q] after %d ticks: total=%lld "
                       "(initial=%lld) departed=%lld landed=%lld "
                       "final_drained=%lld\n",
                       NUM_TIMESTEPS, global_total, global_initial,
                       global_departed, global_landed, global_inflight);
                fflush(stdout);
        }

        /* CONSERVATION: local_ground already includes the just-drained
         * in-flight planes (deliver_local bumped pending for each), so the
         * sum of ground+pending across ranks must equal the initial count.
         * Nothing is created or destroyed. */
        REQUIRE(global_total == global_initial, "planes conserved");
        REQUIRE(global_departed >= global_landed, "departures >= landings");
        REQUIRE(global_landed > 0, "some planes landed");

        free(ctx.host);
        dstm_dqueue_destroy(&q);

        CHECK(dstm_fini(t_last + 1), "dstm_fini");
        dstm_distance_destroy(&dist);

        if (g_rank == 0) {
                printf("[airport_q] PASS\n");
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

        if (g_world_size > 32) {
                if (g_rank == 0)
                        fprintf(stderr,
                                "airport_q: targets world_size <= 32 "
                                "(got %d)\n", g_world_size);
                dstm_finalize();
                MPI_Finalize();
                return 1;
        }

        test_airport_queue();

        dstm_finalize();
        MPI_Finalize();
        return 0;
}
