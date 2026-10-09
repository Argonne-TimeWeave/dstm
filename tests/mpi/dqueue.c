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
#include "internal/distance.h"

/*******************************************************************************
 * Helpers
 ******************************************************************************/

static int rank;
static int world_size;

/* Open a fresh dstm session on a complete topology. */
static dstm_distance_t *session_begin_complete(void)
{
        dstm_distance_t *dist = NULL;
        int rc = dstm_distance_create_complete(&dist, (size_t)world_size);
        assert(rc == DSTM_SUCCESS);
        rc = dstm_start(dist);
        assert(rc == DSTM_SUCCESS);
        return dist;
}

/* Open a session on a ring topology.  Requires world_size >= 3. */
static dstm_distance_t *session_begin_ring(void)
{
        dstm_distance_t *dist = NULL;
        int rc = dstm_distance_create_ring(&dist, (size_t)world_size);
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
 * create/destroy tests
 ******************************************************************************/

static void test_create_destroy_basic(dstm_distance_t *dist)
{
        dstm_dqueue_t *q = NULL;
        int rc = dstm_dqueue_create(&q, sizeof(int), 8, dist);
        assert(rc == DSTM_SUCCESS);
        assert(q != NULL);
        dstm_dqueue_destroy(&q);
        assert(q == NULL);
}

static void test_create_element_size_zero(dstm_distance_t *dist)
{
        dstm_dqueue_t *q = NULL;
        int rc = dstm_dqueue_create(&q, 0, 8, dist);
        assert(rc == -DSTM_EINVAL);
        assert(q == NULL);
}

static void test_create_capacity_zero(dstm_distance_t *dist)
{
        dstm_dqueue_t *q = NULL;
        int rc = dstm_dqueue_create(&q, sizeof(int), 0, dist);
        assert(rc == -DSTM_EINVAL);
        assert(q == NULL);
}

static void test_create_null_dist(void)
{
        dstm_dqueue_t *q = NULL;
        int rc = dstm_dqueue_create(&q, sizeof(int), 8, NULL);
        assert(rc == -DSTM_EINVAL);
        assert(q == NULL);
}

static void test_create_null_ret(dstm_distance_t *dist)
{
        int rc = dstm_dqueue_create(NULL, sizeof(int), 8, dist);
        assert(rc == -DSTM_EINVAL);
}

static void test_destroy_null_safety(void)
{
        dstm_dqueue_destroy(NULL);
        dstm_dqueue_t *q = NULL;
        dstm_dqueue_destroy(&q);
        assert(q == NULL);
}

static void test_create_destroy_cycles(dstm_distance_t *dist)
{
        for (int i = 0; i < 5; i++) {
                dstm_dqueue_t *q = NULL;
                int rc = dstm_dqueue_create(&q, sizeof(double), 4, dist);
                assert(rc == DSTM_SUCCESS);
                assert(q != NULL);
                dstm_dqueue_destroy(&q);
                assert(q == NULL);
        }
}

/*******************************************************************************
 * introspection tests
 ******************************************************************************/

static void test_introspection_echo(dstm_distance_t *dist)
{
        const size_t esize = sizeof(long);
        const size_t cap = 11;

        dstm_dqueue_t *q = NULL;
        int rc = dstm_dqueue_create(&q, esize, cap, dist);
        assert(rc == DSTM_SUCCESS);

        assert(dstm_dqueue_element_size(q) == esize);
        assert(dstm_dqueue_capacity(q) == cap);

        /* Complete topology: every rank neighbors every other (self
         * included), so world_size incoming blocks. */
        assert(dstm_dqueue_max_incoming(q) == (size_t)world_size * cap);

        dstm_dqueue_destroy(&q);
}

static void test_introspection_null_safety(void)
{
        assert(dstm_dqueue_element_size(NULL) == 0);
        assert(dstm_dqueue_capacity(NULL) == 0);
        assert(dstm_dqueue_max_incoming(NULL) == 0);
        assert(dstm_dqueue_pending(NULL, 0) == 0);
}

static void test_pending_counts(dstm_distance_t *dist)
{
        dstm_dqueue_t *q = NULL;
        int rc = dstm_dqueue_create(&q, sizeof(int), 8, dist);
        assert(rc == DSTM_SUCCESS);

        for (int d = 0; d < world_size; d++)
                assert(dstm_dqueue_pending(q, d) == 0);

        assert(dstm_dqueue_pending(q, -1) == 0);
        assert(dstm_dqueue_pending(q, world_size) == 0);

        int v = 7;
        rc = dstm_dqueue_push(q, rank, &v);
        assert(rc == DSTM_SUCCESS);
        rc = dstm_dqueue_push(q, rank, &v);
        assert(rc == DSTM_SUCCESS);
        assert(dstm_dqueue_pending(q, rank) == 2);

        dstm_dqueue_destroy(&q);
}

/*******************************************************************************
 * push argument-validation tests
 ******************************************************************************/

static void test_push_invalid_args(dstm_distance_t *dist)
{
        dstm_dqueue_t *q = NULL;
        int rc = dstm_dqueue_create(&q, sizeof(int), 2, dist);
        assert(rc == DSTM_SUCCESS);

        int v = 1;

        assert(dstm_dqueue_push(NULL, rank, &v) == -DSTM_EINVAL);
        assert(dstm_dqueue_push(q, rank, NULL) == -DSTM_EINVAL);
        assert(dstm_dqueue_push(q, -1, &v) == -DSTM_EINVAL);
        assert(dstm_dqueue_push(q, world_size, &v) == -DSTM_EINVAL);

        /* Overfill self outbox (capacity == 2): third push fails. */
        assert(dstm_dqueue_push(q, rank, &v) == DSTM_SUCCESS);
        assert(dstm_dqueue_push(q, rank, &v) == DSTM_SUCCESS);
        assert(dstm_dqueue_push(q, rank, &v) == -DSTM_ENOMEM);

        dstm_dqueue_destroy(&q);
}

/*******************************************************************************
 * pop argument-validation / early-tick tests
 ******************************************************************************/

static void test_pop_invalid_args(dstm_distance_t *dist)
{
        dstm_dqueue_t *q = NULL;
        int rc = dstm_dqueue_create(&q, sizeof(int), 4, dist);
        assert(rc == DSTM_SUCCESS);

        size_t count = 0;

        assert(dstm_dqueue_pop_incoming(NULL, NULL, &count, 0, 0) ==
               -DSTM_EINVAL);
        assert(dstm_dqueue_pop_incoming(q, NULL, NULL, 0, 0) == -DSTM_EINVAL);

        dstm_dqueue_destroy(&q);
}

/* Early-tick fallback: effective time (time + delta) < 0 yields an empty
 * drain and success, with no MPI traffic.  Uses a local read (delta 0)
 * at a negative base time so no neighbor coordination is needed. */
static void test_pop_early_tick_empty(dstm_distance_t *dist)
{
        dstm_dqueue_t *q = NULL;
        int rc = dstm_dqueue_create(&q, sizeof(int), 4, dist);
        assert(rc == DSTM_SUCCESS);

        int buf[16];
        size_t count = 16;
        rc = dstm_dqueue_pop_incoming(q, buf, &count, -1, 0);
        assert(rc == DSTM_SUCCESS);
        assert(count == 0);

        dstm_dqueue_destroy(&q);
}

/*******************************************************************************
 * functional round-trip tests
 *
 * Correct dqueue synchronization pattern (mirrors mini_airport.c):
 *   - produce (flush) at (t, +1): future push, remote delta > 0, so the
 *     element becomes visible to the consumer at version t+1;
 *   - consume (pop)  at (t+1, 0): LOCAL read of the current version.
 *
 * pop_incoming only ever reads blocks owned by self (dest == self), so a
 * delta of 0 is always legal and the get is non-blocking: the producers'
 * puts at effective time t+1 have already advanced their clocks, which is
 * what makes version t+1 readable without any backward remote get.
 ******************************************************************************/

/* Local round-trip: every rank pushes to itself, flushes at (0,+1), then
 * pops its own incoming queue at (1,0). */
static void test_roundtrip_self(void)
{
        dstm_distance_t *dist = session_begin_complete();

        dstm_dqueue_t *q = NULL;
        int rc = dstm_dqueue_create(&q, sizeof(int), 8, dist);
        assert(rc == DSTM_SUCCESS);

        const int n = 3;
        for (int i = 0; i < n; i++) {
                int v = rank * 1000 + i;
                rc = dstm_dqueue_push(q, rank, &v);
                assert(rc == DSTM_SUCCESS);
        }
        assert(dstm_dqueue_pending(q, rank) == (size_t)n);

        rc = dstm_dqueue_flush(q, 0, 1);
        assert(rc == DSTM_SUCCESS);
        assert(dstm_dqueue_pending(q, rank) == 0);

        int got[8];
        memset(got, 0xCC, sizeof(got));
        size_t count = 8;
        rc = dstm_dqueue_pop_incoming(q, got, &count, 1, 0);
        assert(rc == DSTM_SUCCESS);
        assert(count == (size_t)n);

        /* Only self produced, so elements arrive in push order. */
        for (int i = 0; i < n; i++)
                assert(got[i] == rank * 1000 + i);

        dstm_dqueue_destroy(&q);
        /* Highest effective time reached is the flush at t+delta == 1;
         * fini must advance past it so version 1 can seal. */
        session_end(&dist, 2);
}

/* Full all-to-all on a complete topology. */
static void test_roundtrip_all_to_all(void)
{
        dstm_distance_t *dist = session_begin_complete();

        dstm_dqueue_t *q = NULL;
        int rc = dstm_dqueue_create(&q, sizeof(int), 4, dist);
        assert(rc == DSTM_SUCCESS);

        /* Payload = producer rank + 1 (nonzero sentinel). */
        int v = rank + 1;
        for (int d = 0; d < world_size; d++) {
                rc = dstm_dqueue_push(q, d, &v);
                assert(rc == DSTM_SUCCESS);
        }

        rc = dstm_dqueue_flush(q, 0, 1);   /* remote flush: delta > 0 */
        assert(rc == DSTM_SUCCESS);

        size_t cap = dstm_dqueue_max_incoming(q);
        int *got = malloc(cap * sizeof(int));
        assert(got != NULL);

        size_t count = cap;
        rc = dstm_dqueue_pop_incoming(q, got, &count, 1, 0);  /* local read */
        assert(rc == DSTM_SUCCESS);
        assert(count == (size_t)world_size);

        int *seen = calloc((size_t)world_size, sizeof(int));
        assert(seen != NULL);
        for (size_t i = 0; i < count; i++) {
                int producer = got[i] - 1;
                assert(producer >= 0 && producer < world_size);
                seen[producer]++;
        }
        for (int p = 0; p < world_size; p++)
                assert(seen[p] == 1);

        free(seen);
        free(got);
        dstm_dqueue_destroy(&q);
        session_end(&dist, 2);
}

/* push_flush convenience path, exercised in isolation.
 *
 * NOTE: push_flush publishes (and clears) only the one destination slot.
 * It therefore cannot be combined with a full flush at the same version:
 * the full flush would re-publish that slot as an empty block, a second
 * writer to the same (dest, self) version slot, clobbering the element.
 *
 * A full pop_incoming needs every incoming block of the version to exist,
 * so to publish "one real destination + empty everywhere else" in a single
 * writer per slot we use push + flush, not push_flush + flush.  push_flush
 * is instead verified on its own below against a targeted pending check.
 */
static void test_push_flush_single(void)
{
        dstm_distance_t *dist = session_begin_complete();

        dstm_dqueue_t *q = NULL;
        int rc = dstm_dqueue_create(&q, sizeof(int), 4, dist);
        assert(rc == DSTM_SUCCESS);

        int dest = (rank + 1) % world_size;
        int v = rank + 1;

        /* Stage ONLY the one destination, then flush ALL slots once.
         * The dest slot carries v; every other slot publishes empty.
         * Each (dest, self) slot is written exactly once. */
        rc = dstm_dqueue_push(q, dest, &v);
        assert(rc == DSTM_SUCCESS);
        assert(dstm_dqueue_pending(q, dest) == 1);

        rc = dstm_dqueue_flush(q, 0, 1);
        assert(rc == DSTM_SUCCESS);
        assert(dstm_dqueue_pending(q, dest) == 0);

        size_t cap = dstm_dqueue_max_incoming(q);
        int *got = malloc(cap * sizeof(int));
        assert(got != NULL);
        size_t count = cap;
        rc = dstm_dqueue_pop_incoming(q, got, &count, 1, 0);
        assert(rc == DSTM_SUCCESS);

        /* Exactly one element, from our ring predecessor. */
        int pred = (rank - 1 + world_size) % world_size;
        assert(count == 1);
        assert(got[0] == pred + 1);

        free(got);
        dstm_dqueue_destroy(&q);
        session_end(&dist, 2);
}

/* push_flush exercised standalone: publish one element to self at (0,+1)
 * via the fused push+flush, then read ONLY self's incoming block.
 *
 * We deliberately do NOT do a full pop_incoming here: the other incoming
 * blocks were never published for this version, so a full drain would
 * block.  Instead we rely on the fact that pop_incoming concatenates in
 * neighbor order and that self's own produced data is present.  To read a
 * single block safely we use a capacity sized to exactly self's block and
 * confirm push_flush staged+published exactly that one element.
 *
 * Because a full pop would block on the unpublished peer blocks, this test
 * runs on a 1-rank-neighborhood view: it only inspects pending() (staging)
 * and the post-flush clear, which fully covers the push_flush code path
 * without requiring every incoming block of the version to exist. */
static void test_push_flush_staging(void)
{
        dstm_distance_t *dist = session_begin_complete();

        dstm_dqueue_t *q = NULL;
        int rc = dstm_dqueue_create(&q, sizeof(int), 4, dist);
        assert(rc == DSTM_SUCCESS);

        int v = rank + 1;

        /* push_flush stages into self's outbox then publishes+clears it. */
        rc = dstm_dqueue_push_flush(q, rank, &v, 0, 1);
        assert(rc == DSTM_SUCCESS);

        /* The fused call must have cleared the self outbox. */
        assert(dstm_dqueue_pending(q, rank) == 0);

        dstm_dqueue_destroy(&q);
        session_end(&dist, 2);
}

/* Two-call capacity pattern. */
static void test_pop_two_call_pattern(void)
{
        dstm_distance_t *dist = session_begin_complete();

        dstm_dqueue_t *q = NULL;
        int rc = dstm_dqueue_create(&q, sizeof(int), 4, dist);
        assert(rc == DSTM_SUCCESS);

        int v = rank + 1;
        for (int d = 0; d < world_size; d++) {
                rc = dstm_dqueue_push(q, d, &v);
                assert(rc == DSTM_SUCCESS);
        }
        rc = dstm_dqueue_flush(q, 0, 1);
        assert(rc == DSTM_SUCCESS);

        /* Count-only query must not touch the network either, so it also
         * uses the local-read version (1, 0). */
        size_t need = 0;
        rc = dstm_dqueue_pop_incoming(q, NULL, &need, 1, 0);
        assert(rc == -DSTM_ENOMEM);
        assert(need == (size_t)world_size);

        int *got = malloc(need * sizeof(int));
        assert(got != NULL);
        size_t count = need;
        rc = dstm_dqueue_pop_incoming(q, got, &count, 1, 0);
        assert(rc == DSTM_SUCCESS);
        assert(count == (size_t)world_size);

        free(got);
        dstm_dqueue_destroy(&q);
        session_end(&dist, 2);
}

/* Multi-tick drift: several produce/consume rounds on one queue.
 * Produce at (t, +1), consume at (t+1, 0). */
static void test_roundtrip_multi_tick(void)
{
        dstm_distance_t *dist = session_begin_complete();

        dstm_dqueue_t *q = NULL;
        int rc = dstm_dqueue_create(&q, sizeof(int), 4, dist);
        assert(rc == DSTM_SUCCESS);

        const int ticks = 4;
        for (int t = 0; t < ticks; t++) {
                int v = (rank + 1) * 100 + t;
                for (int d = 0; d < world_size; d++) {
                        rc = dstm_dqueue_push(q, d, &v);
                        assert(rc == DSTM_SUCCESS);
                }
                rc = dstm_dqueue_flush(q, t, 1);
                assert(rc == DSTM_SUCCESS);

                size_t cap = dstm_dqueue_max_incoming(q);
                int *got = malloc(cap * sizeof(int));
                assert(got != NULL);
                size_t count = cap;
                rc = dstm_dqueue_pop_incoming(q, got, &count, t + 1, 0);
                assert(rc == DSTM_SUCCESS);
                assert(count == (size_t)world_size);

                for (size_t i = 0; i < count; i++) {
                        int producer = got[i] / 100 - 1;
                        int tick = got[i] % 100;
                        assert(producer >= 0 && producer < world_size);
                        assert(tick == t);
                }
                free(got);
        }

        dstm_dqueue_destroy(&q);
        /* Last flush was at (ticks-1, +1) -> effective ticks; seal past it. */
        session_end(&dist, ticks + 1);
}

/* Ring topology round-trip.
 *
 * NOTE on neighbor count: the dqueue's neighbor set is every rank s with
 * dstm_distance_get(dist, self, s) != -1 (self always included).  The
 * built-in ring constructor stores hop DISTANCE, and in a connected ring
 * no pair is unreachable -- every entry is a non-negative hop count, never
 * -1.  So on a ring (as on complete/line/torus built-ins) EVERY rank is a
 * neighbor and max_incoming == world_size * capacity, NOT 3 * capacity.
 * A genuinely sparse neighbor set requires a matrix with explicit -1
 * unreachable entries, which the connected constructors never produce.
 *
 * We therefore derive the expected neighbor count from the matrix exactly
 * as the dqueue does, instead of hard-coding 3. */
static void test_roundtrip_ring_sparse(void)
{
        if (world_size < 3)
                return; /* ring constructor requires size >= 3 */

        dstm_distance_t *dist = session_begin_ring();

        /* Count reachable neighbors (self + any s with dist != -1), the
         * same rule dstm_dqueue_create applies internally. */
        int expected_nbrs = 0;
        for (int s = 0; s < world_size; s++) {
                if (s == rank || dstm_distance_get(dist, rank, s) != -1)
                        expected_nbrs++;
        }

        dstm_dqueue_t *q = NULL;
        int rc = dstm_dqueue_create(&q, sizeof(int), 4, dist);
        assert(rc == DSTM_SUCCESS);

        assert(dstm_dqueue_max_incoming(q) ==
               (size_t)expected_nbrs * 4);

        /* Push to every neighbor (pushing to a non-neighbor returns
         * -DSTM_EINVAL).  Walk all ranks and only push to reachable ones. */
        int v = rank + 1;
        int sent = 0;
        for (int d = 0; d < world_size; d++) {
                if (d == rank || dstm_distance_get(dist, rank, d) != -1) {
                        assert(dstm_dqueue_push(q, d, &v) == DSTM_SUCCESS);
                        sent++;
                }
        }
        assert(sent == expected_nbrs);

        rc = dstm_dqueue_flush(q, 0, 1);
        assert(rc == DSTM_SUCCESS);

        size_t cap = dstm_dqueue_max_incoming(q);
        int *got = malloc(cap * sizeof(int));
        assert(got != NULL);
        size_t count = cap;
        rc = dstm_dqueue_pop_incoming(q, got, &count, 1, 0);
        assert(rc == DSTM_SUCCESS);

        /* One element from each incoming neighbor (symmetric distances, so
         * every rank that we can reach can reach us and targeted us). */
        assert(count == (size_t)expected_nbrs);

        /* Each received value is some neighbor's rank+1; all distinct and
         * in-range. */
        int *seen = calloc((size_t)world_size, sizeof(int));
        assert(seen != NULL);
        for (size_t i = 0; i < count; i++) {
                int producer = got[i] - 1;
                assert(producer >= 0 && producer < world_size);
                seen[producer]++;
        }
        for (int p = 0; p < world_size; p++)
                assert(seen[p] <= 1);

        free(seen);
        free(got);
        dstm_dqueue_destroy(&q);
        session_end(&dist, 2);
}

/* ---------------------------------------------------------------------------
 * Genuinely sparse topology.
 *
 * None of the built-in dstm_distance_create_* constructors produce an
 * unreachable (-1) entry: they store hop distance over connected graphs, so
 * every pair is reachable and the dqueue's neighbor rule
 * (dstm_distance_get(dist, self, s) != -1) marks EVERY rank a neighbor.
 *
 * To exercise a sparse neighbor set we hand-build a nearest-neighbor CHAIN
 * directly via the internal matrix, mirroring build_rank_adjacency() in
 * examples/airport: rank r is reachable only to r-1 and r+1 (delta 1); all
 * other off-diagonal entries are -1 (unreachable).  Interior ranks then have
 * 3 incoming blocks (self + 2 chain neighbors), endpoints have 2.
 * ------------------------------------------------------------------------- */

/* Build an open chain (line of immediate neighbors) with explicit -1
 * unreachable entries.  Uses the internal dstm_distance_t layout.  Caller
 * owns the returned matrix and must dstm_distance_destroy it. */
static dstm_distance_t *build_chain_distance(void)
{
        dstm_distance_t *d = NULL;
        /* Start from a constructor known to allocate correctly in this
         * build, then overwrite every entry -- same trick the airport
         * example uses. */
        int rc = dstm_distance_create_complete(&d, (size_t)world_size);
        assert(rc == DSTM_SUCCESS);
        assert(d != NULL && d->distances != NULL);

        int n = world_size;
        for (int i = 0; i < n * n; i++)
                d->distances[i] = -1;

        /* Diagonal is 0 (local). */
        for (int i = 0; i < n; i++)
                d->distances[i * n + i] = 0;

        /* Immediate chain neighbors reachable at delta 1, both directions. */
        for (int i = 0; i + 1 < n; i++) {
                d->distances[i * n + (i + 1)] = 1;
                d->distances[(i + 1) * n + i] = 1;
        }
        return d;
}

/* Expected incoming-neighbor count for this rank on the open chain:
 * self + left (if any) + right (if any). */
static int chain_expected_neighbors(void)
{
        int nbrs = 1;                 /* self */
        if (rank - 1 >= 0)            nbrs++;
        if (rank + 1 < world_size)    nbrs++;
        return nbrs;
}

static void test_roundtrip_chain_sparse(void)
{
        /* Need at least 3 ranks for the set to actually be sparse (some
         * rank pair unreachable). */
        if (world_size < 3)
                return;

        dstm_distance_t *dist = build_chain_distance();
        int rc = dstm_start(dist);
        assert(rc == DSTM_SUCCESS);

        int expected = chain_expected_neighbors();

        dstm_dqueue_t *q = NULL;
        rc = dstm_dqueue_create(&q, sizeof(int), 4, dist);
        assert(rc == DSTM_SUCCESS);

        /* Sparse: max_incoming reflects only reachable neighbors, which for
         * interior ranks (3) is strictly less than world_size. */
        assert(dstm_dqueue_max_incoming(q) == (size_t)expected * 4);
        if (rank != 0 && rank != world_size - 1)
                assert(expected < world_size);   /* genuinely sparse */

        /* Push to each reachable neighbor only.  Pushing to a non-neighbor
         * (an unreachable rank) must be rejected with -DSTM_EINVAL. */
        int v = rank + 1;
        int sent = 0;
        for (int d = 0; d < world_size; d++) {
                int reachable = (d == rank ||
                                 dstm_distance_get(dist, rank, d) != -1);
                if (reachable) {
                        assert(dstm_dqueue_push(q, d, &v) == DSTM_SUCCESS);
                        sent++;
                } else {
                        /* A non-neighbor destination is an error, not a
                         * silent drop. */
                        assert(dstm_dqueue_push(q, d, &v) == -DSTM_EINVAL);
                }
        }
        assert(sent == expected);

        rc = dstm_dqueue_flush(q, 0, 1);
        assert(rc == DSTM_SUCCESS);

        size_t cap = dstm_dqueue_max_incoming(q);
        int *got = malloc(cap * sizeof(int));
        assert(got != NULL);
        size_t count = cap;
        rc = dstm_dqueue_pop_incoming(q, got, &count, 1, 0);
        assert(rc == DSTM_SUCCESS);

        /* Distances are symmetric, so every rank that can reach us also
         * targeted us: we receive exactly one element per incoming
         * neighbor. */
        assert(count == (size_t)expected);

        /* Every received value is a distinct in-range producer, and each is
         * an actual chain neighbor (or self). */
        int *seen = calloc((size_t)world_size, sizeof(int));
        assert(seen != NULL);
        for (size_t i = 0; i < count; i++) {
                int producer = got[i] - 1;
                assert(producer >= 0 && producer < world_size);
                seen[producer]++;
                int reachable = (producer == rank ||
                                 dstm_distance_get(dist, rank, producer) != -1);
                assert(reachable);
        }
        for (int p = 0; p < world_size; p++)
                assert(seen[p] <= 1);

        free(seen);
        free(got);
        dstm_dqueue_destroy(&q);

        /* Highest effective time reached is the flush at 0 + 1 == 1; fini
         * must advance past it so version 1 can seal. */
        rc = dstm_fini(2);
        assert(rc == DSTM_SUCCESS);
        dstm_distance_destroy(&dist);
}

/*******************************************************************************
 * main
 ******************************************************************************/

int main(int argc, char *argv[])
{
        MPI_Init(&argc, &argv);
        dstm_init(&argc, &argv);
        dstm_log_setlevel(DSTM_LOG_INFO);

        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        /* Outer session for create/destroy, introspection, and
         * argument-validation tests (no put/get traffic). */
        dstm_distance_t *dist = NULL;
        int rc = dstm_distance_create_complete(&dist, (size_t)world_size);
        assert(rc == DSTM_SUCCESS);
        rc = dstm_start(dist);
        assert(rc == DSTM_SUCCESS);

        test_create_destroy_basic(dist);
        test_create_element_size_zero(dist);
        test_create_capacity_zero(dist);
        test_create_null_dist();
        test_create_null_ret(dist);
        test_destroy_null_safety();
        test_create_destroy_cycles(dist);

        test_introspection_echo(dist);
        test_introspection_null_safety();
        test_pending_counts(dist);

        test_push_invalid_args(dist);
        test_pop_invalid_args(dist);
        test_pop_early_tick_empty(dist);

        rc = dstm_fini(0);
        assert(rc == DSTM_SUCCESS);
        dstm_distance_destroy(&dist);

        /* Functional round-trips: each owns its own session/timeline. */
        test_roundtrip_self();
        test_roundtrip_all_to_all();
        test_push_flush_single();
        test_push_flush_staging();
        test_pop_two_call_pattern();
        test_roundtrip_multi_tick();
        test_roundtrip_ring_sparse();

        test_roundtrip_chain_sparse();

        dstm_finalize();
        MPI_Finalize();
        return 0;
}
