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
 * dqueue unit tests -- single append-based queue edition.
 *
 * The queue is now ONE region / ONE slot per rank, versioned per timestamp.
 * All producers APPEND into the same (owner, timestamp) slot in MPI arrival
 * order; there are no per-neighbor blocks.  Consequences reflected here:
 *
 *   - max_incoming collapses to `capacity` (NOT n_neighbors * capacity);
 *   - pop_incoming is a single local get of this rank's slot;
 *   - flush self-seeds the owner's version, so a consumer's pop never blocks
 *     on a "missing" version even when nobody pushed to us this tick;
 *   - push only validates argument ranges; reachability from the distance
 *     matrix bounds who actually pushes, not the addressability of a dest.
 *
 * Synchronization discipline (unchanged):
 *   - produce (flush) at (t, +1): future push, remote delta > 0, visible to
 *     the consumer at version t+1;
 *   - consume (pop)  at (t+1, 0): LOCAL read of the current version.
 *
 * Append ordering is nondeterministic (MPI arrival order), so functional
 * tests verify SET/COUNT conservation, never positional ordering across
 * distinct producers.
 *
 * Run: mpirun -n 4 ./dqueue
 */

#include <assert.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "dstm.h"
#include "internal/distance.h"

/*******************************************************************************
 * Globals
 ******************************************************************************/

static int rank;
static int world_size;

/*******************************************************************************
 * Session helpers
 ******************************************************************************/

static dstm_distance_t *session_begin_complete(void)
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

        /* Single append-based slot: the WHOLE incoming queue holds at most
         * `capacity` elements, regardless of neighbor count. */
        assert(dstm_dqueue_max_incoming(q) == cap);

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

        /* A single producer cannot stage more than `capacity` elements for
         * one destination (capacity == 2): the third push fails. */
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
 * drain and success, with no MPI traffic. */
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
 ******************************************************************************/

/* Local round-trip: every rank appends to its OWN slot, flushes at (0,+1),
 * then pops its own incoming queue at (1,0).  Only self produced, so arrival
 * order equals push order and we can assert positions. */
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

        /* Single producer (self): one append, so push order is preserved. */
        for (int i = 0; i < n; i++)
                assert(got[i] == rank * 1000 + i);

        dstm_dqueue_destroy(&q);
        session_end(&dist, 2);
}

/* Full all-to-all on a complete topology: every rank appends one element to
 * every rank's slot.  Each rank must receive exactly one element from each
 * producer (including itself), in SOME order. */
static void test_roundtrip_all_to_all(void)
{
        dstm_distance_t *dist = session_begin_complete();

        dstm_dqueue_t *q = NULL;
        /* Aggregate capacity: world_size producers each contribute 1. */
        int rc = dstm_dqueue_create(&q, sizeof(int), (size_t)world_size, dist);
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

/* push_flush: fused push + single-destination flush.  With append
 * semantics this composes with later flushes at the same version, so we can
 * use it alongside a full flush and still full-drain.
 *
 * Each rank push_flushes ONE element to its ring successor at (0,+1), then
 * also flushes (self-seed + any staged).  Consumer pops at (1,0) and must
 * see exactly one element, from its ring predecessor. */
static void test_push_flush_single(void)
{
        dstm_distance_t *dist = session_begin_complete();

        dstm_dqueue_t *q = NULL;
        int rc = dstm_dqueue_create(&q, sizeof(int), (size_t)world_size, dist);
        assert(rc == DSTM_SUCCESS);

        int dest = (rank + 1) % world_size;
        int v = rank + 1;

        /* Fused push+flush publishes v into dest's slot at version 1. */
        rc = dstm_dqueue_push_flush(q, dest, &v, 0, 1);
        assert(rc == DSTM_SUCCESS);
        assert(dstm_dqueue_pending(q, dest) == 0);

        /* Also flush so our OWN slot's version 1 is seeded (self-seed) and
         * any remaining staging (none here) is published.  Appends compose,
         * so re-flushing dest with 0 staged is a safe no-op. */
        rc = dstm_dqueue_flush(q, 0, 1);
        assert(rc == DSTM_SUCCESS);

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

/* push_flush staging/clear semantics in isolation. */
static void test_push_flush_staging(void)
{
        dstm_distance_t *dist = session_begin_complete();

        dstm_dqueue_t *q = NULL;
        int rc = dstm_dqueue_create(&q, sizeof(int), 4, dist);
        assert(rc == DSTM_SUCCESS);

        int v = rank + 1;

        rc = dstm_dqueue_push_flush(q, rank, &v, 0, 1);
        assert(rc == DSTM_SUCCESS);

        /* The fused call must have cleared the self outbox. */
        assert(dstm_dqueue_pending(q, rank) == 0);

        /* Seed our version fully and drain: self produced exactly 1. */
        rc = dstm_dqueue_flush(q, 0, 1);
        assert(rc == DSTM_SUCCESS);

        int got[4];
        size_t count = 4;
        rc = dstm_dqueue_pop_incoming(q, got, &count, 1, 0);
        assert(rc == DSTM_SUCCESS);
        assert(count == 1);
        assert(got[0] == rank + 1);

        dstm_dqueue_destroy(&q);
        session_end(&dist, 2);
}

/* Two-call capacity pattern: query the required count, then drain. */
static void test_pop_two_call_pattern(void)
{
        dstm_distance_t *dist = session_begin_complete();

        dstm_dqueue_t *q = NULL;
        int rc = dstm_dqueue_create(&q, sizeof(int), (size_t)world_size, dist);
        assert(rc == DSTM_SUCCESS);

        int v = rank + 1;
        for (int d = 0; d < world_size; d++) {
                rc = dstm_dqueue_push(q, d, &v);
                assert(rc == DSTM_SUCCESS);
        }
        rc = dstm_dqueue_flush(q, 0, 1);
        assert(rc == DSTM_SUCCESS);

        /* Count-only query (buffer NULL, capacity 0) returns -DSTM_ENOMEM
         * with the exact total needed. */
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

/* Undersized buffer: pop reports the required count via -DSTM_ENOMEM. */
static void test_pop_buffer_too_small(void)
{
        dstm_distance_t *dist = session_begin_complete();

        dstm_dqueue_t *q = NULL;
        int rc = dstm_dqueue_create(&q, sizeof(int), (size_t)world_size, dist);
        assert(rc == DSTM_SUCCESS);

        int v = rank + 1;
        for (int d = 0; d < world_size; d++)
                assert(dstm_dqueue_push(q, d, &v) == DSTM_SUCCESS);
        rc = dstm_dqueue_flush(q, 0, 1);
        assert(rc == DSTM_SUCCESS);

        /* Buffer for only 1 element but world_size are incoming (when
         * world_size > 1). */
        if (world_size > 1) {
                int small[1];
                size_t count = 1;
                rc = dstm_dqueue_pop_incoming(q, small, &count, 1, 0);
                assert(rc == -DSTM_ENOMEM);
                assert(count == (size_t)world_size);
        }

        dstm_dqueue_destroy(&q);
        session_end(&dist, 2);
}

/* Multi-tick drift: several produce/consume rounds on one queue.
 * Produce at (t, +1), consume at (t+1, 0). */
static void test_roundtrip_multi_tick(void)
{
        dstm_distance_t *dist = session_begin_complete();

        dstm_dqueue_t *q = NULL;
        int rc = dstm_dqueue_create(&q, sizeof(int), (size_t)world_size, dist);
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
        session_end(&dist, ticks + 1);
}

/* Conservation over several ticks with asymmetric per-tick volumes:
 * producer rank r sends (r % 3) + 1 elements to every rank each tick.
 * Verify, globally, that total popped equals total pushed. */
static void test_conservation_multi(void)
{
        dstm_distance_t *dist = session_begin_complete();

        /* Aggregate worst case per slot per tick:
         * sum over producers of ((p % 3) + 1) <= 3 * world_size. */
        size_t cap = (size_t)(3 * world_size);
        dstm_dqueue_t *q = NULL;
        int rc = dstm_dqueue_create(&q, sizeof(int), cap, dist);
        assert(rc == DSTM_SUCCESS);

        const int ticks = 5;
        long long sent = 0, received = 0;

        for (int t = 0; t < ticks; t++) {
                int per = (rank % 3) + 1;
                int v = rank + 1;
                for (int d = 0; d < world_size; d++) {
                        for (int k = 0; k < per; k++) {
                                rc = dstm_dqueue_push(q, d, &v);
                                assert(rc == DSTM_SUCCESS);
                                sent++;
                        }
                }
                rc = dstm_dqueue_flush(q, t, 1);
                assert(rc == DSTM_SUCCESS);

                int *got = malloc(cap * sizeof(int));
                assert(got != NULL);
                size_t count = cap;
                rc = dstm_dqueue_pop_incoming(q, got, &count, t + 1, 0);
                assert(rc == DSTM_SUCCESS);
                for (size_t i = 0; i < count; i++) {
                        int producer = got[i] - 1;
                        assert(producer >= 0 && producer < world_size);
                        received++;
                }
                free(got);
        }

        long long g_sent = 0, g_recv = 0;
        MPI_Allreduce(&sent, &g_sent, 1, MPI_LONG_LONG, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(&received, &g_recv, 1, MPI_LONG_LONG, MPI_SUM,
                      MPI_COMM_WORLD);
        assert(g_sent == g_recv);

        dstm_dqueue_destroy(&q);
        session_end(&dist, ticks + 1);
}

/*******************************************************************************
 * main
 ******************************************************************************/

int main(int argc, char *argv[])
{
        MPI_Init(&argc, &argv);
        dstm_init(&argc, &argv);
        dstm_log_setlevel(DSTM_LOG_ERROR);

        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        /* Outer session for create/destroy, introspection, and
         * argument-validation tests (no cross-rank put/get traffic). */
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
        test_pop_buffer_too_small();
        test_roundtrip_multi_tick();
        test_conservation_multi();

        dstm_finalize();
        MPI_Finalize();
        return 0;
}
