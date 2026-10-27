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
 * dqueue unit tests -- single append-based queue, immediate-push edition.
 *
 * push publishes immediately (one dstm_append) at the (time, delta) given on
 * the call; there is no staging and no flush.  Discipline unchanged:
 *   - produce (push) at (t, +1): future push, remote delta > 0;
 *   - consume (pop)  at (t+1, 0): LOCAL read of the current version.
 *
 * max_incoming == capacity (one slot).  Append order across distinct
 * producers is MPI arrival order (nondeterministic), so functional tests
 * check set/count conservation, not positions.
 *
 * Run: mpirun -n 4 ./dqueue
 */

#include <assert.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "dstm.h"
#include "internal/distance.h"

static int rank;
static int world_size;

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
 * create/destroy
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
 * introspection
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
        /* Single slot: whole queue holds at most `capacity` elements. */
        assert(dstm_dqueue_max_incoming(q) == cap);

        dstm_dqueue_destroy(&q);
}

static void test_introspection_null_safety(void)
{
        assert(dstm_dqueue_element_size(NULL) == 0);
        assert(dstm_dqueue_capacity(NULL) == 0);
        assert(dstm_dqueue_max_incoming(NULL) == 0);
}

/*******************************************************************************
 * push argument-validation
 ******************************************************************************/

static void test_push_invalid_args(dstm_distance_t *dist)
{
        dstm_dqueue_t *q = NULL;
        int rc = dstm_dqueue_create(&q, sizeof(int), 4, dist);
        assert(rc == DSTM_SUCCESS);

        int v = 1;

        /* Immediate push now carries (time, delta).  Validate arg checks
         * with a local push at (0, 0) which is legal for the local rank. */
        assert(dstm_dqueue_push(NULL, rank, &v, 0, 0) == -DSTM_EINVAL);
        assert(dstm_dqueue_push(q, rank, NULL, 0, 0) == -DSTM_EINVAL);
        assert(dstm_dqueue_push(q, -1, &v, 0, 0) == -DSTM_EINVAL);
        assert(dstm_dqueue_push(q, world_size, &v, 0, 0) == -DSTM_EINVAL);

        /* Remote push with delta == 0 is rejected by the append layer. */
        if (world_size > 1) {
                int other = (rank + 1) % world_size;
                assert(dstm_dqueue_push(q, other, &v, 0, 0) == -DSTM_EDOM);
        }

        dstm_dqueue_destroy(&q);
}

/*******************************************************************************
 * pop argument-validation / early-tick
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
 * functional round-trips
 ******************************************************************************/

/* Local round-trip: every rank pushes to ITSELF at (0, +1) and pops at
 * (1, 0).  Single producer, so append order == push order. */
static void test_roundtrip_self(void)
{
        dstm_distance_t *dist = session_begin_complete();

        dstm_dqueue_t *q = NULL;
        int rc = dstm_dqueue_create(&q, sizeof(int), 8, dist);
        assert(rc == DSTM_SUCCESS);

        const int n = 3;
        for (int i = 0; i < n; i++) {
                int v = rank * 1000 + i;
                rc = dstm_dqueue_push(q, rank, &v, 0, 1);  /* future push */
                assert(rc == DSTM_SUCCESS);
        }

        int got[8];
        memset(got, 0xCC, sizeof(got));
        size_t count = 8;
        rc = dstm_dqueue_pop_incoming(q, got, &count, 1, 0);
        assert(rc == DSTM_SUCCESS);
        assert(count == (size_t)n);

        /* Append concatenation order is NOT guaranteed to equal push order
         * (even for a single producer: equal-keyed requests are not FIFO in
         * the progress list).  Assert the multiset, not positions. */
        int seen[3] = {0, 0, 0};
        for (int i = 0; i < n; i++) {
                int idx = got[i] - rank * 1000;
                assert(idx >= 0 && idx < n);
                seen[idx]++;
        }
        for (int i = 0; i < n; i++)
                assert(seen[i] == 1);

        dstm_dqueue_destroy(&q);
        session_end(&dist, 2);
}

/* All-to-all on complete topology: every rank pushes one element to every
 * rank at (0, +1), pops at (1, 0).  Each rank receives exactly one element
 * per producer, in some order. */
static void test_roundtrip_all_to_all(void)
{
        dstm_distance_t *dist = session_begin_complete();

        dstm_dqueue_t *q = NULL;
        int rc = dstm_dqueue_create(&q, sizeof(int), (size_t)world_size, dist);
        assert(rc == DSTM_SUCCESS);

        int v = rank + 1;  /* nonzero sentinel = producer rank + 1 */
        for (int d = 0; d < world_size; d++) {
                /* local self-push may use delta 0 or 1; use 1 uniformly so
                 * every element lands at effective time 1. */
                rc = dstm_dqueue_push(q, d, &v, 0, 1);
                assert(rc == DSTM_SUCCESS);
        }

        size_t cap = dstm_dqueue_max_incoming(q);
        int *got = malloc(cap * sizeof(int));
        assert(got != NULL);
        size_t count = cap;
        rc = dstm_dqueue_pop_incoming(q, got, &count, 1, 0);
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

/* Two-call capacity pattern. */
static void test_pop_two_call_pattern(void)
{
        dstm_distance_t *dist = session_begin_complete();

        dstm_dqueue_t *q = NULL;
        int rc = dstm_dqueue_create(&q, sizeof(int), (size_t)world_size, dist);
        assert(rc == DSTM_SUCCESS);

        int v = rank + 1;
        for (int d = 0; d < world_size; d++)
                assert(dstm_dqueue_push(q, d, &v, 0, 1) == DSTM_SUCCESS);

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

/* Undersized buffer reports required count. */
static void test_pop_buffer_too_small(void)
{
        dstm_distance_t *dist = session_begin_complete();

        dstm_dqueue_t *q = NULL;
        int rc = dstm_dqueue_create(&q, sizeof(int), (size_t)world_size, dist);
        assert(rc == DSTM_SUCCESS);

        int v = rank + 1;
        for (int d = 0; d < world_size; d++)
                assert(dstm_dqueue_push(q, d, &v, 0, 1) == DSTM_SUCCESS);

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

/* Multi-tick: produce (t, +1), consume (t+1, 0). */
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
                        rc = dstm_dqueue_push(q, d, &v, t, 1);
                        assert(rc == DSTM_SUCCESS);
                }

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

/* Conservation with asymmetric per-tick volumes. */
static void test_conservation_multi(void)
{
        dstm_distance_t *dist = session_begin_complete();

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
                                rc = dstm_dqueue_push(q, d, &v, t, 1);
                                assert(rc == DSTM_SUCCESS);
                                sent++;
                        }
                }

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
        dstm_log_setlevel(DSTM_LOG_DEBUG);

        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

        test_push_invalid_args(dist);
        test_pop_invalid_args(dist);
        test_pop_early_tick_empty(dist);

        rc = dstm_fini(0);
        assert(rc == DSTM_SUCCESS);
        dstm_distance_destroy(&dist);

        test_roundtrip_self();
        test_roundtrip_all_to_all();
        test_pop_two_call_pattern();
        test_pop_buffer_too_small();
        test_roundtrip_multi_tick();
        test_conservation_multi();

        dstm_finalize();
        MPI_Finalize();
        return 0;
}
