/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

/*
 * MPI coverage for the function-pointer reducer accumulate API:
 *
 *   dstm_reduce_register
 *   dstm_accumulate
 *   dstm_darray_accumulate
 *   dstm_dgraph_vertex_accumulate
 *   dstm_dgraph_edge_accumulate
 *
 * The reducer is a user function pointer, registered once per logical
 * datatype during setup and named on the wire by a small integer id.  This
 * lets a single accumulate fold arbitrary POD structs, not just scalars.
 *
 * Functional tests use a complete distance matrix and the pattern:
 *
 *   - every rank issues accumulates at (time=0, delta=+1), targeting ver 1;
 *   - every rank then performs local GETs at (time=1, delta=0).
 *
 * Having every rank operate at logical time 1 advances every rank's DSTM
 * heartbeat, so version 1 becomes READABLE without an asymmetric deadlock.
 *
 * Version buffers are calloc'd, so an untouched cell starts as all-zero
 * bytes.  All reducers here use all-zero as their identity.
 */

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dstm.h"

static int rank;
static int world_size;

/*******************************************************************************
 * User reducers
 ******************************************************************************/

/* Elementwise int32 sum over a cell of (size/4) int32 lanes. */
static void reduce_i32_sum(void *dst, const void *src, size_t size, void *user)
{
        (void)user;
        int32_t       *d = dst;
        const int32_t *s = src;
        size_t         n = size / sizeof(int32_t);

        for (size_t i = 0; i < n; i++)
                d[i] += s[i];
}

/* Elementwise int32 max over a cell of (size/4) int32 lanes. */
static void reduce_i32_max(void *dst, const void *src, size_t size, void *user)
{
        (void)user;
        int32_t       *d = dst;
        const int32_t *s = src;
        size_t         n = size / sizeof(int32_t);

        for (size_t i = 0; i < n; i++)
                if (s[i] > d[i])
                        d[i] = s[i];
}

/* Elementwise double sum, demonstrating a second datatype through the
 * same mechanism. */
static void reduce_f64_sum(void *dst, const void *src, size_t size, void *user)
{
        (void)user;
        double       *d = dst;
        const double *s = src;
        size_t        n = size / sizeof(double);

        for (size_t i = 0; i < n; i++)
                d[i] += s[i];
}

/* The whole point: reduce a non-trivial struct.  count accumulates, value
 * sums, best tracks the maximum, and best_owner carries the rank that
 * produced the current best (tie-broken by lowest rank, which makes the
 * fold commutative). */
typedef struct {
        int32_t count;
        int64_t value;
        int32_t best;
        int32_t best_owner;
} agg_t;

static void reduce_agg(void *dst, const void *src, size_t size, void *user)
{
        (void)user;
        assert(size == sizeof(agg_t));
        agg_t       *d = dst;
        const agg_t *s = src;

        d->count += s->count;
        d->value += s->value;

        if (s->best > d->best ||
            (s->best == d->best && s->best_owner < d->best_owner)) {
                d->best       = s->best;
                d->best_owner = s->best_owner;
        }
}

/* Reducer ids, registered once in main and shared by every test. */
static dstm_reduce_id_t RID_I32_SUM;
static dstm_reduce_id_t RID_I32_MAX;
static dstm_reduce_id_t RID_F64_SUM;
static dstm_reduce_id_t RID_AGG;

/*******************************************************************************
 * Session helpers
 ******************************************************************************/

static dstm_distance_t *session_begin(void)
{
        dstm_distance_t *dist = NULL;

        int rc = dstm_distance_create_complete(&dist, (size_t)world_size);
        assert(rc == DSTM_SUCCESS);
        assert(dist != NULL);

        rc = dstm_start(dist);
        assert(rc == DSTM_SUCCESS);

        return dist;
}

static void session_end(dstm_distance_t **dist, dstm_time_t last_time)
{
        assert(dist != NULL);
        assert(*dist != NULL);

        int rc = dstm_fini(last_time);
        assert(rc == DSTM_SUCCESS);

        dstm_distance_destroy(dist);
        assert(*dist == NULL);
}

/*******************************************************************************
 * Expected-value helpers
 ******************************************************************************/

static int32_t expected_i32_sum(void)
{
        int64_t n = world_size;
        return (int32_t)(n * (n + 1) / 2);
}

/*******************************************************************************
 * Registration / argument validation
 ******************************************************************************/

static void test_register_validation(void)
{
        dstm_reduce_id_t id = DSTM_REDUCE_ID_INVALID;

        assert(dstm_reduce_register(NULL, NULL, &id) == -DSTM_EINVAL);
        assert(dstm_reduce_register(reduce_i32_sum, NULL, NULL) ==
               -DSTM_EINVAL);
}

static void test_darray_invalid_arguments(void)
{
        dstm_distance_t *dist = session_begin();

        dstm_darray_t *a = NULL;
        int rc = dstm_darray_create(&a, sizeof(int32_t), 1);
        assert(rc == DSTM_SUCCESS);
        assert(a != NULL);

        size_t local_index = dstm_darray_global_offset(a);
        int32_t value = 1;

        assert(dstm_darray_accumulate(NULL, local_index, &value, 0, 1,
                                      RID_I32_SUM) == -DSTM_EINVAL);

        assert(dstm_darray_accumulate(a, local_index, NULL, 0, 1,
                                      RID_I32_SUM) == -DSTM_EINVAL);

        assert(dstm_darray_accumulate(a, local_index, &value, 0, -1,
                                      RID_I32_SUM) == -DSTM_EINVAL);

        /* Unregistered reducer id. */
        assert(dstm_darray_accumulate(a, local_index, &value, 0, 1,
                                      9999) == -DSTM_EINVAL);

        /* Negative / invalid reducer id. */
        assert(dstm_darray_accumulate(a, local_index, &value, 0, 1,
                                      DSTM_REDUCE_ID_INVALID) == -DSTM_EINVAL);

        /* Remote delta zero rejected by the underlying dstm_accumulate. */
        if (world_size > 1) {
                int peer = (rank + 1) % world_size;
                size_t remote_index = (size_t)peer;

                assert(dstm_darray_accumulate(a, remote_index, &value, 0, 0,
                                              RID_I32_SUM) == -DSTM_EDOM);
        }

        dstm_darray_destroy(&a);
        assert(a == NULL);

        session_end(&dist, 0);
}

/*******************************************************************************
 * Scalar darray: many writers merge into one cell
 ******************************************************************************/

static void test_darray_scalar_sum_and_max(void)
{
        dstm_distance_t *dist = session_begin();

        /* Two cells per owner: [sum, max]. */
        enum { SUM = 0, MAX = 1, N = 2 };

        dstm_darray_t *a = NULL;
        int rc = dstm_darray_create(&a, sizeof(int32_t), N);
        assert(rc == DSTM_SUCCESS);

        for (int owner = 0; owner < world_size; owner++) {
                size_t base = (size_t)owner * N;
                int32_t contribution = (int32_t)(rank + 1);

                rc = dstm_darray_accumulate(a, base + SUM, &contribution,
                                            0, 1, RID_I32_SUM);
                assert(rc == DSTM_SUCCESS);

                rc = dstm_darray_accumulate(a, base + MAX, &contribution,
                                            0, 1, RID_I32_MAX);
                assert(rc == DSTM_SUCCESS);
        }

        size_t base = dstm_darray_global_offset(a);
        int32_t got_sum = INT32_MIN, got_max = INT32_MIN;

        rc = dstm_darray_get(a, base + SUM, &got_sum, 1, 0);
        assert(rc == DSTM_SUCCESS);
        rc = dstm_darray_get(a, base + MAX, &got_max, 1, 0);
        assert(rc == DSTM_SUCCESS);

        assert(got_sum == expected_i32_sum());
        assert(got_max == (int32_t)world_size);

        dstm_darray_destroy(&a);
        session_end(&dist, 1);
}

/*******************************************************************************
 * Multi-lane cell (count derived purely from byte size)
 ******************************************************************************/

static void test_darray_elementwise_vector(void)
{
        dstm_distance_t *dist = session_begin();

        /* Each element is four int32 lanes; the reducer folds all 16 bytes. */
        dstm_darray_t *a = NULL;
        int rc = dstm_darray_create(&a, 4 * sizeof(int32_t), 1);
        assert(rc == DSTM_SUCCESS);

        int32_t contribution[4] = {
                (int32_t)(rank + 1),
                1,
                (int32_t)(2 * (rank + 1)),
                -(int32_t)(rank + 1),
        };

        for (int owner = 0; owner < world_size; owner++) {
                rc = dstm_darray_accumulate(a, (size_t)owner, contribution,
                                            0, 1, RID_I32_SUM);
                assert(rc == DSTM_SUCCESS);
        }

        int32_t result[4] = {INT32_MIN, INT32_MIN, INT32_MIN, INT32_MIN};
        size_t local_index = dstm_darray_global_offset(a);

        rc = dstm_darray_get(a, local_index, result, 1, 0);
        assert(rc == DSTM_SUCCESS);

        assert(result[0] == expected_i32_sum());
        assert(result[1] == (int32_t)world_size);
        assert(result[2] == 2 * expected_i32_sum());
        assert(result[3] == -expected_i32_sum());

        dstm_darray_destroy(&a);
        session_end(&dist, 1);
}

/*******************************************************************************
 * Struct reducer: the reason the API exists
 ******************************************************************************/

static void test_darray_struct_reducer(void)
{
        dstm_distance_t *dist = session_begin();

        dstm_darray_t *a = NULL;
        int rc = dstm_darray_create(&a, sizeof(agg_t), 1);
        assert(rc == DSTM_SUCCESS);

        agg_t contribution = {
                .count      = 1,
                .value      = (int64_t)(rank + 1),
                .best       = (int32_t)(rank + 1),
                .best_owner = rank,
        };

        for (int owner = 0; owner < world_size; owner++) {
                rc = dstm_darray_accumulate(a, (size_t)owner, &contribution,
                                            0, 1, RID_AGG);
                assert(rc == DSTM_SUCCESS);
        }

        agg_t result = {0};
        size_t local_index = dstm_darray_global_offset(a);
        rc = dstm_darray_get(a, local_index, &result, 1, 0);
        assert(rc == DSTM_SUCCESS);

        assert(result.count == world_size);
        assert(result.value == (int64_t)expected_i32_sum());
        assert(result.best == (int32_t)world_size);
        assert(result.best_owner == world_size - 1);

        dstm_darray_destroy(&a);
        session_end(&dist, 1);
}

/*******************************************************************************
 * Double datatype through the same mechanism
 ******************************************************************************/

static void test_darray_double_sum(void)
{
        dstm_distance_t *dist = session_begin();

        dstm_darray_t *a = NULL;
        int rc = dstm_darray_create(&a, sizeof(double), 1);
        assert(rc == DSTM_SUCCESS);

        double contribution = (double)(rank + 1);

        for (int owner = 0; owner < world_size; owner++) {
                rc = dstm_darray_accumulate(a, (size_t)owner, &contribution,
                                            0, 1, RID_F64_SUM);
                assert(rc == DSTM_SUCCESS);
        }

        double result = NAN;
        size_t local_index = dstm_darray_global_offset(a);
        rc = dstm_darray_get(a, local_index, &result, 1, 0);
        assert(rc == DSTM_SUCCESS);

        assert(result == (double)expected_i32_sum());

        dstm_darray_destroy(&a);
        session_end(&dist, 1);
}

/*******************************************************************************
 * Version isolation
 ******************************************************************************/

static void test_darray_multiple_versions(void)
{
        dstm_distance_t *dist = session_begin();

        dstm_darray_t *a = NULL;
        int rc = dstm_darray_create(&a, sizeof(int32_t), 1);
        assert(rc == DSTM_SUCCESS);

        int32_t at_v1 = (int32_t)(rank + 1);
        int32_t at_v2 = 10 * (int32_t)(rank + 1);

        for (int owner = 0; owner < world_size; owner++) {
                rc = dstm_darray_accumulate(a, (size_t)owner, &at_v1,
                                            0, 1, RID_I32_SUM);
                assert(rc == DSTM_SUCCESS);

                rc = dstm_darray_accumulate(a, (size_t)owner, &at_v2,
                                            0, 2, RID_I32_SUM);
                assert(rc == DSTM_SUCCESS);
        }

        size_t local_index = dstm_darray_global_offset(a);

        int32_t got_v1 = -1;
        rc = dstm_darray_get(a, local_index, &got_v1, 1, 0);
        assert(rc == DSTM_SUCCESS);
        assert(got_v1 == expected_i32_sum());

        int32_t got_v2 = -1;
        rc = dstm_darray_get(a, local_index, &got_v2, 2, 0);
        assert(rc == DSTM_SUCCESS);
        assert(got_v2 == 10 * expected_i32_sum());

        dstm_darray_destroy(&a);
        session_end(&dist, 2);
}

/*******************************************************************************
 * Raw dstm_accumulate on a registered region
 ******************************************************************************/

static void test_raw_accumulate_local(void)
{
        dstm_distance_t *dist = session_begin();

        /* Every rank folds into its OWN local region cell at offset 0. */
        int region = dstm_get_local_region_id();
        dstm_ptr_t cell = {region, 0};

        int32_t contribution = (int32_t)(rank + 1);

        /* Single local writer, delta +1 -> version 1. */
        int rc = dstm_accumulate(rank, region, 0, 1, cell, &contribution,
                                 sizeof(int32_t), RID_I32_SUM);
        assert(rc == DSTM_SUCCESS);

        int32_t result = -1;
        rc = dstm_get(rank, region, 1, 0, &result, cell, sizeof(int32_t));
        assert(rc == DSTM_SUCCESS);
        assert(result == (int32_t)(rank + 1));

        session_end(&dist, 1);
}

/*******************************************************************************
 * Dgraph builder: one vertex per rank, complete out-edges.
 ******************************************************************************/

static dstm_dgraph_t *build_complete_graph(size_t vertex_state_size,
                                           size_t edge_state_size)
{
        size_t degrees[1];
        degrees[0] = (size_t)world_size;

        size_t *dsts = malloc((size_t)world_size * sizeof(size_t));
        assert(dsts != NULL);

        for (int r = 0; r < world_size; r++)
                dsts[r] = (size_t)r;

        dstm_dgraph_t *g = NULL;
        int rc = dstm_dgraph_create(&g, vertex_state_size, edge_state_size,
                                    1, degrees, dsts, NULL, NULL);
        assert(rc == DSTM_SUCCESS);
        assert(g != NULL);

        free(dsts);
        return g;
}

static void test_dgraph_invalid_arguments(void)
{
        dstm_distance_t *dist = session_begin();
        dstm_dgraph_t *g =
                build_complete_graph(sizeof(int32_t), sizeof(int32_t));

        size_t local_vertex = dstm_dgraph_local_vertex_offset(g);
        int32_t value = 1;

        assert(dstm_dgraph_vertex_accumulate(NULL, local_vertex, &value,
                                             0, 1, RID_I32_SUM) ==
               -DSTM_EINVAL);

        assert(dstm_dgraph_vertex_accumulate(g, local_vertex, NULL,
                                             0, 1, RID_I32_SUM) ==
               -DSTM_EINVAL);

        assert(dstm_dgraph_vertex_accumulate(g, local_vertex, &value,
                                             0, -1, RID_I32_SUM) ==
               -DSTM_EINVAL);

        assert(dstm_dgraph_vertex_accumulate(g, local_vertex, &value,
                                             0, 1, 9999) == -DSTM_EINVAL);

        assert(dstm_dgraph_edge_accumulate(NULL, local_vertex, local_vertex,
                                           &value, 0, 1, RID_I32_SUM) ==
               -DSTM_EINVAL);

        assert(dstm_dgraph_edge_accumulate(g, local_vertex, local_vertex,
                                           NULL, 0, 1, RID_I32_SUM) ==
               -DSTM_EINVAL);

        size_t invalid_vertex = (size_t)world_size;

        assert(dstm_dgraph_vertex_accumulate(g, invalid_vertex, &value,
                                             0, 1, RID_I32_SUM) ==
               -DSTM_EINVAL);

        assert(dstm_dgraph_edge_accumulate(g, invalid_vertex, local_vertex,
                                           &value, 0, 1, RID_I32_SUM) !=
               DSTM_SUCCESS);

        dstm_dgraph_destroy(&g);
        session_end(&dist, 0);
}

static void test_dgraph_vertex_and_edge_accumulate(void)
{
        dstm_distance_t *dist = session_begin();

        /* Vertex state is a struct; edge state is two int32 lanes. */
        dstm_dgraph_t *g =
                build_complete_graph(sizeof(agg_t), 2 * sizeof(int32_t));

        agg_t vertex_contribution = {
                .count      = 1,
                .value      = (int64_t)(rank + 1),
                .best       = (int32_t)(rank + 1),
                .best_owner = rank,
        };

        int32_t edge_contribution[2] = {
                2 * (int32_t)(rank + 1),
                (int32_t)(rank + 1),
        };

        for (int owner = 0; owner < world_size; owner++) {
                size_t vertex = (size_t)owner;

                int rc = dstm_dgraph_vertex_accumulate(g, vertex,
                                                       &vertex_contribution,
                                                       0, 1, RID_AGG);
                assert(rc == DSTM_SUCCESS);

                rc = dstm_dgraph_edge_accumulate(g, vertex, vertex,
                                                 edge_contribution,
                                                 0, 1, RID_I32_SUM);
                assert(rc == DSTM_SUCCESS);
        }

        size_t local_vertex = dstm_dgraph_local_vertex_offset(g);

        agg_t vertex_result = {0};
        int rc = dstm_dgraph_vertex_get(g, local_vertex, &vertex_result,
                                        1, 0);
        assert(rc == DSTM_SUCCESS);
        assert(vertex_result.count == world_size);
        assert(vertex_result.value == (int64_t)expected_i32_sum());
        assert(vertex_result.best == (int32_t)world_size);
        assert(vertex_result.best_owner == world_size - 1);

        int32_t edge_result[2] = {-1, -1};
        rc = dstm_dgraph_edge_get(g, local_vertex, local_vertex,
                                  edge_result, 1, 0);
        assert(rc == DSTM_SUCCESS);
        assert(edge_result[0] == 2 * expected_i32_sum());
        assert(edge_result[1] == expected_i32_sum());

        dstm_dgraph_destroy(&g);
        session_end(&dist, 1);
}

static void test_dgraph_zero_sized_state(void)
{
        dstm_distance_t *dist = session_begin();

        dstm_dgraph_t *g = build_complete_graph(0, 0);
        size_t local_vertex = dstm_dgraph_local_vertex_offset(g);
        int32_t ignored = 123;

        int rc = dstm_dgraph_vertex_accumulate(g, local_vertex, &ignored,
                                               0, 1, RID_I32_SUM);
        assert(rc == DSTM_SUCCESS);

        rc = dstm_dgraph_edge_accumulate(g, local_vertex, local_vertex,
                                         &ignored, 0, 1, RID_I32_SUM);
        assert(rc == DSTM_SUCCESS);

        rc = dstm_dgraph_vertex_accumulate(g, (size_t)world_size, &ignored,
                                           0, 1, RID_I32_SUM);
        assert(rc == -DSTM_EINVAL);

        dstm_dgraph_destroy(&g);
        session_end(&dist, 0);
}

/*******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char **argv)
{
        MPI_Init(&argc, &argv);

        int rc = dstm_init(&argc, &argv);
        assert(rc == DSTM_SUCCESS);

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        dstm_log_setlevel(DSTM_LOG_ERROR);

        assert(world_size >= 1);

        /* Register reducers in the SAME order on every rank, before any
         * session.  This is the SPMD contract that makes reduce ids portable
         * across ranks. */
        rc = dstm_reduce_register(reduce_i32_sum, NULL, &RID_I32_SUM);
        assert(rc == DSTM_SUCCESS);
        rc = dstm_reduce_register(reduce_i32_max, NULL, &RID_I32_MAX);
        assert(rc == DSTM_SUCCESS);
        rc = dstm_reduce_register(reduce_f64_sum, NULL, &RID_F64_SUM);
        assert(rc == DSTM_SUCCESS);
        rc = dstm_reduce_register(reduce_agg, NULL, &RID_AGG);
        assert(rc == DSTM_SUCCESS);

        test_register_validation();

        test_darray_invalid_arguments();
        test_darray_scalar_sum_and_max();
        test_darray_elementwise_vector();
        test_darray_struct_reducer();
        test_darray_double_sum();
        test_darray_multiple_versions();

        test_raw_accumulate_local();

        test_dgraph_invalid_arguments();
        test_dgraph_vertex_and_edge_accumulate();
        test_dgraph_zero_sized_state();

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
                printf("PASS: accumulate coverage (world_size=%d)\n",
                       world_size);
                fflush(stdout);
        }

        rc = dstm_finalize();
        assert(rc == DSTM_SUCCESS);

        MPI_Finalize();
        return EXIT_SUCCESS;
}
