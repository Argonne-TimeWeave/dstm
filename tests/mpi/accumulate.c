/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

/*
 * MPI coverage for:
 *
 *   dstm_accumulate
 *   dstm_darray_accumulate
 *   dstm_dgraph_vertex_accumulate
 *   dstm_dgraph_edge_accumulate
 *
 * Functional tests use a complete distance matrix and the following pattern:
 *
 *   - every rank issues accumulates at (time=0, delta=+1);
 *   - every accumulate therefore targets version 1;
 *   - every rank subsequently performs local GETs at (time=1, delta=0).
 *
 * Having every rank perform an operation at logical time 1 advances every
 * rank's DSTM heartbeat. This allows version 1 to become READABLE without
 * creating an asymmetric GET/barrier deadlock.
 *
 * Most reduction cells begin as zero because version buffers are calloc'd.
 * Consequently:
 *
 *   - SUM, MAX, BOR and BXOR can be tested naturally;
 *   - MIN is tested with negative signed/floating-point inputs;
 *   - PROD and BAND are exercised and expected to remain zero.
 *
 * Testing nonzero identities for PROD/BAND would require explicit seeding of
 * the same version. That is deliberately kept separate from this test because
 * seeding introduces ordering requirements between PUT and ACCUMULATE requests.
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

static int64_t expected_i64_sum(void)
{
        int64_t n = world_size;
        return n * (n + 1) / 2;
}

static uint32_t expected_u32_sum(void)
{
        uint64_t n = (uint64_t)world_size;
        return (uint32_t)(n * (n + 1) / 2);
}

static uint64_t expected_u64_sum(void)
{
        uint64_t n = (uint64_t)world_size;
        return n * (n + 1) / 2;
}

static uint32_t expected_u32_or(void)
{
        uint32_t result = 0;

        for (int r = 0; r < world_size; r++)
                result |= (uint32_t)(r + 1);

        return result;
}

static uint64_t expected_u64_or(void)
{
        uint64_t result = 0;

        for (int r = 0; r < world_size; r++)
                result |= (uint64_t)(r + 1);

        return result;
}

static uint32_t expected_u32_xor(void)
{
        uint32_t result = 0;

        for (int r = 0; r < world_size; r++)
                result ^= (uint32_t)(r + 1);

        return result;
}

static uint64_t expected_u64_xor(void)
{
        uint64_t result = 0;

        for (int r = 0; r < world_size; r++)
                result ^= (uint64_t)(r + 1);

        return result;
}

/*******************************************************************************
 * Darray argument validation
 ******************************************************************************/

static void test_darray_invalid_arguments(void)
{
        dstm_distance_t *dist = session_begin();

        dstm_darray_t *a = NULL;
        int rc = dstm_darray_create(&a, sizeof(int32_t), 1);
        assert(rc == DSTM_SUCCESS);
        assert(a != NULL);

        size_t local_index = dstm_darray_global_offset(a);
        int32_t value = 1;

        assert(dstm_darray_accumulate(NULL,
                                     local_index,
                                     &value,
                                     0,
                                     1,
                                     DSTM_DT_INT32,
                                     DSTM_OP_SUM) == -DSTM_EINVAL);

        assert(dstm_darray_accumulate(a,
                                     local_index,
                                     NULL,
                                     0,
                                     1,
                                     DSTM_DT_INT32,
                                     DSTM_OP_SUM) == -DSTM_EINVAL);

        assert(dstm_darray_accumulate(a,
                                     local_index,
                                     &value,
                                     0,
                                     -1,
                                     DSTM_DT_INT32,
                                     DSTM_OP_SUM) == -DSTM_EINVAL);

        assert(dstm_darray_accumulate(a,
                                     local_index,
                                     &value,
                                     0,
                                     1,
                                     DSTM_DT_INT32,
                                     DSTM_OP_APPEND) == -DSTM_EINVAL);

        assert(dstm_darray_accumulate(a,
                                     local_index,
                                     &value,
                                     0,
                                     1,
                                     DSTM_DT_INT32,
                                     DSTM_OP_MAX_SENTINEL) == -DSTM_EINVAL);

        assert(dstm_darray_accumulate(a,
                                     local_index,
                                     &value,
                                     0,
                                     1,
                                     DSTM_DT_MAX_SENTINEL,
                                     DSTM_OP_SUM) == -DSTM_EINVAL);

        assert(dstm_darray_accumulate(a,
                                     local_index,
                                     &value,
                                     0,
                                     1,
                                     (dstm_datatype_t)-1,
                                     DSTM_OP_SUM) == -DSTM_EINVAL);

        assert(dstm_darray_accumulate(a,
                                     local_index,
                                     &value,
                                     0,
                                     1,
                                     DSTM_DT_INT32,
                                     (dstm_reduce_op_t)-1) == -DSTM_EINVAL);

        /*
         * Remote delta zero must be rejected by the underlying
         * dstm_accumulate call.
         */
        if (world_size > 1) {
                int peer = (rank + 1) % world_size;
                size_t remote_index = (size_t)peer;

                assert(dstm_darray_accumulate(a,
                                             remote_index,
                                             &value,
                                             0,
                                             0,
                                             DSTM_DT_INT32,
                                             DSTM_OP_SUM) == -DSTM_EDOM);
        }

        dstm_darray_destroy(&a);
        assert(a == NULL);

        /*
         * element_size=6 is not divisible by either a four-byte or an
         * eight-byte datatype.
         */
        dstm_darray_t *bad_width = NULL;
        rc = dstm_darray_create(&bad_width, 6, 1);
        assert(rc == DSTM_SUCCESS);

        unsigned char payload[8] = {0};
        local_index = dstm_darray_global_offset(bad_width);

        assert(dstm_darray_accumulate(bad_width,
                                     local_index,
                                     payload,
                                     0,
                                     1,
                                     DSTM_DT_INT32,
                                     DSTM_OP_SUM) == -DSTM_EINVAL);

        assert(dstm_darray_accumulate(bad_width,
                                     local_index,
                                     payload,
                                     0,
                                     1,
                                     DSTM_DT_INT64,
                                     DSTM_OP_SUM) == -DSTM_EINVAL);

        dstm_darray_destroy(&bad_width);

        session_end(&dist, 0);
}

/*******************************************************************************
 * All integer operators and integer datatypes
 ******************************************************************************/

enum {
        INT_OP_SUM = 0,
        INT_OP_PROD,
        INT_OP_MIN,
        INT_OP_MAX,
        INT_OP_BOR,
        INT_OP_BAND,
        INT_OP_BXOR,
        INT_OP_COUNT
};

static const dstm_reduce_op_t integer_ops[INT_OP_COUNT] = {
        DSTM_OP_SUM,
        DSTM_OP_PROD,
        DSTM_OP_MIN,
        DSTM_OP_MAX,
        DSTM_OP_BOR,
        DSTM_OP_BAND,
        DSTM_OP_BXOR,
};

static void test_int32_operators(void)
{
        dstm_distance_t *dist = session_begin();

        dstm_darray_t *a = NULL;
        int rc = dstm_darray_create(&a, sizeof(int32_t), INT_OP_COUNT);
        assert(rc == DSTM_SUCCESS);

        /*
         * Every rank contributes to every owner's seven operator cells.
         */
        for (int owner = 0; owner < world_size; owner++) {
                size_t owner_base = (size_t)owner * INT_OP_COUNT;

                for (size_t op_index = 0;
                     op_index < INT_OP_COUNT;
                     op_index++) {
                        int32_t contribution;

                        if (op_index == INT_OP_MIN)
                                contribution = -(int32_t)(rank + 1);
                        else
                                contribution = (int32_t)(rank + 1);

                        rc = dstm_darray_accumulate(
                                a,
                                owner_base + op_index,
                                &contribution,
                                0,
                                1,
                                DSTM_DT_INT32,
                                integer_ops[op_index]);
                        assert(rc == DSTM_SUCCESS);
                }
        }

        size_t local_base = dstm_darray_global_offset(a);
        int32_t result[INT_OP_COUNT];

        for (size_t i = 0; i < INT_OP_COUNT; i++) {
                result[i] = INT32_MIN;

                rc = dstm_darray_get(a,
                                     local_base + i,
                                     &result[i],
                                     1,
                                     0);
                assert(rc == DSTM_SUCCESS);
        }

        assert(result[INT_OP_SUM] == expected_i32_sum());
        assert(result[INT_OP_PROD] == 0);
        assert(result[INT_OP_MIN] == -(int32_t)world_size);
        assert(result[INT_OP_MAX] == (int32_t)world_size);
        assert((uint32_t)result[INT_OP_BOR] == expected_u32_or());
        assert(result[INT_OP_BAND] == 0);
        assert((uint32_t)result[INT_OP_BXOR] == expected_u32_xor());

        dstm_darray_destroy(&a);
        session_end(&dist, 1);
}

static void test_int64_operators(void)
{
        dstm_distance_t *dist = session_begin();

        dstm_darray_t *a = NULL;
        int rc = dstm_darray_create(&a, sizeof(int64_t), INT_OP_COUNT);
        assert(rc == DSTM_SUCCESS);

        for (int owner = 0; owner < world_size; owner++) {
                size_t owner_base = (size_t)owner * INT_OP_COUNT;

                for (size_t op_index = 0;
                     op_index < INT_OP_COUNT;
                     op_index++) {
                        int64_t contribution;

                        if (op_index == INT_OP_MIN)
                                contribution = -(int64_t)(rank + 1);
                        else
                                contribution = (int64_t)(rank + 1);

                        rc = dstm_darray_accumulate(
                                a,
                                owner_base + op_index,
                                &contribution,
                                0,
                                1,
                                DSTM_DT_INT64,
                                integer_ops[op_index]);
                        assert(rc == DSTM_SUCCESS);
                }
        }

        size_t local_base = dstm_darray_global_offset(a);
        int64_t result[INT_OP_COUNT];

        for (size_t i = 0; i < INT_OP_COUNT; i++) {
                result[i] = INT64_MIN;

                rc = dstm_darray_get(a,
                                     local_base + i,
                                     &result[i],
                                     1,
                                     0);
                assert(rc == DSTM_SUCCESS);
        }

        assert(result[INT_OP_SUM] == expected_i64_sum());
        assert(result[INT_OP_PROD] == 0);
        assert(result[INT_OP_MIN] == -(int64_t)world_size);
        assert(result[INT_OP_MAX] == (int64_t)world_size);
        assert((uint64_t)result[INT_OP_BOR] == expected_u64_or());
        assert(result[INT_OP_BAND] == 0);
        assert((uint64_t)result[INT_OP_BXOR] == expected_u64_xor());

        dstm_darray_destroy(&a);
        session_end(&dist, 1);
}

static void test_uint32_operators(void)
{
        dstm_distance_t *dist = session_begin();

        dstm_darray_t *a = NULL;
        int rc = dstm_darray_create(&a, sizeof(uint32_t), INT_OP_COUNT);
        assert(rc == DSTM_SUCCESS);

        for (int owner = 0; owner < world_size; owner++) {
                size_t owner_base = (size_t)owner * INT_OP_COUNT;

                for (size_t op_index = 0;
                     op_index < INT_OP_COUNT;
                     op_index++) {
                        uint32_t contribution = (uint32_t)(rank + 1);

                        rc = dstm_darray_accumulate(
                                a,
                                owner_base + op_index,
                                &contribution,
                                0,
                                1,
                                DSTM_DT_UINT32,
                                integer_ops[op_index]);
                        assert(rc == DSTM_SUCCESS);
                }
        }

        size_t local_base = dstm_darray_global_offset(a);
        uint32_t result[INT_OP_COUNT];

        for (size_t i = 0; i < INT_OP_COUNT; i++) {
                result[i] = UINT32_MAX;

                rc = dstm_darray_get(a,
                                     local_base + i,
                                     &result[i],
                                     1,
                                     0);
                assert(rc == DSTM_SUCCESS);
        }

        assert(result[INT_OP_SUM] == expected_u32_sum());
        assert(result[INT_OP_PROD] == 0);
        assert(result[INT_OP_MIN] == 0);
        assert(result[INT_OP_MAX] == (uint32_t)world_size);
        assert(result[INT_OP_BOR] == expected_u32_or());
        assert(result[INT_OP_BAND] == 0);
        assert(result[INT_OP_BXOR] == expected_u32_xor());

        dstm_darray_destroy(&a);
        session_end(&dist, 1);
}

static void test_uint64_operators(void)
{
        dstm_distance_t *dist = session_begin();

        dstm_darray_t *a = NULL;
        int rc = dstm_darray_create(&a, sizeof(uint64_t), INT_OP_COUNT);
        assert(rc == DSTM_SUCCESS);

        for (int owner = 0; owner < world_size; owner++) {
                size_t owner_base = (size_t)owner * INT_OP_COUNT;

                for (size_t op_index = 0;
                     op_index < INT_OP_COUNT;
                     op_index++) {
                        uint64_t contribution = (uint64_t)(rank + 1);

                        rc = dstm_darray_accumulate(
                                a,
                                owner_base + op_index,
                                &contribution,
                                0,
                                1,
                                DSTM_DT_UINT64,
                                integer_ops[op_index]);
                        assert(rc == DSTM_SUCCESS);
                }
        }

        size_t local_base = dstm_darray_global_offset(a);
        uint64_t result[INT_OP_COUNT];

        for (size_t i = 0; i < INT_OP_COUNT; i++) {
                result[i] = UINT64_MAX;

                rc = dstm_darray_get(a,
                                     local_base + i,
                                     &result[i],
                                     1,
                                     0);
                assert(rc == DSTM_SUCCESS);
        }

        assert(result[INT_OP_SUM] == expected_u64_sum());
        assert(result[INT_OP_PROD] == 0);
        assert(result[INT_OP_MIN] == 0);
        assert(result[INT_OP_MAX] == (uint64_t)world_size);
        assert(result[INT_OP_BOR] == expected_u64_or());
        assert(result[INT_OP_BAND] == 0);
        assert(result[INT_OP_BXOR] == expected_u64_xor());

        dstm_darray_destroy(&a);
        session_end(&dist, 1);
}

/*******************************************************************************
 * Floating-point datatypes and operators
 ******************************************************************************/

enum {
        FP_OP_SUM = 0,
        FP_OP_PROD,
        FP_OP_MIN,
        FP_OP_MAX,
        FP_OP_COUNT
};

static const dstm_reduce_op_t fp_ops[FP_OP_COUNT] = {
        DSTM_OP_SUM,
        DSTM_OP_PROD,
        DSTM_OP_MIN,
        DSTM_OP_MAX,
};

static void test_float_operators(void)
{
        dstm_distance_t *dist = session_begin();

        dstm_darray_t *a = NULL;
        int rc = dstm_darray_create(&a, sizeof(float), FP_OP_COUNT);
        assert(rc == DSTM_SUCCESS);

        for (int owner = 0; owner < world_size; owner++) {
                size_t owner_base = (size_t)owner * FP_OP_COUNT;

                for (size_t op_index = 0;
                     op_index < FP_OP_COUNT;
                     op_index++) {
                        float contribution;

                        if (op_index == FP_OP_MIN)
                                contribution = -(float)(rank + 1);
                        else
                                contribution = (float)(rank + 1);

                        rc = dstm_darray_accumulate(
                                a,
                                owner_base + op_index,
                                &contribution,
                                0,
                                1,
                                DSTM_DT_FLOAT,
                                fp_ops[op_index]);
                        assert(rc == DSTM_SUCCESS);
                }
        }

        size_t local_base = dstm_darray_global_offset(a);
        float result[FP_OP_COUNT];

        for (size_t i = 0; i < FP_OP_COUNT; i++) {
                result[i] = NAN;

                rc = dstm_darray_get(a,
                                     local_base + i,
                                     &result[i],
                                     1,
                                     0);
                assert(rc == DSTM_SUCCESS);
        }

        assert(result[FP_OP_SUM] == (float)expected_i64_sum());
        assert(result[FP_OP_PROD] == 0.0f);
        assert(result[FP_OP_MIN] == -(float)world_size);
        assert(result[FP_OP_MAX] == (float)world_size);

        dstm_darray_destroy(&a);
        session_end(&dist, 1);
}

static void test_double_operators(void)
{
        dstm_distance_t *dist = session_begin();

        dstm_darray_t *a = NULL;
        int rc = dstm_darray_create(&a, sizeof(double), FP_OP_COUNT);
        assert(rc == DSTM_SUCCESS);

        for (int owner = 0; owner < world_size; owner++) {
                size_t owner_base = (size_t)owner * FP_OP_COUNT;

                for (size_t op_index = 0;
                     op_index < FP_OP_COUNT;
                     op_index++) {
                        double contribution;

                        if (op_index == FP_OP_MIN)
                                contribution = -(double)(rank + 1);
                        else
                                contribution = (double)(rank + 1);

                        rc = dstm_darray_accumulate(
                                a,
                                owner_base + op_index,
                                &contribution,
                                0,
                                1,
                                DSTM_DT_DOUBLE,
                                fp_ops[op_index]);
                        assert(rc == DSTM_SUCCESS);
                }
        }

        size_t local_base = dstm_darray_global_offset(a);
        double result[FP_OP_COUNT];

        for (size_t i = 0; i < FP_OP_COUNT; i++) {
                result[i] = NAN;

                rc = dstm_darray_get(a,
                                     local_base + i,
                                     &result[i],
                                     1,
                                     0);
                assert(rc == DSTM_SUCCESS);
        }

        assert(result[FP_OP_SUM] == (double)expected_i64_sum());
        assert(result[FP_OP_PROD] == 0.0);
        assert(result[FP_OP_MIN] == -(double)world_size);
        assert(result[FP_OP_MAX] == (double)world_size);

        dstm_darray_destroy(&a);
        session_end(&dist, 1);
}

/*******************************************************************************
 * Elementwise count > 1
 ******************************************************************************/

static void test_darray_elementwise_vector(void)
{
        dstm_distance_t *dist = session_begin();

        /*
         * Each distributed-array element contains four int32_t values.
         * The wrapper must derive count=4 from element_size/datatype_size.
         */
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
                rc = dstm_darray_accumulate(a,
                                            (size_t)owner,
                                            contribution,
                                            0,
                                            1,
                                            DSTM_DT_INT32,
                                            DSTM_OP_SUM);
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
 * Version isolation
 ******************************************************************************/

static void test_darray_multiple_versions(void)
{
        dstm_distance_t *dist = session_begin();

        dstm_darray_t *a = NULL;
        int rc = dstm_darray_create(&a, sizeof(int32_t), 1);
        assert(rc == DSTM_SUCCESS);

        /*
         * Version 1 receives rank+1.
         * Version 2 receives 10*(rank+1).
         */
        int32_t at_v1 = (int32_t)(rank + 1);
        int32_t at_v2 = 10 * (int32_t)(rank + 1);

        for (int owner = 0; owner < world_size; owner++) {
                rc = dstm_darray_accumulate(a,
                                            (size_t)owner,
                                            &at_v1,
                                            0,
                                            1,
                                            DSTM_DT_INT32,
                                            DSTM_OP_SUM);
                assert(rc == DSTM_SUCCESS);

                rc = dstm_darray_accumulate(a,
                                            (size_t)owner,
                                            &at_v2,
                                            0,
                                            2,
                                            DSTM_DT_INT32,
                                            DSTM_OP_SUM);
                assert(rc == DSTM_SUCCESS);
        }

        size_t local_index = dstm_darray_global_offset(a);

        int32_t got_v1 = -1;
        rc = dstm_darray_get(a, local_index, &got_v1, 1, 0);
        assert(rc == DSTM_SUCCESS);
        assert(got_v1 == expected_i32_sum());

        /*
         * Every rank now advances to time 2 before reading version 2.
         */
        int32_t got_v2 = -1;
        rc = dstm_darray_get(a, local_index, &got_v2, 2, 0);
        assert(rc == DSTM_SUCCESS);
        assert(got_v2 == 10 * expected_i32_sum());

        dstm_darray_destroy(&a);
        session_end(&dist, 2);
}

/*******************************************************************************
 * Dgraph builder
 *
 * Each rank owns one vertex. Every vertex has an outgoing edge to every
 * vertex, including itself. This makes every source rank a graph neighbor
 * and allows all ranks to resolve all tested edges.
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
        int rc = dstm_dgraph_create(&g,
                                    vertex_state_size,
                                    edge_state_size,
                                    1,
                                    degrees,
                                    dsts,
                                    NULL,
                                    NULL);
        assert(rc == DSTM_SUCCESS);
        assert(g != NULL);

        free(dsts);
        return g;
}

/*******************************************************************************
 * Dgraph validation
 ******************************************************************************/

static void test_dgraph_invalid_arguments(void)
{
        dstm_distance_t *dist = session_begin();
        dstm_dgraph_t *g =
                build_complete_graph(sizeof(int32_t), sizeof(int32_t));

        size_t local_vertex = dstm_dgraph_local_vertex_offset(g);
        int32_t value = 1;

        assert(dstm_dgraph_vertex_accumulate(NULL,
                                            local_vertex,
                                            &value,
                                            0,
                                            1,
                                            DSTM_DT_INT32,
                                            DSTM_OP_SUM) == -DSTM_EINVAL);

        assert(dstm_dgraph_vertex_accumulate(g,
                                            local_vertex,
                                            NULL,
                                            0,
                                            1,
                                            DSTM_DT_INT32,
                                            DSTM_OP_SUM) == -DSTM_EINVAL);

        assert(dstm_dgraph_vertex_accumulate(g,
                                            local_vertex,
                                            &value,
                                            0,
                                            -1,
                                            DSTM_DT_INT32,
                                            DSTM_OP_SUM) == -DSTM_EINVAL);

        assert(dstm_dgraph_vertex_accumulate(g,
                                            local_vertex,
                                            &value,
                                            0,
                                            1,
                                            DSTM_DT_INT32,
                                            DSTM_OP_APPEND) == -DSTM_EINVAL);

        assert(dstm_dgraph_edge_accumulate(NULL,
                                          local_vertex,
                                          local_vertex,
                                          &value,
                                          0,
                                          1,
                                          DSTM_DT_INT32,
                                          DSTM_OP_SUM) == -DSTM_EINVAL);

        assert(dstm_dgraph_edge_accumulate(g,
                                          local_vertex,
                                          local_vertex,
                                          NULL,
                                          0,
                                          1,
                                          DSTM_DT_INT32,
                                          DSTM_OP_SUM) == -DSTM_EINVAL);

        /*
         * Every vertex has edges to all valid vertices. world_size is
         * therefore outside the declared vertex-id range.
         */
        size_t invalid_vertex = (size_t)world_size;

        assert(dstm_dgraph_vertex_accumulate(g,
                                            invalid_vertex,
                                            &value,
                                            0,
                                            1,
                                            DSTM_DT_INT32,
                                            DSTM_OP_SUM) == -DSTM_EINVAL);

        assert(dstm_dgraph_edge_accumulate(g,
                                          invalid_vertex,
                                          local_vertex,
                                          &value,
                                          0,
                                          1,
                                          DSTM_DT_INT32,
                                          DSTM_OP_SUM) != DSTM_SUCCESS);

        dstm_dgraph_destroy(&g);
        session_end(&dist, 0);
}

static void test_dgraph_bad_state_width(void)
{
        dstm_distance_t *dist = session_begin();

        dstm_dgraph_t *g = build_complete_graph(6, 6);
        size_t local_vertex = dstm_dgraph_local_vertex_offset(g);
        unsigned char value[8] = {0};

        assert(dstm_dgraph_vertex_accumulate(g,
                                            local_vertex,
                                            value,
                                            0,
                                            1,
                                            DSTM_DT_INT32,
                                            DSTM_OP_SUM) == -DSTM_EINVAL);

        assert(dstm_dgraph_edge_accumulate(g,
                                          local_vertex,
                                          local_vertex,
                                          value,
                                          0,
                                          1,
                                          DSTM_DT_INT32,
                                          DSTM_OP_SUM) == -DSTM_EINVAL);

        dstm_dgraph_destroy(&g);
        session_end(&dist, 0);
}

/*******************************************************************************
 * Dgraph vertex and edge functional coverage
 ******************************************************************************/

static void test_dgraph_vertex_and_edge_accumulate(void)
{
        dstm_distance_t *dist = session_begin();

        /*
         * Both vertex and edge states contain two int32_t components.
         */
        dstm_dgraph_t *g = build_complete_graph(2 * sizeof(int32_t),
                                                2 * sizeof(int32_t));

        int32_t vertex_contribution[2] = {
                (int32_t)(rank + 1),
                1,
        };

        int32_t edge_contribution[2] = {
                2 * (int32_t)(rank + 1),
                -(int32_t)(rank + 1),
        };

        /*
         * Every producer contributes to the one vertex owned by each rank
         * and to that owner's self-edge.
         */
        for (int owner = 0; owner < world_size; owner++) {
                size_t vertex = (size_t)owner;

                int rc = dstm_dgraph_vertex_accumulate(
                        g,
                        vertex,
                        vertex_contribution,
                        0,
                        1,
                        DSTM_DT_INT32,
                        DSTM_OP_SUM);
                assert(rc == DSTM_SUCCESS);

                rc = dstm_dgraph_edge_accumulate(
                        g,
                        vertex,
                        vertex,
                        edge_contribution,
                        0,
                        1,
                        DSTM_DT_INT32,
                        DSTM_OP_SUM);
                assert(rc == DSTM_SUCCESS);
        }

        size_t local_vertex = dstm_dgraph_local_vertex_offset(g);

        int32_t vertex_result[2] = {-1, -1};
        int rc = dstm_dgraph_vertex_get(g,
                                        local_vertex,
                                        vertex_result,
                                        1,
                                        0);
        assert(rc == DSTM_SUCCESS);
        assert(vertex_result[0] == expected_i32_sum());
        assert(vertex_result[1] == (int32_t)world_size);

        int32_t edge_result[2] = {-1, -1};
        rc = dstm_dgraph_edge_get(g,
                                  local_vertex,
                                  local_vertex,
                                  edge_result,
                                  1,
                                  0);
        assert(rc == DSTM_SUCCESS);
        assert(edge_result[0] == 2 * expected_i32_sum());
        assert(edge_result[1] == -expected_i32_sum());

        dstm_dgraph_destroy(&g);
        session_end(&dist, 1);
}

/*******************************************************************************
 * Dgraph zero-sized state behavior
 ******************************************************************************/

static void test_dgraph_zero_sized_state(void)
{
        dstm_distance_t *dist = session_begin();

        dstm_dgraph_t *g = build_complete_graph(0, 0);
        size_t local_vertex = dstm_dgraph_local_vertex_offset(g);
        int32_t ignored = 123;

        /*
         * The current wrappers define accumulation on an existing zero-sized
         * state as a successful no-op.
         */
        int rc = dstm_dgraph_vertex_accumulate(g,
                                               local_vertex,
                                               &ignored,
                                               0,
                                               1,
                                               DSTM_DT_INT32,
                                               DSTM_OP_SUM);
        assert(rc == DSTM_SUCCESS);

        rc = dstm_dgraph_edge_accumulate(g,
                                         local_vertex,
                                         local_vertex,
                                         &ignored,
                                         0,
                                         1,
                                         DSTM_DT_INT32,
                                         DSTM_OP_SUM);
        assert(rc == DSTM_SUCCESS);

        /*
         * An invalid vertex must still be rejected even when state size is
         * zero.
         */
        rc = dstm_dgraph_vertex_accumulate(g,
                                           (size_t)world_size,
                                           &ignored,
                                           0,
                                           1,
                                           DSTM_DT_INT32,
                                           DSTM_OP_SUM);
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

        /* Validation surface. */
        test_darray_invalid_arguments();

        /* All datatypes and all valid operators. */
        test_int32_operators();
        test_int64_operators();
        test_uint32_operators();
        test_uint64_operators();
        test_float_operators();
        test_double_operators();

        /* Wrapper-derived element count and version semantics. */
        test_darray_elementwise_vector();
        test_darray_multiple_versions();

        /* Dgraph wrappers. */
        test_dgraph_invalid_arguments();
        test_dgraph_bad_state_width();
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
