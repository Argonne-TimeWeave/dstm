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

/* Verify a memory region is writable and holds the expected byte pattern. */
static int verify_memory(void *ptr, size_t size, unsigned char pattern)
{
	if (ptr == NULL)
		return 0;
	unsigned char *bytes = (unsigned char *)ptr;
	for (size_t i = 0; i < size; i++) {
		if (bytes[i] != pattern)
			return 0;
	}
	return 1;
}

/*******************************************************************************
 * create_destroy tests
 ******************************************************************************/

/* Basic create + destroy cycle. */
static void test_create_destroy_basic(void)
{
	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create(&a, sizeof(int), 20);
	assert(rc == 0);
	assert(a != NULL);
	dstm_darray_destroy(&a);
	assert(a == NULL);
}

/* element_size=0 must return -DSTM_EINVAL and leave the pointer NULL. */
static void test_create_element_size_zero(void)
{
	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create(&a, 0, 20);
	assert(rc == -DSTM_EINVAL);
	assert(a == NULL);
}

/* local_elements=0 must return -DSTM_EINVAL and leave the pointer NULL. */
static void test_create_local_elements_zero(void)
{
	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create(&a, sizeof(int), 0);
	assert(rc == -DSTM_EINVAL);
	assert(a == NULL);
}

/* Both element_size=0 and local_elements=0 must return -DSTM_EINVAL. */
static void test_create_both_zero(void)
{
	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create(&a, 0, 0);
	assert(rc == -DSTM_EINVAL);
	assert(a == NULL);
}

/* dstm_darray_destroy must not crash when given NULL or &NULL. */
static void test_destroy_null_safety(void)
{
	dstm_darray_destroy(NULL);
	dstm_darray_t *a = NULL;
	dstm_darray_destroy(&a);
	assert(a == NULL);
}

/* Multiple create/destroy cycles must not corrupt state. */
static void test_create_destroy_cycles(void)
{
	for (int i = 0; i < 5; i++) {
		dstm_darray_t *a = NULL;
		int rc = dstm_darray_create(&a, sizeof(double), 10);
		assert(rc == 0);
		assert(a != NULL);
		dstm_darray_destroy(&a);
		assert(a == NULL);
	}
}

/*******************************************************************************
 * accessor tests
 ******************************************************************************/

/* dstm_darray_local_count must return exactly what was passed to create. */
static void test_local_count(void)
{
	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create(&a, sizeof(int), 17);
	assert(rc == 0);
	assert(dstm_darray_local_count(a) == 17);
	dstm_darray_destroy(&a);
}

/* Rank 0 must always have global_offset == 0. */
static void test_global_offset_rank0(void)
{
	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create(&a, sizeof(int), 20);
	assert(rc == 0);
	if (rank == 0)
		assert(dstm_darray_global_offset(a) == 0);
	dstm_darray_destroy(&a);
}

/* The global_offset of each rank must equal the prefix-sum of all previous
 * ranks' local counts (verified using MPI_Exscan). */
static void test_global_offsets_consistency(void)
{
	size_t local_n = 7 + (size_t)rank; /* different per rank */
	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create(&a, sizeof(int), local_n);
	assert(rc == 0);

	size_t my_offset = dstm_darray_global_offset(a);

	/* MPI_Exscan gives the exclusive prefix-sum of local_n across ranks. */
	size_t expected_offset = 0;
	MPI_Exscan(&local_n, &expected_offset, 1, MPI_UNSIGNED_LONG_LONG,
	           MPI_SUM, MPI_COMM_WORLD);
	/* On rank 0 MPI_Exscan leaves the output buffer undefined; treat as 0.
	 */
	if (rank == 0)
		expected_offset = 0;

	assert(my_offset == expected_offset);
	dstm_darray_destroy(&a);
}

/* get_local_ptr must return a non-NULL, writable pointer. */
static void test_get_local_ptr_writable(void)
{
	size_t n = 32;
	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create(&a, sizeof(int), n);
	assert(rc == 0);

	void *ptr = dstm_darray_get_local_ptr(a);
	assert(ptr != NULL);

	/* Write a pattern then read it back. */
	memset(ptr, 0xAB, n * sizeof(int));
	assert(verify_memory(ptr, n * sizeof(int), 0xAB));

	dstm_darray_destroy(&a);
}

/*******************************************************************************
 * create_equal tests
 ******************************************************************************/

/* Even division: total_elements divisible by world_size. */
static void test_create_equal_even(void)
{
	size_t total = (size_t)world_size * 4;
	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create_equal(&a, sizeof(int), total);
	assert(rc == 0);
	assert(dstm_darray_local_count(a) == 4);
	dstm_darray_destroy(&a);
}

/* Uneven division: remainder distributed to lower-ranked ranks.
 * With total = world_size*4 + 1, rank 0 gets 5, all others get 4. */
static void test_create_equal_uneven(void)
{
	size_t total = (size_t)world_size * 4 + 1;
	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create_equal(&a, sizeof(int), total);
	assert(rc == 0);

	size_t expected = (rank == 0) ? 5 : 4;
	assert(dstm_darray_local_count(a) == expected);
	dstm_darray_destroy(&a);
}

/* Sum of all local counts must equal the total passed to create_equal. */
static void test_create_equal_total_matches(void)
{
	size_t total = (size_t)world_size * 7 + 3;
	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create_equal(&a, sizeof(int), total);
	assert(rc == 0);

	size_t local = dstm_darray_local_count(a);
	size_t sum = 0;
	MPI_Allreduce(&local, &sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
	              MPI_COMM_WORLD);
	assert(sum == total);
	dstm_darray_destroy(&a);
}

/* total_elements=0 must propagate -DSTM_EINVAL from dstm_darray_create. */
static void test_create_equal_total_zero(void)
{
	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create_equal(&a, sizeof(int), 0);
	assert(rc == -DSTM_EINVAL);
	assert(a == NULL);
}

/*******************************************************************************
 * put/get invalid argument tests
 ******************************************************************************/

/* dstm_darray_put with a NULL src pointer must return -DSTM_EINVAL. */
static void test_put_null_src(void)
{
	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create(&a, sizeof(int), 10);
	assert(rc == 0);

	size_t idx = dstm_darray_global_offset(a);
	rc = dstm_darray_put(a, idx, NULL, 0, 0);
	assert(rc == -DSTM_EINVAL);

	dstm_darray_destroy(&a);
}

/* dstm_darray_put with a negative delta must return -DSTM_EINVAL. */
static void test_put_negative_delta(void)
{
	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create(&a, sizeof(int), 10);
	assert(rc == 0);

	int val = 42;
	size_t idx = dstm_darray_global_offset(a);
	rc = dstm_darray_put(a, idx, &val, 0, -1);
	assert(rc == -DSTM_EINVAL);

	dstm_darray_destroy(&a);
}

/* dstm_darray_get with a NULL dst pointer must return -DSTM_EINVAL. */
static void test_get_null_dst(void)
{
	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create(&a, sizeof(int), 10);
	assert(rc == 0);

	size_t idx = dstm_darray_global_offset(a);
	rc = dstm_darray_get(a, idx, NULL, 0, 0);
	assert(rc == -DSTM_EINVAL);

	dstm_darray_destroy(&a);
}

/* dstm_darray_get with a positive delta must return -DSTM_EINVAL. */
static void test_get_positive_delta(void)
{
	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create(&a, sizeof(int), 10);
	assert(rc == 0);

	int val = 0;
	size_t idx = dstm_darray_global_offset(a);
	rc = dstm_darray_get(a, idx, &val, 0, 1);
	assert(rc == -DSTM_EINVAL);

	dstm_darray_destroy(&a);
}

/*******************************************************************************
 * range/rangev invalid argument tests
 ******************************************************************************/

/* count == 0 is a successful no-op even with all-NULL buffers. */
static void test_range_zero_count_noop(void)
{
	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create(&a, sizeof(int), 10);
	assert(rc == 0);

	assert(dstm_darray_get_range(a, NULL, 0, NULL, 0, 0) == DSTM_SUCCESS);
	assert(dstm_darray_put_range(a, NULL, 0, NULL, 0, 0) == DSTM_SUCCESS);
	assert(dstm_darray_get_rangev(a, NULL, 0, NULL, 0, 0) == DSTM_SUCCESS);
	assert(dstm_darray_put_rangev(a, NULL, 0, NULL, 0, 0) == DSTM_SUCCESS);

	dstm_darray_destroy(&a);
}

/* NULL darray must return -DSTM_EINVAL even with count == 0. */
static void test_range_null_darray(void)
{
	size_t idx = 0;
	int buf = 0;
	void *p = &buf;
	const void *cp = &buf;

	assert(dstm_darray_get_range(NULL, NULL, 0, NULL, 0, 0) ==
	       -DSTM_EINVAL);
	assert(dstm_darray_put_range(NULL, NULL, 0, NULL, 0, 0) ==
	       -DSTM_EINVAL);
	assert(dstm_darray_get_rangev(NULL, NULL, 0, NULL, 0, 0) ==
	       -DSTM_EINVAL);
	assert(dstm_darray_put_rangev(NULL, NULL, 0, NULL, 0, 0) ==
	       -DSTM_EINVAL);

	assert(dstm_darray_get_range(NULL, &idx, 1, &buf, 0, 0) ==
	       -DSTM_EINVAL);
	assert(dstm_darray_put_range(NULL, &idx, 1, &buf, 0, 0) ==
	       -DSTM_EINVAL);
	assert(dstm_darray_get_rangev(NULL, &idx, 1, &p, 0, 0) ==
	       -DSTM_EINVAL);
	assert(dstm_darray_put_rangev(NULL, &idx, 1, &cp, 0, 0) ==
	       -DSTM_EINVAL);
}

/* For count > 0, NULL indices or NULL buffer-pointer must return
 * -DSTM_EINVAL before any underlying call happens. */
static void test_range_null_args(void)
{
	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create(&a, sizeof(int), 10);
	assert(rc == 0);

	int buf = 0;
	size_t idx = dstm_darray_global_offset(a);
	void *p = &buf;
	const void *cp = &buf;

	/* NULL indices */
	assert(dstm_darray_get_range(a, NULL, 1, &buf, 0, 0) == -DSTM_EINVAL);
	assert(dstm_darray_put_range(a, NULL, 1, &buf, 0, 0) == -DSTM_EINVAL);
	assert(dstm_darray_get_rangev(a, NULL, 1, &p, 0, 0) == -DSTM_EINVAL);
	assert(dstm_darray_put_rangev(a, NULL, 1, &cp, 0, 0) == -DSTM_EINVAL);

	/* NULL buffer / pointer array */
	assert(dstm_darray_get_range(a, &idx, 1, NULL, 0, 0) == -DSTM_EINVAL);
	assert(dstm_darray_put_range(a, &idx, 1, NULL, 0, 0) == -DSTM_EINVAL);
	assert(dstm_darray_get_rangev(a, &idx, 1, NULL, 0, 0) == -DSTM_EINVAL);
	assert(dstm_darray_put_rangev(a, &idx, 1, NULL, 0, 0) == -DSTM_EINVAL);

	dstm_darray_destroy(&a);
}

/* Negative delta on put_range must propagate -DSTM_EINVAL from dstm_put. */
static void test_put_range_negative_delta(void)
{
	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create(&a, sizeof(int), 10);
	assert(rc == 0);

	int val = 42;
	size_t idx = dstm_darray_global_offset(a);
	rc = dstm_darray_put_range(a, &idx, 1, &val, 0, -1);
	assert(rc == -DSTM_EINVAL);

	const void *p = &val;
	rc = dstm_darray_put_rangev(a, &idx, 1, &p, 0, -1);
	assert(rc == -DSTM_EINVAL);

	dstm_darray_destroy(&a);
}

/* Positive delta on get_range must propagate -DSTM_EINVAL from dstm_get. */
static void test_get_range_positive_delta(void)
{
	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create(&a, sizeof(int), 10);
	assert(rc == 0);

	int val = 0;
	size_t idx = dstm_darray_global_offset(a);
	rc = dstm_darray_get_range(a, &idx, 1, &val, 0, 1);
	assert(rc == -DSTM_EINVAL);

	void *p = &val;
	rc = dstm_darray_get_rangev(a, &idx, 1, &p, 0, 1);
	assert(rc == -DSTM_EINVAL);

	dstm_darray_destroy(&a);
}

/*******************************************************************************
 * range/rangev functional round-trip tests
 *
 * These use the same time-driven pattern as the allgatherv stencil:
 *   - put at time t with delta = +1 publishes the local slice for time
 *     t+1;
 *   - get at time t with delta = -1 reads each element as published at
 *     time t-1.
 *
 * Pre-populate every global index with a known initial value via puts
 * advancing time, then read back via get_range/get_rangev at a strictly
 * later time with delta = -1, with each rank only reading its own
 * partition (delta == 0 is only permitted for the local rank).
 ******************************************************************************/

/* Open a fresh dstm session so each functional test runs on its own
 * timeline starting at logical time 0.  Mirrors the pattern used by
 * tests/mpi/multi_session.c. */
static dstm_distance_t *session_begin(void)
{
	dstm_distance_t *dist = NULL;
	int rc = dstm_distance_create_complete(&dist, (size_t)world_size);
	assert(rc == DSTM_SUCCESS);
	rc = dstm_start(dist);
	assert(rc == DSTM_SUCCESS);
	return dist;
}

/* Close the session opened by session_begin.  @p last_time must be >=
 * the highest effective (time + delta) any put or get reached during
 * this session. */
static void session_end(dstm_distance_t **dist, dstm_time_t last_time)
{
	int rc = dstm_fini(last_time);
	assert(rc == DSTM_SUCCESS);
	dstm_distance_destroy(dist);
}

/* Build a darray of @p per_rank ints per rank.  Each rank writes its
 * own slice with the pattern v[i] = (int)(global_index * 10 + tag) at
 * time @p time_put with delta = +1.  Returns the darray. */
static dstm_darray_t *build_and_populate(size_t per_rank,
                                         dstm_time_t time_put,
                                         int tag)
{
	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create(&a, sizeof(int), per_rank);
	assert(rc == 0);

	size_t off = dstm_darray_global_offset(a);
	for (size_t i = 0; i < per_rank; i++) {
		int val = (int)((off + i) * 10 + (size_t)tag);
		rc = dstm_darray_put(a, off + i, &val, time_put, 1);
		assert(rc == DSTM_SUCCESS);
	}
	return a;
}

/* get_range over a contiguous slice of indices owned by the local rank. */
static void test_get_range_contiguous_local(void)
{
	const size_t per_rank = 8;
	const dstm_time_t t_put = 0;
	const dstm_time_t t_get = 2;
	const int tag = 1;

	dstm_distance_t *dist = session_begin();

	dstm_darray_t *a = build_and_populate(per_rank, t_put, tag);
	size_t off = dstm_darray_global_offset(a);

	size_t indices[per_rank];
	for (size_t i = 0; i < per_rank; i++)
		indices[i] = off + i;

	int got[per_rank];
	memset(got, 0xCC, sizeof(got));
	int rc = dstm_darray_get_range(a, indices, per_rank, got, t_get, -1);
	assert(rc == DSTM_SUCCESS);

	for (size_t i = 0; i < per_rank; i++) {
		int expected = (int)((off + i) * 10 + (size_t)tag);
		assert(got[i] == expected);
	}

	dstm_darray_destroy(&a);
	session_end(&dist, t_put + 1);
}

/* get_range with a discontinuous index set (every other element of the
 * local partition, in reverse order). */
static void test_get_range_discontinuous(void)
{
	const size_t per_rank = 8;
	const dstm_time_t t_put = 0;
	const dstm_time_t t_get = 2;
	const int tag = 2;

	dstm_distance_t *dist = session_begin();

	dstm_darray_t *a = build_and_populate(per_rank, t_put, tag);
	size_t off = dstm_darray_global_offset(a);

	/* Pick reverse-order even-indexed elements: off+6, off+4, off+2, off+0
	 */
	const size_t count = 4;
	size_t indices[4] = { off + 6, off + 4, off + 2, off + 0 };

	int got[4];
	memset(got, 0xCC, sizeof(got));
	int rc = dstm_darray_get_range(a, indices, count, got, t_get, -1);
	assert(rc == DSTM_SUCCESS);

	for (size_t i = 0; i < count; i++) {
		int expected = (int)(indices[i] * 10 + (size_t)tag);
		assert(got[i] == expected);
	}

	dstm_darray_destroy(&a);
	session_end(&dist, t_put + 1);
}

/* get_range that spans every rank's partition: each rank reads the
 * first element of every rank's slice.  This forces cross-rank
 * resolution inside the loop. */
static void test_get_range_cross_rank(void)
{
	const size_t per_rank = 4;
	const dstm_time_t t_put = 0;
	const dstm_time_t t_get = 2;
	const int tag = 3;

	dstm_distance_t *dist = session_begin();

	dstm_darray_t *a = build_and_populate(per_rank, t_put, tag);

	size_t *indices = malloc((size_t)world_size * sizeof(size_t));
	int *got = malloc((size_t)world_size * sizeof(int));
	assert(indices != NULL && got != NULL);

	for (int r = 0; r < world_size; r++)
		indices[r] = (size_t)r * per_rank; /* first elt of rank r */

	int rc = dstm_darray_get_range(a, indices, (size_t)world_size, got,
	                               t_get, -1);
	assert(rc == DSTM_SUCCESS);

	for (int r = 0; r < world_size; r++) {
		int expected = (int)(indices[r] * 10 + (size_t)tag);
		assert(got[r] == expected);
	}

	free(got);
	free(indices);
	dstm_darray_destroy(&a);
	session_end(&dist, t_put + 1);
}

/* put_range followed by get_range round-trip on the local partition. */
static void test_put_range_roundtrip(void)
{
	const size_t per_rank = 6;
	const dstm_time_t t_put = 0;
	const dstm_time_t t_get = 2;
	const int tag = 4;

	dstm_distance_t *dist = session_begin();

	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create(&a, sizeof(int), per_rank);
	assert(rc == 0);

	size_t off = dstm_darray_global_offset(a);
	size_t indices[per_rank];
	int payload[per_rank];
	for (size_t i = 0; i < per_rank; i++) {
		indices[i] = off + i;
		payload[i] = (int)((off + i) * 100 + (size_t)tag);
	}

	rc = dstm_darray_put_range(a, indices, per_rank, payload, t_put, 1);
	assert(rc == DSTM_SUCCESS);

	int got[per_rank];
	memset(got, 0xDD, sizeof(got));
	rc = dstm_darray_get_range(a, indices, per_rank, got, t_get, -1);
	assert(rc == DSTM_SUCCESS);

	for (size_t i = 0; i < per_rank; i++)
		assert(got[i] == payload[i]);

	dstm_darray_destroy(&a);
	session_end(&dist, t_put + 1);
}

/* put_rangev/get_rangev round-trip using scattered per-element buffers
 * (each element lives in its own malloc'd block to actually exercise
 * the per-element pointer indirection). */
static void test_rangev_scattered_roundtrip(void)
{
	const size_t per_rank = 5;
	const dstm_time_t t_put = 0;
	const dstm_time_t t_get = 2;
	const int tag = 5;

	dstm_distance_t *dist = session_begin();

	dstm_darray_t *a = NULL;
	int rc = dstm_darray_create(&a, sizeof(int), per_rank);
	assert(rc == 0);

	size_t off = dstm_darray_global_offset(a);

	size_t indices[per_rank];
	int *src_blocks[per_rank];
	int *dst_blocks[per_rank];
	const void *src_ptrs[per_rank];
	void *dst_ptrs[per_rank];

	for (size_t i = 0; i < per_rank; i++) {
		indices[i] = off + i;
		src_blocks[i] = malloc(sizeof(int));
		dst_blocks[i] = malloc(sizeof(int));
		assert(src_blocks[i] != NULL && dst_blocks[i] != NULL);
		*src_blocks[i] = (int)((off + i) * 1000 + (size_t)tag);
		*dst_blocks[i] = -1;
		src_ptrs[i] = src_blocks[i];
		dst_ptrs[i] = dst_blocks[i];
	}

	rc = dstm_darray_put_rangev(a, indices, per_rank, src_ptrs, t_put, 1);
	assert(rc == DSTM_SUCCESS);

	rc = dstm_darray_get_rangev(a, indices, per_rank, dst_ptrs, t_get, -1);
	assert(rc == DSTM_SUCCESS);

	for (size_t i = 0; i < per_rank; i++) {
		assert(*dst_blocks[i] == *src_blocks[i]);
		free(src_blocks[i]);
		free(dst_blocks[i]);
	}

	dstm_darray_destroy(&a);
	session_end(&dist, t_put + 1);
}

/* get_rangev with a discontinuous index set and scattered destinations.
 * Mixes the two flexibility axes the v form is meant to enable. */
static void test_get_rangev_discontinuous_scattered(void)
{
	const size_t per_rank = 8;
	const dstm_time_t t_put = 0;
	const dstm_time_t t_get = 2;
	const int tag = 6;

	dstm_distance_t *dist = session_begin();

	dstm_darray_t *a = build_and_populate(per_rank, t_put, tag);
	size_t off = dstm_darray_global_offset(a);

	const size_t count = 3;
	size_t indices[3] = { off + 5, off + 1, off + 7 };
	int *blocks[3];
	void *ptrs[3];
	for (size_t i = 0; i < count; i++) {
		blocks[i] = malloc(sizeof(int));
		assert(blocks[i] != NULL);
		*blocks[i] = -1;
		ptrs[i] = blocks[i];
	}

	int rc = dstm_darray_get_rangev(a, indices, count, ptrs, t_get, -1);
	assert(rc == DSTM_SUCCESS);

	for (size_t i = 0; i < count; i++) {
		int expected = (int)(indices[i] * 10 + (size_t)tag);
		assert(*blocks[i] == expected);
		free(blocks[i]);
	}

	dstm_darray_destroy(&a);
	session_end(&dist, t_put + 1);
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

	/* Outer session covers everything that doesn't actually drive
	 * put/get traffic: create/destroy, accessors, create_equal, and
	 * argument-validation tests (which always early-return with an
	 * error before touching the timeline). */
	dstm_distance_t *dist;
	dstm_distance_create_complete(&dist, world_size);
	dstm_start(dist);

	/* create_destroy */
	test_create_destroy_basic();
	test_create_element_size_zero();
	test_create_local_elements_zero();
	test_create_both_zero();
	test_destroy_null_safety();
	test_create_destroy_cycles();

	/* accessors */
	test_local_count();
	test_global_offset_rank0();
	test_global_offsets_consistency();
	test_get_local_ptr_writable();

	/* create_equal */
	test_create_equal_even();
	test_create_equal_uneven();
	test_create_equal_total_matches();
	test_create_equal_total_zero();

	/* put/get invalid args */
	test_put_null_src();
	test_put_negative_delta();
	test_get_null_dst();
	test_get_positive_delta();

	/* range/rangev invalid args */
	test_range_zero_count_noop();
	test_range_null_darray();
	test_range_null_args();
	test_put_range_negative_delta();
	test_get_range_positive_delta();

	dstm_fini(0);
	dstm_distance_destroy(&dist);

	/* Functional round-trips: each test owns its own start/fini
	 * session so they don't have to share a single timeline. */
	test_get_range_contiguous_local();
	test_get_range_discontinuous();
	test_get_range_cross_rank();
	test_put_range_roundtrip();
	test_rangev_scattered_roundtrip();
	test_get_rangev_discontinuous_scattered();

	dstm_finalize();
	MPI_Finalize();
	return 0;
}
