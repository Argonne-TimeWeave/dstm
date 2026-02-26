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
 * main
 ******************************************************************************/

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);
	dstm_init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

	dstm_fini();
	dstm_finalize();
	MPI_Finalize();
	return 0;
}
