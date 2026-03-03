/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "config.h"

#include <mpi.h>
#include <stdlib.h>

#include "dstm.h"

#include "internal/globals.h"
#include "internal/region.h"

struct dstm_darray_s {
	size_t element_size;
	size_t total_elements;
	size_t local_elements;
	dstm_ptr_t local_ptr;
	dstm_region_t *region;
	/* per-rank offset from the global indices */
	size_t *rank_offsets;
	/* per-rank size of the local part, needed for gather operations */
	size_t *rank_elements;
};

/* from a global index into the darray, figure out the rank to query.
 * Can be done with a binary search into the per-rank offsets: just find
 * witch rank has the largest lower than index offset. */
static int darray_find_rank(dstm_darray_t *darray, size_t global_index)
{
	int lo = 0;
	int hi = dstm_global_context.world_size - 1;

	while (lo < hi) {
		int mid = lo + ((hi - lo + 1) / 2);
		if (darray->rank_offsets[mid] <= global_index)
			lo = mid;
		else
			hi = mid - 1;
	}
	return lo;
}

int dstm_darray_create(dstm_darray_t **ret,
                       size_t element_size,
                       size_t local_elements)
{
	if (ret == NULL || element_size == 0 || local_elements == 0) {
		if (ret != NULL)
			*ret = NULL;
		return -DSTM_EINVAL;
	}

	int err = 0;
	int world_size = dstm_global_context.world_size;
	MPI_Comm comm = dstm_global_context.comm;

	dstm_darray_t *darray = malloc(sizeof(*darray));
	if (darray == NULL)
		return -DSTM_ENOMEM;

	darray->rank_elements = calloc(world_size, sizeof(size_t));
	if (darray->rank_elements == NULL) {
		err = -DSTM_ENOMEM;
		goto err;
	}

	darray->rank_offsets = calloc(world_size, sizeof(size_t));
	if (darray->rank_offsets == NULL) {
		err = -DSTM_ENOMEM;
		goto err_elements;
	}

	dstm_region_t *region = dstm_get_local_region();
	darray->region = region;

	size_t local_size = element_size * local_elements;
	darray->local_ptr = dstm_region_malloc(region, local_size);
	if (dstm_ptr_isnull(darray->local_ptr)) {
		err = -DSTM_ENOMEM;
		goto err_off;
	}

	/* gather on all ranks the size of each local part */
	darray->element_size = element_size;
	darray->local_elements = local_elements;

	err = MPI_Allgather(&local_elements, 1, MPI_UNSIGNED_LONG_LONG,
	                    darray->rank_elements, 1, MPI_UNSIGNED_LONG_LONG,
	                    comm);
	if (err != MPI_SUCCESS) {
		err = -DSTM_FAILURE;
		goto err_mpi;
	}

	size_t total = 0;
	for (int i = 0; i < world_size; i++)
		total += darray->rank_elements[i];
	darray->total_elements = total;

	for (int i = 0; i < world_size; i++) {
		darray->rank_offsets[i] = 0;
		for (int j = 0; j < i; j++)
			darray->rank_offsets[i] += darray->rank_elements[j];
	}

	*ret = darray;
	return 0;

err_mpi:
	dstm_region_free(darray->region, darray->local_ptr);
err_off:
	free(darray->rank_offsets);
err_elements:
	free(darray->rank_elements);
err:
	free(darray);
	*ret = NULL;
	return err;
}

int dstm_darray_create_equal(dstm_darray_t **ret,
                             size_t element_size,
                             size_t total_elements)
{
	int world_size = dstm_global_context.world_size;
	int rank = dstm_global_context.rank;

	size_t local_elements = total_elements / (size_t)world_size;
	size_t remainder = total_elements % (size_t)world_size;
	if (rank < (int)remainder)
		local_elements++;

	return dstm_darray_create(ret, element_size, local_elements);
}

void dstm_darray_destroy(dstm_darray_t **darray)
{
	if (darray == NULL || *darray == NULL)
		return;

	dstm_darray_t *d = *darray;
	if (!dstm_ptr_isnull(d->local_ptr))
		dstm_region_free(d->region, d->local_ptr);

	free(d->rank_offsets);
	free(d->rank_elements);
	free(d);
	*darray = NULL;
}

void *dstm_darray_get_local_ptr(dstm_darray_t *darray)
{
	return dstm_region_ptr_tovoid(darray->region, darray->local_ptr);
}

size_t dstm_darray_local_count(dstm_darray_t *darray)
{
	return darray->local_elements;
}

size_t dstm_darray_global_offset(dstm_darray_t *darray)
{
	int rank = dstm_global_context.rank;
	return darray->rank_offsets[rank];
}

int dstm_darray_get(dstm_darray_t *darray,
                    size_t global_index,
                    void *dst,
                    dstm_time_t time,
                    dstm_time_t delta)
{
	int rank = darray_find_rank(darray, global_index);
	int local_offset =
	        (int)(global_index - (size_t)darray->rank_offsets[rank]);

	dstm_ptr_t src;
	src.region = rank;
	src.offset = local_offset * darray->element_size;

	return dstm_get(rank, src.region, time, delta, dst, src,
	                darray->element_size);
}

int dstm_darray_put(dstm_darray_t *darray,
                    size_t global_index,
                    void *src,
                    dstm_time_t time,
                    dstm_time_t delta)
{
	int rank = darray_find_rank(darray, global_index);
	int local_offset =
	        (int)(global_index - (size_t)darray->rank_offsets[rank]);

	dstm_ptr_t dst;
	dst.region = rank;
	dst.offset = local_offset * darray->element_size;

	return dstm_put(rank, dst.region, time, delta, dst, src,
	                darray->element_size);
}

int dstm_darray_allgatherv(dstm_darray_t *darray,
                           void *src,
                           void *dst,
                           dstm_time_t time,
                           const dstm_time_t *deltas)
{
	if (darray == NULL || src == NULL || dst == NULL || deltas == NULL)
		return -DSTM_EINVAL;

	int self = dstm_global_context.rank;
	int world_size = dstm_global_context.world_size;

	/* Validate deltas: local rank must have delta >= 0 (put constraint),
	 * all remote ranks must have delta <= 0 (get constraint). */
	if (deltas[self] < 0)
		return -DSTM_EINVAL;
	for (int r = 0; r < world_size; r++) {
		if (r != self && deltas[r] > 0)
			return -DSTM_EINVAL;
	}

	/* Local put: write src into this rank's own darray partition. */
	int rc = dstm_put(self, self, time, deltas[self], darray->local_ptr,
	                  src,
	                  darray->rank_elements[self] * darray->element_size);
	if (rc != DSTM_SUCCESS)
		return rc;

	/* Remote gets: read each other rank's full slice into dst. */
	for (int r = 0; r < world_size; r++) {
		if (r == self)
			continue;

		void *out = (char *)dst +
		            darray->rank_offsets[r] * darray->element_size;
		dstm_ptr_t remote;
		remote.region = r;
		remote.offset = 0;
		size_t size = darray->rank_elements[r] * darray->element_size;

		rc = dstm_get(r, r, time, deltas[r], out, remote, size);
		if (rc != DSTM_SUCCESS)
			return rc;
	}

	return DSTM_SUCCESS;
}
