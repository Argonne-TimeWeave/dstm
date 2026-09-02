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
#include <string.h>

#include "dstm.h"

#include "internal/globals.h"
#include "internal/region.h"

struct dstm_darray_s {
	size_t element_size;
	size_t total_elements;
	size_t local_elements;
	dstm_region_t *region;
    /* this rank's local region id */
    int region_id;
	/* per-rank offset from the global indices */
	size_t *rank_offsets;
	/* per-rank size of the local part, needed for gather operations */
	size_t *rank_elements;
	/* per-rank ptr for the local part */
	dstm_ptr_t *rank_ptrs;
    /* [world_size] each rank's region id */
    int *rank_region_id;
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

/* Resolve a global index to the owning (rank, dstm_ptr_t) pair: rank is
 * the MPI rank that owns the element, and ptr is the dstm_ptr_t that
 * addresses that element inside the owning rank's local partition. */
static void darray_resolve(dstm_darray_t *darray,
                           size_t global_index,
                           int *out_rank,
                           int *out_region,
                           dstm_ptr_t *out_ptr)
{
	int rank = darray_find_rank(darray, global_index);
	size_t local_offset = global_index - darray->rank_offsets[rank];
	dstm_ptr_t ptr = darray->rank_ptrs[rank];
	ptr.offset += (intptr_t)(local_offset * darray->element_size);
    /* stamp the owning rank's local region id into the ptr so it and
     * the region argument to dstm_get/dstm_put agree. */
    ptr.region = darray->rank_region_id[rank];
	*out_rank = rank;
    *out_region = darray->rank_region_id[rank];
	*out_ptr = ptr;
}

int dstm_darray_create_on_region(dstm_darray_t **ret,
                                 size_t element_size,
                                 size_t local_elements,
                                 int region_id)
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

	darray->rank_ptrs = calloc(world_size, sizeof(dstm_ptr_t));
	if (darray->rank_ptrs == NULL) {
		err = -DSTM_ENOMEM;
		goto err_offsets;
	}

	darray->rank_region_id = calloc(world_size, sizeof(int));
    if (darray->rank_region_id == NULL) {
        err = -DSTM_ENOMEM;
        goto err_ptrs;
    }
    
    dstm_region_t *region = dstm_get_local_region(region_id);
    darray->region = region;
    darray->region_id = region_id;
    if (region == NULL) {
            err = -DSTM_EINVAL;
            goto err_region_id;
    }

	size_t local_size = element_size * local_elements;
	dstm_ptr_t local_ptr = dstm_region_malloc(region, local_size);
	if (dstm_ptr_isnull(local_ptr)) {
		err = -DSTM_ENOMEM;
		goto err_ptrs;
	}
	dstm_log_debug("local alloc: region: %d, offset: %td, size: %zu\n",
	               local_ptr.region, local_ptr.offset, local_size);
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

	err = MPI_Allgather(&local_ptr, 2, MPI_UNSIGNED_LONG_LONG,
	                    darray->rank_ptrs, 2, MPI_UNSIGNED_LONG_LONG, comm);
	if (err != MPI_SUCCESS) {
		err = -DSTM_FAILURE;
		goto err_mpi;
	}

    err = MPI_Allgather(&region_id, 1, MPI_INT,
                        darray->rank_region_id, 1, MPI_INT, comm);
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
	dstm_region_free(darray->region, local_ptr);
err_region_id:
    free(darray->rank_region_id);
err_ptrs:
	free(darray->rank_ptrs);
err_offsets:
	free(darray->rank_offsets);
err_elements:
	free(darray->rank_elements);
err:
	free(darray);
	*ret = NULL;
	return err;
}

int dstm_darray_create(dstm_darray_t **ret,
                       size_t element_size,
                       size_t local_elements) 
{
    return dstm_darray_create_on_region(ret, element_size, local_elements,
                                            dstm_get_local_region_id());
}

int dstm_darray_create_equal_on_region(dstm_darray_t **ret,
                                       size_t element_size,
                                       size_t total_elements,
                                       int region_id)
{
	int world_size = dstm_global_context.world_size;
	int rank = dstm_global_context.rank;

	size_t local_elements = total_elements / (size_t)world_size;
	size_t remainder = total_elements % (size_t)world_size;
	if (rank < (int)remainder)
		local_elements++;

	return dstm_darray_create_on_region(ret, element_size, local_elements, region_id);
}

int dstm_darray_create_equal(dstm_darray_t **ret,
                             size_t element_size,
                             size_t total_elements)
{
    return dstm_darray_create_equal_on_region(ret, element_size,
                                              total_elements,
                                              dstm_get_local_region_id());
}

void dstm_darray_destroy(dstm_darray_t **darray)
{
	if (darray == NULL || *darray == NULL)
		return;

	dstm_darray_t *d = *darray;
	int self = dstm_global_context.rank;
	if (!dstm_ptr_isnull(d->rank_ptrs[self]))
		dstm_region_free(d->region, d->rank_ptrs[self]);

    free(d->rank_region_id);
	free(d->rank_ptrs);
	free(d->rank_offsets);
	free(d->rank_elements);
	free(d);
	*darray = NULL;
}

void *dstm_darray_get_local_ptr(dstm_darray_t *darray)
{
	int self = dstm_global_context.rank;
	return dstm_region_ptr_tovoid(darray->region, darray->rank_ptrs[self]);
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
	dstm_log_debug("darray: %p, index: %zu, dst: %p, time: %" PRId64
	               ", delta: %" PRId64 "\n",
	               darray, global_index, dst, time, delta);
	int rank, region;
	dstm_ptr_t src;
	darray_resolve(darray, global_index, &rank, &region, &src);
	dstm_log_debug("rank: %d, region: %d, src: %d:%td\n", rank, region, 
                   src.region, src.offset);

	return dstm_get(rank, region, time, delta, dst, src,
	                darray->element_size);
}

int dstm_darray_put(dstm_darray_t *darray,
                    size_t global_index,
                    void *src,
                    dstm_time_t time,
                    dstm_time_t delta)
{
	dstm_log_debug("darray: %p, index: %zu, src: %p, time: %" PRId64
	               ", delta: %" PRId64 "\n",
	               darray, global_index, src, time, delta);
	int rank, region;
	dstm_ptr_t dst;
	darray_resolve(darray, global_index, &rank, &region, &dst);

	return dstm_put(rank, region, time, delta, dst, src,
	                darray->element_size);
}

int dstm_darray_get_range(dstm_darray_t *darray,
                          const size_t *indices,
                          size_t count,
                          void *dst,
                          dstm_time_t time,
                          dstm_time_t delta)
{
	dstm_log_debug("darray: %p, count: %zu, dst: %p, time: %" PRId64
	               ", delta: %" PRId64 "\n",
	               darray, count, dst, time, delta);
	if (darray == NULL)
		return -DSTM_EINVAL;
	if (count == 0)
		return DSTM_SUCCESS;
	if (indices == NULL || dst == NULL)
		return -DSTM_EINVAL;

	size_t esize = darray->element_size;
	for (size_t i = 0; i < count; i++) {
		int rank, region;
		dstm_ptr_t src;
		darray_resolve(darray, indices[i], &rank, &region, &src);
		void *out = (char *)dst + i * esize;
		int rc = dstm_get(rank, region, time, delta, out, src, esize);
		if (rc != DSTM_SUCCESS)
			return rc;
	}
	return DSTM_SUCCESS;
}

int dstm_darray_put_range(dstm_darray_t *darray,
                          const size_t *indices,
                          size_t count,
                          const void *src,
                          dstm_time_t time,
                          dstm_time_t delta)
{
	dstm_log_debug("darray: %p, count: %zu, src: %p, time: %" PRId64
	               ", delta: %" PRId64 "\n",
	               darray, count, src, time, delta);
	if (darray == NULL)
		return -DSTM_EINVAL;
	if (count == 0)
		return DSTM_SUCCESS;
	if (indices == NULL || src == NULL)
		return -DSTM_EINVAL;

	size_t esize = darray->element_size;
	for (size_t i = 0; i < count; i++) {
		int rank, region;
		dstm_ptr_t dst;
		darray_resolve(darray, indices[i], &rank, &region, &dst);
		void *in = (void *)((const char *)src + i * esize);
		int rc = dstm_put(rank, region, time, delta, dst, in, esize);
		if (rc != DSTM_SUCCESS)
			return rc;
	}
	return DSTM_SUCCESS;
}

int dstm_darray_get_rangev(dstm_darray_t *darray,
                           const size_t *indices,
                           size_t count,
                           void *const *dsts,
                           dstm_time_t time,
                           dstm_time_t delta)
{
	dstm_log_debug("darray: %p, count: %zu, dsts: %p, time: %" PRId64
	               ", delta: %" PRId64 "\n",
	               darray, count, (const void *)dsts, time, delta);
	if (darray == NULL)
		return -DSTM_EINVAL;
	if (count == 0)
		return DSTM_SUCCESS;
	if (indices == NULL || dsts == NULL)
		return -DSTM_EINVAL;

	size_t esize = darray->element_size;
	for (size_t i = 0; i < count; i++) {
		int rank, region;
		dstm_ptr_t src;
		darray_resolve(darray, indices[i], &rank, &region, &src);
		int rc = dstm_get(rank, region, time, delta, dsts[i], src, esize);
		if (rc != DSTM_SUCCESS)
			return rc;
	}
	return DSTM_SUCCESS;
}

int dstm_darray_put_rangev(dstm_darray_t *darray,
                           const size_t *indices,
                           size_t count,
                           const void *const *srcs,
                           dstm_time_t time,
                           dstm_time_t delta)
{
	dstm_log_debug("darray: %p, count: %zu, srcs: %p, time: %" PRId64
	               ", delta: %" PRId64 "\n",
	               darray, count, (const void *)srcs, time, delta);
	if (darray == NULL)
		return -DSTM_EINVAL;
	if (count == 0)
		return DSTM_SUCCESS;
	if (indices == NULL || srcs == NULL)
		return -DSTM_EINVAL;

	size_t esize = darray->element_size;
	for (size_t i = 0; i < count; i++) {
		int rank, region;
		dstm_ptr_t dst;
		darray_resolve(darray, indices[i], &rank, &region, &dst);
		int rc = dstm_put(rank, region, time, delta, dst,
		                  (void *)srcs[i], esize);
		if (rc != DSTM_SUCCESS)
			return rc;
	}
	return DSTM_SUCCESS;
}

int dstm_darray_allgatherv(dstm_darray_t *darray,
                           void *src,
                           void *dst,
                           dstm_time_t time,
                           const dstm_time_t *deltas)
{
	dstm_log_debug("darray: %p, src: %p, dst: %p, time: %" PRId64 "\n",
	               darray, src, dst, time);
	if (darray == NULL || src == NULL || dst == NULL || deltas == NULL)
		return -DSTM_EINVAL;

	int self = dstm_global_context.rank;
	int world_size = dstm_global_context.world_size;

	/* Validate deltas: local rank must have delta > 0 (put constraint),
	 * all remote ranks must have delta < 0 (get constraint).  Zero is
	 * not allowed in any slot: delta == 0 is only meaningful for the
	 * local rank, but the local slot is itself a put that requires a
	 * strictly positive delta, so the local-rank rule that would
	 * otherwise permit delta == 0 cannot apply here. */
	if (deltas[self] <= 0)
		return -DSTM_EDOM;
	for (int r = 0; r < world_size; r++) {
		if (r != self && deltas[r] >= 0)
			return -DSTM_EDOM;
	}

	/* Local put: write src into this rank's own darray partition. */
	for (size_t i = 0; i < darray->rank_elements[self]; i++) {
		size_t idx = darray->rank_offsets[self] + i;
		void *s = (void *)((char *)src + i * darray->element_size);
		int rc = dstm_darray_put(darray, idx, s, time, deltas[self]);
		if (rc != DSTM_SUCCESS)
			return rc;
	}

	/* Remote gets: read each other rank's full slice into dst. */
	for (int r = 0; r < world_size; r++) {
		if (r == self)
			continue;

		for (size_t i = 0; i < darray->rank_elements[r]; i++) {
			size_t idx = darray->rank_offsets[r] + i;
			void *out = (char *)dst + idx * darray->element_size;
			int rc = dstm_darray_get(darray, idx, out, time,
			                         deltas[r]);
			if (rc != DSTM_SUCCESS)
				return rc;
		}
	}

	return DSTM_SUCCESS;
}

int dstm_darray_get_or_default(dstm_darray_t *darray,
                               size_t global_index,
                               void *dst,
                               dstm_time_t time,
                               dstm_time_t delta,
                               const void *default_value)
{
	if (darray == NULL || dst == NULL || default_value == NULL)
		return -DSTM_EINVAL;

	if (time + delta < 0) {
		memcpy(dst, default_value, darray->element_size);
		return DSTM_SUCCESS;
	}

	return dstm_darray_get(darray, global_index, dst, time, delta);
}

/*******************************************************************************
 * Formatting helpers
 ******************************************************************************/

json_t *dstm_darray_to_json(dstm_darray_t *a)
{
	if (a == NULL)
		return json_null();
	return json_pack("{s:I, s:I, s:I}", "element_size",
	                 (json_int_t)a->element_size, "local_elements",
	                 (json_int_t)a->local_elements, "total_elements",
	                 (json_int_t)a->total_elements);
}
