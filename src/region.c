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

#include "dstm.h"

#include "internal/region.h"

/*******************************************************************************
 * Init functions
 ******************************************************************************/

int dstm_region_create(dstm_region_t **region, size_t size)
{
	int err;
	dstm_region_t *ret;
	(void)size;
	ret = calloc(1, sizeof(dstm_region_t));
	if (ret == NULL)
		return -DSTM_ENOMEM;

	err = aml_allocator_buddy_create(&ret->allocator, &aml_area_linux,
				   NULL);
	if (err != AML_SUCCESS)
		goto end;

	ret->time = 0;
	ret->ninfos = NULL;
	*region = ret;
	return 0;
end:
	free(ret);
	return err;
}

void dstm_region_destroy(dstm_region_t **region)
{
	dstm_region_t *r;
	if (region == NULL)
		return;

	r = *region;
	if (r == NULL)
		return;

	dstm_region_ninfo_t *ni, *tmp;
	HASH_ITER(hh, r->ninfos, ni, tmp) {
		HASH_DEL(r->ninfos, ni);
		free(ni);
	}
	aml_allocator_buddy_destroy(&r->allocator);
	*region = NULL;
}

int dstm_region_set_ninfo(dstm_region_t *region, int rank, dstm_time_t time)
{
	if (region == NULL)
		return -DSTM_EINVAL;

	dstm_region_ninfo_t *ni;
	HASH_FIND_INT(region->ninfos, &rank, ni);
	if (ni == NULL) {
		ni = calloc(1, sizeof(*ni));
		assert(ni != NULL);
		ni->rank = rank;
		HASH_ADD_INT(region->ninfos, rank, ni);
	}
	ni->time = time;
	return 0;
}
