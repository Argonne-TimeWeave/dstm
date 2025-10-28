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
#include <aml.h>
#include <aml/area/linux.h>
#include <aml/higher/allocator.h>
#include <aml/higher/buddy.h>

#include "internal/region.h"
#include "internal/utlish.h"

#define DSTM_REQUEST_TYPE_PUT 0
#define DSTM_REQUEST_TYPE_GET 1
#define DSTM_REQUEST_TYPE_MAX 2

struct dstm_region_s {
	struct aml_allocator *allocator;
	dstm_time_t time;
};

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
		return err;

	ret->time = 0;
	ret->reqs = NULL;
	*region = ret;
	return 0;
}

void dstm_region_destroy(dstm_region_t **region)
{
	dstm_region_t *r;
	if (region == NULL)
		return;
	
	r = *region;
	if (r == NULL)
		return;

	aml_allocator_buddy_destroy(&r->allocator);
	*region = NULL;
}
