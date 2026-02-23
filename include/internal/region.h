/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

/**
 * Region management, internal
 **/

#ifndef DSTM_INTERNAL_REGION_H
#define DSTM_INTERNAL_REGION_H 1

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************
 * CONSTANTS
 ******************************************************************************/

#define DSTM_DEFAULT_REGION_SIZE (1<<30)

#include <aml.h>
#include <aml/area/linux.h>
#include <aml/higher/allocator.h>
#include <aml/higher/buddy.h>
#include "internal/utlist.h"
#include "internal/uthash.h"

struct dstm_region_ninfo_s {
	int rank;
	dstm_time_t time;
	UT_hash_handle hh;
};
typedef struct dstm_region_ninfo_s dstm_region_ninfo_t;

struct dstm_region_s {
	struct aml_allocator *allocator;
	dstm_time_t time;
	dstm_region_ninfo_t *ninfos;
};

/*******************************************************************************
 * Create/Destroy
 ******************************************************************************/

int dstm_region_create(dstm_region_t **region, size_t size);
void dstm_region_destroy(dstm_region_t **region);

int dstm_region_set_ninfo(dstm_region_t *region, int rank, dstm_time_t time);


void *dstm_region_malloc(dstm_region_t *r, size_t size);
void dstm_region_free(dstm_region_t *r, void *ptr);

#ifdef __cplusplus
}
#endif

#endif
