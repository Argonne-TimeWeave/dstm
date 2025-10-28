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
 * Create/Destroy
 ******************************************************************************/

#define DSTM_DEFAULT_REGION_SIZE (1<<30)

int dstm_region_create(dstm_region_t **region, size_t size);
void dstm_region_destroy(dstm_region_t **region);

#ifdef __cplusplus
}
#endif

#endif
