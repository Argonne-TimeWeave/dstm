/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

#ifndef DSTM_INTERNAL_DISTANCE_H
#define DSTM_INTERNAL_DISTANCE_H 1

#ifdef __cplusplus
extern "C" {
#endif

struct dstm_distance_s {
	int *distances;
	size_t size;
};

#ifdef __cplusplus
}
#endif

#endif
