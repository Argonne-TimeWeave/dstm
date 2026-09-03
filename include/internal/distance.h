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

#include <stddef.h>
#include <stdlib.h>

#include <jansson.h>

#include "dstm.h"

#include "internal/log.h"

#ifdef __cplusplus
extern "C" {
#endif

struct dstm_distance_s {
    /* Min sync distance for each rank pair */
    /* WRITABLE -> READABLE */
	int *distances;
    /* Max sync distance for each rank pair */
    /* READABLE -> DISCARDABLE */
    int *discard_distances;
	size_t size;
};

json_t *dstm_distance_to_json(dstm_distance_t *d);

#define dstm_log_distance(level, label, d)                                     \
	dstm_log_json((level), (label), dstm_distance_to_json((d)))

#ifdef __cplusplus
}
#endif

#endif
