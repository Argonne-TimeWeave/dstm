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

#ifdef __cplusplus
extern "C" {
#endif

struct dstm_distance_s {
	int *distances;
	size_t size;
};

json_t *dstm_distance_to_json(dstm_distance_t *d);

#define dstm_log_distance(level, label, d)                                     \
	do {                                                                   \
		if ((level) <= dstm_log_level) {                               \
			json_t *__j = dstm_distance_to_json((d));              \
			char *__s = json_dumps(                                \
			    __j, JSON_COMPACT | JSON_SORT_KEYS |               \
				     JSON_ENCODE_ANY);                         \
			dstm_log_printf((level), __FILE__, __func__, __LINE__, \
			                "%s: %s\n", (label), __s);             \
			free(__s);                                             \
			json_decref(__j);                                      \
		}                                                              \
	} while (0)

#ifdef __cplusplus
}
#endif

#endif
