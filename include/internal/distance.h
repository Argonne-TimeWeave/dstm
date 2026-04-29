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

#include "dstm.h"

#ifdef __cplusplus
extern "C" {
#endif

struct dstm_distance_s {
	int *distances;
	size_t size;
};

/* Format: { size: N, distances[0][0]: V, distances[0][1]: V, ... (first row
 * only) } Only the first row of the matrix is printed to keep the output
 * bounded. */
#define DSTM_DISTANCE_SNPRINTF_SIZE 256
int dstm_distance_snprintf(char *buf, size_t n, dstm_distance_t *d);

#define dstm_log_distance(level, label, d)                                     \
	do {                                                                   \
		if ((level) <= dstm_log_level) {                               \
			char __buf[DSTM_DISTANCE_SNPRINTF_SIZE];               \
			dstm_distance_snprintf(__buf, sizeof(__buf), (d));     \
			dstm_log_printf((level), __FILE__, __func__, __LINE__, \
			                "%s: %s\n", (label), __buf);           \
		}                                                              \
	} while (0)

#ifdef __cplusplus
}
#endif

#endif
