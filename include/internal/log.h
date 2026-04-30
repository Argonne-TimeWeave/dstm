/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

/**
 * Logging utilities, internal
 **/

#ifndef DSTM_INTERNAL_LOG_H
#define DSTM_INTERNAL_LOG_H 1

#include <stdlib.h>

#include <jansson.h>

#include "dstm.h"

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************
 * JSON logging
 ******************************************************************************/

/**
 * Emit a log message at the given level with "label: <json>".
 * The json expression is only evaluated when the level is active.
 */
#define dstm_log_json(level, label, json_expr)                                 \
	do {                                                                   \
		if ((level) <= dstm_log_level) {                               \
			json_t *__j = (json_expr);                             \
			char *__s = json_dumps(                                \
			        __j, JSON_COMPACT | JSON_ENCODE_ANY);          \
			dstm_log_printf((level), __FILE__, __func__, __LINE__, \
			                "%s: %s\n", (label), __s);             \
			free(__s);                                             \
			json_decref(__j);                                      \
		}                                                              \
	} while (0)

/*******************************************************************************
 * Enum-to-string utilities
 ******************************************************************************/

/**
 * Generic integer-to-string lookup table entry.
 * Tables must be terminated by a {0, NULL} sentinel.
 */
typedef struct dstm_enum_table_s {
	int value;
	const char *name;
} dstm_enum_table_t;

/**
 * Look up the string name for an integer value in a table.
 * Returns "UNKNOWN" if the value is negative or not found.
 */
static inline const char *dstm_enum_to_string(int value,
                                              const dstm_enum_table_t *table)
{
	if (value < 0)
		return "UNKNOWN";
	for (int i = 0; table[i].name != NULL; i++)
		if (table[i].value == value)
			return table[i].name;
	return "UNKNOWN";
}

#ifdef __cplusplus
}
#endif

#endif
