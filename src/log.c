/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

#include "config.h"

#include <assert.h>
#include <stdio.h>

#include "dstm.h"

#include "internal/globals.h"

/*******************************************************************************
 * Logging functions
 *******************************************************************************/

int dstm_log_level = DSTM_LOG_NORMAL;
int dstm_log_initialized = 0;
FILE *dstm_log_fd = NULL;
static const char *dstm_log_namespace;

static const char *dstm_log_level_string[] = {
        "quiet", "error", "warning", "normal", "info", "debug",
};

int dstm_log_init(FILE *f, const char *nm)
{
	if (f == NULL || nm == NULL)
		return -DSTM_EINVAL;
	dstm_log_fd = f;
	dstm_log_level = DSTM_LOG_NORMAL;
	dstm_log_namespace = nm;
	dstm_log_initialized = 1;
	return 0;
}

int dstm_log_setlevel(int level)
{
	if (level < 0)
		level = 0;
	if (level > DSTM_LOG_DEBUG)
		level = DSTM_LOG_DEBUG;
	dstm_log_level = level;
	return 0;
}

void dstm_log_printf(int level,
                     const char *file,
                     const char *func,
                     unsigned int line,
                     const char *format,
                     ...)
{
	va_list ap;
	if (level > dstm_log_level)
		return;

	int rank = dstm_global_context.rank;
	int ws = dstm_global_context.world_size;
	if (dstm_log_level == DSTM_LOG_DEBUG)
		fprintf(dstm_log_fd, "%s:\t%s:%u/%u:\t%s:\t%s:\t%u:\t",
		        dstm_log_namespace, dstm_log_level_string[level], rank,
		        ws, file, func, line);
	else
		fprintf(dstm_log_fd, "%s:\t%s:%u/%u:\t", dstm_log_namespace,
		        dstm_log_level_string[level], rank, ws);
	va_start(ap, format);
	vfprintf(dstm_log_fd, format, ap);
	va_end(ap);
}
