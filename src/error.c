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

#include <stdio.h>

#include "dstm.h"

static const char *const dstm_error_strings[] = {
        [DSTM_SUCCESS] = "Success",
        [DSTM_FAILURE] = "Generic error",
        [DSTM_ENOMEM] = "Not enough memory",
        [DSTM_EINVAL] = "Invalid argument",
        [DSTM_EDOM] = "Value out of bound",
        [DSTM_EBUSY] = "Underlying resource is not available for operation",
        [DSTM_ENOTSUP] = "Operation not supported",
        [DSTM_EPERM] = "Insufficient permissions",
        [DSTM_ENOTFOUND] = "Value not found",
};

const char *dstm_strerror(const int err)
{
	if (err < 0 || err >= DSTM_ERROR_MAX)
		return "Unknown error";
	return dstm_error_strings[err];
}

void dstm_perror(const char *msg)
{
	fprintf(stderr, "%s:%s\n", msg, dstm_strerror(dstm_errno));
}
