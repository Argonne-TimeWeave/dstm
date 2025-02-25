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
#include <abt.h>
#include <aml.h>
#include <stdlib.h>

const int dstm_version_major = DSTM_VERSION_MAJOR;
const int dstm_version_minor = DSTM_VERSION_MINOR;
const int dstm_version_patch = DSTM_VERSION_PATCH;
const char *dstm_version_revision = DSTM_VERSION_REVISION;
const char *dstm_version_string = DSTM_VERSION_STRING;

int dstm_errno = 0;

int dstm_init(int *argc, char **argv[])
{
	(void)argc;
	(void)argv;
	int err;

	/* setup a default log config to handle errors in this part of the
	 * initialization
	 */
	if (!dstm_log_initialized)
		dstm_log_init(stderr, "dstm");

	err = aml_init(argc, argv);
	if (err != AML_SUCCESS)
		return err;

	/* ABT wants the raw stuff */
	int ac = (argc == NULL)? 0: *argc;
	char **av = (argv == NULL)? 0: *argv;
	ABT_init(ac, av);

	return DSTM_SUCCESS;
}

int dstm_finalize(void)
{
	ABT_finalize();
	aml_finalize();
	return 0;
}
