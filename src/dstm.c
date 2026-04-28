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

#include <stdlib.h>

#include "dstm.h"

#include "internal/globals.h"
#include "internal/msg.h"
#include "internal/region.h"

const int dstm_version_major = DSTM_VERSION_MAJOR;
const int dstm_version_minor = DSTM_VERSION_MINOR;
const int dstm_version_patch = DSTM_VERSION_PATCH;
const char *dstm_version_revision = DSTM_VERSION_REVISION;
const char *dstm_version_string = DSTM_VERSION_STRING;

int dstm_errno = 0;

/*******************************************************************************
 * Context
 * Should be in a user controlled struct, but a singleton will do for now.
 ******************************************************************************/

dstm_context_t dstm_global_context = {0, 0, 0, NULL, NULL};

/*******************************************************************************
 * Utils
 ******************************************************************************/
int dstm_get_local_region_id(void)
{
	return dstm_global_context.rank;
}

dstm_region_t *dstm_get_local_region(void)
{
	return dstm_global_context.region;
}

/*******************************************************************************
 * Init functions
 ******************************************************************************/

int dstm_start(dstm_distance_t *dist)
{
	int err;

	if (dstm_distance_size(dist) != (size_t)dstm_global_context.world_size)
		return -DSTM_EINVAL;

	err = dstm_distance_create_copy(&dstm_global_context.distances, dist);
	if (err)
		return err;

	/* create a region locally, let's go for 1GB of memory available */
	err = dstm_region_create(&dstm_global_context.region,
	                         DSTM_DEFAULT_REGION_SIZE);
	if (err)
		return err;
	return 0;
}

int dstm_fini(dstm_time_t last_time)
{
	/* Run the distributed ring barrier to ensure all in-flight requests
	 * across all ranks have completed before any rank tears down its
	 * local region. */
	dstm_barrier(last_time);

	dstm_region_destroy(&dstm_global_context.region);
	return 0;
}

int dstm_init(int *argc, char **argv[])
{
	(void)argc;
	(void)argv;

	/* setup a default log config to handle errors in this part of the
	 * initialization
	 */
	if (!dstm_log_initialized)
		dstm_log_init(stderr, "dstm");

	int world_size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &dstm_global_context.comm);

	dstm_global_context.world_size = world_size;
	dstm_global_context.rank = rank;

	dstm_msg_init();

	return DSTM_SUCCESS;
}

int dstm_finalize(void)
{
	dstm_msg_finalize();
	dstm_distance_destroy(&dstm_global_context.distances);
	return 0;
}
