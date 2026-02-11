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
#include "internal/globals.h"
#include "internal/region.h"
#include "internal/msg.h"
#include <aml.h>
#include <mpi.h>
#include <stdlib.h>

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

dstm_context_t dstm_global_context = { 0, 0 , 0, NULL, { NULL, 0}};

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

dstm_distance_t DSTM_DISTANCE_IDENTITY = { NULL, 0};

int dstm_distance_deepcopy(dstm_distance_t dst, dstm_distance_t src)
{
	if (src.size == 0 || src.distances == NULL)
		return -DSTM_EINVAL;

	dst.distances = calloc(dst.size * dst.size, sizeof(int));
	if (dst.distances == NULL)
		return -DSTM_ENOMEM;

	memcpy(dst.distances, src.distances, dst.size * dst.size * sizeof(int));
	return DSTM_SUCCESS;
}

int dstm_distance_get(dstm_distance_t dst, int i, int j)
{
	return dst.distances[dst.size * i + j];
}

int dstm_start(dstm_distance_t dist)
{
	int err;

	/* sanity check, the matrix needs to be size * size */
	if (dist.size != (size_t )dstm_global_context.world_size)
		return -DSTM_EINVAL;
	
	err = dstm_distance_deepcopy(dstm_global_context.distances, dist);
	if (err)
		return err;

	/* create a region locally, let's go for 1GB of memory available */
	err = dstm_region_create(&dstm_global_context.region, DSTM_DEFAULT_REGION_SIZE);
	if (err)
		return err;
	return 0;
}

int dstm_fini(void)
{
	dstm_region_destroy(&dstm_global_context.region);
	return 0;
}

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

	err = MPI_Init(NULL, NULL);
	if (err)
		return err;

	int world_size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &dstm_global_context.comm);

	DSTM_DISTANCE_IDENTITY.distances = calloc(world_size * world_size,
					 sizeof(int));
	DSTM_DISTANCE_IDENTITY.size = world_size;
	for (int i = 0; i < world_size; i++)
		DSTM_DISTANCE_IDENTITY.distances[i * world_size + i] = 1;

	dstm_global_context.world_size = world_size;
	dstm_global_context.rank = rank;

	dstm_msg_init();

	return DSTM_SUCCESS;
}

int dstm_finalize(void)
{
	dstm_msg_finalize();
	MPI_Finalize();
	if (DSTM_DISTANCE_IDENTITY.distances != NULL)
		free(DSTM_DISTANCE_IDENTITY.distances);
	aml_finalize();
	return 0;
}
