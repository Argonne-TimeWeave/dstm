/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

/**
 * Global declarations, internal
 **/

#ifndef DSTM_INTERNAL_GLOBALS_H
#define DSTM_INTERNAL_GLOBALS_H 1

#ifdef __cplusplus
extern "C" {
#endif

#include <mpi.h>

struct dstm_context_s {
	MPI_Comm comm;
	int world_size;
	int rank;
	dstm_region_t *region;
	dstm_distance_t distances;
};
typedef struct dstm_context_s dstm_context_t;

extern dstm_context_t dstm_global_context;

#ifdef __cplusplus
}
#endif

#endif
