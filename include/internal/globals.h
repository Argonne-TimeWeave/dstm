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

#include <mpi.h>
#include <stdio.h>

#include "dstm.h"

#ifdef __cplusplus
extern "C" {
#endif

struct dstm_context_s {
	/* Active communicator used by all session-level traffic
	 * (dstm_msg_*, darray collectives, etc.).  During an active
	 * session (between dstm_start and dstm_fini) this points to a
	 * fresh per-session comm duplicated from `user_comm`.  Outside
	 * any session it aliases `user_comm` so that data-structure
	 * creations (e.g. dstm_darray_create) issued between
	 * dstm_init/dstm_finalize still have a valid communicator. */
	MPI_Comm comm;
	/* Long-lived communicator duplicated from MPI_COMM_WORLD by
	 * dstm_init and freed by dstm_finalize.  Acts as the parent
	 * communicator from which each session's `comm` is duplicated. */
	MPI_Comm user_comm;
	int world_size;
	int rank;
	dstm_region_t *region;
	dstm_distance_t *distances;
};
typedef struct dstm_context_s dstm_context_t;

extern dstm_context_t dstm_global_context;

extern int dstm_log_level;
extern FILE *dstm_log_fd;

#ifdef __cplusplus
}
#endif

#endif
