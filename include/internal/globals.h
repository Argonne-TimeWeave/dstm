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

#define DSTM_MAX_REGIONS 64

struct dstm_region_s;
typedef struct dstm_region_s dstm_region_t;

struct dstm_region_ninfo_s;
typedef struct dstm_region_ninfo_s dstm_region_ninfo_t;

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

    /* Local region table, indexed by local region id. Slot 0 is default */
	dstm_region_t *regions[DSTM_MAX_REGIONS];
    int region_count;

	dstm_distance_t *distances;

    /* ---- rank-level versioning / clock state ----
     * These were previously per-region.  They are now shared across ALL
     * of this rank's regions: neighbor clocks are keyed by (rank,
     * region_id) inside `ninfos`, while the barrier and heartbeat
     * watermark are genuinely rank-wide.  All reset per session. */
    dstm_region_ninfo_t *ninfos;    /* neighbor last-known times 
                                        by (rank, region_id) */

    /* Number of barrier tokens received but not yet consumed by the
    * barrier DFA.  The BARRIER message handler increments this on
    * every token arrival; the barrier DFA decrements it when it
    * consumes a token to advance to the next state. */
    int barrier_token_count;

    /* Set to 1 after the first barrier round completes (all ranks
    * have drained their outgoing requests).  When set, the version
    * transition checks (WRITABLE->READABLE, READABLE->DISCARDABLE)
    * skip ALL neighbor distance checks, since no rank can issue new
    * PUT/GET messages.  Reset to 0 when the barrier exits. */
    int barrier_all_drained;

    /* Highest logical timestamp for which we have already broadcast a
    * HEARTBEAT to all neighbors.  Used to suppress redundant
    * heartbeats: dstm_put/dstm_get only emit a heartbeat when the
    * caller's `time` strictly exceeds this watermark. Rank-wide; reset to 0 on each
    * dstm_start. */
    dstm_time_t local_heartbeat_time;
};
typedef struct dstm_context_s dstm_context_t;

extern dstm_context_t dstm_global_context;

extern int dstm_log_level;
extern FILE *dstm_log_fd;

#ifdef __cplusplus
}
#endif

#endif
