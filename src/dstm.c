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

#include <assert.h>
#include <stdlib.h>

#include "dstm.h"

#include "internal/globals.h"
#include "internal/msg.h"
#include "internal/region.h"
#include "internal/request.h"

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

dstm_context_t dstm_global_context = {0, 0, 0, 0, NULL, NULL};

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

	/* Reset the per-session monotonic tag counter so each session
	 * starts from a deterministic base. */
	dstm_request_tag_reset();

	err = dstm_distance_create_copy(&dstm_global_context.distances, dist);
	if (err)
		return err;

	/* Duplicate user_comm into a fresh per-session communicator.  All
	 * session-level traffic (puts, gets, heartbeats, barrier tokens)
	 * will be matched against this private comm; messages from a
	 * previous session — sent on a now-freed comm — cannot leak into
	 * this one. */
	MPI_Comm_dup(dstm_global_context.user_comm, &dstm_global_context.comm);

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

	/* Invariant: after a successful barrier, the global request list
	 * must be empty.  If this fires, the barrier protocol failed to
	 * drain something (likely a regression). */
	assert(requests == NULL);

	dstm_region_destroy(&dstm_global_context.region);
	dstm_distance_destroy(&dstm_global_context.distances);

	/* Rendezvous before tearing down the session communicator.  The ring
	 * barrier above only guarantees each rank has completed *locally*; a
	 * straggler token it sent may still be in flight.  Freeing the comm
	 * now would let MPI recycle its context id, and the next session's
	 * MPI_Comm_dup could reuse it — aliasing a stale message into the new
	 * session and corrupting its protocol.  A collective MPI_Barrier
	 * ensures all ranks have arrived and no session message is in transit
	 * before the comm (and its context id) is released. */
	MPI_Barrier(dstm_global_context.comm);

	/* Free the session communicator and restore the fallback so any
	 * subsequent data-structure calls (e.g. dstm_darray_create) made
	 * before the next dstm_start still have a valid `comm`. */
	MPI_Comm_free(&dstm_global_context.comm);
	dstm_global_context.comm = dstm_global_context.user_comm;

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

	/* Duplicate MPI_COMM_WORLD into the long-lived user_comm.  This
	 * gives DSTM a private message namespace for the entire process
	 * lifetime, isolated from any other MPI traffic the application
	 * may have.  Between sessions (or before the first dstm_start),
	 * `comm` aliases user_comm so calls like dstm_darray_create can
	 * still run their collectives. */
	MPI_Comm_dup(MPI_COMM_WORLD, &dstm_global_context.user_comm);
	dstm_global_context.comm = dstm_global_context.user_comm;

	dstm_global_context.world_size = world_size;
	dstm_global_context.rank = rank;

	dstm_msg_init();

	return DSTM_SUCCESS;
}

int dstm_finalize(void)
{
	dstm_msg_finalize();
	dstm_distance_destroy(&dstm_global_context.distances);

	/* Free the long-lived user communicator.  At this point `comm`
	 * aliases user_comm (set by the last dstm_fini, or by dstm_init
	 * if no session was ever started), so a single free suffices. */
	MPI_Comm_free(&dstm_global_context.user_comm);
	dstm_global_context.comm = MPI_COMM_NULL;
	return 0;
}
