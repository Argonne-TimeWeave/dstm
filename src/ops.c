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
#include "internal/region.h"
#include "internal/globals.h"
#include "internal/utlist.h"
#include <mpi.h>
#include <stdlib.h>

/* we need to make the distinction between local, incoming, or outgoing request */
#define DSTM_REQUEST_TYPE_LOCAL_PUT 0
#define DSTM_REQUEST_TYPE_LOCAL_GET 1
#define DSTM_REQUEST_TYPE_IN_PUT 2
#define DSTM_REQUEST_TYPE_IN_GET 3
#define DSTM_REQUEST_TYPE_OUT_PUT 4
#define DSTM_REQUEST_TYPE_OUT_GET 5
#define DSTM_REQUEST_TYPE_MAX 6

struct dstm_request_s {
	int type;
	dstm_time_t time;
	dstm_time_t delta;
	int rank;
	intptr_t dst;
	intptr_t src;
	size_t size;
	struct dstm_request_s *prev, *next;
};
typedef struct dstm_request_s dstm_request_t;

dstm_request_t *requests = NULL;

/* sort by r->time + r->delta */
int dstm_request_cmp_arrival(dstm_request_t *a, dstm_request_t *b)
{
	dstm_time_t ta, tb;
	ta = a->time + a->delta;
	tb = b->time + b->delta;
	return ta - tb;
}

/* sort by r->time and then by r->delta */
int dstm_request_cmp_departure(dstm_request_t *a, dstm_request_t *b)
{
	dstm_time_t ta;
	ta = a->time - b->time;
	if (ta)
		return ta;

	return a->delta - b->delta;
}
/*******************************************************************************
 * Basic request management
 ******************************************************************************/

int dstm_req_enqueue(int type, dstm_time_t time, dstm_time_t delta, int rank,
		     intptr_t dst, intptr_t src, size_t size)
{
	assert(type < DSTM_REQUEST_TYPE_MAX);

	dstm_request_t *req = calloc(1, sizeof(dstm_request_t));
	if (req == NULL)
		return -DSTM_ENOMEM;

	req->type = type;
	req->time = time;
	req->delta = delta;
	req->rank = rank;
	req->dst = dst;
	req->src = src;
	req->size = size;

	DL_INSERT_INORDER(requests, req, dstm_request_cmp_departure);
	return 0;
}

/*******************************************************************************
 * Distributed mechanics
 ******************************************************************************/

int dstm_progress_engine(dstm_time_t now)
{
	/* we don't enter this unless there's progress to be made
	 *
	 * Progress means moving forward in time, meaning that everything that
	 * needed to happen at 'now' is done:
	 *
	 * - executing requests
	 * - sending out heartbeat
	 * - waiting for neighbors to reach the end of this 'now' too
	 */
	(void)now;
	int done = 0;
	while(!done) {

		/* receive messages from neighbors, we can receive either a
		 * heartbeat or a request
		 */
		//MPI_Recv(MPI_ANY_SOURCE, MPI_ANY_TAG);
	

	// then wait for all sends to be flushed out, for all the request still waiting for the current time.
	
	// update neighbors on new time

		done = 1;
	}
	return 0;
}

int dstm_put(int rank, int region_id, dstm_time_t time, dstm_time_t delta,
	     intptr_t dst, void *src, size_t size)
{
	(void)region_id;
	if (src == NULL)
		return -DSTM_EINVAL;

	/* request tracking part:
	 * making sure the request is being sent to the right place
	 */


	/* first, figure out if this is a local request or a remote one */
	if (rank == dstm_global_context.rank) {
		/* local put:
		 * we just enqueue it locally
		 * */
		dstm_req_enqueue(DSTM_REQUEST_TYPE_LOCAL_PUT, time, delta,
				 rank, dst, (intptr_t)src, size);
	}
	else {
		/* outgoing put
		 * we enqueue it for tracking purposes, and push it out to the
		 * appropriate rank.
		 */
		dstm_req_enqueue(DSTM_REQUEST_TYPE_OUT_PUT, time, delta,
				 rank, dst, (intptr_t)src, size);
		// TODO: MPI ISend
	}

	/* synchronization part:
	 * figuring out if we can make progress on some requests.
	 */

	// did we just enqueue a request that has a time further along that the
	// local time?
	if (time > dstm_global_context.region->time) {
		dstm_progress_engine(dstm_global_context.region->time);		
	}

	/* TODO: flushing, do we need to wait for the request to be sent out? */
	// by the time we return, the request is at least registered.
	return 0;
}

int dstm_get(int rank, int region_id, dstm_time_t time, dstm_time_t delta,
	     void *dst, intptr_t src, size_t size)
{
	(void)region_id;
	if (dst == NULL)
		return -DSTM_EINVAL;

	/* request tracking part:
	 * making sure the request is being sent to the right place
	 */
	

	/* first, figure out if this is a local request or a remote one */
	if (rank == dstm_global_context.rank) {
		/* local get:
		 * we just enqueue it locally
		 * */
		dstm_req_enqueue(DSTM_REQUEST_TYPE_LOCAL_GET, time, delta,
				 rank, (intptr_t)dst, src, size);
	}
	else {
		/* outgoing get
		 * we enqueue it for tracking purposes, and push it out to the
		 * appropriate rank.
		 */
		dstm_req_enqueue(DSTM_REQUEST_TYPE_OUT_GET, time, delta,
				 rank, (intptr_t)dst, src, size);
		// TODO: MPI Isend
	}

	/* synchronization part:
	 * figuring out if we can make progress on some requests.
	 */

	// did we just enqueue a request that has a time further along that the
	// local time?
	if (time > dstm_global_context.region->time) {
		dstm_progress_engine(dstm_global_context.region->time);		
	}

	/* TODO: flushing, do we need to wait for the request to be sent out? */
	// by the time we return, the request is at least registered.
	return 0;
}

