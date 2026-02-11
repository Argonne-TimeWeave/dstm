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

#include <mpi.h>
#include <stdlib.h>

#include "dstm.h"

#include "internal/globals.h"
#include "internal/msg.h"
#include "internal/region.h"
#include "internal/utlist.h"

/* we need to make the distinction between local, incoming, or outgoing request
 */
#define DSTM_REQUEST_TYPE_LOCAL_PUT 0
#define DSTM_REQUEST_TYPE_LOCAL_GET 1
#define DSTM_REQUEST_TYPE_IN_PUT 2
#define DSTM_REQUEST_TYPE_IN_GET 3
#define DSTM_REQUEST_TYPE_OUT_PUT 4
#define DSTM_REQUEST_TYPE_OUT_GET 5
#define DSTM_REQUEST_TYPE_MAX 6

#define DSTM_REQUEST_STATUS_INIT 0
#define DSTM_REQUEST_STATUS_ISEND 1
#define DSTM_REQUEST_STATUS_IRECV 2
#define DSTM_REQUEST_STATUS_DONE 3
#define DSTM_REQUEST_STATUS_MAX 4

struct dstm_request_s {
	int type;
	int status;
	dstm_time_t time;
	dstm_time_t delta;
	int rank;
	intptr_t dst;
	intptr_t src;
	size_t size;
	dstm_msg_t msg;
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

int dstm_request_create(dstm_request_t **r,
                        int type,
                        dstm_time_t time,
                        dstm_time_t delta,
                        int rank,
                        intptr_t dst,
                        intptr_t src,
                        size_t size)
{
	assert(type < DSTM_REQUEST_TYPE_MAX);
	assert(r != NULL);

	dstm_request_t *req = calloc(1, sizeof(dstm_request_t));
	if (req == NULL)
		return -DSTM_ENOMEM;

	req->type = type;
	req->status = DSTM_REQUEST_STATUS_INIT;
	req->time = time;
	req->delta = delta;
	req->rank = rank;
	req->dst = dst;
	req->src = src;
	req->size = size;
	*r = req;
	return 0;
}

int dstm_request_destroy(dstm_request_t **r)
{
	if (r == NULL || *r == NULL)
		return 0;
	free(*r);
	*r = NULL;
	return 0;
}

/*******************************************************************************
 * Request Mechanics
 ******************************************************************************/

int dstm_request_isdone(dstm_request_t *req)
{
	return req->status == DSTM_REQUEST_STATUS_DONE;
}

typedef struct dstm_request_progress_s {
	int type;
	int (*progress)(dstm_request_t *req, dstm_time_t step);
} dstm_request_progress_t;

/* TODO: implement all those transitions */

int dstm_request_progress_local_put(dstm_request_t *req, dstm_time_t step)
{
	/* logic of a local put:
	 * - starts in INIT state, enqueued as a request to put data associated
	 *   with the request buffer into the local region memory.
	 *
	 * - when step == req->time + delta, apply the request to the local
	 *   memory.
	 */
	assert(req->status == DSTM_REQUEST_STATUS_INIT);
	if (req->time + req->delta == step) {
		/* TODO: is that a real pointer or an offset into a region */
		memcpy((void *)req->dst, (void *)req->src, req->size);
		req->status = DSTM_REQUEST_STATUS_DONE;
	}
	return 0;
}

int dstm_request_progress_local_get(dstm_request_t *req, dstm_time_t step)
{
	/* logic of a local put:
	 * - starts in INIT state, enqueued as a request to put data associated
	 *   with the request buffer into the local region memory.
	 *
	 * - when step == req->time + delta, apply the request to the local
	 *   memory.
	 */
	assert(req->status == DSTM_REQUEST_STATUS_INIT);
	if (req->time + req->delta == step) {
		/* TODO: is that a real pointer or an offset into a region */
		memcpy((void *)req->dst, (void *)req->src, req->size);
		req->status = DSTM_REQUEST_STATUS_DONE;
	}
	return 0;
}

int dstm_request_progress_in_put(dstm_request_t *req, dstm_time_t step)
{
	/* logic:
	 * - init: input request received from external rank
	 * - done: applied put
	 */
	assert(req->status == DSTM_REQUEST_STATUS_INIT);
	if (req->time + req->delta == step) {
		memcpy((void *)req->dst, req->msg.data, req->size);
		req->status = DSTM_REQUEST_STATUS_DONE;
	}
	return 0;
}

int dstm_request_progress_in_get(dstm_request_t *req, dstm_time_t step)
{
	/* logic:
	 * - init: input request received from external rank
	 * -     : filled out get, sent data back
	 *       : send completed
	 */
	if (req->status == DSTM_REQUEST_STATUS_INIT) {
		if (req->time + req->delta == step) {
			dstm_msg_fill_hdr(&req->msg, DSTM_MSG_TYPE_GET_REP,
			                  req->time, req->delta,
			                  dstm_global_context.rank, req->rank);
			dstm_msg_fill_getrep(&req->msg, (void *)req->src,
			                     req->size);
			dstm_msg_isend(&req->msg);
			req->status = DSTM_REQUEST_STATUS_ISEND;
		}
	} else if (req->status == DSTM_REQUEST_STATUS_ISEND) {
		if (dstm_msg_test(req->msg)) {
			req->status = DSTM_REQUEST_STATUS_DONE;
		}
	}
	return 0;
}

int dstm_request_progress_out_put(dstm_request_t *req, dstm_time_t step)
{
	/* logic:
	 * - isend: data sent out, isend posted
	 * - done: send completed.
	 */
	(void)step;
	assert(req->status == DSTM_REQUEST_STATUS_ISEND);
	if (dstm_msg_test(req->msg)) {
		req->status = DSTM_REQUEST_STATUS_DONE;
	}
	return 0;
}

int dstm_request_progress_out_get(dstm_request_t *req, dstm_time_t step)
{
	/* logic:
	 * - irecv: request sent, recv posted, waiting for completion
	 * - done: recv completed, get filled out
	 */
	(void)step;
	assert(req->status == DSTM_REQUEST_STATUS_IRECV);
	if (dstm_msg_test(req->msg)) {
		memcpy((void *)req->dst, (void *)req->msg.data, req->src);
		req->status = DSTM_REQUEST_STATUS_DONE;
	}
	return 0;
}

dstm_request_progress_t progress_sm[DSTM_REQUEST_TYPE_MAX] = {
        {DSTM_REQUEST_TYPE_LOCAL_PUT, dstm_request_progress_local_put},
        {DSTM_REQUEST_TYPE_LOCAL_GET, dstm_request_progress_local_get},
        {DSTM_REQUEST_TYPE_IN_PUT, dstm_request_progress_in_put},
        {DSTM_REQUEST_TYPE_IN_GET, dstm_request_progress_in_get},
        {DSTM_REQUEST_TYPE_OUT_PUT, dstm_request_progress_out_put},
        {DSTM_REQUEST_TYPE_OUT_GET, dstm_request_progress_out_get},
};

int dstm_request_progress(dstm_request_t *req, dstm_time_t step)
{
	assert(req != NULL);
	assert(progress_sm[req->type].progress != NULL);
	return progress_sm[req->type].progress(req, step);
}

dstm_time_t dstm_max_time_progress(int rank,
                                   dstm_time_t currenttime,
                                   dstm_time_t maxtime_requested)
{
	assert(currenttime <= maxtime_requested);
	/* compute across neighbors the maximum time delta that can be
	 * progressed without drifting too much:
	 *
	 * each neighbor can be at most at a fixed distance from us, we compute
	 * the largest value smaller than that distance, i.e. the min of all
	 * distances.
	 *
	 * TODO: this is static once past init, so pre-compute and store it
	 */
	dstm_time_t min = maxtime_requested;
	for (int i = 0; i < dstm_global_context.world_size; i++) {
		int dist = dstm_distance_get(dstm_global_context.distances,
		                             rank, i);

		/* skip non neighbors */
		if (dist <= 0)
			continue;

		if (currenttime + dist <= min)
			min = currenttime + dist;
	}
	return min;
}

/*******************************************************************************
 * Distributed mechanics
 ******************************************************************************/

int dstm_put(int rank,
             int region_id,
             dstm_time_t time,
             dstm_time_t delta,
             intptr_t dst,
             void *src,
             size_t size)
{
	(void)region_id;
	if (src == NULL)
		return -DSTM_EINVAL;

	/* no idea what the semantic of a put in the past would be for now */
	if (delta < 0)
		return -DSTM_EINVAL;

	/* request tracking part:
	 * making sure the request is being sent to the right place
	 */

	/* first, figure out if this is a local request or a remote one */
	dstm_request_t *req;
	if (rank == dstm_global_context.rank) {
		/* local put:
		 * we just enqueue it locally
		 * */
		dstm_request_create(&req, DSTM_REQUEST_TYPE_LOCAL_PUT, time,
		                    delta, rank, dst, (intptr_t)src, size);
	} else {
		/* outgoing put
		 * we enqueue it for tracking purposes, and push it out to the
		 * appropriate rank.
		 */
		dstm_request_create(&req, DSTM_REQUEST_TYPE_OUT_PUT, time,
		                    delta, rank, dst, (intptr_t)src, size);

		/* send out the operation asynchronously, with an extra copy of
		 * the source buffer
		 */
		dstm_msg_fill_hdr(&req->msg, DSTM_MSG_TYPE_PUT, time, delta,
		                  rank, dstm_global_context.rank);
		dstm_msg_fill_put(&req->msg, dst, src, size);
		dstm_msg_isend(&req->msg);
		req->status = DSTM_REQUEST_STATUS_ISEND;
	}
	DL_INSERT_INORDER(requests, req, dstm_request_cmp_departure);

	/* synchronization part:
	 * figuring out if we can make progress on some requests.
	 *
	 * In order to force progress, we consider that if the current put is at
	 * a timestep later than the current region, then we need to update that
	 * region and flush all the requests that would be at an earlier time.
	 *
	 * TODO: this entire loop is quite convoluted, there's a better version
	 * of that that only goes through events that actually matter, by
	 * alternating between handling all pending messages and the list of
	 * requests that can be processed in the allowed interval.
	 */
	int done = 1;
	if (time > dstm_global_context.region->time)
		done = 0;

	while (!done) {
		dstm_request_t *nr;
		dstm_msg_t m;
		/* block until we receive one incoming message */
		dstm_msg_recvany(&m);

		/* Update neighbor's highest received timestamp */
		dstm_region_set_ninfo(dstm_global_context.region,
		                      m.header.src_rank, m.header.time);

		switch (m.header.type) {
		case DSTM_MSG_TYPE_PUT:
			/* request for a put, enqueue the request as an
			 * incoming put
			 */
			dstm_request_create(&nr, DSTM_REQUEST_TYPE_IN_PUT,
			                    m.header.time, m.header.delta,
			                    m.header.dst_rank, m.u.put.dst,
			                    (intptr_t)m.data, m.u.put.size);
			nr->msg = m;
			DL_INSERT_INORDER(requests, nr,
			                  dstm_request_cmp_departure);
			break;
		case DSTM_MSG_TYPE_GET_REQ:
			/* request for a get, enqueue as an incoming
			 * get
			 */
			dstm_request_create(&nr, DSTM_REQUEST_TYPE_IN_GET,
			                    m.header.time, m.header.delta,
			                    m.header.dst_rank, (intptr_t)m.data,
			                    m.u.get.ptr, m.u.get.size);
			nr->msg = m;
			DL_INSERT_INORDER(requests, nr,
			                  dstm_request_cmp_departure);
			break;
		case DSTM_MSG_TYPE_GET_REP:
			/* response for a get, not supposed to arrive
			 * here?
			 * TODO figure out how existing irecvs and  mpi_probe
			 * interact
			 */
		default:
			assert(0);
		}

		/* there's a chance that we cannot progress by as much as the
		 * new timestamp: if the neighbors are not as far along as
		 * expected. So compute the largest delta we can progress
		 * on based on current neighbor data before engaging in request
		 * progress. */
		dstm_time_t maxtime = dstm_max_time_progress(
		        dstm_global_context.rank,
		        dstm_global_context.region->time, time);
		assert(maxtime <= time);

		/* loop over all requests, see what can be progressed, one
		 * timestep at a time. We have to go through each timestep to
		 * make sure that we progress requests in the right order.
		 */
		for (dstm_time_t nextt = dstm_global_context.region->time;
		     nextt <= maxtime; nextt++) {

			dstm_request_t *it, *tmp;
			DL_FOREACH_SAFE(requests, it, tmp)
			{
				/* don't go further than new time */
				if (it->time >= maxtime)
					break;

				dstm_request_progress(it, nextt);
				if (dstm_request_isdone(it)) {
					DL_DELETE(requests, it);
					if (it == req) {
						/* we're done, post process */
						done = 1;
					}
					dstm_request_destroy(&it);
				}
			}
		}
	}
	// by the time we return, the request is at least registered.
	return 0;
}

int dstm_get(int rank,
             int region_id,
             dstm_time_t time,
             dstm_time_t delta,
             void *dst,
             intptr_t src,
             size_t size)
{
	(void)region_id;
	if (dst == NULL)
		return -DSTM_EINVAL;

	/* no idea what the semantic of a get in the future would be for now */
	if (delta > 0)
		return -DSTM_EINVAL;

	/* request tracking part:
	 * making sure the request is being sent to the right place
	 */

	/* first, figure out if this is a local request or a remote one */
	dstm_request_t *req;
	if (rank == dstm_global_context.rank) {
		/* local get:
		 * we just enqueue it locally
		 * */
		dstm_request_create(&req, DSTM_REQUEST_TYPE_LOCAL_GET, time,
		                    delta, rank, (intptr_t)dst, src, size);
	} else {
		/* outgoing get
		 * we enqueue it for tracking purposes, and push it out to the
		 * appropriate rank.
		 */
		dstm_request_create(&req, DSTM_REQUEST_TYPE_OUT_GET, time,
		                    delta, rank, (intptr_t)dst, src, size);

		/* this is the place where MPI RMA might make sense, if we could
		 * use windows to make sure the MPI processes are in the right
		 * state.
		 *
		 * But they're not, because the dstm logic doesn't enforce that
		 * only one process would be doing gets with the right
		 * timeline.
		 *
		 * There might be a weird version of this possible by forcing
		 * fences for every step, basically making sure that all ranks
		 * that will ever access a window have done the same amount of
		 * fences on it. I'm not sure it would work in practice.
		 *
		 * Instead, we end up using MPI almost like a messaging system,
		 * sending remote requests and getting answers back.  It's a
		 * pretty poor use of MPI really, and should probably be done in
		 * a different way (tagged RMA extension, or maybe just use
		 * libfabric directly).
		 */

		/* we're sending to the destination rank the fact that it needs
		 * to give us data, and then preparing to receive it in the
		 * original window. We're keeping the MPI semantics that a get
		 * is not supposed to touch its buffer until the operation
		 * completes.
		 */

		dstm_msg_fill_hdr(&req->msg, DSTM_MSG_TYPE_GET_REQ, time, delta,
		                  rank, dstm_global_context.rank);
		dstm_msg_fill_getreq(&req->msg, src, size);
		dstm_msg_send(&req->msg);

		dstm_msg_fill_hdr(&req->msg, DSTM_MSG_TYPE_GET_REP, time, delta,
		                  rank, dstm_global_context.rank);
		dstm_msg_fill_getrep(&req->msg, dst, size);
		dstm_msg_irecv(&req->msg);
		req->status = DSTM_REQUEST_STATUS_IRECV;
	}
	DL_INSERT_INORDER(requests, req, dstm_request_cmp_departure);

	/* synchronization part:
	 * this is a blocking call, so we take advantage of it to make progress
	 * on stuff, until we can verify that the request itself is completed:
	 */
	int done = 0;
	while (!done) {
		dstm_request_t *nr;
		dstm_msg_t m;
		/* block until we receive one incoming message */
		dstm_msg_recvany(&m);

		/* Update neighbor's highest received timestamp */
		dstm_region_set_ninfo(dstm_global_context.region,
		                      m.header.src_rank, m.header.time);

		switch (m.header.type) {
		case DSTM_MSG_TYPE_PUT:
			/* request for a put, enqueue the request as an
			 * incoming put
			 */
			dstm_request_create(&nr, DSTM_REQUEST_TYPE_IN_PUT,
			                    m.header.time, m.header.delta,
			                    m.header.dst_rank, m.u.put.dst,
			                    (intptr_t)m.data, m.u.put.size);
			nr->msg = m;
			DL_INSERT_INORDER(requests, nr,
			                  dstm_request_cmp_departure);
			break;
		case DSTM_MSG_TYPE_GET_REQ:
			/* request for a get, enqueue as an incoming
			 * get
			 */
			dstm_request_create(&nr, DSTM_REQUEST_TYPE_IN_PUT,
			                    m.header.time, m.header.delta,
			                    m.header.dst_rank, (intptr_t)m.data,
			                    m.u.get.ptr, m.u.get.size);
			nr->msg = m;
			DL_INSERT_INORDER(requests, nr,
			                  dstm_request_cmp_departure);
			break;
		case DSTM_MSG_TYPE_GET_REP:
			/* response for a get, not supposed to arrive
			 * here?
			 * TODO figure out how existing irecvs and  mpi_probe
			 * interact
			 */
		default:
			assert(0);
		}

		/* there's a chance that we cannot progress by as much as the
		 * new timestamp: if the neighbors are not as far along as
		 * expected. So compute the largest delta we can progress
		 * on based on current neighbor data before engaging in request
		 * progress. */
		dstm_time_t maxtime = dstm_max_time_progress(
		        dstm_global_context.rank,
		        dstm_global_context.region->time, time);
		assert(maxtime <= time);

		/* loop over all requests, see what can be progressed */
		dstm_request_t *it, *tmp;
		DL_FOREACH_SAFE(requests, it, tmp)
		{
			/* don't go further than new time */
			if (it->time >= maxtime)
				break;
			dstm_request_progress(it, maxtime);
			if (dstm_request_isdone(it)) {
				DL_DELETE(requests, it);
				if (it == req) {
					/* we're done, post process */
					done = 1;
				}
				dstm_request_destroy(&it);
			}
		}
	}
	return 0;
}
