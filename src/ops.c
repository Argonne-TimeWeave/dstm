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
	dstm_ptr_t dst;
	dstm_ptr_t src;
	size_t size;
	dstm_msg_t msg;
	struct dstm_request_s *prev, *next;
};
typedef struct dstm_request_s dstm_request_t;

dstm_request_t *requests = NULL;

void dstm_log_printrequest(int level, dstm_request_t *req);
void dstm_request_fprintf(FILE *f, dstm_request_t *req);

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
                        dstm_ptr_t dst,
                        dstm_ptr_t src,
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
 * Versioning logic: single copy edition.
 *
 * Two states for a region:
 * - writable:
 * - readable
 ******************************************************************************/

#define DSTM_REGION_STATUS_WRITABLE 0
#define DSTM_REGION_STATUS_READABLE 1

static int dstm_local_region_status = DSTM_REGION_STATUS_WRITABLE;

int dstm_region_status_update(dstm_time_t time)
{
	(void)time;
	if (dstm_local_region_status == DSTM_REGION_STATUS_WRITABLE) {
		/* we're in the process of accepting writes, check if we should
		 * stay in that state or start accepting reads
		 */

		/* do we have pending puts? */
		dstm_request_t *it, *tmp;
		DL_FOREACH_SAFE(requests, it, tmp)
		{
			/* skip any put not related to current region time */
			if (it->time + it->delta >
			    dstm_global_context.region->time)
				continue;

			if (it->type == DSTM_REQUEST_TYPE_LOCAL_PUT &&
			    it->status != DSTM_REQUEST_STATUS_DONE)
				return 0;

			if (it->type == DSTM_REQUEST_TYPE_IN_PUT &&
			    it->status != DSTM_REQUEST_STATUS_DONE)
				return 0;
		}

		/* could we receive more puts?
		 * i.e. are my neighbors all after currenttime - distance
		 */
		for (int i = 0; i < dstm_global_context.world_size; i++) {
			int dist =
			        dstm_distance_get(dstm_global_context.distances,
			                          dstm_global_context.rank, i);

			dstm_time_t nt;
			dstm_region_get_ntime(dstm_global_context.region, i,
			                      &nt);

			/* skip non neighbors */
			if (dist <= 0)
				continue;

			if (nt + dist <= dstm_global_context.region->time)
				return 0;
		}
		dstm_log_debug("region readable, time: %" PRIu64 "\n",
		               dstm_global_context.region->time);

		/* switching to read, not changing time */
		dstm_local_region_status = DSTM_REGION_STATUS_READABLE;
		return 0;
	} else {
		/* we're in the progress of accepting reads, check if we should
		 * stay in that state or start accepting writes
		 */

		/* do we have pending gets */
		dstm_request_t *it, *tmp;
		DL_FOREACH_SAFE(requests, it, tmp)
		{
			/* skip any get not related to current region time */
			if (it->time + it->delta >
			    dstm_global_context.region->time)
				continue;

			if (it->type == DSTM_REQUEST_TYPE_LOCAL_GET &&
			    it->status != DSTM_REQUEST_STATUS_DONE)
				return 0;

			if (it->type == DSTM_REQUEST_TYPE_IN_GET &&
			    it->status != DSTM_REQUEST_STATUS_DONE)
				return 0;
		}

		/* could we receive more gets?
		 * i.e. are my neighbors closer than currenttime - distance
		 */
		for (int i = 0; i < dstm_global_context.world_size; i++) {
			int dist =
			        dstm_distance_get(dstm_global_context.distances,
			                          dstm_global_context.rank, i);

			dstm_time_t nt;
			dstm_region_get_ntime(dstm_global_context.region, i,
			                      &nt);

			/* skip non neighbors */
			if (dist <= 0)
				continue;

			if (nt - dist <= dstm_global_context.region->time)
				return 0;
		}

		/* switching to write, should only happen when moving in time.
		 * */
		assert(dstm_global_context.region->time < time);
		dstm_global_context.region->time++;
		dstm_log_debug("region writable, time: %" PRIu64 "\n",
		               dstm_global_context.region->time);
		dstm_local_region_status = DSTM_REGION_STATUS_WRITABLE;
		return 0;
	}
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
	int (*progress)(dstm_request_t *req);
} dstm_request_progress_t;

int dstm_request_progress_local_put(dstm_request_t *req)
{
	/* logic of a local put:
	 * - starts in INIT state, enqueued as a request to put data associated
	 *   with the request buffer into the local region memory.
	 *
	 * Can be completed if current region is available.
	 * - when step == req->time + delta, apply the request to the local
	 *   memory.
	 */
	dstm_region_t *rg = dstm_get_local_region();
	assert(req->status == DSTM_REQUEST_STATUS_INIT);
	if (req->time + req->delta == rg->time &&
	    dstm_local_region_status == DSTM_REGION_STATUS_WRITABLE) {
		void *dptr = dstm_region_ptr_tovoid(rg, req->dst);
		void *sptr = dstm_region_ptr_tovoid(rg, req->src);
		memcpy(dptr, sptr, req->size);
		req->status = DSTM_REQUEST_STATUS_DONE;
	}
	return 0;
}

int dstm_request_progress_local_get(dstm_request_t *req)
{
	/* logic of a local put:
	 * - starts in INIT state, enqueued as a request to put data associated
	 *   with the request buffer into the local region memory.
	 *
	 * - when step == req->time + delta, apply the request to the local
	 *   memory.
	 */
	dstm_region_t *rg = dstm_get_local_region();
	assert(req->status == DSTM_REQUEST_STATUS_INIT);
	if (req->time + req->delta == rg->time &&
	    dstm_local_region_status == DSTM_REGION_STATUS_READABLE) {
		void *dptr = dstm_region_ptr_tovoid(rg, req->dst);
		void *sptr = dstm_region_ptr_tovoid(rg, req->src);
		memcpy(dptr, sptr, req->size);
		req->status = DSTM_REQUEST_STATUS_DONE;
	}
	return 0;
}

int dstm_request_progress_in_put(dstm_request_t *req)
{
	/* logic:
	 * - init: input request received from external rank
	 * - done: applied put
	 */
	dstm_region_t *rg = dstm_get_local_region();
	assert(req->status == DSTM_REQUEST_STATUS_INIT);
	if (req->time + req->delta == rg->time &&
	    dstm_local_region_status == DSTM_REGION_STATUS_WRITABLE) {
		void *dptr = dstm_region_ptr_tovoid(rg, req->dst);
		void *sptr = dstm_region_ptr_tovoid(rg, req->src);
		memcpy(dptr, sptr, req->size);
		req->status = DSTM_REQUEST_STATUS_DONE;
	}
	return 0;
}

int dstm_request_progress_in_get(dstm_request_t *req)
{
	/* logic:
	 * - init: input request received from external rank
	 * -     : filled out get, sent data back
	 *       : send completed
	 */
	dstm_region_t *rg = dstm_get_local_region();
	if (req->status == DSTM_REQUEST_STATUS_INIT) {
		if (req->time + req->delta == rg->time &&
		    dstm_local_region_status == DSTM_REGION_STATUS_READABLE) {
			void *sptr = dstm_region_ptr_tovoid(rg, req->src);
			dstm_log_debug("getrep: %u %u %p %u %u\n",
			               req->src.region, req->src.offset, sptr,
			               req->size, rg->size);
			dstm_msg_fill_hdr(&req->msg, DSTM_MSG_TYPE_GET_REP,
			                  req->time, req->delta,
			                  dstm_global_context.rank, req->rank);
			dstm_msg_fill_getrep(&req->msg,
					     sptr,
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

int dstm_request_progress_out_put(dstm_request_t *req)
{
	/* logic:
	 * - isend: data sent out, isend posted
	 * - done: send completed.
	 */
	assert(req->status == DSTM_REQUEST_STATUS_ISEND);
	if (dstm_msg_test(req->msg)) {
		req->status = DSTM_REQUEST_STATUS_DONE;
	}
	return 0;
}

int dstm_request_progress_out_get(dstm_request_t *req)
{
	/* logic:
	 * - irecv: request sent, recv posted, waiting for completion
	 * - done: recv completed, get filled out
	 */
	dstm_region_t *rg = dstm_get_local_region();
	assert(req->status == DSTM_REQUEST_STATUS_IRECV);
	if (dstm_msg_test(req->msg)) {
		void *dptr = dstm_region_ptr_tovoid(rg, req->dst);
		dstm_log_debug("applying: %p %zu\n", dptr, req->size);
		memcpy(dptr, (void *)req->msg.data, req->size);
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

int dstm_request_progress(dstm_request_t *req)
{
	assert(req != NULL);
	assert(progress_sm[req->type].progress != NULL);
	dstm_log_debug("request:");
	dstm_log_printrequest(DSTM_LOG_DEBUG, req);
	dstm_log_debug("region:");
	dstm_log_printregion(DSTM_LOG_DEBUG, dstm_get_local_region());
	return progress_sm[req->type].progress(req);
}

/* Progress loop: recv messages and try to make progress on any request.
 *
 * We try to progress as much as we can each call. An alternative design
 * would have returned as soon as one thing made progress, but that could let
 * to longer latencies on same requests.
 *
 * TODO: actually evaluate that.
 */
int dstm_do_progress(dstm_time_t time)
{
	dstm_log_info("progress loop:  reqtime: %d, region time: %d\n", time,
	              dstm_global_context.region->time);
	dstm_request_t *nr;
	dstm_ptr_t dptr, sptr;
	dstm_msg_t m;
	MPI_Status status;

	/* test if we have any message to deal with */
	int ret = dstm_msg_iprobe(&status);

	if (ret) {
		dstm_log_debug("incoming message: %d,%d\n", status.MPI_SOURCE,
		               status.MPI_TAG);
		/* block until we receive the incoming message */
		dstm_msg_recvany(status, &m);

		/* Update neighbor's highest received timestamp */
		dstm_region_set_ninfo(dstm_global_context.region,
		                      m.header.src_rank, m.header.time);
		dstm_log_debug("received: ");
		dstm_log_printmsg(DSTM_LOG_DEBUG, &m);
		switch (m.header.type) {
		case DSTM_MSG_TYPE_PUT:
			/* request for a put, enqueue the request as an
			 * incoming put
			 */
			dptr = (dstm_ptr_t){m.u.put.region, m.u.put.dst};
			sptr = (dstm_ptr_t){DSTM_REGIONID_LOCAL,
			                    (intptr_t)m.data};
			dstm_request_create(&nr, DSTM_REQUEST_TYPE_IN_PUT,
			                    m.header.time, m.header.delta,
			                    m.header.dst_rank, dptr, sptr,
			                    m.u.put.size);
			nr->msg = m;
			DL_INSERT_INORDER(requests, nr,
			                  dstm_request_cmp_departure);
			break;
		case DSTM_MSG_TYPE_GET_REQ:
			/* request for a get, enqueue as an incoming
			 * get
			 */
			dptr = (dstm_ptr_t){DSTM_REGIONID_LOCAL,
			                    (intptr_t)m.data};
			sptr = (dstm_ptr_t){m.u.get.region, m.u.get.ptr};
			dstm_request_create(&nr, DSTM_REQUEST_TYPE_IN_GET,
			                    m.header.time, m.header.delta,
			                    m.header.src_rank, dptr, sptr,
			                    m.u.get.size);
			nr->msg = m;
			DL_INSERT_INORDER(requests, nr,
			                  dstm_request_cmp_departure);
			break;
		case DSTM_MSG_TYPE_GET_REP:
			/* response for a get, not supposed to arrive
			 * here?
			 * TODO figure out how existing irecvs and
			 * mpi_iprobe interact
			 */
		default:
			assert(0);
		}
	}

	/* Progressing requests is quite counter-intuitive. We're not really
	 * constrained by time, not directly. Sending out and Receiving messages
	 * is just about making progress on MPI operations. And applying puts
	 * and gets is about "availability" of that version in time of memory.
	 *
	 * Assume we maintain a copy of each local region per time stamp.  -
	 * ignore unwritten bytes, assume arbitrary invalid data at init - when
	 * all the puts directed at it have been applied, and no more can be
	 * received, then this specific version is "ready" - gets to a region
	 * can progress when the version is "ready".
	 *
	 * The challenge is then mostly a question of extra constraints, in
	 * particular:
	 *  1. An optimized version of that were we remove old copies as we go.
	 *  2. A version with limited amount of copies per region (a fixed
	 *     number of timestamps available at once).
	 *  3. A version with a single copy per region (only one timestamp
	 *     available at once).
	 *
	 * The unlimited case is almost too easy: apply any put as soon as
	 * possible, and only track readiness of each region for gets.
	 * A more constrained version introduces additional synchronizations,
	 * since it requires discarding old copies, and that can only be done
	 * once all gets are done and no more can arrive.
	 *
	 * For now we implement the single copy version.
	 */

	dstm_region_status_update(time);
	/* loop over all requests, see what can be progressed */
	dstm_request_t *it, *tmp;
	DL_FOREACH_SAFE(requests, it, tmp)
	{
		dstm_request_progress(it);
		if (dstm_request_isdone(it)) {
			DL_DELETE(requests, it);
		}
	}
	return 0;
}

/*******************************************************************************
 * Distributed mechanics
 ******************************************************************************/

int dstm_put(int rank,
             int region_id,
             dstm_time_t time,
             dstm_time_t delta,
             dstm_ptr_t dst,
             void *src,
             size_t size)
{
	dstm_log_info("put: %d, %d, %" PRIu64 ", %" PRId64 ", %d:%d, %p, %d\n",
	              rank, region_id, time, delta, dst.region, dst.offset, src,
	              size);
	dstm_ptr_t sptr;

	assert(rank == region_id);
	assert(rank == dst.region);

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
		sptr = (dstm_ptr_t){ DSTM_REGIONID_LOCAL, (intptr_t ) src };
		dstm_request_create(&req, DSTM_REQUEST_TYPE_LOCAL_PUT, time,
		                    delta, rank, dst, sptr, size);
	} else {
		/* outgoing put
		 * we enqueue it for tracking purposes, and push it out to the
		 * appropriate rank.
		 */
		sptr = (dstm_ptr_t){ DSTM_REGIONID_LOCAL, (intptr_t ) src };
		dstm_request_create(&req, DSTM_REQUEST_TYPE_OUT_PUT, time,
		                    delta, rank, dst, sptr, size);

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
	// by the time we return, the request is at least registered.
	return 0;
}

int dstm_get(int rank,
             int region_id,
             dstm_time_t time,
             dstm_time_t delta,
             void *dst,
             dstm_ptr_t src,
             size_t size)
{
	dstm_log_info("get: %d, %d, %" PRIu64 ", %" PRId64 ", %p, %d:%d, %d\n",
	              rank, region_id, time, delta, dst, src.region, src.offset,
	              size);
	dstm_ptr_t dptr;

	assert(rank == region_id);
	assert(rank == src.region);
	
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
		dptr = (dstm_ptr_t){ DSTM_REGIONID_LOCAL, (intptr_t ) dst };
		dstm_request_create(&req, DSTM_REQUEST_TYPE_LOCAL_GET, time,
		                    delta, rank, dptr, src, size);
	} else {
		/* outgoing get
		 * we enqueue it for tracking purposes, and push it out to the
		 * appropriate rank.
		 */
		dptr = (dstm_ptr_t){ DSTM_REGIONID_LOCAL, (intptr_t ) dst };
		dstm_request_create(&req, DSTM_REQUEST_TYPE_OUT_GET, time,
		                    delta, rank, dptr, src, size);

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
		                  dstm_global_context.rank, rank);
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
		dstm_do_progress(time);
		if (dstm_request_isdone(req))
			done = 1;
	}
	dstm_request_destroy(&req);
	return 0;
}

/*******************************************************************************
 * Logging helpers
 ******************************************************************************/

void dstm_log_printrequest(int level, dstm_request_t *request)
{
	if (level > dstm_log_level)
		return;
	if (request == NULL) {
		fprintf(dstm_log_fd, "NULL\n");
		return;
	}
	dstm_request_fprintf(dstm_log_fd, request);
}

typedef struct dstm_request_type_table_s {
	int type;
	const char *s;
} dstm_request_type_table_t;

static const dstm_request_type_table_t dstm_request_type_table[] = {
        {DSTM_REQUEST_TYPE_LOCAL_PUT, "LOCAL_PUT"},
        {DSTM_REQUEST_TYPE_LOCAL_GET, "LOCAL_GET"},
        {DSTM_REQUEST_TYPE_IN_PUT, "IN_PUT"},
        {DSTM_REQUEST_TYPE_IN_GET, "IN_GET"},
        {DSTM_REQUEST_TYPE_OUT_PUT, "OUT_GET"},
        {DSTM_REQUEST_TYPE_OUT_GET, "OUT_GET"},
        {0, NULL},
};

const char *dstm_request_type_t2s(int type,
                                  const dstm_request_type_table_t *table)
{
	if (type < 0)
		return "UNKNOWN";
	for (int i = 0; table[i].s != NULL; i++)
		if (table[i].type == type)
			return table[i].s;
	return "UNKNOWN";
}

void dstm_request_fprintf(FILE *f, dstm_request_t *req)
{
	assert(f != NULL);
	assert(req != NULL);
	fprintf(f,
	        "{ type: %s, status: %d, time: %" PRIu64 ", delta: %" PRId64
	        ", rank: %d, size: %zu\n",
	        dstm_request_type_t2s(req->type, dstm_request_type_table),
	        req->status, req->time, req->delta, req->rank, req->size);
	dstm_ptr_fprintf(f, req->dst);
	dstm_ptr_fprintf(f, req->src);
	dstm_msg_fprintf(f, &req->msg);
}
