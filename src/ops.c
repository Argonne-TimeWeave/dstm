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

#include <limits.h>
#include <mpi.h>
#include <stdlib.h>

#include "dstm.h"

#include "internal/globals.h"
#include "internal/msg.h"
#include "internal/region.h"
#include "internal/request.h"
#include "internal/utlist.h"

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

static int dstm_request_tag_count = 1;

int dstm_request_new_tag(void)
{
	int ret = -1;
	int tag_range = dstm_global_context.world_size * 1000;
	ret = tag_range * dstm_global_context.rank + dstm_request_tag_count;
	dstm_request_tag_count++;
	return ret;
}

int dstm_request_create(dstm_request_t **r,
                        int type,
                        int tag,
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
	req->tag = tag;
	req->status = DSTM_REQUEST_STATUS_INIT;
	req->time = time;
	req->delta = delta;
	req->rank = rank;
	req->dst = dst;
	req->src = src;
	req->size = size;
	req->buf = NULL;
	*r = req;
	return 0;
}

int dstm_request_destroy(dstm_request_t **r)
{
	if (r == NULL || *r == NULL)
		return 0;
	dstm_log_request(DSTM_LOG_DEBUG, "destroy", *r);
	/* free the owned payload buffer, if any */
	free((*r)->buf);
	/* free any MPI pack buffer that is still outstanding */
	free((*r)->msg.packbuf);
	free(*r);
	*r = NULL;
	return 0;
}

/*******************************************************************************
 * Versioning logic: multi-version edition.
 *
 * Each region maintains a list of timestamped versions, each with its own
 * state machine:  WRITABLE -> READABLE -> DISCARDABLE.
 *
 * PUTs are applied immediately to the target version (creating it on demand).
 * GETs wait for the version to become READABLE.  Old versions are evicted
 * (DISCARDABLE) once all GETs are done and no neighbor can still send a GET
 * for that timestamp (distance-based check).
 ******************************************************************************/

int dstm_region_send_heartbeat_to_all(dstm_time_t time, int status)
{
	for (int i = 0; i < dstm_global_context.world_size; i++) {
		int dist = dstm_distance_get(dstm_global_context.distances,
		                             dstm_global_context.rank, i);
		if (dist <= 0)
			continue;

		dstm_msg_t m;
		dstm_msg_fill_hdr(&m, DSTM_MSG_TYPE_HEARTBEAT, time, 0, i,
		                  dstm_global_context.rank);
		dstm_msg_fill_heartbeat(&m, status);

		int tag = dstm_request_new_tag();
		dstm_msg_send(&m, tag);
	}
	return 0;
}

static int dstm_local_heartbeat_time = 0;

/* Check whether a WRITABLE version can transition to READABLE.
 * Returns 1 if the transition should happen, 0 otherwise. */
static int dstm_version_can_become_readable(dstm_region_version_t *v)
{
	dstm_region_t *region = dstm_global_context.region;

	/* Check for pending PUT requests targeting this version's time. */
	dstm_request_t *it;
	DL_FOREACH(requests, it)
	{
		dstm_time_t target = it->time + it->delta;
		if (target != v->time)
			continue;

		if (it->type == DSTM_REQUEST_TYPE_LOCAL_PUT &&
		    it->status != DSTM_REQUEST_STATUS_DONE)
			return 0;

		if (it->type == DSTM_REQUEST_TYPE_IN_PUT &&
		    it->status != DSTM_REQUEST_STATUS_DONE)
			return 0;
	}

	/* Could we receive more PUTs from neighbors?
	 * A neighbor at distance dist whose last-known time is nt can still
	 * send a PUT targeting v->time if nt + dist <= v->time.
	 *
	 * Barrier optimization: after barrier round 1 completes, all ranks
	 * have finished their outgoing work.  No new PUT messages will
	 * arrive, so skip all neighbor distance checks. */
	if (!region->barrier_all_drained) {
		for (int i = 0; i < dstm_global_context.world_size; i++) {
			int dist = dstm_distance_get(
			        dstm_global_context.distances,
			        dstm_global_context.rank, i);
			if (dist <= 0)
				continue;

			dstm_time_t nt;
			int s;
			dstm_region_get_ninfo(region, i, &nt, &s);

			dstm_log_debug("checkr v%" PRId64 ": nt=%" PRId64
			               " dist=%d\n",
			               v->time, nt, dist);
			if (nt + dist <= v->time)
				return 0;
		}
	}

	return 1;
}

/* Check whether a READABLE version can transition to DISCARDABLE.
 * Returns 1 if the transition should happen, 0 otherwise. */
static int dstm_version_can_become_discardable(dstm_region_version_t *v)
{
	dstm_region_t *region = dstm_global_context.region;

	/* Check for pending GET requests targeting this version's time. */
	dstm_request_t *it;
	DL_FOREACH(requests, it)
	{
		dstm_time_t target = it->time + it->delta;
		if (target != v->time)
			continue;

		if (it->type == DSTM_REQUEST_TYPE_LOCAL_GET &&
		    it->status != DSTM_REQUEST_STATUS_DONE)
			return 0;

		if (it->type == DSTM_REQUEST_TYPE_IN_GET &&
		    it->status != DSTM_REQUEST_STATUS_DONE)
			return 0;
	}

	/* Could we receive more GETs from neighbors?
	 * A neighbor at distance dist whose last-known time is nt can still
	 * send a GET targeting v->time if nt - dist <= v->time.
	 *
	 * Barrier optimization: after barrier round 1 completes, no new
	 * GET messages will arrive, so skip all neighbor distance checks. */
	if (!region->barrier_all_drained) {
		for (int i = 0; i < dstm_global_context.world_size; i++) {
			int dist = dstm_distance_get(
			        dstm_global_context.distances,
			        dstm_global_context.rank, i);
			if (dist <= 0)
				continue;

			dstm_time_t nt;
			int s;
			dstm_region_get_ninfo(region, i, &nt, &s);

			dstm_log_debug("checkd v%" PRId64 ": nt=%" PRId64
			               " dist=%d\n",
			               v->time, nt, dist);
			if (nt - dist <= v->time)
				return 0;
		}
	}

	return 1;
}

int dstm_region_status_update(dstm_time_t time)
{
	dstm_region_t *region = dstm_global_context.region;
	dstm_region_version_t *v, *vtmp;

	DL_FOREACH_SAFE(region->versions, v, vtmp)
	{
		switch (v->status) {
		case DSTM_REGION_STATUS_WRITABLE:
			if (dstm_version_can_become_readable(v)) {
				v->status = DSTM_REGION_STATUS_READABLE;
				dstm_log_info("version readable, time: %" PRId64
				              "\n",
				              v->time);
			}
			break;

		case DSTM_REGION_STATUS_READABLE:
			if (dstm_version_can_become_discardable(v)) {
				v->status = DSTM_REGION_STATUS_DISCARDABLE;
				dstm_log_info("version discardable, time: %"
				              PRId64 "\n",
				              v->time);
			}
			break;

		case DSTM_REGION_STATUS_DISCARDABLE:
			dstm_log_info("evicting version, time: %" PRId64 "\n",
			              v->time);
			dstm_region_version_destroy(region, v);
			break;
		}
	}

	/* Update region->time to the minimum active version timestamp.
	 * If no versions remain, advance to the caller's time. */
	if (region->versions != NULL)
		region->time = region->versions->time;
	else
		region->time = time;

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
	int (*progress)(dstm_request_t *req);
} dstm_request_progress_t;

int dstm_request_progress_local_put(dstm_request_t *req)
{
	/* logic of a local put:
	 * - starts in INIT state, with payload buffered in req->buf.
	 *
	 * Applied immediately by finding or creating the target version.
	 */
	dstm_region_t *rg = dstm_get_local_region();
	assert(req->status == DSTM_REQUEST_STATUS_INIT);
	dstm_time_t target = req->time + req->delta;
	dstm_region_version_t *v = dstm_region_version_find_or_create(rg,
	                                                              target);
	assert(v != NULL);
	void *dptr = (char *)v->buffer + req->dst.offset;
	memcpy(dptr, req->buf, req->size);
	req->status = DSTM_REQUEST_STATUS_DONE;
	return 0;
}

int dstm_request_progress_local_get(dstm_request_t *req)
{
	/* logic of a local get:
	 * - starts in INIT state, enqueued as a request to copy data from a
	 *   version's buffer into the caller's buffer (req->dst,
	 *   REGIONID_LOCAL).
	 *
	 * - completes when the target version exists and is READABLE.
	 */
	dstm_region_t *rg = dstm_get_local_region();
	assert(req->status == DSTM_REQUEST_STATUS_INIT);
	dstm_time_t target = req->time + req->delta;
	dstm_region_version_t *v = dstm_region_version_find(rg, target);
	if (v != NULL && v->status == DSTM_REGION_STATUS_READABLE) {
		void *dptr = dstm_region_ptr_tovoid(rg, req->dst);
		void *sptr = (char *)v->buffer + req->src.offset;
		memcpy(dptr, sptr, req->size);
		req->status = DSTM_REQUEST_STATUS_DONE;
	}
	return 0;
}

int dstm_request_progress_in_put(dstm_request_t *req)
{
	/* logic:
	 * - init: payload already buffered in req->buf (copied on receipt).
	 * - done: payload applied immediately to the target version.
	 */
	dstm_region_t *rg = dstm_get_local_region();
	assert(req->status == DSTM_REQUEST_STATUS_INIT);
	dstm_time_t target = req->time + req->delta;
	dstm_region_version_t *v = dstm_region_version_find_or_create(rg,
	                                                              target);
	assert(v != NULL);
	void *dptr = (char *)v->buffer + req->dst.offset;
	memcpy(dptr, req->buf, req->size);
	req->status = DSTM_REQUEST_STATUS_DONE;
	return 0;
}

int dstm_request_progress_in_get(dstm_request_t *req)
{
	/* logic:
	 * - init: GET_REQ received, waiting for the version to become
	 *   readable.
	 * - isend: reply buffered in req->buf and MPI_Isend posted.
	 * - done: send completed.
	 */
	dstm_region_t *rg = dstm_get_local_region();
	if (req->status == DSTM_REQUEST_STATUS_INIT) {
		dstm_time_t target = req->time + req->delta;
		dstm_region_version_t *v = dstm_region_version_find(rg,
		                                                    target);
		if (v != NULL &&
		    v->status == DSTM_REGION_STATUS_READABLE) {
			void *sptr = (char *)v->buffer + req->src.offset;

			/* Free the packbuf from the original received GET_REQ
			 * before reusing the message for the reply. */
			dstm_msg_free_packbuf(&req->msg);

			/* Buffer the reply payload in req->buf. */
			req->buf = malloc(req->size);
			assert(req->buf != NULL);
			memcpy(req->buf, sptr, req->size);

			dstm_ptr_t null_ptr = {DSTM_REGIONID_INVALID, 0};
			dstm_msg_fill_hdr(&req->msg, DSTM_MSG_TYPE_GET_REP,
			                  req->time, req->delta, req->rank,
			                  dstm_global_context.rank);
			dstm_msg_fill_payload(&req->msg, null_ptr, req->buf,
			                      req->size);
			dstm_msg_isend(&req->msg, req->tag);
			req->status = DSTM_REQUEST_STATUS_ISEND;
		}
	} else if (req->status == DSTM_REQUEST_STATUS_ISEND) {
		if (dstm_msg_test(req->msg)) {
			/* packbuf freed by dstm_request_destroy */
			req->status = DSTM_REQUEST_STATUS_DONE;
		}
	}
	return 0;
}

int dstm_request_progress_out_put(dstm_request_t *req)
{
	/* logic:
	 * - isend: data sent out, isend posted.
	 * - done: send completed; packbuf freed by dstm_request_destroy.
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
	 * - irecv: GET_REQ sent, reply received and buffered in req->buf.
	 * - done: payload copied to caller's destination.
	 */
	dstm_region_t *rg = dstm_get_local_region();
	if (req->status == DSTM_REQUEST_STATUS_IRECV) {
		void *dptr = dstm_region_ptr_tovoid(rg, req->dst);
		dstm_log_debug("applying: %p %zu\n", dptr, req->size);
		memcpy(dptr, req->buf, req->size);
		req->status = DSTM_REQUEST_STATUS_DONE;
	}
	return 0;
}

/*******************************************************************************
 * Barrier request DFA
 *
 * The barrier is implemented as a first-class request driven by the standard
 * dstm_do_progress() loop.  A BARRIER request is created by dstm_barrier()
 * and inserted into the global request list; dstm_request_progress_barrier()
 * advances its state machine on every progress iteration.
 *
 * 3-round ring protocol (rank 0 -> 1 -> ... -> N-1 -> 0):
 *
 * Round 1 -- drain outgoing:
 *   Each rank waits until its OUT_PUT/OUT_GET requests have completed, then
 *   forwards the token.  When the token returns to rank 0, all ranks have
 *   finished their outgoing work: no new PUT or GET messages will be issued
 *   by any rank.
 *
 *   BARRIER_DRAIN_OUT
 *     Wait for all OUT_PUT/OUT_GET requests to finish.
 *     -> rank 0:  inject R1 token, go to R1_FORWARDED
 *        rank >0: go to R1_WAIT
 *
 *   BARRIER_R1_WAIT  (non-zero ranks only)
 *     Wait for a token (barrier_token_count > 0).  Consume it, send the
 *     R1 token to the successor, go to R1_FORWARDED.
 *
 *   BARRIER_R1_FORWARDED
 *     Wait for a token.  Consume it.
 *     -> rank 0:  inject R2 token
 *     -> all:     set barrier_all_drained = 1, go to R2_DRAINING
 *
 * Round 2 -- complete all pending:
 *   barrier_all_drained is now set, which bypasses all neighbor distance
 *   checks in version transitions.  Versions rapidly advance through
 *   WRITABLE -> READABLE -> DISCARDABLE, unblocking any pending IN_GET
 *   and LOCAL_GET requests.
 *
 *   BARRIER_R2_DRAINING
 *     Drain all non-barrier requests.  Once the queue is empty, send the
 *     R2 token to the successor, go to R2_FORWARDED.
 *
 *   BARRIER_R2_FORWARDED
 *     Wait for a token.  Consume it.
 *     -> rank 0:  inject exit token (if world_size > 1)
 *     -> rank >0 (< N-1): forward exit token to successor
 *     -> all:     reset barrier_all_drained = 0, go to DONE
 *
 * Helper: dstm_barrier_send_token() -- sends a BARRIER message to the ring
 * successor carrying the barrier's last_time.
 ******************************************************************************/

/* Send the barrier token to the ring successor. */
static void dstm_barrier_send_token(dstm_time_t last_time)
{
	int next =
	        (dstm_global_context.rank + 1) % dstm_global_context.world_size;
	dstm_msg_t m;
	memset(&m, 0, sizeof(m));
	dstm_msg_fill_hdr(&m, DSTM_MSG_TYPE_BARRIER, last_time, 0, next,
	                  dstm_global_context.rank);
	dstm_msg_fill_barrier(&m, last_time);
	int tag = dstm_request_new_tag();
	dstm_msg_send(&m, tag);
}

/* Return 1 if any outgoing request (OUT_PUT or OUT_GET) is not yet done. */
static int dstm_has_pending_outgoing(void)
{
	dstm_request_t *it;
	DL_FOREACH(requests, it)
	{
		if ((it->type == DSTM_REQUEST_TYPE_OUT_PUT ||
		     it->type == DSTM_REQUEST_TYPE_OUT_GET) &&
		    it->status != DSTM_REQUEST_STATUS_DONE)
			return 1;
	}
	return 0;
}

/* Return 1 if any request other than `barrier_req` is still in the list. */
static int dstm_has_non_barrier_requests(dstm_request_t *barrier_req)
{
	dstm_request_t *it;
	DL_FOREACH(requests, it)
	{
		if (it != barrier_req)
			return 1;
	}
	return 0;
}

int dstm_request_progress_barrier(dstm_request_t *req)
{
	int rank = dstm_global_context.rank;
	int world_size = dstm_global_context.world_size;
	dstm_time_t last_time = req->time; /* last_time stored in req->time */
	dstm_region_t *region = dstm_global_context.region;

	switch (req->status) {

	/* ---- Round 1: drain outgoing requests ---- */

	case DSTM_REQUEST_STATUS_BARRIER_DRAIN_OUT:
		/* Wait for all outgoing PUT/GET requests to finish. */
		if (dstm_has_pending_outgoing())
			break;

		if (rank == 0) {
			/* Rank 0 injects the R1 token immediately. */
			dstm_barrier_send_token(last_time);
			dstm_log_info("barrier: rank 0 sent R1 token, "
			              "last_time: %" PRId64 "\n",
			              last_time);
			req->status =
			        DSTM_REQUEST_STATUS_BARRIER_R1_FORWARDED;
		} else {
			req->status = DSTM_REQUEST_STATUS_BARRIER_R1_WAIT;
		}
		break;

	case DSTM_REQUEST_STATUS_BARRIER_R1_WAIT:
		/* Non-zero ranks: wait for the R1 token from predecessor. */
		if (region->barrier_token_count == 0)
			break;

		region->barrier_token_count--;
		dstm_log_info("barrier: rank %d received R1 token\n", rank);

		/* Forward R1 token to successor. */
		dstm_barrier_send_token(last_time);
		dstm_log_info("barrier: rank %d sent R1 token\n", rank);
		req->status = DSTM_REQUEST_STATUS_BARRIER_R1_FORWARDED;
		break;

	case DSTM_REQUEST_STATUS_BARRIER_R1_FORWARDED:
		/* Wait for the R1 ring to close (rank 0) or for the R2
		 * token (non-zero ranks). */
		if (region->barrier_token_count == 0)
			break;

		region->barrier_token_count--;
		dstm_log_info("barrier: rank %d R1 complete\n", rank);

		/* Rank 0: inject R2 token. */
		if (rank == 0) {
			dstm_barrier_send_token(last_time);
			dstm_log_info("barrier: rank 0 sent R2 token\n");
		}

		/* All ranks: enable barrier optimization so version
		 * transitions bypass neighbor distance checks. */
		region->barrier_all_drained = 1;
		req->status = DSTM_REQUEST_STATUS_BARRIER_R2_DRAINING;
		break;

	/* ---- Round 2: complete all pending requests ---- */

	case DSTM_REQUEST_STATUS_BARRIER_R2_DRAINING:
		/* Drain all non-barrier requests. With barrier_all_drained
		 * set, version transitions are no longer blocked by stale
		 * neighbor heartbeat times. */
		if (dstm_has_non_barrier_requests(req))
			break;

		/* Queue is empty: forward the R2 token. */
		dstm_barrier_send_token(last_time);
		dstm_log_info("barrier: rank %d sent R2 token\n", rank);
		req->status = DSTM_REQUEST_STATUS_BARRIER_R2_FORWARDED;
		break;

	case DSTM_REQUEST_STATUS_BARRIER_R2_FORWARDED:
		/* Wait for the R2 ring to close (rank 0) or for the exit
		 * token (non-zero ranks). */
		if (region->barrier_token_count == 0)
			break;

		region->barrier_token_count--;

		if (rank == 0) {
			/* R2 ring complete.  Inject exit token so all other
			 * ranks can leave.  Single-rank case needs no token. */
			dstm_log_info("barrier: rank 0 R2 complete\n");
			if (world_size > 1)
				dstm_barrier_send_token(last_time);
		} else {
			dstm_log_info("barrier: rank %d received exit "
			              "token\n",
			              rank);
			/* Forward exit token to successor (all but last). */
			if (rank < world_size - 1)
				dstm_barrier_send_token(last_time);
		}

		/* Clean up barrier state. */
		region->barrier_all_drained = 0;
		req->status = DSTM_REQUEST_STATUS_DONE;
		break;

	default:
		/* DONE is handled by dstm_do_progress; nothing else expected. */
		break;
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
        {DSTM_REQUEST_TYPE_BARRIER, dstm_request_progress_barrier},
};

int dstm_request_progress(dstm_request_t *req)
{
	assert(req != NULL);
	assert(progress_sm[req->type].progress != NULL);
	dstm_log_request(DSTM_LOG_DEBUG, "request", req);
	dstm_log_region(DSTM_LOG_DEBUG, "region", dstm_get_local_region());
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
	dstm_log_debug("progress loop: reqtime: %" PRId64
	               ", region time: %" PRId64 "\n",
	               time, dstm_global_context.region->time);
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
		dstm_region_set_ninfo(dstm_global_context.region,
		                      m.header.src_rank, m.header.time,
		                      DSTM_REGION_STATUS_WRITABLE);

		dstm_log_msg(DSTM_LOG_DEBUG, "received", &m);
		switch (m.header.type) {
		case DSTM_MSG_TYPE_PUT:
			/* Incoming PUT: copy the payload into a request-owned
			 * buffer so we can free the MPI packbuf immediately.
			 */
			dptr = (dstm_ptr_t){m.u.putget.ptr.region,
			                    m.u.putget.ptr.offset};
			dstm_request_create(&nr, DSTM_REQUEST_TYPE_IN_PUT,
			                    status.MPI_TAG, m.header.time,
			                    m.header.delta, m.header.dst_rank,
			                    dptr, (dstm_ptr_t){0, 0},
			                    m.u.putget.size);
			nr->buf = malloc(m.u.putget.size);
			assert(nr->buf != NULL);
			memcpy(nr->buf, m.data, m.u.putget.size);
			dstm_msg_free_packbuf(&m);
			DL_INSERT_INORDER(requests, nr,
			                  dstm_request_cmp_departure);
			break;
		case DSTM_MSG_TYPE_GET_REQ:
			/* Incoming GET request: store enough info to send the
			 * reply later (when the region is readable).  The
			 * received packbuf is kept in nr->msg until the reply
			 * is built, at which point it is freed.
			 */
			sptr = (dstm_ptr_t){m.u.putget.ptr.region,
			                    m.u.putget.ptr.offset};
			dstm_request_create(&nr, DSTM_REQUEST_TYPE_IN_GET,
			                    status.MPI_TAG, m.header.time,
			                    m.header.delta, m.header.src_rank,
			                    (dstm_ptr_t){0, 0}, sptr,
			                    m.u.putget.size);
			nr->msg = m;
			DL_INSERT_INORDER(requests, nr,
			                  dstm_request_cmp_departure);
			break;
		case DSTM_MSG_TYPE_GET_REP: {
			/* Reply to an outgoing GET: find the matching request,
			 * copy the data into req->buf, and free the packbuf.
			 */
			dstm_request_t *it = NULL;
			DL_FOREACH(requests, it)
			{
				if (it->tag == status.MPI_TAG)
					break;
			}
			assert(it != NULL);
			it->buf = malloc(m.u.putget.size);
			assert(it->buf != NULL);
			memcpy(it->buf, m.data, m.u.putget.size);
			dstm_msg_free_packbuf(&m);
			it->status = DSTM_REQUEST_STATUS_IRECV;
			break;
		}
		case DSTM_MSG_TYPE_HEARTBEAT:
			dstm_region_set_ninfo(dstm_global_context.region,
			                      m.header.src_rank, m.header.time,
			                      m.u.heartbeat.status);
			dstm_msg_free_packbuf(&m);
			break;
		case DSTM_MSG_TYPE_BARRIER:
			/* Increment the token count so the barrier DFA can
			 * detect and consume this token on its next
			 * progress iteration. */
			dstm_global_context.region->barrier_token_count++;
			dstm_log_info("barrier token received from rank %d, "
			              "last_time: %" PRId64 "\n",
			              m.header.src_rank,
			              m.u.barrier.last_time);
			dstm_msg_free_packbuf(&m);
			break;
		default:
			assert(0);
		}
	}

	/* Multi-version progress model:
	 *
	 * Each region maintains multiple timestamped versions.  PUTs are
	 * applied immediately to the target version (creating it on demand
	 * with an uninitialized buffer).  GETs wait for the target version
	 * to become READABLE (all PUTs applied, no more can arrive).
	 *
	 * Each version has its own state machine:
	 *   WRITABLE -> READABLE -> DISCARDABLE
	 *
	 * The WRITABLE -> READABLE transition uses distance-based neighbor
	 * checks to ensure no more PUTs can arrive.  The READABLE ->
	 * DISCARDABLE transition uses the same check for GETs.
	 * DISCARDABLE versions are evicted (buffer freed).
	 *
	 * region->time tracks the minimum active version timestamp and
	 * advances as old versions are evicted.
	 */

	dstm_region_status_update(time);
	/* loop over all requests, see what can be progressed */
	dstm_request_t *it, *tmp;
	DL_FOREACH_SAFE(requests, it, tmp)
	{
		dstm_request_progress(it);
		if (dstm_request_isdone(it)) {
			DL_DELETE(requests, it);
			/* Caller-owned requests (e.g. the one held by
			 * dstm_get) are removed from the list but not freed
			 * here; the caller checks for DONE status via its own
			 * pointer and calls dstm_request_destroy itself. */
			if (!it->caller_owned)
				dstm_request_destroy(&it);
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
	dstm_log_info("put: %d, %d, %" PRId64 ", %" PRId64
	              ", %d:%td, %p, %zu\n",
	              rank, region_id, time, delta, dst.region, dst.offset, src,
	              size);

	assert(rank == region_id);
	assert(rank == dst.region);

	if (src == NULL)
		return -DSTM_EINVAL;

	/* no idea what the semantic of a put in the past would be for now */
	if (delta < 0)
		return -DSTM_EINVAL;


	/* did we increase our local time */
	if (dstm_local_heartbeat_time < time) {
		dstm_region_send_heartbeat_to_all(time,
						  DSTM_REGION_STATUS_WRITABLE);
		dstm_local_heartbeat_time = time;
	}

	/* request tracking part:
	 * making sure the request is being sent to the right place
	 */

	/* first, figure out if this is a local request or a remote one */
	dstm_request_t *req;
	if (rank == dstm_global_context.rank) {
		/* Local put: buffer the payload in req->buf so the caller's
		 * buffer can be reused or freed immediately.
		 */
		dstm_request_create(&req, DSTM_REQUEST_TYPE_LOCAL_PUT, -1, time,
		                    delta, rank, dst, (dstm_ptr_t){0, 0}, size);
		req->buf = malloc(size);
		assert(req->buf != NULL);
		memcpy(req->buf, src, size);
	} else {
		/* Outgoing put: buffer the payload so the caller's buffer can
		 * be reused or freed immediately, then kick off the async send.
		 */
		int tag = dstm_request_new_tag();
		dstm_request_create(&req, DSTM_REQUEST_TYPE_OUT_PUT, tag, time,
		                    delta, rank, dst, (dstm_ptr_t){0, 0}, size);
		req->buf = malloc(size);
		assert(req->buf != NULL);
		memcpy(req->buf, src, size);

		dstm_msg_fill_hdr(&req->msg, DSTM_MSG_TYPE_PUT, time, delta,
		                  rank, dstm_global_context.rank);
		dstm_msg_fill_payload(&req->msg, dst, req->buf, size);
		dstm_msg_isend(&req->msg, tag);
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
	dstm_log_info("get: %d, %d, %" PRId64 ", %" PRId64
	              ", %p, %d:%td, %zu\n",
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

	/* did we increase our local time */
	if (dstm_local_heartbeat_time < time) {
		dstm_region_send_heartbeat_to_all(time,
						  DSTM_REGION_STATUS_WRITABLE);
		dstm_local_heartbeat_time = time;
	}

	/* request tracking part:
	 * making sure the request is being sent to the right place
	 */

	/* first, figure out if this is a local request or a remote one */
	dstm_request_t *req;
	if (rank == dstm_global_context.rank) {
		/* local get:
		 * we just enqueue it locally
		 * */
		dptr = (dstm_ptr_t){DSTM_REGIONID_LOCAL, (intptr_t)dst};
		dstm_request_create(&req, DSTM_REQUEST_TYPE_LOCAL_GET, -1, time,
		                    delta, rank, dptr, src, size);
		req->caller_owned = 1;
	} else {
		/* outgoing get
		 * we enqueue it for tracking purposes, and push it out to the
		 * appropriate rank.
		 */
		dptr = (dstm_ptr_t){DSTM_REGIONID_LOCAL, (intptr_t)dst};
		int tag = dstm_request_new_tag();
		dstm_request_create(&req, DSTM_REQUEST_TYPE_OUT_GET, tag, time,
		                    delta, rank, dptr, src, size);
		req->caller_owned = 1;

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

		dstm_msg_fill_hdr(&req->msg, DSTM_MSG_TYPE_GET_REQ, time, delta,
		                  rank, dstm_global_context.rank);
		dstm_msg_fill_payload(&req->msg, src, NULL, size);
		dstm_msg_send(&req->msg, tag);
		req->status = DSTM_REQUEST_STATUS_ISEND;
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
 * Barrier: distributed termination via a three-round ring token algorithm.
 *
 * dstm_barrier() creates a BARRIER request and inserts it into the global
 * request list.  The barrier DFA (dstm_request_progress_barrier()) is then
 * driven by the standard dstm_do_progress() loop until the request reaches
 * DONE status, at which point dstm_barrier() returns.
 *
 * See the comment above dstm_request_progress_barrier() for the full protocol.
 ******************************************************************************/

int dstm_barrier(dstm_time_t last_time)
{
	/* Create a caller-owned BARRIER request.  We store last_time in
	 * req->time (delta and rank are unused for barrier requests). */
	dstm_request_t *req;
	dstm_ptr_t null_ptr = {DSTM_REGIONID_INVALID, 0};
	dstm_request_create(&req, DSTM_REQUEST_TYPE_BARRIER, 0, last_time, 0,
	                    0, null_ptr, null_ptr, 0);
	assert(req != NULL);
	req->status = DSTM_REQUEST_STATUS_BARRIER_DRAIN_OUT;
	req->caller_owned = 1;

	/* Insert at the end of the list so existing requests are processed
	 * first (the barrier DFA itself handles ordering internally). */
	DL_APPEND(requests, req);

	/* Spin the unified progress loop until the barrier is done. */
	while (!dstm_request_isdone(req))
		dstm_do_progress(last_time);

	dstm_request_destroy(&req);
	return 0;
}

/*******************************************************************************
 * Formatting helpers
 ******************************************************************************/

typedef struct dstm_request_type_table_s {
	int type;
	const char *s;
} dstm_request_type_table_t;

static const dstm_request_type_table_t dstm_request_type_table[] = {
        {DSTM_REQUEST_TYPE_LOCAL_PUT, "LOCAL_PUT"},
        {DSTM_REQUEST_TYPE_LOCAL_GET, "LOCAL_GET"},
        {DSTM_REQUEST_TYPE_IN_PUT, "IN_PUT"},
        {DSTM_REQUEST_TYPE_IN_GET, "IN_GET"},
        {DSTM_REQUEST_TYPE_OUT_PUT, "OUT_PUT"},
        {DSTM_REQUEST_TYPE_OUT_GET, "OUT_GET"},
        {DSTM_REQUEST_TYPE_BARRIER, "BARRIER"},
        {0, NULL},
};

static const char *dstm_request_type_t2s(int type,
                                         const dstm_request_type_table_t *table)
{
	if (type < 0)
		return "UNKNOWN";
	for (int i = 0; table[i].s != NULL; i++)
		if (table[i].type == type)
			return table[i].s;
	return "UNKNOWN";
}

int dstm_request_snprintf(char *buf, size_t n, dstm_request_t *req)
{
	assert(buf != NULL);
	if (req == NULL)
		return snprintf(buf, n, "NULL");

	/* Header line */
	int total = snprintf(buf, n,
	                     "{ type: %s, tag: %d, status: %d, time: %" PRIu64
	                     ", delta: %" PRId64 ", rank: %d, size: %zu, dst: ",
	                     dstm_request_type_t2s(req->type,
	                                           dstm_request_type_table),
	                     req->tag, req->status, req->time, req->delta,
	                     req->rank, req->size);

#define ADVANCE(written)                                                       \
	do {                                                                   \
		int _w = (written);                                            \
		buf += (_w < (int)n ? _w : (int)n - 1);                        \
		n = (_w < (int)n ? n - (size_t)_w : 1);                        \
		total += _w;                                                   \
	} while (0)

	ADVANCE(total);

	/* dst pointer */
	int w = dstm_ptr_snprintf(buf, n, req->dst);
	ADVANCE(w);

	w = snprintf(buf, n, ", src: ");
	ADVANCE(w);

	/* src pointer */
	w = dstm_ptr_snprintf(buf, n, req->src);
	ADVANCE(w);

	w = snprintf(buf, n, ", msg: ");
	ADVANCE(w);

	/* embedded message */
	w = dstm_msg_snprintf(buf, n, &req->msg);
	ADVANCE(w);

	w = snprintf(buf, n, "}");
	total += w;

#undef ADVANCE

	return total;
}

