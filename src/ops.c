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

static int dstm_region_send_heartbeat_to_all(dstm_time_t time)
{
	for (int i = 0; i < dstm_global_context.world_size; i++) {
		int dist = dstm_distance_get(dstm_global_context.distances,
		                             dstm_global_context.rank, i);
		if (dist <= 0)
			continue;

		dstm_msg_t m;
		dstm_msg_fill_hdr(&m, DSTM_MSG_TYPE_HEARTBEAT, time, 0, i,
		                  dstm_global_context.rank, 0);

		int tag = dstm_request_new_tag();
		dstm_msg_send(&m, tag);
	}
	return 0;
}

/* Check whether a WRITABLE version can transition to READABLE.
 * Returns 1 if the transition should happen, 0 otherwise. */
static int dstm_version_can_become_readable(dstm_region_t *region, int region_id,
                                            dstm_region_version_t *v)
{	
    /* Check for pending PUT requests targeting this version's time. */
	dstm_request_t *it;
	DL_FOREACH(requests, it)
	{
		dstm_time_t target = it->time + it->delta;
		if (target != v->time)
			continue;

        if (it->region_id != region_id) continue; /* other region: unrelated */

		if (it->type == DSTM_REQUEST_TYPE_LOCAL_PUT &&
		    it->status != DSTM_REQUEST_STATUS_DONE)
			return 0;

		if (it->type == DSTM_REQUEST_TYPE_IN_PUT &&
		    it->status != DSTM_REQUEST_STATUS_DONE)
			return 0;

        /* Appends are writers too: a still-pending append means
		 * more bytes can still land in this version. */
		if (it->type == DSTM_REQUEST_TYPE_LOCAL_APPEND &&
		    it->status != DSTM_REQUEST_STATUS_DONE)
			return 0;

		if (it->type == DSTM_REQUEST_TYPE_IN_APPEND &&
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
	if (!dstm_global_context.barrier_all_drained) {
		for (int i = 0; i < dstm_global_context.world_size; i++) {
			int dist =
			        dstm_distance_get(dstm_global_context.distances,
			                          dstm_global_context.rank, i);
			if (dist <= 0)
				continue;

			dstm_time_t nt;
			dstm_ninfo_get(i, region_id, &nt);

			dstm_log_debug("checkr r%d v%" PRId64 ": nt=%" PRId64
			               " dist=%d\n",
			               region_id, v->time, nt, dist);
			if (nt + dist <= v->time)
				return 0;
		}

        /* The local rank can still issue local puts (delta >= 0)
         * targeting any effective time >= its own current logical
         * time.  Until our own clock has moved strictly past v->time,
         * a local put at v->time is still legal and the version must
         * remain WRITABLE.
         *
         * Also gated by barrier_all_drained: after barrier round 1,
         * no rank (including this one) will issue more puts.
         */
        if (dstm_global_context.local_heartbeat_time < v->time)
            return 0;
	}

	return 1;
}

/* Check whether a READABLE version can transition to DISCARDABLE.
 * Returns 1 if the transition should happen, 0 otherwise. */
static int dstm_version_can_become_discardable(dstm_region_t *region, int region_id,
                                               dstm_region_version_t *v)
{
	/* Check for pending GET requests targeting this version's time. */
	dstm_request_t *it;
	DL_FOREACH(requests, it)
	{
		dstm_time_t target = it->time + it->delta;
		if (target != v->time)
			continue;

        if (it->region_id != region_id) continue; /* other region: unrelated */

		if (it->type == DSTM_REQUEST_TYPE_LOCAL_GET &&
		    it->status != DSTM_REQUEST_STATUS_DONE)
			return 0;

		if (it->type == DSTM_REQUEST_TYPE_IN_GET &&
		    it->status != DSTM_REQUEST_STATUS_DONE)
			return 0;
	}

	/* Could we receive more GETs from neighbors?
	 * A neighbor at distance dist whose last-known time is nt can still
	 * send a GET at time nt with max dist, so targeting nt - dist.
	 * Thus, v->time greater (or equal) than nt - dist cannot be discarded.
	 *
	 * Barrier optimization: after barrier round 1 completes, no new
	 * GET messages will arrive, so skip all neighbor distance checks. */
	if (!dstm_global_context.barrier_all_drained) {
		for (int i = 0; i < dstm_global_context.world_size; i++) {
			int dist =
			        dstm_distance_get_discard(dstm_global_context.distances,
                                      dstm_global_context.rank, i);
			if (dist <= 0)
				continue;

			dstm_time_t nt;
			dstm_ninfo_get(i, region_id, &nt);

			dstm_log_debug("checkd r%d v%" PRId64 ": nt=%" PRId64
			               " dist=%d\n",
			               region_id, v->time, nt, dist);
			if (nt - dist <= v->time)
				return 0;
		}

        if (dstm_global_context.local_heartbeat_time < v->time)
            return 0;
	}

	return 1;
}

static int dstm_region_status_update(dstm_time_t time)
{
    for (int rid = 0; rid < dstm_global_context.region_count; rid++) {
        dstm_region_t *region = dstm_global_context.regions[rid];
        if (region == NULL)
            continue; 

        dstm_region_version_t *v, *vtmp;

        DL_FOREACH_SAFE(region->versions, v, vtmp)
        {
            switch (v->status) {
            case DSTM_REGION_STATUS_WRITABLE:
                if (dstm_version_can_become_readable(region, rid, v)) {
                    v->status = DSTM_REGION_STATUS_READABLE;
                    dstm_log_info("version readable, time: %" PRId64
                                  "\n",
                                  v->time);
                }
                break;

            case DSTM_REGION_STATUS_READABLE:
                if (dstm_version_can_become_discardable(region, rid, v)) {
                    v->status = DSTM_REGION_STATUS_DISCARDABLE;
                    dstm_log_info(
                            "version discardable, time: %" PRId64
                            "\n",
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
    }
	return 0;
}

/*******************************************************************************
 * Request progress functions
 ******************************************************************************/

static int dstm_request_progress_local_put(dstm_request_t *req)
{
	/* logic of a local put:
	 * - starts in INIT state, with payload buffered in req->buf.
	 *
	 * Applied immediately by finding or creating the target version.
	 *
	 * Put-before-get phase rule enforcement: the target version MUST
	 * still be WRITABLE when the put applies.  If a get on the same
	 * effective time has already driven the version to READABLE (or
	 * beyond), this put is a contract violation: the caller issued a
	 * put after a get on the same (rank, region, effective time).
	 * We abort the program loudly rather than silently overwriting a
	 * buffer that has already been observed.
	 */
	dstm_region_t *rg = dstm_get_local_region(req->region_id);
    assert(rg != NULL);
	assert(req->status == DSTM_REQUEST_STATUS_INIT);
	dstm_time_t target = req->time + req->delta;
	dstm_region_version_t *v =
	        dstm_region_version_find_or_create(rg, target);
	assert(v != NULL);
	if (v->status != DSTM_REGION_STATUS_WRITABLE) {
		dstm_log_error(
		        "dstm_put contract violation: rank %d, effective "
		        "time %" PRId64 ": target version is no longer "
		        "WRITABLE (status=%d).  All puts to a given "
		        "effective time must be issued before any get on "
		        "that time.\n",
		        dstm_global_context.rank, target, v->status);
		abort();
	}
	void *dptr = (char *)v->buffer + req->dst.offset;
	memcpy(dptr, req->buf, req->size);
	req->status = DSTM_REQUEST_STATUS_DONE;
	return 0;
}

static int dstm_request_progress_local_get(dstm_request_t *req)
{
	/* logic of a local get:
	 * - starts in INIT state, enqueued as a request to copy data from a
	 *   version's buffer into the caller's buffer (req->dst,
	 *   REGIONID_LOCAL).
	 *
	 * - completes when the target version exists and is READABLE.
	 */
	dstm_region_t *rg = dstm_get_local_region(req->region_id);
    assert(rg != NULL);
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

static int dstm_request_progress_in_put(dstm_request_t *req)
{
	/* logic:
	 * - init: payload already buffered in req->buf (copied on receipt).
	 * - done: payload applied immediately to the target version.
	 *
	 * Put-before-get phase rule enforcement: same as
	 * dstm_request_progress_local_put.  An incoming PUT from a
	 * remote rank that targets an effective time we have already
	 * promoted to READABLE is a global contract violation -- the
	 * sender issued a put after a get had been observed against
	 * this rank's region for that time.  Abort loudly.
	 */
	dstm_region_t *rg = dstm_get_local_region(req->region_id);
    if (rg == NULL) {
		dstm_log_error("incoming request for unknown local region %d "
		               "on rank %d\n",
		               req->region_id, dstm_global_context.rank);
		abort();
	}
	assert(req->status == DSTM_REQUEST_STATUS_INIT);
	dstm_time_t target = req->time + req->delta;
	dstm_region_version_t *v =
	        dstm_region_version_find_or_create(rg, target);
	assert(v != NULL);
	if (v->status != DSTM_REGION_STATUS_WRITABLE) {
		dstm_log_error(
		        "incoming put contract violation on rank %d at "
		        "effective time %" PRId64 ": target version is no "
		        "longer WRITABLE (status=%d).  All puts to a given "
		        "effective time must be issued before any get on "
		        "that time.\n",
		        dstm_global_context.rank, target, v->status);
		abort();
	}
	void *dptr = (char *)v->buffer + req->dst.offset;
	memcpy(dptr, req->buf, req->size);
	req->status = DSTM_REQUEST_STATUS_DONE;
	return 0;
}

/* Shared accumulate helper: append req->buf (req->size bytes) into the
 * target version's slot at req->dst.offset, which is laid out as
 * [ uint64_t byte_count | accumulated bytes ].  Creates the version on
 * demand, exactly like IN_PUT, and enforces the same WRITABLE phase rule.
 *
 * The region buffer is calloc'd per version, so an as-yet-untouched slot
 * reads byte_count == 0 and we start accumulating at offset 0. */
static void dstm_region_append_apply(dstm_region_t *rg, dstm_request_t *req,
                                     dstm_time_t target)
{
	dstm_region_version_t *v =
	        dstm_region_version_find_or_create(rg, target);
	assert(v != NULL);
	if (v->status != DSTM_REGION_STATUS_WRITABLE) {
		dstm_log_error(
		        "append contract violation on rank %d at effective "
		        "time %" PRId64 ": target version is no longer "
		        "WRITABLE (status=%d).  All appends/puts to a given "
		        "effective time must be issued before any get on "
		        "that time.\n",
		        dstm_global_context.rank, target, v->status);
		abort();
	}

	/* Slot base within the version buffer. */
	char    *slot  = (char *)v->buffer + req->dst.offset;
	uint64_t count;
	memcpy(&count, slot, sizeof(uint64_t));

	/* Overflow guard: the accumulated region must stay inside the
	 * region buffer.  The data-structure layer (dqueue) bound-checks
	 * before issuing, but a corrupt or oversized producer must not be
	 * allowed to scribble past the region. */
	size_t head    = (size_t)req->dst.offset + sizeof(uint64_t);
	size_t written = (size_t)count;
	if (head + written + req->size > rg->size) {
		dstm_log_error(
		        "append overflow on rank %d: slot at offset %td would "
		        "exceed region size %zu (count=%" PRIu64
		        ", add=%zu)\n",
		        dstm_global_context.rank, req->dst.offset, rg->size,
		        count, req->size);
		abort();
	}

	memcpy(slot + sizeof(uint64_t) + written, req->buf, req->size);
	count += (uint64_t)req->size;
	memcpy(slot, &count, sizeof(uint64_t));
}

static int dstm_request_progress_local_append(dstm_request_t *req)
{
	dstm_region_t *rg = dstm_get_local_region(req->region_id);
	assert(rg != NULL);
	assert(req->status == DSTM_REQUEST_STATUS_INIT);
	dstm_time_t target = req->time + req->delta;
	dstm_region_append_apply(rg, req, target);
	req->status = DSTM_REQUEST_STATUS_DONE;
	return 0;
}

static int dstm_request_progress_in_append(dstm_request_t *req)
{
	dstm_region_t *rg = dstm_get_local_region(req->region_id);
	if (rg == NULL) {
		dstm_log_error("incoming append for unknown local region %d "
		               "on rank %d\n",
		               req->region_id, dstm_global_context.rank);
		abort();
	}
	assert(req->status == DSTM_REQUEST_STATUS_INIT);
	dstm_time_t target = req->time + req->delta;
	dstm_region_append_apply(rg, req, target);
	req->status = DSTM_REQUEST_STATUS_DONE;
	return 0;
}

/* OUT_APPEND behaves exactly like OUT_PUT: an async MPI send that
 * completes when MPI_Test reports done. */
static int dstm_request_progress_out_append(dstm_request_t *req)
{
	assert(req->status == DSTM_REQUEST_STATUS_ISEND);
	if (dstm_msg_test(req->msg))
		req->status = DSTM_REQUEST_STATUS_DONE;
	return 0;
}

static int dstm_request_progress_in_get(dstm_request_t *req)
{
	/* logic:
	 * - init: GET_REQ received, waiting for the version to become
	 *   readable.
	 * - isend: reply buffered in req->buf and MPI_Isend posted.
	 * - done: send completed.
	 */
	dstm_region_t *rg = dstm_get_local_region(req->region_id);
    assert(rg != NULL);
    if (req->status == DSTM_REQUEST_STATUS_INIT) {
        dstm_time_t target = req->time + req->delta;
        dstm_region_version_t *v = dstm_region_version_find(rg, target);

        /* Distinguish "not created yet" from "already evicted".
         *
         * A GET for `target` can only ever be satisfied if the version
         * for `target` still exists or has yet to be created.  If it is
         * missing AND a strictly-newer version is already live, then the
         * target slot was passed over and reclaimed: no future PUT can
         * recreate an older version (puts only move the clock forward),
         * so this GET can never complete.  That is the genuine
         * premature-eviction bug; fail loudly rather than deadlock.
         *
         * Merely having target < rg->time is NOT sufficient: an empty
         * version list advances rg->time to the caller's clock, which
         * would false-positive on legitimate early/out-of-order GETs. */
        if (v == NULL && rg->versions != NULL &&
            rg->versions->time > target) {
                dstm_log_error(
                        "incoming GET for region %d target time %"
                        PRId64 " on rank %d: version already evicted "
                        "(oldest live version=%" PRId64 ").  Premature "
                        "eviction / distance-check bug.\n",
                        req->region_id, target,
                        dstm_global_context.rank, rg->versions->time);
                abort();
        }

        if (v != NULL && v->status == DSTM_REGION_STATUS_READABLE) {
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
			                  target, 0, req->rank,
			                  dstm_global_context.rank, req->region_id);
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

static int dstm_request_progress_out_put(dstm_request_t *req)
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

static int dstm_request_progress_out_get(dstm_request_t *req)
{
	/* logic:
	 * - irecv: GET_REQ sent, reply received and buffered in req->buf.
	 * - done: payload copied to caller's destination.
	 */

    /* req->region_id is the remote source region id. 
     *   the source region id, may not exist localy when region ids
     *   are asymmetric. */
	dstm_region_t *region = 
                    dstm_get_local_region(dstm_get_local_region_id());

	if (req->status == DSTM_REQUEST_STATUS_IRECV) {
		assert(req->dst.region == DSTM_REGIONID_LOCAL);
		void *dptr = dstm_region_ptr_tovoid(region, req->dst);
		assert(dptr != NULL);
		dstm_log_debug("applying: %p %zu\n", dptr, req->size);
		memcpy(dptr, req->buf, req->size);
		req->status = DSTM_REQUEST_STATUS_DONE;
	}
	return 0;
}

/*******************************************************************************
 * Progress dispatch
 ******************************************************************************/

typedef struct dstm_request_progress_s {
	int type;
	int (*progress)(dstm_request_t *req);
} dstm_request_progress_t;

static dstm_request_progress_t progress_sm[DSTM_REQUEST_TYPE_MAX] = {
        {DSTM_REQUEST_TYPE_LOCAL_PUT, dstm_request_progress_local_put},
        {DSTM_REQUEST_TYPE_LOCAL_GET, dstm_request_progress_local_get},
        {DSTM_REQUEST_TYPE_LOCAL_APPEND, dstm_request_progress_local_append},
        {DSTM_REQUEST_TYPE_IN_PUT, dstm_request_progress_in_put},
        {DSTM_REQUEST_TYPE_IN_GET, dstm_request_progress_in_get},
        {DSTM_REQUEST_TYPE_IN_APPEND, dstm_request_progress_in_append},
        {DSTM_REQUEST_TYPE_OUT_PUT, dstm_request_progress_out_put},
        {DSTM_REQUEST_TYPE_OUT_GET, dstm_request_progress_out_get},
        {DSTM_REQUEST_TYPE_OUT_APPEND, dstm_request_progress_out_append},
        {DSTM_REQUEST_TYPE_BARRIER, dstm_request_progress_barrier},
};

int dstm_request_progress(dstm_request_t *req)
{
	assert(req != NULL);
	assert(progress_sm[req->type].progress != NULL);
	dstm_log_request(DSTM_LOG_DEBUG, "request", req);
	dstm_log_region(DSTM_LOG_DEBUG, "region", dstm_get_local_region(req->region_id));
	return progress_sm[req->type].progress(req);
}

/*******************************************************************************
 * Progress engine
 ******************************************************************************/

/* Handle a single incoming MPI message: dispatch by type and either create
 * a new request or update an existing one. */
static void dstm_handle_incoming_msg(MPI_Status status)
{
	dstm_request_t *nr;
	dstm_ptr_t dptr, sptr;
	dstm_msg_t m;

	dstm_log_debug("incoming message: %d,%d\n", status.MPI_SOURCE,
	               status.MPI_TAG);

	/* block until we receive the incoming message */
	dstm_msg_recvany(status, &m);

	/* Update neighbor clocks.  PUT/GET/GET_REP pertain to a specific
     * region and must only advance that region's neighbor time, so
     * that traffic on one region cannot retire versions on another.
     * HEARTBEAT announces a whole-rank clock advance and therefore
     * fans out to every registered region.  BARRIER carries no region
     * and does not participate in the distance-based checks. */
    switch (m.header.type) {
    case DSTM_MSG_TYPE_PUT:
    case DSTM_MSG_TYPE_APPEND:
    case DSTM_MSG_TYPE_GET_REQ:
    case DSTM_MSG_TYPE_GET_REP:
            dstm_ninfo_set(m.header.src_rank, m.header.region_id,
                           m.header.time);
            break;
    case DSTM_MSG_TYPE_HEARTBEAT:
            for (int rid = 0; rid < dstm_global_context.region_count; rid++) {
                    if (dstm_global_context.regions[rid] != NULL)
                            dstm_ninfo_set(m.header.src_rank, rid,
                                           m.header.time);
            }
            break;
    default:
            break;
    }

    dstm_log_msg(DSTM_LOG_DEBUG, "received", &m);

	switch (m.header.type) {
	case DSTM_MSG_TYPE_PUT:
		/* Incoming PUT: copy the payload into a request-owned
		 * buffer so we can free the MPI packbuf immediately. */
		dptr = (dstm_ptr_t){m.u.putget.ptr.region,
		                    m.u.putget.ptr.offset};
		dstm_request_create(&nr, DSTM_REQUEST_TYPE_IN_PUT,
		                    status.MPI_TAG, m.header.time,
		                    m.header.delta, m.header.dst_rank, 
                            m.header.region_id, dptr,
		                    (dstm_ptr_t){0, 0}, m.u.putget.size);
		nr->buf = malloc(m.u.putget.size);
		assert(nr->buf != NULL);
		memcpy(nr->buf, m.data, m.u.putget.size);
		dstm_msg_free_packbuf(&m);
		DL_INSERT_INORDER(requests, nr, dstm_request_cmp_departure);
		break;

	case DSTM_MSG_TYPE_APPEND:
		/* Incoming APPEND: identical plumbing to PUT, but the
		 * request accumulates into the slot instead of
		 * overwriting it.  Concatenation order is the order these
		 * IN_APPEND requests are applied, i.e. MPI arrival order. */
		dptr = (dstm_ptr_t){m.u.putget.ptr.region,
		                    m.u.putget.ptr.offset};
		dstm_request_create(&nr, DSTM_REQUEST_TYPE_IN_APPEND,
		                    status.MPI_TAG, m.header.time,
		                    m.header.delta, m.header.dst_rank,
		                    m.header.region_id, dptr,
		                    (dstm_ptr_t){0, 0}, m.u.putget.size);
		nr->buf = malloc(m.u.putget.size);
		assert(nr->buf != NULL);
		memcpy(nr->buf, m.data, m.u.putget.size);
		dstm_msg_free_packbuf(&m);
		DL_INSERT_INORDER(requests, nr, dstm_request_cmp_departure);
		break;

	case DSTM_MSG_TYPE_GET_REQ:
		/* Incoming GET request: store enough info to send the
		 * reply later (when the region is readable).  The
		 * received packbuf is kept in nr->msg until the reply
		 * is built, at which point it is freed. */
		sptr = (dstm_ptr_t){m.u.putget.ptr.region,
		                    m.u.putget.ptr.offset};
		dstm_request_create(&nr, DSTM_REQUEST_TYPE_IN_GET,
		                    status.MPI_TAG, m.header.time,
		                    m.header.delta, m.header.src_rank, m.header.region_id,
		                    (dstm_ptr_t){0, 0}, sptr, m.u.putget.size);
		nr->msg = m;
		DL_INSERT_INORDER(requests, nr, dstm_request_cmp_departure);
		break;

	case DSTM_MSG_TYPE_GET_REP: {
		/* Reply to an outgoing GET: find the matching request,
		 * copy the data into req->buf, and free the packbuf. */
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
		/* ninfo already updated above from the header */
		dstm_msg_free_packbuf(&m);
		break;

	case DSTM_MSG_TYPE_BARRIER:
		/* Increment the token count so the barrier DFA can
		 * detect and consume this token on its next
		 * progress iteration. */
		dstm_global_context.barrier_token_count++;
		dstm_log_info("barrier token received from rank %d, "
		              "last_time: %" PRId64 "\n",
		              m.header.src_rank, m.u.barrier.last_time);
		dstm_msg_free_packbuf(&m);
		break;

	default:
		assert(0);
	}
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
	dstm_log_debug("progress loop: reqtime: %" PRId64 "\n", time);

	/* test if we have any message to deal with */
	MPI_Status status;
	int ret = dstm_msg_iprobe(&status);
	if (ret)
		dstm_handle_incoming_msg(status);

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

    if (region_id < 0 || region_id >= DSTM_MAX_REGIONS)
        return -DSTM_EINVAL;

	if (src == NULL)
		return -DSTM_EINVAL;

    if (region_id != dst.region)
        return -DSTM_EINVAL;

	/* no idea what the semantic of a put in the past would be for now */
	if (delta < 0)
		return -DSTM_EINVAL;

	/* delta == 0 means "right now"; only the local rank can
	 * meaningfully observe its own current state, so a remote put
	 * with delta == 0 has no well-defined ordering semantics. */
	if (delta == 0 && rank != dstm_global_context.rank)
		return -DSTM_EDOM;

	/* did we increase our local time */
    /* rank-level heartbeat/clock (context, not region) */
	if (dstm_global_context.local_heartbeat_time < time) {
		dstm_region_send_heartbeat_to_all(time);
		dstm_global_context.local_heartbeat_time = time;
	}

	/* first, figure out if this is a local request or a remote one */
	dstm_request_t *req;
	if (rank == dstm_global_context.rank) {
		/* Local put: buffer the payload in req->buf so the caller's
		 * buffer can be reused or freed immediately.
		 */
		dstm_request_create(&req, DSTM_REQUEST_TYPE_LOCAL_PUT, -1, time,
		                    delta, rank, region_id, dst, (dstm_ptr_t){0, 0}, size);
		req->buf = malloc(size);
		assert(req->buf != NULL);
		memcpy(req->buf, src, size);
	} else {
		/* Outgoing put: buffer the payload so the caller's buffer can
		 * be reused or freed immediately, then kick off the async send.
		 */
		int tag = dstm_request_new_tag();
		dstm_request_create(&req, DSTM_REQUEST_TYPE_OUT_PUT, tag, time,
		                    delta, rank, region_id, dst, (dstm_ptr_t){0, 0}, size);
		req->buf = malloc(size);
		assert(req->buf != NULL);
		memcpy(req->buf, src, size);

		dstm_msg_fill_hdr(&req->msg, DSTM_MSG_TYPE_PUT, time, delta,
		                  rank, dstm_global_context.rank, region_id);
		dstm_msg_fill_payload(&req->msg, dst, req->buf, size);
		dstm_msg_isend(&req->msg, tag);
		req->status = DSTM_REQUEST_STATUS_ISEND;
	}
	DL_INSERT_INORDER(requests, req, dstm_request_cmp_departure);
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

    if (region_id < 0 || region_id >= DSTM_MAX_REGIONS)
        return -DSTM_EINVAL;

	if (dst == NULL)
		return -DSTM_EINVAL;

    if (region_id != src.region)
        return -DSTM_EINVAL;

	/* no idea what the semantic of a get in the future would be for now */
	if (delta > 0)
		return -DSTM_EINVAL;

	/* delta == 0 means "right now"; only the local rank can
	 * meaningfully observe its own current state, so a remote get
	 * with delta == 0 has no well-defined ordering semantics. */
	if (delta == 0 && rank != dstm_global_context.rank)
		return -DSTM_EDOM;

	/* did we increase our local time */
    /* rank-level heartbeat/clock (context, not region) */
	if (dstm_global_context.local_heartbeat_time < time) {
		dstm_region_send_heartbeat_to_all(time);
		dstm_global_context.local_heartbeat_time = time;
	}

	/* first, figure out if this is a local request or a remote one */
	dstm_request_t *req;
	if (rank == dstm_global_context.rank) {
		/* local get: we just enqueue it locally */
		dptr = (dstm_ptr_t){DSTM_REGIONID_LOCAL, (intptr_t)dst};
		dstm_request_create(&req, DSTM_REQUEST_TYPE_LOCAL_GET, -1, time,
		                    delta, rank, region_id, dptr, src, size);
		req->caller_owned = 1;
	} else {
		/* outgoing get: enqueue for tracking, push to the
		 * appropriate rank. */
		dptr = (dstm_ptr_t){DSTM_REGIONID_LOCAL, (intptr_t)dst};
		int tag = dstm_request_new_tag();
		dstm_request_create(&req, DSTM_REQUEST_TYPE_OUT_GET, tag, time,
		                    delta, rank, region_id, dptr, src, size);
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
		                  rank, dstm_global_context.rank, region_id);
		dstm_msg_fill_payload(&req->msg, src, NULL, size);
		dstm_msg_send(&req->msg, tag);
		req->status = DSTM_REQUEST_STATUS_ISEND;
	}
	DL_INSERT_INORDER(requests, req, dstm_request_cmp_departure);

	/* synchronization: this is a blocking call, so we drive progress
	 * until the request completes. */
	while (!dstm_request_isdone(req))
		dstm_do_progress(time);

	dstm_request_destroy(&req);
	return 0;
}

int dstm_get_or_default(int rank,
                        int region_id,
                        dstm_time_t time,
                        dstm_time_t delta,
                        void *dst,
                        dstm_ptr_t src,
                        size_t size,
                        const void *default_value)
{
	if (default_value == NULL || dst == NULL || size == 0)
		return -DSTM_EINVAL;

	if (time + delta < 0) {
		memcpy(dst, default_value, size);
		return DSTM_SUCCESS;
	}

	return dstm_get(rank, region_id, time, delta, dst, src, size);
}

int dstm_append(int rank,
                int region_id,
                dstm_time_t time,
                dstm_time_t delta,
                dstm_ptr_t dst,
                void *src,
                size_t size)
{
	dstm_log_info("append: %d, %d, %" PRId64 ", %" PRId64
	              ", %d:%td, %p, %zu\n",
	              rank, region_id, time, delta, dst.region, dst.offset, src,
	              size);

	if (region_id < 0 || region_id >= DSTM_MAX_REGIONS)
		return -DSTM_EINVAL;

	if (src == NULL)
		return -DSTM_EINVAL;

	if (region_id != dst.region)
		return -DSTM_EINVAL;

	/* An append, like a put, cannot target the past. */
	if (delta < 0)
		return -DSTM_EINVAL;

	/* delta == 0 is only meaningful for the local rank. */
	if (delta == 0 && rank != dstm_global_context.rank)
		return -DSTM_EDOM;

	/* Same clock-advance heartbeat as put/get. */
	if (dstm_global_context.local_heartbeat_time < time) {
		dstm_region_send_heartbeat_to_all(time);
		dstm_global_context.local_heartbeat_time = time;
	}

	dstm_request_t *req;
	if (rank == dstm_global_context.rank) {
		/* Local append: buffer the payload so the caller's buffer
		 * can be reused or freed immediately. */
		dstm_request_create(&req, DSTM_REQUEST_TYPE_LOCAL_APPEND, -1,
		                    time, delta, rank, region_id, dst,
		                    (dstm_ptr_t){0, 0}, size);
		req->buf = malloc(size);
		assert(req->buf != NULL);
		memcpy(req->buf, src, size);
	} else {
		/* Outgoing append: async send, APPEND message type. */
		int tag = dstm_request_new_tag();
		dstm_request_create(&req, DSTM_REQUEST_TYPE_OUT_APPEND, tag,
		                    time, delta, rank, region_id, dst,
		                    (dstm_ptr_t){0, 0}, size);
		req->buf = malloc(size);
		assert(req->buf != NULL);
		memcpy(req->buf, src, size);

		dstm_msg_fill_hdr(&req->msg, DSTM_MSG_TYPE_APPEND, time, delta,
		                  rank, dstm_global_context.rank, region_id);
		dstm_msg_fill_payload(&req->msg, dst, req->buf, size);
		dstm_msg_isend(&req->msg, tag);
		req->status = DSTM_REQUEST_STATUS_ISEND;
	}
	DL_INSERT_INORDER(requests, req, dstm_request_cmp_departure);
	return 0;
}
