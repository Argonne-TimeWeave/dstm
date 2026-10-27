/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

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

#include "config.h"

#include <stdlib.h>

#include "dstm.h"

#include "internal/globals.h"
#include "internal/msg.h"
#include "internal/region.h"
#include "internal/request.h"
#include "internal/utlist.h"

/* Send the barrier token to the ring successor. */
static void dstm_barrier_send_token(dstm_time_t last_time)
{
	int next =
	        (dstm_global_context.rank + 1) % dstm_global_context.world_size;
	dstm_msg_t m;
	memset(&m, 0, sizeof(m));
	dstm_msg_fill_hdr(&m, DSTM_MSG_TYPE_BARRIER, last_time, 0, next,
	                  dstm_global_context.rank, 0); /* region_id unused */
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
             it->type == DSTM_REQUEST_TYPE_OUT_GET ||
		     it->type == DSTM_REQUEST_TYPE_OUT_APPEND) &&
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
		if (dstm_global_context.barrier_token_count == 0)
			break;

		dstm_global_context.barrier_token_count--;
		dstm_log_info("barrier: rank %d received R1 token\n", rank);

		/* Forward R1 token to successor. */
		dstm_barrier_send_token(last_time);
		dstm_log_info("barrier: rank %d sent R1 token\n", rank);
		req->status = DSTM_REQUEST_STATUS_BARRIER_R1_FORWARDED;
		break;

	case DSTM_REQUEST_STATUS_BARRIER_R1_FORWARDED:
		/* Wait for the R1 ring to close (rank 0) or for the R2
		 * token (non-zero ranks). */
		if (dstm_global_context.barrier_token_count == 0)
			break;

		dstm_global_context.barrier_token_count--;
		dstm_log_info("barrier: rank %d R1 complete\n", rank);

		/* Rank 0: inject R2 token. */
		if (rank == 0) {
			dstm_barrier_send_token(last_time);
			dstm_log_info("barrier: rank 0 sent R2 token\n");
		}

		/* All ranks: enable barrier optimization so version
		 * transitions bypass neighbor distance checks. */
		dstm_global_context.barrier_all_drained = 1;
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
		if (dstm_global_context.barrier_token_count == 0)
			break;

		dstm_global_context.barrier_token_count--;

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
		dstm_global_context.barrier_all_drained = 0;
		req->status = DSTM_REQUEST_STATUS_DONE;
		break;

	default:
		/* DONE is handled by dstm_do_progress; nothing else expected. */
		break;
	}
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
	                    0, 0, null_ptr, null_ptr, 0);
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
