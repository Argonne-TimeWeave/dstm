/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

/**
 * Request management, internal: tracks in-flight PUT/GET operations.
 **/

#ifndef DSTM_INTERNAL_REQUEST_H
#define DSTM_INTERNAL_REQUEST_H 1

#ifdef __cplusplus
extern "C" {
#endif

#include "dstm.h"

#include "internal/log.h"
#include "internal/msg.h"

/*******************************************************************************
 * Request type constants
 ******************************************************************************/

/* we need to make the distinction between local, incoming, or outgoing request
 */
#define DSTM_REQUEST_TYPE_LOCAL_PUT 0
#define DSTM_REQUEST_TYPE_LOCAL_GET 1
#define DSTM_REQUEST_TYPE_IN_PUT 2
#define DSTM_REQUEST_TYPE_IN_GET 3
#define DSTM_REQUEST_TYPE_OUT_PUT 4
#define DSTM_REQUEST_TYPE_OUT_GET 5
#define DSTM_REQUEST_TYPE_BARRIER 6
#define DSTM_REQUEST_TYPE_MAX 7

/* Standard request status values (shared by all request types) */
#define DSTM_REQUEST_STATUS_INIT 0
#define DSTM_REQUEST_STATUS_ISEND 1
#define DSTM_REQUEST_STATUS_IRECV 2
#define DSTM_REQUEST_STATUS_DONE 3

/* Barrier-request DFA states (values >= 4, only used by BARRIER requests).
 *
 * The barrier uses a 3-round ring protocol (rank 0 -> 1 -> ... -> N-1 -> 0).
 *
 * Round 1 -- drain outgoing:
 *   Each rank waits for its own OUT_PUT/OUT_GET requests to finish, then
 *   forwards the token.  When the token returns to rank 0, all ranks have
 *   finished their outgoing work and no new PUT/GET messages will be issued.
 *
 *   BARRIER_DRAIN_OUT    - wait for OUT_PUT/OUT_GET to complete
 *   BARRIER_R1_WAIT      - non-zero ranks: wait for R1 token from predecessor
 *   BARRIER_R1_FORWARDED - R1 token sent; waiting for the R1 ring to close
 *                          (rank 0) or for the R2 token (non-zero ranks)
 *
 * Round 2 -- complete all pending:
 *   Now barrier_all_drained is set, which bypasses all neighbor distance
 *   checks in version transitions.  Each rank drains all remaining requests
 *   (IN_GET, LOCAL_GET, etc.) then forwards the token.
 *
 *   BARRIER_R2_DRAINING  - drain all non-barrier requests
 *   BARRIER_R2_FORWARDED - R2 token sent; waiting for the R2 ring to close
 *                          (rank 0) or for the exit token (non-zero ranks)
 *
 * Round 3 -- exit:
 *   Rank 0 injects the exit token; each rank forwards it and transitions
 *   to DONE.
 *
 * Terminal state (shared):
 *   DONE (== 3)          - barrier finished; request is removed from the list
 */
#define DSTM_REQUEST_STATUS_BARRIER_DRAIN_OUT 4
#define DSTM_REQUEST_STATUS_BARRIER_R1_WAIT 5
#define DSTM_REQUEST_STATUS_BARRIER_R1_FORWARDED 6
#define DSTM_REQUEST_STATUS_BARRIER_R2_DRAINING 7
#define DSTM_REQUEST_STATUS_BARRIER_R2_FORWARDED 8

#define DSTM_REQUEST_STATUS_MAX 9

/*******************************************************************************
 * Request struct
 ******************************************************************************/

struct dstm_request_s {
	int type;
	int tag;
	int status;
	dstm_time_t time;
	dstm_time_t delta;
	int rank;
	dstm_ptr_t dst;
	dstm_ptr_t src;
	size_t size;
	/* owned payload buffer (malloc'd copy of the data to write/send/reply),
	 * or NULL if this request type carries no payload. Freed by
	 * dstm_request_destroy.
	 */
	void *buf;
	/* When set, dstm_do_progress will not destroy this request on
	 * completion; the caller that set this flag is responsible for calling
	 * dstm_request_destroy once it has observed DONE status.
	 */
	int caller_owned;
	/* embedded message used for MPI send/recv bookkeeping */
	dstm_msg_t msg;
	struct dstm_request_s *prev, *next;
};
typedef struct dstm_request_s dstm_request_t;

/*******************************************************************************
 * Request API
 ******************************************************************************/

/* Global pending request list.  All in-flight requests (LOCAL_PUT, LOCAL_GET,
 * IN_PUT, IN_GET, OUT_PUT, OUT_GET) are kept on this list until they reach
 * DONE status. */
extern dstm_request_t *requests;

/* Drive the progress loop one step: probe for incoming messages, update the
 * region state machine, and advance all pending requests.
 * @param time  The caller's current logical timestep (passed to the region
 *              state machine to control clock advancement). */
int dstm_do_progress(dstm_time_t time);

int dstm_request_new_tag(void);

int dstm_request_cmp_departure(dstm_request_t *a, dstm_request_t *b);

int dstm_request_create(dstm_request_t **r,
                        int type,
                        int tag,
                        dstm_time_t time,
                        dstm_time_t delta,
                        int rank,
                        dstm_ptr_t dst,
                        dstm_ptr_t src,
                        size_t size);

int dstm_request_destroy(dstm_request_t **r);

int dstm_request_isdone(dstm_request_t *req);

int dstm_request_progress(dstm_request_t *req);

int dstm_request_progress_barrier(dstm_request_t *req);

json_t *dstm_request_to_json(dstm_request_t *req);

#define dstm_log_request(level, label, req)                                    \
	dstm_log_json((level), (label), dstm_request_to_json((req)))

#ifdef __cplusplus
}
#endif

#endif
