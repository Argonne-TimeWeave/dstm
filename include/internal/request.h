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
 * Forward pass (all ranks):
 *   BARRIER_DRAIN        - waiting until all OUT_PUT isends have completed
 *   BARRIER_WAIT         - non-zero ranks: waiting for token from predecessor
 *                          (rank 0 skips this state and goes straight to
 *                          BARRIER_DRAINING after injecting the token)
 *   BARRIER_DRAINING     - draining remaining requests now that predecessor
 *                          is known to have finished; also sends the forward
 *                          token once the queue is empty
 *
 * Return pass (all ranks):
 *   BARRIER_FORWARDED    - forward token sent; rank 0 now waits for the ring
 *                          to complete (token returns from rank N-1); non-zero
 *                          ranks wait for the completion token
 *   BARRIER_COMPLETE     - completion token received; non-zero ranks forward
 *                          it to their successor then transition to DONE
 *
 * Terminal state (shared):
 *   DONE (== 3)          - barrier finished; request is removed from the list
 */
#define DSTM_REQUEST_STATUS_BARRIER_DRAIN 4
#define DSTM_REQUEST_STATUS_BARRIER_WAIT 5
#define DSTM_REQUEST_STATUS_BARRIER_DRAINING 6
#define DSTM_REQUEST_STATUS_BARRIER_FORWARDED 7
#define DSTM_REQUEST_STATUS_BARRIER_COMPLETE 8

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

#define DSTM_REQUEST_SNPRINTF_SIZE 576
int dstm_request_snprintf(char *buf, size_t n, dstm_request_t *req);

#define dstm_log_request(level, label, req)                                    \
	do {                                                                   \
		if ((level) <= dstm_log_level) {                               \
			char __buf[DSTM_REQUEST_SNPRINTF_SIZE];                \
			dstm_request_snprintf(__buf, sizeof(__buf), (req));    \
			dstm_log_printf((level), __FILE__, __func__, __LINE__, \
			                "%s: %s\n", (label), __buf);           \
		}                                                              \
	} while (0)

#ifdef __cplusplus
}
#endif

#endif
