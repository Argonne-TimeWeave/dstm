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
#define DSTM_REQUEST_TYPE_MAX 6

#define DSTM_REQUEST_STATUS_INIT 0
#define DSTM_REQUEST_STATUS_ISEND 1
#define DSTM_REQUEST_STATUS_IRECV 2
#define DSTM_REQUEST_STATUS_DONE 3
#define DSTM_REQUEST_STATUS_MAX 4

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

void dstm_log_printrequest(int level, dstm_request_t *req);
void dstm_request_fprintf(FILE *f, dstm_request_t *req);

#ifdef __cplusplus
}
#endif

#endif
