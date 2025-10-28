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
#include "internal/utlist.h"
#include <mpi.h>
#include <stdlib.h>

#define DSTM_REQUEST_TYPE_PUT 0
#define DSTM_REQUEST_TYPE_GET 1
#define DSTM_REQUEST_TYPE_MAX 2

struct dstm_request_s {
	int type;
	dstm_time_t time;
	dstm_time_t delta;
	int rank;
	void *dst;
	void *src;
	size_t size;
	struct dstm_request_s *prev, *next;
};
typedef struct dstm_request_s dstm_request_t;

dstm_request_t *local_requests = NULL;
dstm_request_t *outgoing_requests = NULL;

int dstm_request_cmp(dstm_request_t *a, dstm_request_t *b)
{
	dstm_time_t ta, tb;
	ta = a->time + a->delta;
	tb = b->time + b->delta;
	return ta - tb;
}

/*******************************************************************************
 * Basic request management
 ******************************************************************************/

int dstm_req_enqueue(dstm_request_t *rq, int type, dstm_time_t time, dstm_time_t delta, int rank,
		     void *dst, void *src, size_t size)
{
	assert(type < DSTM_REQUEST_TYPE_MAX);
	assert(rq != NULL);
	assert(dst != NULL && src != NULL);

	dstm_req_t *req = calloc(1, sizeof(dstm_req_t));
	if (req == NULL)
		return -DSTM_ENOMEM;

	req->type = type;
	req->time = r->time;
	req->delta = r->delta;
	req->rank = rank;
	req->dst = dst;
	req->src = src;
	req->size = size;

	DL_INSERT_INORDER(rq, req, dstm_request_cmp);
	return 0;
}

/*******************************************************************************
 * Distributed mechanics
 ******************************************************************************/

/* executes all requests that are before time now */
int dstm_req_local_apply(dstm_time_t now)
{
	dstm_request_t *req, *tmp;
	DL_FOREACH_SAFE(local_requests, req, tmp) {
		dstm_time_t when = req->time + req->delta;
		if (when > now)
			break;

		/* TODO: that's basically a FIFO order, not sure that what we
		 * want. */
		memcpy(req->dst, req->src, req->size);
		free(req);
	}
	return 0;
}

int dstm_req_remote_push(dstm_time_t now)
{


}

int dstm_put_local(dstm_region_t *r, dstm_time_t delta, void *dst, void *src, size_t
		   size)
{
	if (dst == NULL || r == NULL || src == NULL)
		return -DSTM_EINVAL;

	return dstm_req_enqueue(local_requests, DSTM_REQUEST_TYPE_PUT, r->time,
			 delta, dstm_global_context.rank,
			 dst, src, size);
}

int dstm_get_local(dstm_region_t *r, dstm_time_t delta, void *dst, void *src, size_t
		   size)
{

	if (dst == NULL || r == NULL || src == NULL)
		return -DSTM_EINVAL;

	return dstm_req_enqueue(local_requests, DSTM_REQUEST_TYPE_GET, r->time,
			 delta, dstm_global_context.rank,
			 dst, src, size);
}


int dstm_put_remote(int rank, int regionid, dstm_time_t time,  dstm_time_t delta, void *dst, void *src, size_t
		   size)
{
	if (dst == NULL || src == NULL)
		return -DSTM_EINVAL;

	return dstm_req_enqueue(outgoing_requests, DSTM_REQUEST_TYPE_PUT, r->time,
			 delta, dstm_global_context.rank,
			 dst, src, size);
}

int dstm_get_remote(int rank, int regionid, dstm_time_t time, dstm_time_t delta, void *dst, void *src, size_t
		   size)
{

	if (dst == NULL || src == NULL)
		return -DSTM_EINVAL;

	return dstm_req_enqueue(outgoing_requests, DSTM_REQUEST_TYPE_GET, time,
			 delta, dstm_global_context.rank,
			 dst, src, size);
}


