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

#include <stdlib.h>

#include "dstm.h"

#include "internal/globals.h"
#include "internal/msg.h"
#include "internal/request.h"
#include "internal/utils.h"

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

int dstm_request_isdone(dstm_request_t *req)
{
	return req->status == DSTM_REQUEST_STATUS_DONE;
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

	DSTM_SNPRINTF_ADVANCE(buf, n, total, total);

	/* dst pointer */
	int w = dstm_ptr_snprintf(buf, n, req->dst);
	DSTM_SNPRINTF_ADVANCE(buf, n, total, w);

	w = snprintf(buf, n, ", src: ");
	DSTM_SNPRINTF_ADVANCE(buf, n, total, w);

	/* src pointer */
	w = dstm_ptr_snprintf(buf, n, req->src);
	DSTM_SNPRINTF_ADVANCE(buf, n, total, w);

	w = snprintf(buf, n, ", msg: ");
	DSTM_SNPRINTF_ADVANCE(buf, n, total, w);

	/* embedded message */
	w = dstm_msg_snprintf(buf, n, &req->msg);
	DSTM_SNPRINTF_ADVANCE(buf, n, total, w);

	w = snprintf(buf, n, "}");
	total += w;

	return total;
}
