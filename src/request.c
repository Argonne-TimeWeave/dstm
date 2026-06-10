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
#include "internal/log.h"
#include "internal/msg.h"
#include "internal/region.h"
#include "internal/request.h"

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

void dstm_request_tag_reset(void)
{
	dstm_request_tag_count = 1;
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

static const dstm_enum_table_t dstm_request_type_table[] = {
        {DSTM_REQUEST_TYPE_LOCAL_PUT, "LOCAL_PUT"},
        {DSTM_REQUEST_TYPE_LOCAL_GET, "LOCAL_GET"},
        {DSTM_REQUEST_TYPE_IN_PUT, "IN_PUT"},
        {DSTM_REQUEST_TYPE_IN_GET, "IN_GET"},
        {DSTM_REQUEST_TYPE_OUT_PUT, "OUT_PUT"},
        {DSTM_REQUEST_TYPE_OUT_GET, "OUT_GET"},
        {DSTM_REQUEST_TYPE_BARRIER, "BARRIER"},
        {0, NULL},
};

static const dstm_enum_table_t dstm_request_status_table[] = {
        {DSTM_REQUEST_STATUS_INIT, "INIT"},
        {DSTM_REQUEST_STATUS_ISEND, "ISEND"},
        {DSTM_REQUEST_STATUS_IRECV, "IRECV"},
        {DSTM_REQUEST_STATUS_DONE, "DONE"},
        {DSTM_REQUEST_STATUS_BARRIER_DRAIN_OUT, "BARRIER_DRAIN_OUT"},
        {DSTM_REQUEST_STATUS_BARRIER_R1_WAIT, "BARRIER_R1_WAIT"},
        {DSTM_REQUEST_STATUS_BARRIER_R1_FORWARDED, "BARRIER_R1_FORWARDED"},
        {DSTM_REQUEST_STATUS_BARRIER_R2_DRAINING, "BARRIER_R2_DRAINING"},
        {DSTM_REQUEST_STATUS_BARRIER_R2_FORWARDED, "BARRIER_R2_FORWARDED"},
        {0, NULL},
};

json_t *dstm_request_to_json(dstm_request_t *req)
{
	if (req == NULL)
		return json_null();

	json_t *obj = json_object();
	json_object_set_new(obj, "type",
	                    json_string(dstm_enum_to_string(
	                            req->type, dstm_request_type_table)));
	json_object_set_new(obj, "time", json_integer(req->time));
	json_object_set_new(obj, "delta", json_integer(req->delta));
	json_object_set_new(obj, "dst", dstm_ptr_to_json(req->dst));
	json_object_set_new(obj, "src", dstm_ptr_to_json(req->src));
	json_object_set_new(obj, "rank", json_integer(req->rank));
	json_object_set_new(obj, "size", json_integer(req->size));
	json_object_set_new(obj, "status",
	                    json_string(dstm_enum_to_string(
	                            req->status, dstm_request_status_table)));
	json_object_set_new(obj, "tag", json_integer(req->tag));
	json_object_set_new(obj, "msg", dstm_msg_to_json(&req->msg));

	return obj;
}
