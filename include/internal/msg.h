/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

/**
 * Messaging management, internal: basically the networking stack
 **/

#ifndef DSTM_INTERNAL_MSG_H
#define DSTM_INTERNAL_MSG_H 1

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************
 * CONSTANTS
 ******************************************************************************/

#define DSTM_MSG_TYPE_PUT 0
#define DSTM_MSG_TYPE_GET 1
#define DSTM_MSG_TYPE_MAX 2

struct dstm_msg_s {
	int type;
	MPI_Request request;
	void *header;
	void *buf;
};
typedef struct dstm_msg_s dstm_msg_t;

struct dstm_msg_put_s {
	dstm_time_t time;
	dstm_time_t delta;
	int rank;
	intptr_t dst;
	size_t size;
};
typedef struct dstm_msg_put_s dstm_msg_put_t;

struct dstm_msg_get_s {
	dstm_time_t time;
	dstm_time_t delta;
	int rank;
	intptr_t src;
	size_t size;
};
typedef struct dstm_msg_get_s dstm_msg_get_t;

/*******************************************************************************
 * Create/Destroy
 ******************************************************************************/

int dstm_msg_create_put(dstm_msg_t **msg, int type, dstm_time_t time, dstm_time_t delta,
		    int rank, intptr_t src, size_t size);
int dstm_msg_create_get(dstm_msg_t **msg, int type, dstm_time_t time, dstm_time_t delta,
		    int rank, intptr_t src, size_t size);
void dstm_msg_destroy(dstm_msg_t **msg);

int dstm_msg_isend(int dst, int src, dstm_msg_t **m);
int dstm_msg_send(int dst, int src, dstm_msg_t **m);
int dstm_msg_irecv(dstm_request_t *req, int src, void *buf, size_t size);

#ifdef __cplusplus
}
#endif

#endif
