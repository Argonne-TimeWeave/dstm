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
#define DSTM_MSG_TYPE_GET_REQ 1
#define DSTM_MSG_TYPE_GET_REP 2
#define DSTM_MSG_TYPE_MAX 3

struct dstm_msg_hdr_s {
	int type;
	int dst_rank;
	int src_rank;
	dstm_time_t time;
	dstm_time_t delta;
};
typedef struct dstm_msg_hdr_s dstm_msg_hdr_t;

struct dstm_msg_putget_s {
	dstm_ptr_t ptr;
	size_t size;
};
typedef struct dstm_msg_putget_s dstm_msg_putget_t;

struct dstm_msg_s {
	/* required for asynchronous requests */
	MPI_Request request;
	dstm_msg_hdr_t header;
	/* we'll need other type for some other operations, might as
	 * well start using a union.
	 */
	union {
		dstm_msg_putget_t putget;
	} u;
	/* data buffer, if needed */
	void *data;
	/* packing buffer, if needed */
	void *packbuf;
};
typedef struct dstm_msg_s dstm_msg_t;

/*******************************************************************************
 * Create/Destroy: No memory allocation implyed
 ******************************************************************************/

int dstm_msg_init(void);
int dstm_msg_finalize(void);

int dstm_msg_fill_hdr(dstm_msg_t *msg, int type, dstm_time_t time, dstm_time_t
		       delta, int dst_rank, int src_rank);

int dstm_msg_fill_put(dstm_msg_t *msg, dstm_ptr_t dst, void *src, size_t size);
int dstm_msg_fill_getreq(dstm_msg_t *msg, dstm_ptr_t src, size_t size);
int dstm_msg_fill_getrep(dstm_msg_t *msg, void *src, size_t size);

int dstm_msg_send(dstm_msg_t *msg, int tag);
int dstm_msg_isend(dstm_msg_t *msg, int tag);

int dstm_msg_iprobe(MPI_Status *status);
int dstm_msg_irecv(dstm_msg_t *msg, int tag);
int dstm_msg_recvany(MPI_Status status, dstm_msg_t *msg);

int dstm_msg_test(dstm_msg_t msg);

void dstm_log_printmsg(int level, dstm_msg_t *msg);
void dstm_msg_fprintf(FILE *f, dstm_msg_t *msg);

#ifdef __cplusplus
}
#endif

#endif
