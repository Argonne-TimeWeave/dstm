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

#include <stddef.h>
#include <stdlib.h>

#include <jansson.h>
#include <mpi.h>

#include "dstm.h"

#include "internal/log.h"

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************
 * CONSTANTS
 ******************************************************************************/

/* Region status values used by the version state machine are defined
 * in internal/region.h. */

#define DSTM_MSG_TYPE_PUT 0
#define DSTM_MSG_TYPE_GET_REQ 1
#define DSTM_MSG_TYPE_GET_REP 2
#define DSTM_MSG_TYPE_HEARTBEAT 3
#define DSTM_MSG_TYPE_BARRIER 4
#define DSTM_MSG_TYPE_MAX 5

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

/* Barrier token payload: carries the target logical time so that all ranks in
 * the ring can verify consistency.  src_rank in the header identifies which
 * rank forwarded this token. */
struct dstm_msg_barrier_s {
	dstm_time_t last_time;
};
typedef struct dstm_msg_barrier_s dstm_msg_barrier_t;

struct dstm_msg_s {
	/* required for asynchronous requests */
	MPI_Request request;
	dstm_msg_hdr_t header;
	/* we'll need other type for some other operations, might as
	 * well start using a union.
	 */
	union {
		dstm_msg_putget_t putget;
		dstm_msg_barrier_t barrier;
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

int dstm_msg_fill_hdr(dstm_msg_t *msg,
                      int type,
                      dstm_time_t time,
                      dstm_time_t delta,
                      int dst_rank,
                      int src_rank);

int dstm_msg_fill_barrier(dstm_msg_t *msg, dstm_time_t last_time);

/* Fill the putget body and set a non-owning data pointer.
 *
 * ptr  - remote pointer stored in u.putget.ptr (pass a zero-initialised
 *        dstm_ptr_t when not applicable, e.g. for GET_REP).
 * src  - pointer to the payload bytes, or NULL when there is no payload
 *        (e.g. GET_REQ).  The caller is responsible for keeping the buffer
 *        alive until the message has been fully sent; ownership stays with
 *        the enclosing dstm_request_t.buf.
 * size - byte count stored in u.putget.size.
 */
int dstm_msg_fill_payload(dstm_msg_t *msg,
                          dstm_ptr_t ptr,
                          void *src,
                          size_t size);

/* Free msg->packbuf and reset the pointer to NULL. */
void dstm_msg_free_packbuf(dstm_msg_t *msg);

int dstm_msg_send(dstm_msg_t *msg, int tag);
int dstm_msg_isend(dstm_msg_t *msg, int tag);

int dstm_msg_iprobe(MPI_Status *status);
int dstm_msg_irecv(dstm_msg_t *msg, int tag);
int dstm_msg_recvany(MPI_Status status, dstm_msg_t *msg);

int dstm_msg_test(dstm_msg_t msg);

/* Collective agreement on failure across @p comm. Given this rank's error
 * code @p err, returns the code this rank should propagate: its own if it
 * failed, -DSTM_FAILURE if any peer failed, 0 if all ranks are healthy.
 *
 * Must be called by every rank at matching points during a collective
 * operation, so that a single rank's local failure cannot leave peers
 * blocked forever in a later collective. */
int dstm_msg_consensus(MPI_Comm comm, int err);

json_t *dstm_msg_to_json(dstm_msg_t *msg);

#define dstm_log_msg(level, label, msg)                                        \
	dstm_log_json((level), (label), dstm_msg_to_json((msg)))

#ifdef __cplusplus
}
#endif

#endif
