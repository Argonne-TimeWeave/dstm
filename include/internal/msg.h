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

/* Region status values, shared between the messaging layer (heartbeat body)
 * and the ops layer (local region state machine). */
#define DSTM_REGION_STATUS_WRITABLE 0
#define DSTM_REGION_STATUS_READABLE 1

#define DSTM_MSG_TYPE_PUT 0
#define DSTM_MSG_TYPE_GET_REQ 1
#define DSTM_MSG_TYPE_GET_REP 2
#define DSTM_MSG_TYPE_HEARTBEAT 3
#define DSTM_MSG_TYPE_MAX 4

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

struct dstm_msg_heartbeat_s {
	int status; /* DSTM_REGION_STATUS_WRITABLE or
	               DSTM_REGION_STATUS_READABLE */
};
typedef struct dstm_msg_heartbeat_s dstm_msg_heartbeat_t;

struct dstm_msg_s {
	/* required for asynchronous requests */
	MPI_Request request;
	dstm_msg_hdr_t header;
	/* we'll need other type for some other operations, might as
	 * well start using a union.
	 */
	union {
		dstm_msg_putget_t putget;
		dstm_msg_heartbeat_t heartbeat;
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

int dstm_msg_fill_heartbeat(dstm_msg_t *msg, int status);

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

int dstm_msg_send(dstm_msg_t *msg, int tag);
int dstm_msg_isend(dstm_msg_t *msg, int tag);

int dstm_msg_iprobe(MPI_Status *status);
int dstm_msg_irecv(dstm_msg_t *msg, int tag);
int dstm_msg_recvany(MPI_Status status, dstm_msg_t *msg);

int dstm_msg_test(dstm_msg_t msg);

#define DSTM_MSG_SNPRINTF_SIZE 320
int dstm_msg_snprintf(char *buf, size_t n, dstm_msg_t *msg);

#define dstm_log_msg(level, label, msg)                                        \
	do {                                                                   \
		if ((level) <= dstm_log_level) {                               \
			char __buf[DSTM_MSG_SNPRINTF_SIZE];                    \
			dstm_msg_snprintf(__buf, sizeof(__buf), (msg));        \
			dstm_log_printf((level), __FILE__, __func__, __LINE__, \
			                "%s: %s\n", (label), __buf);           \
		}                                                              \
	} while (0)

#ifdef __cplusplus
}
#endif

#endif
