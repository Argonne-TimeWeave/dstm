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

#include <mpi.h>
#include <stdlib.h>

#include "dstm.h"

#include "internal/globals.h"
#include "internal/msg.h"
#include "internal/region.h"
#include "internal/utlist.h"

MPI_Datatype dstm_msg_hdr_dt;
MPI_Datatype dstm_msg_putget_dt;
MPI_Datatype dstm_msg_heartbeat_dt;

/*******************************************************************************
 * MPI Datatypes management
 ******************************************************************************/

/* need to create the datatypes at init */
int dstm_msg_init(void)
{
	/* dstm_msg_hdr_t: type(int), dst_rank(int), src_rank(int),
	 * time(int64_t), delta(int64_t) */
	MPI_Datatype dstm_msg_hdr_sig[5] = {MPI_INT, MPI_INT, MPI_INT,
	                                    MPI_INT64_T, MPI_INT64_T};
	int dstm_msg_hdr_bl[5] = {1, 1, 1, 1, 1};
	MPI_Aint dstm_msg_hdr_disp[5] = {offsetof(dstm_msg_hdr_t, type),
	                                 offsetof(dstm_msg_hdr_t, dst_rank),
	                                 offsetof(dstm_msg_hdr_t, src_rank),
	                                 offsetof(dstm_msg_hdr_t, time),
	                                 offsetof(dstm_msg_hdr_t, delta)};
	MPI_Type_create_struct(5, dstm_msg_hdr_bl, dstm_msg_hdr_disp,
	                       dstm_msg_hdr_sig, &dstm_msg_hdr_dt);
	MPI_Type_commit(&dstm_msg_hdr_dt);

	/* dstm_msg_putget_t: region(int), ptr(intptr_t), size(size_t) */
	MPI_Datatype dstm_msg_putget_sig[3] = {MPI_INT, MPI_AINT, MPI_AINT};
	int dstm_msg_putget_bl[3] = {1, 1, 1};
	MPI_Aint dstm_msg_putget_disp[3] = {
	        offsetof(dstm_msg_putget_t, ptr.region),
	        offsetof(dstm_msg_putget_t, ptr.offset),
	        offsetof(dstm_msg_putget_t, size)};
	MPI_Type_create_struct(3, dstm_msg_putget_bl, dstm_msg_putget_disp,
	                       dstm_msg_putget_sig, &dstm_msg_putget_dt);
	MPI_Type_commit(&dstm_msg_putget_dt);

	/* dstm_msg_heartbeat_t: status(int) */
	MPI_Datatype dstm_msg_heartbeat_sig[1] = {MPI_INT};
	int dstm_msg_heartbeat_bl[1] = {1};
	MPI_Aint dstm_msg_heartbeat_disp[1] = {
	        offsetof(dstm_msg_heartbeat_t, status)};
	MPI_Type_create_struct(1, dstm_msg_heartbeat_bl,
	                       dstm_msg_heartbeat_disp, dstm_msg_heartbeat_sig,
	                       &dstm_msg_heartbeat_dt);
	MPI_Type_commit(&dstm_msg_heartbeat_dt);

	return 0;
}

int dstm_msg_finalize(void)
{
	MPI_Type_free(&dstm_msg_hdr_dt);
	MPI_Type_free(&dstm_msg_putget_dt);
	MPI_Type_free(&dstm_msg_heartbeat_dt);
	return 0;
}

/*******************************************************************************
 * Message cleanup
 ******************************************************************************/

void dstm_msg_free_packbuf(dstm_msg_t *msg)
{
	free(msg->packbuf);
	msg->packbuf = NULL;
}

/*******************************************************************************
 * Message creation/filling
 ******************************************************************************/

int dstm_msg_fill_hdr(dstm_msg_t *m,
                      int type,
                      dstm_time_t time,
                      dstm_time_t delta,
                      int src_rank,
                      int dst_rank)
{
	if (type < 0 || type >= DSTM_MSG_TYPE_MAX) {
		dstm_errno = DSTM_EINVAL;
		return -DSTM_EINVAL;
	}
	m->header.type = type;
	m->header.dst_rank = dst_rank;
	m->header.src_rank = src_rank;
	m->header.time = time;
	m->header.delta = delta;
	return 0;
}

int dstm_msg_fill_payload(dstm_msg_t *m, dstm_ptr_t ptr, void *src, size_t size)
{
	m->u.putget.ptr = ptr;
	m->u.putget.size = size;
	/* src may be NULL for messages that carry no payload (GET_REQ).
	 * When non-NULL it is a non-owning pointer into the request's buf. */
	m->data = src;
	return 0;
}

int dstm_msg_fill_heartbeat(dstm_msg_t *m, int status)
{
	m->u.heartbeat.status = status;
	return 0;
}

/*******************************************************************************
 * Sending Messages
 ******************************************************************************/

/* Pack header + putget body + payload bytes into msg->packbuf and post an
 * async send.  Used for both PUT and GET_REP, which share the same wire
 * format: [hdr][putget][data bytes].
 *
 * msg->packbuf is allocated here and must be freed by the caller once
 * MPI_Test/MPI_Wait on msg->request reports completion.
 */
static int dstm_msg_isend_with_data(dstm_msg_t *msg, int tag)
{
	assert(msg != NULL);
	assert(msg->data != NULL);

	int hsz, psz, dsz;
	MPI_Pack_size(1, dstm_msg_hdr_dt, dstm_global_context.comm, &hsz);
	MPI_Pack_size(1, dstm_msg_putget_dt, dstm_global_context.comm, &psz);
	MPI_Pack_size(msg->u.putget.size, MPI_CHAR, dstm_global_context.comm,
	              &dsz);
	size_t bufsz = hsz + psz + dsz;

	msg->packbuf = calloc(bufsz, sizeof(char));
	if (msg->packbuf == NULL)
		return -DSTM_ENOMEM;

	int pos = 0;
	MPI_Pack(&msg->header, 1, dstm_msg_hdr_dt, msg->packbuf, bufsz, &pos,
	         dstm_global_context.comm);
	MPI_Pack(&msg->u.putget, 1, dstm_msg_putget_dt, msg->packbuf, bufsz,
	         &pos, dstm_global_context.comm);
	MPI_Pack(msg->data, msg->u.putget.size, MPI_CHAR, msg->packbuf, bufsz,
	         &pos, dstm_global_context.comm);

	MPI_Isend(msg->packbuf, pos, MPI_PACKED, msg->header.dst_rank, tag,
	          dstm_global_context.comm, &msg->request);
	return 0;
}

int dstm_msg_isend(dstm_msg_t *msg, int tag)
{
	assert(msg != NULL);
	switch (msg->header.type) {
	case DSTM_MSG_TYPE_PUT:
	case DSTM_MSG_TYPE_GET_REP:
		return dstm_msg_isend_with_data(msg, tag);
	default:
		assert(0);
	}
	return 0;
}

int dstm_msg_send_getreq(dstm_msg_t *msg, int tag)
{
	/* deal with copying the data and packing it into a buffer */
	int hsz, gsz;
	MPI_Pack_size(1, dstm_msg_hdr_dt, dstm_global_context.comm, &hsz);
	MPI_Pack_size(1, dstm_msg_putget_dt, dstm_global_context.comm, &gsz);
	size_t bufsz = hsz + gsz;

	msg->packbuf = calloc(bufsz, sizeof(char));
	if (msg->packbuf == NULL)
		return -DSTM_ENOMEM;

	int pos = 0;
	MPI_Pack(&msg->header, 1, dstm_msg_hdr_dt, msg->packbuf, bufsz, &pos,
	         dstm_global_context.comm);
	MPI_Pack(&msg->u.putget, 1, dstm_msg_putget_dt, msg->packbuf, bufsz,
	         &pos, dstm_global_context.comm);

	/* send the message */
	MPI_Send(msg->packbuf, pos, MPI_PACKED, msg->header.dst_rank, tag,
	         dstm_global_context.comm);

	dstm_msg_free_packbuf(msg);
	return 0;
}

static int dstm_msg_send_heartbeat(dstm_msg_t *msg, int tag)
{
	int hsz, bsz;
	MPI_Pack_size(1, dstm_msg_hdr_dt, dstm_global_context.comm, &hsz);
	MPI_Pack_size(1, dstm_msg_heartbeat_dt, dstm_global_context.comm, &bsz);
	size_t bufsz = hsz + bsz;

	msg->packbuf = calloc(bufsz, sizeof(char));
	if (msg->packbuf == NULL)
		return -DSTM_ENOMEM;

	int pos = 0;
	MPI_Pack(&msg->header, 1, dstm_msg_hdr_dt, msg->packbuf, bufsz, &pos,
	         dstm_global_context.comm);
	MPI_Pack(&msg->u.heartbeat, 1, dstm_msg_heartbeat_dt, msg->packbuf,
	         bufsz, &pos, dstm_global_context.comm);

	MPI_Send(msg->packbuf, pos, MPI_PACKED, msg->header.dst_rank, tag,
	         dstm_global_context.comm);

	dstm_msg_free_packbuf(msg);
	return 0;
}

int dstm_msg_send(dstm_msg_t *msg, int tag)
{
	assert(msg != NULL);
	switch (msg->header.type) {
	case DSTM_MSG_TYPE_GET_REQ:
		return dstm_msg_send_getreq(msg, tag);
	case DSTM_MSG_TYPE_HEARTBEAT:
		return dstm_msg_send_heartbeat(msg, tag);
	default:
		assert(0);
	}
	return 0;
}

/*******************************************************************************
 * Receiving Unknown Messages
 ******************************************************************************/

int dstm_msg_iprobe(MPI_Status *status)
{
	int flag = 0;

	MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, dstm_global_context.comm, &flag,
	           status);
	return flag;
}

int dstm_msg_recvany(MPI_Status status, dstm_msg_t *msg)
{
	int source, tag;

	source = status.MPI_SOURCE;
	tag = status.MPI_TAG;

	/* there's a message to receive, let's receive it */
	/* figure out the size of the message that we need to receive */
	int bufsz;
	MPI_Get_count(&status, MPI_PACKED, &bufsz);

	msg->packbuf = calloc(bufsz, sizeof(char));
	assert(msg->packbuf != NULL);
	MPI_Recv(msg->packbuf, bufsz, MPI_PACKED, source, tag,
	         dstm_global_context.comm, &status);

	/* unpack message header first, so that we know what we received */
	int pos = 0;
	MPI_Unpack(msg->packbuf, bufsz, &pos, &msg->header, 1, dstm_msg_hdr_dt,
	           dstm_global_context.comm);

	switch (msg->header.type) {
	case DSTM_MSG_TYPE_PUT:
		MPI_Unpack(msg->packbuf, bufsz, &pos, &msg->u.putget, 1,
		           dstm_msg_putget_dt, dstm_global_context.comm);
		msg->data = (char *)msg->packbuf + pos;
		break;
	case DSTM_MSG_TYPE_GET_REQ:
		MPI_Unpack(msg->packbuf, bufsz, &pos, &msg->u.putget, 1,
		           dstm_msg_putget_dt, dstm_global_context.comm);
		break;
	case DSTM_MSG_TYPE_GET_REP:
		MPI_Unpack(msg->packbuf, bufsz, &pos, &msg->u.putget, 1,
		           dstm_msg_putget_dt, dstm_global_context.comm);
		msg->data = (char *)msg->packbuf + pos;
		break;
	case DSTM_MSG_TYPE_HEARTBEAT:
		MPI_Unpack(msg->packbuf, bufsz, &pos, &msg->u.heartbeat, 1,
		           dstm_msg_heartbeat_dt, dstm_global_context.comm);
		break;
	default:
		assert(0);
	}
	return 0;
}

/*******************************************************************************
 * Checking for progress
 ******************************************************************************/

int dstm_msg_test(dstm_msg_t msg)
{
	MPI_Status status;
	int flag = 0;
	MPI_Test(&msg.request, &flag, &status);
	dstm_log_debug("MPI_Test: %d\n", flag);
	return flag;
}

/*******************************************************************************
 * Formatting helpers
 ******************************************************************************/

int dstm_msg_snprintf(char *buf, size_t n, dstm_msg_t *msg)
{
	assert(buf != NULL);
	if (msg == NULL)
		return snprintf(buf, n, "NULL");

	int written = snprintf(
	        buf, n,
	        "{ header: { type: %d, dst_rank: %d, src_rank: %d, time: %" PRId64
	        ", delta: %" PRId64 "}, data: %p, packbuf: %p, ",
	        msg->header.type, msg->header.dst_rank, msg->header.src_rank,
	        msg->header.time, msg->header.delta, msg->data, msg->packbuf);

	/* Advance pointer and shrink remaining capacity for the suffix. */
	char *p = buf + (written < (int)n ? written : (int)n - 1);
	size_t rem = written < (int)n ? n - (size_t)written : 1;

	int suffix = 0;

	switch (msg->header.type) {
	case DSTM_MSG_TYPE_PUT:
		suffix = snprintf(p, rem,
		                  "put: { region: %d, dst: %zu, size: %zu}}",
		                  msg->u.putget.ptr.region,
		                  (size_t)msg->u.putget.ptr.offset,
		                  msg->u.putget.size);
		break;
	case DSTM_MSG_TYPE_GET_REQ:
		suffix = snprintf(p, rem,
		                  "getreq: { region: %d, ptr: %zu, size: %zu}}",
		                  msg->u.putget.ptr.region,
		                  (size_t)msg->u.putget.ptr.offset,
		                  msg->u.putget.size);
		break;
	case DSTM_MSG_TYPE_GET_REP:
		suffix = snprintf(p, rem,
		                  "getrep: { region: %d, ptr: %zu, size: %zu}}",
		                  msg->u.putget.ptr.region,
		                  (size_t)msg->u.putget.ptr.offset,
		                  msg->u.putget.size);
		break;
	case DSTM_MSG_TYPE_HEARTBEAT:
		suffix = snprintf(p, rem, "heartbeat: { status: %s}}",
		                  msg->u.heartbeat.status ==
		                                  DSTM_REGION_STATUS_READABLE ?
		                          "readable" :
		                          "writable");
		break;
	default:
		suffix = snprintf(p, rem, "unknown}}");
		break;
	}

	return written + suffix;
}
