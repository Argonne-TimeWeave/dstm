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
#include "internal/log.h"
#include "internal/msg.h"
#include "internal/profile.h"
#include "internal/region.h"
#include "internal/utlist.h"

MPI_Datatype dstm_msg_hdr_dt;
MPI_Datatype dstm_msg_putget_dt;
MPI_Datatype dstm_msg_barrier_dt;

/*******************************************************************************
 * MPI Datatypes management
 ******************************************************************************/

/* need to create the datatypes at init */
int dstm_msg_init(void)
{
	/* dstm_msg_hdr_t: type(int), dst_rank(int), src_rank(int),
	 * time(int64_t), delta(int64_t), region_id(int) */
	MPI_Datatype dstm_msg_hdr_sig[6] = {MPI_INT, MPI_INT, MPI_INT,
	                                    MPI_INT64_T, MPI_INT64_T, MPI_INT};
	int dstm_msg_hdr_bl[6] = {1, 1, 1, 1, 1, 1};
	MPI_Aint dstm_msg_hdr_disp[6] = {offsetof(dstm_msg_hdr_t, type),
	                                 offsetof(dstm_msg_hdr_t, dst_rank),
	                                 offsetof(dstm_msg_hdr_t, src_rank),
	                                 offsetof(dstm_msg_hdr_t, time),
	                                 offsetof(dstm_msg_hdr_t, delta),
                                     offsetof(dstm_msg_hdr_t, region_id)};
	MPI_Type_create_struct(6, dstm_msg_hdr_bl, dstm_msg_hdr_disp,
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

	/* dstm_msg_barrier_t: last_time(int64_t) */
	MPI_Datatype dstm_msg_barrier_sig[1] = {MPI_INT64_T};
	int dstm_msg_barrier_bl[1] = {1};
	MPI_Aint dstm_msg_barrier_disp[1] = {
	        offsetof(dstm_msg_barrier_t, last_time)};
	MPI_Type_create_struct(1, dstm_msg_barrier_bl, dstm_msg_barrier_disp,
	                       dstm_msg_barrier_sig, &dstm_msg_barrier_dt);
	MPI_Type_commit(&dstm_msg_barrier_dt);

	return 0;
}

int dstm_msg_finalize(void)
{
	MPI_Type_free(&dstm_msg_hdr_dt);
	MPI_Type_free(&dstm_msg_putget_dt);
	MPI_Type_free(&dstm_msg_barrier_dt);
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
                      int dst_rank,
                      int src_rank,
                      int region_id)
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
    m->header.region_id = region_id;
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

int dstm_msg_fill_barrier(dstm_msg_t *m, dstm_time_t last_time)
{
	m->u.barrier.last_time = last_time;
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
    dstm_profile_record(DSTM_PROFILE_MPI_ISEND, (size_t)pos);
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
    dstm_profile_record(DSTM_PROFILE_MPI_SEND, (size_t)pos);

	dstm_msg_free_packbuf(msg);
	return 0;
}

static int dstm_msg_send_heartbeat(dstm_msg_t *msg, int tag)
{
	int hsz;
	MPI_Pack_size(1, dstm_msg_hdr_dt, dstm_global_context.comm, &hsz);
	size_t bufsz = hsz;

	msg->packbuf = calloc(bufsz, sizeof(char));
	if (msg->packbuf == NULL)
		return -DSTM_ENOMEM;

	int pos = 0;
	MPI_Pack(&msg->header, 1, dstm_msg_hdr_dt, msg->packbuf, bufsz, &pos,
	         dstm_global_context.comm);

	MPI_Send(msg->packbuf, pos, MPI_PACKED, msg->header.dst_rank, tag,
	         dstm_global_context.comm);
    dstm_profile_record(DSTM_PROFILE_MPI_SEND, (size_t)pos);

	dstm_msg_free_packbuf(msg);
	return 0;
}

static int dstm_msg_send_barrier(dstm_msg_t *msg, int tag)
{
	int hsz, bsz;
	MPI_Pack_size(1, dstm_msg_hdr_dt, dstm_global_context.comm, &hsz);
	MPI_Pack_size(1, dstm_msg_barrier_dt, dstm_global_context.comm, &bsz);
	size_t bufsz = hsz + bsz;

	msg->packbuf = calloc(bufsz, sizeof(char));
	if (msg->packbuf == NULL)
		return -DSTM_ENOMEM;

	int pos = 0;
	MPI_Pack(&msg->header, 1, dstm_msg_hdr_dt, msg->packbuf, bufsz, &pos,
	         dstm_global_context.comm);
	MPI_Pack(&msg->u.barrier, 1, dstm_msg_barrier_dt, msg->packbuf, bufsz,
	         &pos, dstm_global_context.comm);

	MPI_Send(msg->packbuf, pos, MPI_PACKED, msg->header.dst_rank, tag,
	         dstm_global_context.comm);
    dstm_profile_record(DSTM_PROFILE_MPI_SEND, (size_t)pos);

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
	case DSTM_MSG_TYPE_BARRIER:
		return dstm_msg_send_barrier(msg, tag);
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
    dstm_profile_record(DSTM_PROFILE_MPI_RECV, 0);

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
		/* heartbeat is header-only, nothing to unpack */
		break;
	case DSTM_MSG_TYPE_BARRIER:
		MPI_Unpack(msg->packbuf, bufsz, &pos, &msg->u.barrier, 1,
		           dstm_msg_barrier_dt, dstm_global_context.comm);
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
 * Collective error consensus
 ******************************************************************************/

int dstm_msg_consensus(MPI_Comm comm, int err)
{
        int local_bad = (err != 0) ? 1 : 0;
        int any_bad   = 0;

        int mpi_err = MPI_Allreduce(&local_bad, &any_bad, 1, MPI_INT, MPI_MAX, comm);
        dstm_profile_record_collective(DSTM_PROFILE_MPI_ALLREDUCE, sizeof(local_bad));

        if (mpi_err != MPI_SUCCESS)
                return (err != 0) ? err : -DSTM_FAILURE;

        if (err != 0)
                return err;
        return any_bad ? -DSTM_FAILURE : 0;
}

/*******************************************************************************
 * Formatting helpers
 ******************************************************************************/

static const dstm_enum_table_t dstm_msg_type_table[] = {
        {DSTM_MSG_TYPE_PUT, "PUT"},
        {DSTM_MSG_TYPE_GET_REQ, "GET_REQ"},
        {DSTM_MSG_TYPE_GET_REP, "GET_REP"},
        {DSTM_MSG_TYPE_HEARTBEAT, "HEARTBEAT"},
        {DSTM_MSG_TYPE_BARRIER, "BARRIER"},
        {0, NULL},
};

json_t *dstm_msg_to_json(dstm_msg_t *msg)
{
	if (msg == NULL)
		return json_null();

	char data_buf[32];
	char pack_buf[32];
	snprintf(data_buf, sizeof(data_buf), "%p", msg->data);
	snprintf(pack_buf, sizeof(pack_buf), "%p", msg->packbuf);

	json_t *header = json_pack("{s:s, s:I, s:I, s:i, s:i}", "type",
	                           dstm_enum_to_string(msg->header.type,
	                                               dstm_msg_type_table),
	                           "time", (json_int_t)msg->header.time,
	                           "delta", (json_int_t)msg->header.delta,
	                           "dst_rank", msg->header.dst_rank,
	                           "src_rank", msg->header.src_rank);

	json_t *obj = json_object();
	json_object_set_new(obj, "header", header);

	switch (msg->header.type) {
	case DSTM_MSG_TYPE_PUT:
		json_object_set_new(
		        obj, "put",
		        json_pack("{s:I, s:i, s:I}", "dst",
		                  (json_int_t)msg->u.putget.ptr.offset, "region",
		                  msg->u.putget.ptr.region, "size",
		                  (json_int_t)msg->u.putget.size));
		break;
	case DSTM_MSG_TYPE_GET_REQ:
		json_object_set_new(
		        obj, "getreq",
		        json_pack("{s:I, s:i, s:I}", "ptr",
		                  (json_int_t)msg->u.putget.ptr.offset, "region",
		                  msg->u.putget.ptr.region, "size",
		                  (json_int_t)msg->u.putget.size));
		break;
	case DSTM_MSG_TYPE_GET_REP:
		json_object_set_new(
		        obj, "getrep",
		        json_pack("{s:I, s:i, s:I}", "ptr",
		                  (json_int_t)msg->u.putget.ptr.offset, "region",
		                  msg->u.putget.ptr.region, "size",
		                  (json_int_t)msg->u.putget.size));
		break;
	case DSTM_MSG_TYPE_HEARTBEAT:
		json_object_set_new(obj, "heartbeat", json_object());
		break;
	case DSTM_MSG_TYPE_BARRIER:
		json_object_set_new(obj, "barrier",
		                    json_pack("{s:I}", "last_time",
		                             (json_int_t)msg->u.barrier
		                                     .last_time));
		break;
	default:
		json_object_set_new(obj, "type", json_string("unknown"));
		break;
	}

	json_object_set_new(obj, "data", json_string(data_buf));
	json_object_set_new(obj, "packbuf", json_string(pack_buf));

	return obj;
}
