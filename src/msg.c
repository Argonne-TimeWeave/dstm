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
#include "internal/region.h"
#include "internal/globals.h"
#include "internal/utlist.h"
#include "internal/msg.h"
#include <mpi.h>
#include <stdlib.h>

MPI_Datatype dstm_msg_hdr_dt;
MPI_Datatype dstm_msg_put_dt;
MPI_Datatype dstm_msg_get_dt;

/*******************************************************************************
 * MPI Datatypes management
 ******************************************************************************/

/* need to create the datatypes at init */
int dstm_msg_init(void)
{
	MPI_Datatype dstm_msg_hdr_sig[5] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT}; 
	int dstm_msg_hdr_bl[5] = {1,1,1,1,1};
	MPI_Aint dstm_msg_hdr_disp[5] = {
		offsetof(dstm_msg_hdr_t, type),
		offsetof(dstm_msg_hdr_t, dst_rank),
		offsetof(dstm_msg_hdr_t, src_rank),
		offsetof(dstm_msg_hdr_t, time),
		offsetof(dstm_msg_hdr_t, delta)};
	MPI_Type_create_struct(5, dstm_msg_hdr_bl, dstm_msg_hdr_disp, dstm_msg_hdr_sig,
			       &dstm_msg_hdr_dt);
	return 0;
}

int dstm_msg_finalize(void)
{
	MPI_Type_free(&dstm_msg_hdr_dt);
}

/*******************************************************************************
 * Message creation/filling
 ******************************************************************************/

int dstm_msg_fill_hdr(dstm_msg_t *m, int type, dstm_time_t time, dstm_time_t delta, int src_rank, int dst_rank)
{
	assert(type < DSTM_MSG_TYPE_MAX);
	msg->header.type = DSTM_MSG_TYPE_PUT;
	msg->header.dst_rank = dst_rank;
	msg->header.src_rank = src_rank;
	msg->header.time = time;
	msg->header.delta = delta;
	return 0;
}

int dstm_msg_fill_put(dstm_msg_t *m, intptr_t dst, intptr_t src, size_t size)
{
	assert(src != NULL);
	msg->u.put.dst = dst;
	msg->u.put.size = size;

	/* TODO: should work, but needs to be tested */
	msg->data = src;
	return 0;
}

int dstm_msg_fill_getreq(dstm_msg_t *m, intptr_t src, size_t size)
{
	assert(src != NULL);
	msg->u.get.ptr = src;
	msg->u.get.size = size;

	msg->data = NULL;
	return 0;
}

int dstm_msg_fill_getrep(dstm_msg_t *m, intptr_t dst, size_t size)
{
	assert(src != NULL);
	msg->u.get.ptr = dst;
	msg->u.get.size = size;

	/* TODO: should work, but needs to be tested */
	msg->data = dst;
	return 0;
}

/*******************************************************************************
 * Sending Messages
 ******************************************************************************/

int dstm_msg_isend_put(dstm_msg_t *msg)
{
	assert(msg != NULL);

	/* deal with copying the data and packing it into a buffer */
	int hsz, psz, dsz;
	MPI_Pack_size(1, dstm_msg_hdr_dt, dstm_global_context.comm, &hsz);
	MPI_Pack_size(1, dstm_msg_put_dt, dstm_global_context.comm, &psz);
	MPI_Pack_size(msg->u.put.size, MPI_CHAR, dstm_global_context.comm, &dsz);
	size_t bufsz = hsz + psz + dsz;

	msg->packbuf = calloc(bufsz, sizeof(char));
	if (msg->packbuf == NULL)
		return -DSTM_ENOMEM;

	int pos = 0;
	MPI_Pack(&msg->header, 1, dstm_msg_hdr_dt, msg->packbuf, bufsz, &pos,
		 dstm_global_context.comm);
	MPI_Pack(&msg->u.put, 1, dstm_msg_put_dt, msg->packbuf, bufsz, &pos,
		 dstm_global_context.comm);
	MPI_Pack(&msg->data, msg->u.put.size, MPI_CHAR, msg->packbuf, bufsz, &pos,
		 dstm_global_context.comm);

	/* send the message */
	MPI_Isend(msg->packbuf, pos, MPI_PACKED, msg->header->dst_rank, MPI_TAG_ANY,
		  dstm_global_context.comm, &msg->request);
	return 0;
}

int dstm_msg_isend(dstm_msg_t *msg)
{
	assert(msg != NULL);
	switch(msg->header.type) {
		case DSTM_MSG_TYPE_PUT:
			return dstm_msg_isend_put(msg);
		default:
			assert(0);
	}
	return 0;
}

int dstm_msg_send_getreq(dstm_msg_t *msg)
{
	/* deal with copying the data and packing it into a buffer */
	int hsz, gsz;
	MPI_Pack_size(1, dstm_msg_hdr_dt, dstm_global_context.comm, &hsz);
	MPI_Pack_size(1, dstm_msg_get_dt, dstm_global_context.comm, &gsz);
	size_t bufsz = hsz + gsz;

	msg->packbuf = calloc(bufsz, sizeof(char));
	if (msg->packbuf == NULL)
		return -DSTM_ENOMEM;

	int pos = 0;
	MPI_Pack(&msg->header, 1, dstm_msg_hdr_dt, msg->packbuf, bufsz, &pos,
		 dstm_global_context.comm);
	MPI_Pack(&msg->u.get, 1, dstm_msg_get_dt, msg->packbuf, bufsz, &pos,
		 dstm_global_context.comm);

	/* send the message */
	MPI_Send(msg->packbuf, pos, MPI_PACKED, msg->header->dst_rank, MPI_TAG_ANY,
		  dstm_global_context.comm);
	
	free(msg->packbuf);
	return 0;
}

int dstm_msg_send(dstm_msg_t *msg)
{
	assert(msg != NULL);
	switch(msg->header.type) {
		case DSTM_MSG_TYPE_GET_REQ:
			return dstm_msg_send_getreq(msg);
		default:
			assert(0);
	}
	return 0;
}

/*******************************************************************************
 * Receiving Messages
 ******************************************************************************/

int dstm_msg_irecv_getrep(dstm_msg_t *msg)
{
	/* prepare a receive buffer */
	int hsz, gsz, dsz;
	MPI_Pack_size(1, dstm_msg_hdr_dt, dstm_global_context.comm, &hsz);
	MPI_Pack_size(1, dstm_msg_get_dt, dstm_global_context.comm, &gsz);
	MPI_Pack_size(msg->u.get.size, MPI_CHAR, dstm_global_context.comm, &dsz);
	size_t bufsz = hsz + gsz + dsz;

	msg->packbuf = calloc(bufsz, sizeof(char));
	if (msg->packbuf == NULL)
		return -DSTM_ENOMEM;

	/* receive the message */
	MPI_Irecv(msg->packbuf, bufsz, MPI_PACKED, msg->header->dst_rank, MPI_TAG_ANY,
		  dstm_global_context.comm, &msg->request);
	return 0;
}

int dstm_msg_irecv(dstm_msg_t *msg)
{
	assert(msg != NULL);
	switch(msg->header.type) {
		case DSTM_MSG_TYPE_GET_REP:
			return dstm_msg_irecv_getrep(msg);
		default:
			assert(0);
	}
	return 0;
}

int dstm_msg_recvany(dstm_msg_t **msg)
{

}

