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

MPI_Datatype dstm_msg_dt;

/*******************************************************************************
 * Basic msg  management
 ******************************************************************************/

/* need to create the datatypes at init */
int dstm_msg_init(void)
{
	MPI_Datatype dstm_msg_sig[6] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT,
		MPI_INT, MPI_INT};
	int dstm_msg_bl[1] = {1,1,1,1,1,1};
	MPI_Aint dstm_msg_disp[1] = {0};
	MPI_Type_create_struct(1, dstm_msg_bl, dstm_msg_disp, dstm_msg_sig,
			       &dstm_msg_dt);
	return 0;
}

int dstm_msg_finalize(void)
{
	MPI_Type_free(&dstm_msg_dt);
}

int dstm_msg_create_put(dstm_msg_t **msg, dstm_time_t time, dstm_time_t delta, int rank,
		     intptr_t dst, intptr_t src, size_t size)
{
	assert(type < DSTM_REQUEST_TYPE_MAX);
	dstm_msg_t *msg = calloc(1, sizeof(dstm_msg_t));
	if (msg == NULL)
		return -DSTM_ENOMEM;

	dstm_msg_put_t *put = calloc(1, sizeof(dstm_msg_put_t));
	if (put == NULL)
		goto error;

	msg->type = DSTM_MSG_TYPE_PUT;
	msg->header = (void *)put;
	
	put->time = time;
	put->delta = delta;
	put->rank = rank;
	put->dst = dst;
	put->size = size;

	return 0;
error:
	free(msg);
	return -DSTM_ENOMEM;
}

int dstm_msg_create_get(dstm_msg_t **msg, dstm_time_t time, dstm_time_t delta, int rank,
		     intptr_t src, size_t size)
{
	assert(type < DSTM_REQUEST_TYPE_MAX);
	dstm_msg_t *msg = calloc(1, sizeof(dstm_msg_t));
	if (msg == NULL)
		return -DSTM_ENOMEM;

	dstm_msg_get_t *get = calloc(1, sizeof(dstm_msg_get_t));
	if (get == NULL)
		goto error;

	msg->type = DSTM_MSG_TYPE_GET;
	msg->p = (void *)get;
	
	get->time = time;
	get->delta = delta;
	get->rank = rank;
	get->src = src;
	get->size = size;

	return 0;
error:
	free(msg);
	return -DSTM_ENOMEM;
}

void dstm_msg_destroy(dstm_msg_t **msg)
{
	if(msg == NULL || *msg == NULL)
		return;

	if (msg->p != NULL)
		free(msg->p);
	free(msg);
	*msg = NULL;
}

/*******************************************************************************
 * Networking
 ******************************************************************************/

int dstm_msg_isend(dstm_msg_t *msg, int dst)
{
	/* TODO: doesn't work, need to use MPI_Pack to pack both headers and
	 * data into local buffer
	 */
	assert(msg != NULL);
	MPI_Isend(msg->p, 1, dstm_msg_dt, dst, /*TODO*/ MPI_TAG_ANY,
		  dstm_global_context.comm, &msg->request);
	return 0;
}

int dstm_msg_send(dstm_msg_t **msg, int dst, int src);
int dstm_msg_irecv(int *req, int dst, void *buf, size_t size);
int dstm_msg_recvany(dstm_msg_t **msg);


