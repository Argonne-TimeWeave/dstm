/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

/*
 * MPI messaging roundtrip tests.
 * Must be run with exactly 2 MPI ranks: mpirun -n 2 ./msg
 *
 * Rank 0 acts as sender/requester; rank 1 acts as receiver/responder.
 *
 * Tests:
 *   1. dstm_msg_init / dstm_msg_finalize - MPI datatype lifecycle
 *   2. PUT roundtrip - rank 0 sends (isend), rank 1 receives (recvany)
 *   3. GET_REQ / GET_REP roundtrip - rank 0 sends request (send, blocking),
 *      rank 1 receives and replies (isend); rank 0 posts irecv and polls
 *      with dstm_msg_test until the reply is complete, then unpacks it
 */

#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dstm.h"

#include "internal/globals.h"
#include "internal/msg.h"

/* These MPI datatypes are defined in src/msg.c and initialized by
 * dstm_msg_init.  We declare them extern here so we can use them
 * directly for unpacking in the test. */
extern MPI_Datatype dstm_msg_hdr_dt;
extern MPI_Datatype dstm_msg_get_dt;

#define SENDER 0
#define RECEIVER 1

/* -------------------------------------------------------------------------
 * Test 1: MPI datatype lifecycle
 * Both ranks call dstm_init and dstm_finalize; verify they succeed without
 * crashing. dstm_init sets up the MPI communicator and calls dstm_msg_init
 * internally, so this exercises the full datatype creation path.
 * ------------------------------------------------------------------------- */
static void test_init_finalize(int argc, char *argv[])
{
	int rc;

	rc = dstm_init(&argc, &argv);
	assert(rc == DSTM_SUCCESS);

	rc = dstm_finalize();
	assert(rc == DSTM_SUCCESS);

	/* Re-initialize for subsequent tests */
	rc = dstm_init(&argc, &argv);
	assert(rc == DSTM_SUCCESS);
}

/* -------------------------------------------------------------------------
 * Test 2: PUT roundtrip
 *
 * Rank 0 fills a PUT message with a 16-byte payload and sends it via
 * dstm_msg_isend.  Rank 1 receives it via dstm_msg_recvany and verifies
 * the header fields, put-body fields, and data bytes.
 * ------------------------------------------------------------------------- */
static void test_put_roundtrip(int rank)
{
	static const size_t DATA_SZ = 16;
	static const int REGION = 7;
	static const int OFFSET = 512;

	MPI_Status status;

	if (rank == SENDER) {
		char src[16];
		for (size_t i = 0; i < DATA_SZ; i++)
			src[i] = (char)(i + 1);

		dstm_msg_t msg;
		memset(&msg, 0, sizeof(msg));

		dstm_ptr_t dst = {.region = REGION, .offset = OFFSET};

		int rc;
		rc = dstm_msg_fill_hdr(&msg, DSTM_MSG_TYPE_PUT, 10, 2, SENDER,
		                       RECEIVER);
		assert(rc == 0);
		rc = dstm_msg_fill_put(&msg, dst, src, DATA_SZ);
		assert(rc == 0);
		rc = dstm_msg_isend(&msg);
		assert(rc == 0);

		MPI_Wait(&msg.request, &status);
		free(msg.packbuf);

	} else { /* RECEIVER */
		dstm_msg_t msg;
		memset(&msg, 0, sizeof(msg));

		MPI_Status status;
		int rc = 0;
		do {
			rc = dstm_msg_iprobe(&status);
		} while (!rc);
		rc = dstm_msg_recvany(status, &msg);
		assert(rc == 0);

		/* Verify header */
		assert(msg.header.type == DSTM_MSG_TYPE_PUT);
		assert(msg.header.src_rank == SENDER);
		assert(msg.header.dst_rank == RECEIVER);
		assert(msg.header.time == 10);
		assert(msg.header.delta == 2);

		/* Verify PUT body */
		assert(msg.u.put.region == REGION);
		assert(msg.u.put.dst == OFFSET);
		assert(msg.u.put.size == DATA_SZ);

		/* Verify payload bytes */
		assert(msg.data != NULL);
		char *recv_data = (char *)msg.data;
		for (size_t i = 0; i < DATA_SZ; i++)
			assert(recv_data[i] == (char)(i + 1));

		free(msg.packbuf);
	}

	MPI_Barrier(MPI_COMM_WORLD);
}

/* -------------------------------------------------------------------------
 * Test 3: GET_REQ / GET_REP roundtrip
 *
 * Rank 0 sends a GET_REQ via dstm_msg_send (blocking), then posts a
 * non-blocking receive for the reply via dstm_msg_irecv, and waits for
 * it with MPI_Wait.  It then unpacks the buffer and checks header and
 * data bytes.  This also exercises dstm_msg_test by polling it once
 * before the MPI_Wait.
 *
 * Rank 1 receives the GET_REQ via dstm_msg_recvany, builds a GET_REP
 * with known data, and sends it via dstm_msg_isend + MPI_Wait.
 * ------------------------------------------------------------------------- */
static void test_get_roundtrip(int rank)
{
	static const size_t DATA_SZ = 24;
	static const int REGION = 3;
	static const int PTR_OFF = 1024;

	MPI_Status status;

	if (rank == SENDER) {
		/* --- Phase A: send GET_REQ (blocking) --- */
		dstm_msg_t req;
		memset(&req, 0, sizeof(req));

		dstm_ptr_t src = {.region = REGION, .offset = PTR_OFF};

		int rc;
		rc = dstm_msg_fill_hdr(&req, DSTM_MSG_TYPE_GET_REQ, 20, 1,
		                       SENDER, RECEIVER);
		assert(rc == 0);
		rc = dstm_msg_fill_getreq(&req, src, DATA_SZ);
		assert(rc == 0);
		rc = dstm_msg_send(&req);
		assert(rc == 0);

		/* packbuf freed inside dstm_msg_send_getreq */

		/* --- Phase B: post non-blocking receive for the reply ---
		 *
		 * dstm_msg_irecv_getrep sizes its receive buffer from
		 * msg.u.get.size and posts from msg.header.src_rank.
		 */
		dstm_msg_t rep;
		memset(&rep, 0, sizeof(rep));
		rep.header.type = DSTM_MSG_TYPE_GET_REP;
		rep.header.src_rank = RECEIVER; /* expect reply from RECEIVER */
		rep.u.get.size = DATA_SZ;

		rc = dstm_msg_irecv(&rep);
		assert(rc == 0);

		/* Exercise dstm_msg_test: call it at least once, then wait */
		dstm_msg_test(rep);
		MPI_Wait(&rep.request, &status);

		/* Manually unpack the received buffer to verify contents.
		 * The buffer contains: packed hdr | packed get body | data.
		 * Use dstm_global_context.comm — the same comm used for Pack.
		 */
		int hsz, gsz;
		MPI_Pack_size(1, dstm_msg_hdr_dt, dstm_global_context.comm,
		              &hsz);
		MPI_Pack_size(1, dstm_msg_get_dt, dstm_global_context.comm,
		              &gsz);
		int total_bufsz = hsz + gsz + (int)DATA_SZ;

		dstm_msg_hdr_t hdr;
		dstm_msg_get_t get;
		int pos = 0;
		MPI_Unpack(rep.packbuf, total_bufsz, &pos, &hdr, 1,
		           dstm_msg_hdr_dt, dstm_global_context.comm);
		MPI_Unpack(rep.packbuf, total_bufsz, &pos, &get, 1,
		           dstm_msg_get_dt, dstm_global_context.comm);

		assert(hdr.type == DSTM_MSG_TYPE_GET_REP);
		assert(hdr.src_rank == RECEIVER);
		assert(hdr.dst_rank == SENDER);
		assert(get.size == DATA_SZ);

		/* Data follows immediately after the unpacked get body */
		char *recv_data = (char *)rep.packbuf + pos;
		for (size_t i = 0; i < DATA_SZ; i++)
			assert(recv_data[i] == (char)(i + 10));

		free(rep.packbuf);

	} else { /* RECEIVER */
		/* --- Phase A: receive the GET_REQ --- */
		dstm_msg_t req;
		memset(&req, 0, sizeof(req));
		MPI_Status status;

		int rc = 0;
		do {
			rc = dstm_msg_iprobe(&status);
		} while (!rc);
		rc = dstm_msg_recvany(status, &req);
		assert(rc == 0);

		assert(req.header.type == DSTM_MSG_TYPE_GET_REQ);
		assert(req.header.src_rank == SENDER);
		assert(req.header.dst_rank == RECEIVER);
		assert(req.header.time == 20);
		assert(req.header.delta == 1);
		assert(req.u.get.region == REGION);
		assert(req.u.get.ptr == PTR_OFF);
		assert(req.u.get.size == DATA_SZ);

		free(req.packbuf);

		/* --- Phase B: build and send GET_REP --- */
		char reply_data[24];
		for (size_t i = 0; i < DATA_SZ; i++)
			reply_data[i] = (char)(i + 10);

		dstm_msg_t rep;
		memset(&rep, 0, sizeof(rep));

		rc = dstm_msg_fill_hdr(&rep, DSTM_MSG_TYPE_GET_REP, 20, 1,
		                       RECEIVER, SENDER);
		assert(rc == 0);
		rc = dstm_msg_fill_getrep(&rep, reply_data, DATA_SZ);
		assert(rc == 0);
		rc = dstm_msg_isend(&rep);
		assert(rc == 0);

		MPI_Wait(&rep.request, &status);
		free(rep.packbuf);
	}

	MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size != 2) {
		if (rank == 0)
			fprintf(stderr,
			        "error: this test requires exactly 2 MPI "
			        "ranks (got %d)\n",
			        size);
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	/* Test 1: datatype lifecycle */
	test_init_finalize(argc, argv);
	if (rank == 0)
		printf("PASS: init/finalize\n");

	/* Test 2: PUT roundtrip */
	test_put_roundtrip(rank);
	if (rank == 0)
		printf("PASS: PUT roundtrip\n");

	/* Test 3: GET_REQ / GET_REP roundtrip with dstm_msg_test polling */
	test_get_roundtrip(rank);
	if (rank == 0)
		printf("PASS: GET roundtrip\n");

	dstm_finalize();
	MPI_Finalize();
	return EXIT_SUCCESS;
}
