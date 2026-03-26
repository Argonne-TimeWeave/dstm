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
extern MPI_Datatype dstm_msg_putget_dt;

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
		/* fill_payload sets msg.data to src without copying; the
		 * caller is responsible for keeping src alive until the send
		 * completes. */
		rc = dstm_msg_fill_payload(&msg, dst, src, DATA_SZ);
		assert(rc == 0);
		rc = dstm_msg_isend(&msg, 0);
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
		assert(msg.u.putget.ptr.region == REGION);
		assert(msg.u.putget.ptr.offset == OFFSET);
		assert(msg.u.putget.size == DATA_SZ);

		/* Verify payload bytes */
		assert(msg.data != NULL);
		char *recv_data = (char *)msg.data;
		for (size_t i = 0; i < DATA_SZ; i++)
			assert(recv_data[i] == (char)(i + 1));

		free(msg.packbuf);
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

	dstm_finalize();
	MPI_Finalize();
	return EXIT_SUCCESS;
}
