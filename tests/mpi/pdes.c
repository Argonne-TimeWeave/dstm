/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <stdlib.h>
#include <mpi.h>
#include "dstm.h"

int main(int argc, char *argv[])
{
	dstm_distance_t dist;
	MPI_Init(&argc, &argv);
	dstm_init(&argc, &argv);
	

	int world_size;
	int rank;

	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	/* example only for 4 ranks:
	 *               0
	 *      1               2
	 *                          4
	 *
	 * */
	assert(world_size == 4);
	int distances[16] = { 0, 1, 1, 0, \
		              1, 0, 0, 0, \
			      1, 0, 0, 1, \
			      0, 0, 1, 0};
	dist.distances = (int *)&distances[0];
	dist.size = 4;
	dstm_start(dist);

	/* memory allocation:
	 * out queues, one per rank
	 * messages are ints
	 */
#define MAX_MESSAGES 20
	dstm_region_t *myregion = dstm_get_local_region();
	intptr_t messages = dstm_region_malloc(myregion, sizeof(int) * 4 *
					   MAX_MESSAGES);

	/* we should be awarded the first slot in the region */
	assert(messages == 0);
	/* loop:
	 *
	 * 0 is going to send a message every step to 1 and 2 for 20 steps.
	 * 1 is going to only respond to each message it gets.
	 * 2 is going to forward messages from 0 to 4 after 5 steps
	 * 4 is going to receive messages and do nothing
	 * */
	int msg;
	int ret;
	if (rank == 0) {
		msg = rank+1;
		dstm_put(rank, 0, 0, 1, MAX_MESSAGES, &msg, sizeof(int));
		msg = rank+2;
		dstm_put(rank, 0, 0, 1, MAX_MESSAGES*2, &msg, sizeof(int));
	}
	for(int step = 1; step < 25; step++) {
		if (rank == 0 && step < 20) {
			dstm_get(1, 0, step, -1, &ret, 0, sizeof(int));
			fprintf(stderr, "%d: %d: %d\n", rank, step, ret);
		}
		else if (rank == 1) {
			dstm_get(0, 0, step, -1, &ret, MAX_MESSAGES,
				 sizeof(int));
			fprintf(stderr, "%d: %d: %d\n", rank, step, ret);
			msg = ret + rank*step;
			dstm_put(rank, 0, step, 1, 0, &msg, sizeof(int));
		}
		else if(rank == 2) {
			dstm_get(0, 0, step, -1, &ret, MAX_MESSAGES*2,
				 sizeof(int));
			fprintf(stderr, "%d: %d: %d\n", rank, step, ret);
			msg = ret + rank*step;
			dstm_put(rank, 0, step, 5, MAX_MESSAGES*3, &msg, sizeof(int));
		}
		else if(rank == 3) {
			dstm_get(2, 0, step, -1, &ret, MAX_MESSAGES*3,
				 sizeof(int));
			
			fprintf(stderr, "%d: %d: %d\n", rank, step, ret);
		}
	}
	dstm_region_free(myregion, messages);
	dstm_fini();
	dstm_finalize();
	MPI_Finalize();
	return 0;
}
