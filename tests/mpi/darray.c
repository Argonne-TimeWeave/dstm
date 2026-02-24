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
	dstm_darray_t *a = NULL;
	dstm_darray_create(&a, sizeof(int), 20);
	assert(a != NULL);
	dstm_darray_destroy(&a);
	
	dstm_fini();
	dstm_finalize();
	MPI_Finalize();
	return 0;
}
