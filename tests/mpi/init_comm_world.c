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
 * Verify that dstm_init_comm with MPI_COMM_WORLD behaves identically
 * to dstm_init: same rank, same world_size, session starts cleanly.
 */

#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "dstm.h"

int main(int argc, char *argv[])
{
        MPI_Init(&argc, &argv);

        int world_rank, world_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        int rc = dstm_init_comm(&argc, &argv, MPI_COMM_WORLD);
        assert(rc == DSTM_SUCCESS);
        /* Region id is now an intra-rank concept: the default region is
         * always id 0, independent of rank. */
        assert(dstm_get_local_region_id() == 0);

        /* Run a trivial session. */
        dstm_distance_t *dist = NULL;
        rc = dstm_distance_create_complete(&dist, (size_t)world_size);
        assert(rc == DSTM_SUCCESS);

        rc = dstm_start(dist);
        assert(rc == DSTM_SUCCESS);

        rc = dstm_fini(0);
        assert(rc == DSTM_SUCCESS);

        dstm_distance_destroy(&dist);
        rc = dstm_finalize();
        assert(rc == DSTM_SUCCESS);

        if (world_rank == 0)
                printf("PASS: init_comm with MPI_COMM_WORLD\n");

        MPI_Finalize();
        return EXIT_SUCCESS;
}
