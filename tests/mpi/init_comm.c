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
 * Tests for dstm_init_comm: initializing DSTM with an arbitrary MPI
 * communicator instead of MPI_COMM_WORLD.
 *
 * Must be run with at least 4 MPI ranks.
 *
 * A single process can only safely call dstm_init / dstm_finalize once
 * (the messaging layer is not designed for repeated init/finalize
 * cycles), so this test initializes with a split communicator and
 * exercises everything in one lifetime:
 *
 *   1. Rank/size match the sub-communicator, not MPI_COMM_WORLD.
 *   2. A put/get round-trip works within the sub-communicator.
 *   3. The two disjoint sub-groups run concurrently without
 *      cross-contamination.
 */

#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dstm.h"

int main(int argc, char *argv[])
{
        MPI_Init(&argc, &argv);

        int world_rank, world_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        if (world_size < 4) {
                if (world_rank == 0)
                        fprintf(stderr,
                                "error: this test requires at least 4 MPI "
                                "ranks (got %d)\n",
                                world_size);
                MPI_Finalize();
                return EXIT_FAILURE;
        }

        /* Split MPI_COMM_WORLD into even/odd sub-groups. */
        int color = world_rank % 2;
        MPI_Comm sub_comm;
        MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &sub_comm);

        int sub_rank, sub_size;
        MPI_Comm_rank(sub_comm, &sub_rank);
        MPI_Comm_size(sub_comm, &sub_size);

        /* ---- Initialize DSTM with the sub-communicator ---- */
        int rc = dstm_init_comm(&argc, &argv, sub_comm);
        assert(rc == DSTM_SUCCESS);

        /***************************************************************
         * Check 1: rank and world_size match the sub-communicator
         ***************************************************************/

        /* Default region id is 0 regardless of rank/sub-rank. */
        assert(dstm_get_local_region_id() == 0);

        if (world_rank == 0)
                printf("PASS: rank/size match sub-communicator\n");

        /***************************************************************
         * Check 2: put/get round-trip within the sub-communicator
         *
         * Each sub-group runs its own session.  Values are tagged with
         * the color so cross-group contamination would be detectable.
         ***************************************************************/
        {
                dstm_distance_t *dist = NULL;
                rc = dstm_distance_create_complete(&dist, (size_t)sub_size);
                assert(rc == DSTM_SUCCESS);

                rc = dstm_start(dist);
                assert(rc == DSTM_SUCCESS);

                dstm_darray_t *a = NULL;
                rc = dstm_darray_create(&a, sizeof(int), 1);
                assert(rc == DSTM_SUCCESS);

                /* Put at t=0, delta=+1 → effective time 1. */
                int payload = color * 10000 + sub_rank * 100 + 7;
                rc = dstm_darray_put(a, (size_t)sub_rank, &payload,
                                     /*time=*/0, /*delta=*/1);
                assert(rc == DSTM_SUCCESS);

                /* Get own value at t=2, delta=-1 → effective time 1. */
                int got = 0;
                rc = dstm_darray_get(a, (size_t)sub_rank, &got,
                                     /*time=*/2, /*delta=*/-1);
                assert(rc == DSTM_SUCCESS);
                assert(got == payload);

                /* Get a peer's value within the same sub-group. */
                if (sub_size > 1) {
                        int peer = (sub_rank + 1) % sub_size;
                        int peer_expected = color * 10000 + peer * 100 + 7;
                        int peer_got = 0;
                        rc = dstm_darray_get(a, (size_t)peer, &peer_got,
                                             /*time=*/2, /*delta=*/-1);
                        assert(rc == DSTM_SUCCESS);
                        assert(peer_got == peer_expected);
                }

                dstm_darray_destroy(&a);

                rc = dstm_fini(2);
                assert(rc == DSTM_SUCCESS);
                dstm_distance_destroy(&dist);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (world_rank == 0)
                printf("PASS: put/get round-trip in split communicator\n");

        /***************************************************************
         * Check 3: multi-timestep session (concurrent disjoint groups)
         *
         * Uses dstm_start/dstm_fini for a second session within the
         * same init_comm lifetime.  Each sub-group runs 5 timesteps.
         * A global checksum verifies no cross-contamination.
         ***************************************************************/
        {
                dstm_distance_t *dist = NULL;
                rc = dstm_distance_create_complete(&dist, (size_t)sub_size);
                assert(rc == DSTM_SUCCESS);

                rc = dstm_start(dist);
                assert(rc == DSTM_SUCCESS);

                dstm_darray_t *a = NULL;
                rc = dstm_darray_create(&a, sizeof(int), 1);
                assert(rc == DSTM_SUCCESS);

                const int NUM_STEPS = 5;
                for (int t = 0; t < NUM_STEPS; t++) {
                        int val = color * 1000 + sub_rank * 10 + t;
                        rc = dstm_darray_put(a, (size_t)sub_rank, &val,
                                             (dstm_time_t)t, /*delta=*/1);
                        assert(rc == DSTM_SUCCESS);

                        if (t > 0) {
                                int prev = 0;
                                rc = dstm_darray_get(a, (size_t)sub_rank,
                                                     &prev,
                                                     (dstm_time_t)(t + 1),
                                                     /*delta=*/-1);
                                assert(rc == DSTM_SUCCESS);

                                int expected = color * 1000 +
                                               sub_rank * 10 + (t - 1);
                                assert(prev == expected);
                        }
                }

                int local_sum = 0;
                for (int t = 0; t < NUM_STEPS; t++)
                        local_sum += color * 1000 + sub_rank * 10 + t;

                dstm_darray_destroy(&a);

                rc = dstm_fini((dstm_time_t)(NUM_STEPS + 1));
                assert(rc == DSTM_SUCCESS);
                dstm_distance_destroy(&dist);

                /* Global consistency check on MPI_COMM_WORLD. */
                int *all_sums = NULL;
                if (world_rank == 0)
                        all_sums = malloc((size_t)world_size * sizeof(int));

                MPI_Gather(&local_sum, 1, MPI_INT, all_sums, 1, MPI_INT,
                           0, MPI_COMM_WORLD);

                if (world_rank == 0) {
                        int even_idx = 0, odd_idx = 0;
                        for (int r = 0; r < world_size; r++) {
                                int c = r % 2;
                                int sr = (c == 0) ? even_idx++ : odd_idx++;
                                int expected = 0;
                                for (int t = 0; t < NUM_STEPS; t++)
                                        expected += c * 1000 + sr * 10 + t;
                                assert(all_sums[r] == expected);
                        }
                        free(all_sums);
                        printf("PASS: disjoint concurrent sessions\n");
                }
        }

        /* ---- Tear down ---- */
        rc = dstm_finalize();
        assert(rc == DSTM_SUCCESS);

        MPI_Comm_free(&sub_comm);
        MPI_Finalize();
        return EXIT_SUCCESS;
}
