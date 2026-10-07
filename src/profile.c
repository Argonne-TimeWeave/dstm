/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

#include "config.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "dstm.h"

#include "internal/globals.h"
#include "internal/profile.h"

void dstm_profile_start(void)
{
        memset(&dstm_global_context.profile, 0,
               sizeof(dstm_global_context.profile));
        dstm_global_context.profile.start_time = MPI_Wtime();
        dstm_global_context.profile.active = 1;
}

void dstm_profile_cancel(void)
{
        dstm_global_context.profile.active = 0;
}

void dstm_profile_record(dstm_profile_mpi_call_t call, size_t bytes)
{
        dstm_profile_t *profile = &dstm_global_context.profile;

        if (!profile->active)
                return;

        if (call < 0 || call >= DSTM_PROFILE_MPI_CALL_MAX)
                return;

        profile->mpi_calls[call]++;
        profile->mpi_bytes[call] += (uint64_t)bytes;
}

void dstm_profile_record_collective(dstm_profile_mpi_call_t call,
                                    size_t local_payload_bytes)
{
        uint64_t bytes = 0;

        if (dstm_global_context.world_size > 1) {
                bytes = (uint64_t)local_payload_bytes *
                        (uint64_t)(dstm_global_context.world_size - 1);
        }

        dstm_profile_record(call, (size_t)bytes);
}

void dstm_profile_finish_and_log(MPI_Comm comm, int rank)
{
        dstm_profile_t *profile = &dstm_global_context.profile;
        uint64_t local_calls[DSTM_PROFILE_MPI_CALL_MAX] = {0};
        uint64_t local_bytes[DSTM_PROFILE_MPI_CALL_MAX] = {0};

        uint64_t *rank_calls = NULL;
        uint64_t *rank_bytes = NULL;
        double *rank_runtimes = NULL;

        double local_runtime;
        int world_size;
        int allocation_ok = 1;

        if (!profile->active)
                return;

        local_runtime = MPI_Wtime() - profile->start_time;

        /*
         * Disable collection before gathering results so the profiling
         * collectives do not count themselves.
         */
        profile->active = 0;

        for (int i = 0; i < DSTM_PROFILE_MPI_CALL_MAX; i++) {
                local_calls[i] = profile->mpi_calls[i];
                local_bytes[i] = profile->mpi_bytes[i];
        }

        MPI_Comm_size(comm, &world_size);

        /*
         * Rank 0 owns the gathered profiling data. Each array is laid out
         * as:
         *
         *   rank_calls[rank * DSTM_PROFILE_MPI_CALL_MAX + call]
         *   rank_bytes[rank * DSTM_PROFILE_MPI_CALL_MAX + call]
         *   rank_runtimes[rank]
         */
        if (rank == 0) {
                size_t counter_count =
                        (size_t)world_size * DSTM_PROFILE_MPI_CALL_MAX;

                rank_calls = calloc(counter_count, sizeof(*rank_calls));
                rank_bytes = calloc(counter_count, sizeof(*rank_bytes));
                rank_runtimes =
                        calloc((size_t)world_size, sizeof(*rank_runtimes));

                if (rank_calls == NULL ||
                    rank_bytes == NULL ||
                    rank_runtimes == NULL)
                        allocation_ok = 0;
        }

        /*
         * All ranks must agree whether rank 0 successfully allocated the
         * receive buffers before entering MPI_Gather. Profiling is already
         * disabled, so this broadcast is intentionally excluded.
         */
        MPI_Bcast(&allocation_ok, 1, MPI_INT, 0, comm);

        if (!allocation_ok) {
                if (rank == 0) {
                        dstm_log_normal(
                                "DSTM_PROFILE "
                                "status=unavailable "
                                "reason=out_of_memory\n");
                }

                free(rank_calls);
                free(rank_bytes);
                free(rank_runtimes);
                return;
        }

        MPI_Gather(local_calls, DSTM_PROFILE_MPI_CALL_MAX, MPI_UINT64_T,
                   rank_calls, DSTM_PROFILE_MPI_CALL_MAX, MPI_UINT64_T,
                   0, comm);

        MPI_Gather(local_bytes, DSTM_PROFILE_MPI_CALL_MAX, MPI_UINT64_T,
                   rank_bytes, DSTM_PROFILE_MPI_CALL_MAX, MPI_UINT64_T,
                   0, comm);

        MPI_Gather(&local_runtime, 1, MPI_DOUBLE,
                   rank_runtimes, 1, MPI_DOUBLE,
                   0, comm);

        if (rank == 0) {
                for (int r = 0; r < world_size; r++) {
                        const uint64_t *calls =
                                rank_calls +
                                (size_t)r * DSTM_PROFILE_MPI_CALL_MAX;
                        const uint64_t *bytes =
                                rank_bytes +
                                (size_t)r * DSTM_PROFILE_MPI_CALL_MAX;

                        uint64_t total_calls = 0;
                        uint64_t total_bytes = 0;

                        for (int i = 0;
                             i < DSTM_PROFILE_MPI_CALL_MAX;
                             i++) {
                                total_calls += calls[i];
                                total_bytes += bytes[i];
                        }

                        dstm_log_normal(
                                "DSTM_PROFILE "
                                "scope=rank "
                                "rank=%d "
                                "runtime_wall_seconds=%.9f "
                                "mpi_calls_total=%" PRIu64 " "
                                "mpi_send_calls=%" PRIu64 " "
                                "mpi_isend_calls=%" PRIu64 " "
                                "mpi_recv_calls=%" PRIu64 " "
                                "mpi_irecv_calls=%" PRIu64 " "
                                "mpi_allgather_calls=%" PRIu64 " "
                                "mpi_allreduce_calls=%" PRIu64 " "
                                "mpi_barrier_calls=%" PRIu64 " "
                                "bytes_moved_total=%" PRIu64 " "
                                "mpi_send_bytes=%" PRIu64 " "
                                "mpi_isend_bytes=%" PRIu64 " "
                                "mpi_recv_bytes=%" PRIu64 " "
                                "mpi_irecv_bytes=%" PRIu64 " "
                                "mpi_allgather_bytes=%" PRIu64 " "
                                "mpi_allreduce_bytes=%" PRIu64 " "
                                "mpi_barrier_bytes=%" PRIu64 "\n",
                                r,
                                rank_runtimes[r],
                                total_calls,
                                calls[DSTM_PROFILE_MPI_SEND],
                                calls[DSTM_PROFILE_MPI_ISEND],
                                calls[DSTM_PROFILE_MPI_RECV],
                                calls[DSTM_PROFILE_MPI_IRECV],
                                calls[DSTM_PROFILE_MPI_ALLGATHER],
                                calls[DSTM_PROFILE_MPI_ALLREDUCE],
                                calls[DSTM_PROFILE_MPI_BARRIER],
                                total_bytes,
                                bytes[DSTM_PROFILE_MPI_SEND],
                                bytes[DSTM_PROFILE_MPI_ISEND],
                                bytes[DSTM_PROFILE_MPI_RECV],
                                bytes[DSTM_PROFILE_MPI_IRECV],
                                bytes[DSTM_PROFILE_MPI_ALLGATHER],
                                bytes[DSTM_PROFILE_MPI_ALLREDUCE],
                                bytes[DSTM_PROFILE_MPI_BARRIER]);
                }
        }

        free(rank_calls);
        free(rank_bytes);
        free(rank_runtimes);
}
