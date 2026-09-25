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
        uint64_t global_calls[DSTM_PROFILE_MPI_CALL_MAX] = {0};
        uint64_t local_bytes[DSTM_PROFILE_MPI_CALL_MAX] = {0};
        uint64_t global_bytes[DSTM_PROFILE_MPI_CALL_MAX] = {0};
        uint64_t total_calls = 0;
        uint64_t total_bytes = 0;
        double local_runtime;
        double global_runtime = 0.0;

        if (!profile->active)
                return;

        local_runtime = MPI_Wtime() - profile->start_time;

        /*
         * Disable collection before aggregation so the profiling reductions
         * do not count themselves.
         */
        profile->active = 0;

        for (int i = 0; i < DSTM_PROFILE_MPI_CALL_MAX; i++) {
                local_calls[i] = profile->mpi_calls[i];
                local_bytes[i] = profile->mpi_bytes[i];
        }

        MPI_Reduce(local_calls, global_calls, DSTM_PROFILE_MPI_CALL_MAX,
                   MPI_UINT64_T, MPI_SUM, 0, comm);
        MPI_Reduce(local_bytes, global_bytes, DSTM_PROFILE_MPI_CALL_MAX,
                   MPI_UINT64_T, MPI_SUM, 0, comm);
        MPI_Reduce(&local_runtime, &global_runtime, 1, MPI_DOUBLE,
                   MPI_MAX, 0, comm);

        if (rank != 0)
                return;

        for (int i = 0; i < DSTM_PROFILE_MPI_CALL_MAX; i++) {
                total_calls += global_calls[i];
                total_bytes += global_bytes[i];
        }

        dstm_log_normal(
                "DSTM_PROFILE "
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
                global_runtime,
                total_calls,
                global_calls[DSTM_PROFILE_MPI_SEND],
                global_calls[DSTM_PROFILE_MPI_ISEND],
                global_calls[DSTM_PROFILE_MPI_RECV],
                global_calls[DSTM_PROFILE_MPI_IRECV],
                global_calls[DSTM_PROFILE_MPI_ALLGATHER],
                global_calls[DSTM_PROFILE_MPI_ALLREDUCE],
                global_calls[DSTM_PROFILE_MPI_BARRIER],
                total_bytes,
                global_bytes[DSTM_PROFILE_MPI_SEND],
                global_bytes[DSTM_PROFILE_MPI_ISEND],
                global_bytes[DSTM_PROFILE_MPI_RECV],
                global_bytes[DSTM_PROFILE_MPI_IRECV],
                global_bytes[DSTM_PROFILE_MPI_ALLGATHER],
                global_bytes[DSTM_PROFILE_MPI_ALLREDUCE],
                global_bytes[DSTM_PROFILE_MPI_BARRIER]);
}
