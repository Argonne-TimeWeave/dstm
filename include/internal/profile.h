/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

#ifndef DSTM_INTERNAL_PROFILE_H
#define DSTM_INTERNAL_PROFILE_H 1

#include <stddef.h>
#include <stdint.h>

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum dstm_profile_mpi_call_e {
        DSTM_PROFILE_MPI_SEND = 0,
        DSTM_PROFILE_MPI_ISEND,
        DSTM_PROFILE_MPI_RECV,
        DSTM_PROFILE_MPI_IRECV,
        DSTM_PROFILE_MPI_ALLGATHER,
        DSTM_PROFILE_MPI_ALLREDUCE,
        DSTM_PROFILE_MPI_BARRIER,
        DSTM_PROFILE_MPI_CALL_MAX
} dstm_profile_mpi_call_t;

typedef struct dstm_profile_s {
        double start_time;
        uint64_t mpi_calls[DSTM_PROFILE_MPI_CALL_MAX];
        uint64_t mpi_bytes[DSTM_PROFILE_MPI_CALL_MAX];
        int active;
} dstm_profile_t;

void dstm_profile_start(void);
void dstm_profile_cancel(void);

void dstm_profile_record(dstm_profile_mpi_call_t call, size_t bytes);

void dstm_profile_record_collective(dstm_profile_mpi_call_t call,
                                    size_t local_payload_bytes);

void dstm_profile_finish_and_log(MPI_Comm comm, int rank);

#ifdef __cplusplus
}
#endif

#endif
