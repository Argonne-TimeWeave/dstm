#ifndef SIMULATION_H
#define SIMULATION_H

#include <mpi.h>
#include "airport.h"

typedef struct {
    int rank;
    int num_ranks;

    int block_rows;
    int block_cols;
    int airports_per_rank;

    int grid_rows;
    int grid_cols;
    int num_airports;

    int rank_rows;
    int rank_cols;

    int *my_airports;
    int num_my_airports;
    int *airport_owner;

    airport_state_t *airports;
    airport_tier_t *tiers;

    /* Per-destination-airport inbox of gate slots (only allocated for
       airports owned by this rank, but indexed by global airport id). */
    gate_slot_t **local_inbox;

    /* MPI scratch buffers for all-to-all exchange */
    flight_t *send_buf;
    flight_t *recv_buf;
    int      *send_counts;
    int      *recv_counts;
    int      *send_displs;
    int      *recv_displs;
    int       send_buf_cap;
    int       recv_buf_cap;

    MPI_Datatype flight_type;
} simulation_state_t;

int  simulation_init(simulation_state_t *sim, int rank, int num_ranks,
                     int block_rows, int block_cols, int seed);
void simulation_destroy(simulation_state_t *sim);
void simulation_timestep(simulation_state_t *sim, dstm_time_t t, int quiet);

void simulation_gather_stats(simulation_state_t *sim,
                             int *total_departures,
                             int *total_arrivals,
                             int *total_connections);

#endif
