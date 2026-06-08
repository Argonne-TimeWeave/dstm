#ifndef SIMULATION_H
#define SIMULATION_H

#include <dstm.h>
#include "airport.h"

typedef struct {
    int rank;
    int num_ranks;

    /* 2D rank decomposition */
    int rank_rows;
    int rank_cols;
    int airports_per_rank_row;
    int airports_per_rank_col;

    /* Ownership */
    int *my_airports;
    int num_my_airports;
    int *airport_owner;

    /* Per-airport state (indexed by airport_id) */
    airport_state_t *airports;

    /* Tiers for all airports */
    airport_tier_t *tiers;

    /* DSTM shared gate array (for remote communication) */
    dstm_darray_t *gates;

    /* Local gate buffers for intra-rank flights */
    gate_slot_t **local_inbox;

} simulation_state_t;

int simulation_init(simulation_state_t *sim, int rank, int num_ranks,
                    int seed, dstm_darray_t *gates);
void simulation_destroy(simulation_state_t *sim);
void simulation_timestep(simulation_state_t *sim, dstm_time_t t, int quiet);

void simulation_gather_stats(simulation_state_t *sim,
                             int *total_departures,
                             int *total_arrivals,
                             int *total_connections);

#endif /* SIMULATION_H */
