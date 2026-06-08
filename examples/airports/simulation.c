#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "simulation.h"

static void compute_ownership(simulation_state_t *sim)
{
    int rank = sim->rank;
    int num_ranks = sim->num_ranks;

    int rank_cols = num_ranks;
    for (int r = 1; r * r <= num_ranks; r++) {
        if (num_ranks % r == 0)
            rank_cols = num_ranks / r;
    }
    int rank_rows = num_ranks / rank_cols;

    sim->rank_rows = rank_rows;
    sim->rank_cols = rank_cols;
    sim->grid_rows = sim->block_rows * rank_rows;
    sim->grid_cols = sim->block_cols * rank_cols;
    sim->num_airports = sim->grid_rows * sim->grid_cols;

    sim->num_my_airports = 0;

    for (int r = 0; r < sim->grid_rows; r++) {
        for (int c = 0; c < sim->grid_cols; c++) {
            int airport_id = r * sim->grid_cols + c;

            int owner_rank_row = r / sim->block_rows;
            int owner_rank_col = c / sim->block_cols;
            int owner = owner_rank_row * rank_cols + owner_rank_col;

            sim->airport_owner[airport_id] = owner;

            if (owner == rank) {
                sim->my_airports[sim->num_my_airports++] = airport_id;
            }
        }
    }
}

void simulation_gather_stats(simulation_state_t *sim,
                             int *total_departures,
                             int *total_arrivals,
                             int *total_connections)
{
    *total_departures = 0;
    *total_arrivals = 0;
    *total_connections = 0;

    for (int i = 0; i < sim->num_my_airports; i++) {
        int airport_id = sim->my_airports[i];
        *total_departures += sim->airports[airport_id].departures;
        *total_arrivals += sim->airports[airport_id].arrivals;
        *total_connections += sim->airports[airport_id].connections;
    }
}

int simulation_init(simulation_state_t *sim, int rank, int num_ranks,
                    int block_rows, int block_cols,
                    int seed, dstm_darray_t *gates)
{
    sim->rank = rank;
    sim->num_ranks = num_ranks;
    sim->block_rows = block_rows;
    sim->block_cols = block_cols;
    sim->airports_per_rank = block_rows * block_cols;
    sim->gates = gates;

    int rank_cols = num_ranks;
    for (int r = 1; r * r <= num_ranks; r++) {
        if (num_ranks % r == 0)
            rank_cols = num_ranks / r;
    }
    int rank_rows = num_ranks / rank_cols;
    int grid_rows = block_rows * rank_rows;
    int grid_cols = block_cols * rank_cols;
    int num_airports = grid_rows * grid_cols;

    sim->my_airports   = malloc(num_airports * sizeof(int));
    sim->airport_owner = malloc(num_airports * sizeof(int));
    sim->airports      = malloc(num_airports * sizeof(airport_state_t));
    sim->tiers         = malloc(num_airports * sizeof(airport_tier_t));

    sim->local_inbox = malloc(num_airports * sizeof(gate_slot_t *));
    for (int a = 0; a < num_airports; a++) {
        sim->local_inbox[a] = calloc(NUM_GATES, sizeof(gate_slot_t));
    }

    if (!sim->my_airports || !sim->airport_owner || !sim->airports ||
        !sim->tiers || !sim->local_inbox) {
        dstm_log_error("Failed to allocate simulation memory");
        return -1;
    }

    compute_ownership(sim);

    srand(seed);
    for (int i = 0; i < sim->num_airports; i++) {
        int r = rand() % 100;
        if (r < 15)
            sim->tiers[i] = TIER_HUB;
        else if (r < 50)
            sim->tiers[i] = TIER_REGIONAL;
        else
            sim->tiers[i] = TIER_LOCAL;
    }

    srand(seed + rank);

    for (int i = 0; i < sim->num_my_airports; i++) {
        int airport_id = sim->my_airports[i];
        airport_init(&sim->airports[airport_id], airport_id, sim->num_airports);
    }

    for (int a = 0; a < sim->num_airports; a++) {
        for (int g = 0; g < NUM_GATES; g++) {
            sim->local_inbox[a][g] = empty_gate;
        }
    }

    return 0;
}

void simulation_destroy(simulation_state_t *sim)
{
    for (int i = 0; i < sim->num_my_airports; i++) {
        int airport_id = sim->my_airports[i];
        airport_destroy(&sim->airports[airport_id]);
    }

    for (int a = 0; a < sim->num_airports; a++) {
        free(sim->local_inbox[a]);
    }
    free(sim->local_inbox);

    free(sim->my_airports);
    free(sim->airport_owner);
    free(sim->airports);
    free(sim->tiers);
}

static void merge_into_gate(gate_slot_t *dest, gate_slot_t *src)
{
    for (int f = 0; f < src->n_flights && dest->n_flights < GATE_CAPACITY; f++) {
        dest->flights[dest->n_flights++] = src->flights[f];
    }
}

void simulation_timestep(simulation_state_t *sim, dstm_time_t t, int quiet)
{
    int status;
    int grid_cols = sim->grid_cols;
    int num_airports = sim->num_airports;
    (void)quiet;

    for (int i = 0; i < sim->num_my_airports; i++) {
        int airport_id = sim->my_airports[i];
        size_t idx = (size_t)airport_id * NUM_GATES;
        dstm_darray_put(sim->gates, idx, &empty_gate, t, 0);
    }

    for (int i = 0; i < sim->num_my_airports; i++) {
        int airport_id = sim->my_airports[i];
        airport_clear_outbox(&sim->airports[airport_id]);
    }

    for (int i = 0; i < sim->num_my_airports; i++) {
        int airport_id = sim->my_airports[i];
        gate_slot_t recv_gate;

        for (int g = 0; g < NUM_GATES; g++) {
            size_t idx = (size_t)airport_id * NUM_GATES + g;

            status = dstm_darray_get_or_default(sim->gates, idx,
                                                &recv_gate, t, 0,
                                                &empty_gate);
            if (status != DSTM_SUCCESS) {
                dstm_log_error("GET failed on airport %d gate %d", airport_id, g);
                continue;
            }

            if (recv_gate.n_flights > 0) {
                airport_process_incoming(&sim->airports[airport_id],
                                         &recv_gate, grid_cols, sim->tiers, t);
            }
        }
    }

    for (int i = 0; i < sim->num_my_airports; i++) {
        int airport_id = sim->my_airports[i];

        for (int g = 0; g < NUM_GATES; g++) {
            if (sim->local_inbox[airport_id][g].n_flights > 0) {
                airport_process_incoming(&sim->airports[airport_id],
                                         &sim->local_inbox[airport_id][g],
                                         grid_cols, sim->tiers, t);
                sim->local_inbox[airport_id][g] = empty_gate;
            }
        }
    }

    for (int i = 0; i < sim->num_my_airports; i++) {
        int airport_id = sim->my_airports[i];
        airport_inject_flight(&sim->airports[airport_id], grid_cols, sim->tiers, t);
    }

    for (int i = 0; i < sim->num_my_airports; i++) {
        int airport_id = sim->my_airports[i];
        airport_state_t *ap = &sim->airports[airport_id];

        for (int dest = 0; dest < num_airports; dest++) {
            if (dest == airport_id)
                continue;
            if (ap->outbox[dest].n_flights == 0)
                continue;

            int dest_owner = sim->airport_owner[dest];

            if (dest_owner == sim->rank) {
                int gate = (airport_id * 49) % NUM_GATES;
                merge_into_gate(&sim->local_inbox[dest][gate], &ap->outbox[dest]);

                dstm_log_normal("Airport %d [t=%ld]: LOCAL send %d flights to airport %d\n",
                                airport_id, t, ap->outbox[dest].n_flights, dest);
            } else {
                size_t idx = gate_index(dest, airport_id);
                int delta = airport_distance(airport_id, dest, grid_cols);

                status = dstm_darray_put(sim->gates, idx,
                                         &ap->outbox[dest], t, delta);
                if (status != DSTM_SUCCESS) {
                    dstm_log_error("PUT to airport %d failed", dest);
                }
            }
        }
    }
}
