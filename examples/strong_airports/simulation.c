#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "simulation.h"

static void compute_ownership(simulation_state_t *sim)
{
    int rank = sim->rank;
    int num_ranks = sim->num_ranks;

    /* Find 2D decomposition of ranks */
    int rank_cols = num_ranks;
    for (int r = 1; r * r <= num_ranks; r++) {
        if (num_ranks % r == 0)
            rank_cols = num_ranks / r;
    }
    int rank_rows = num_ranks / rank_cols;

    sim->rank_rows = rank_rows;
    sim->rank_cols = rank_cols;
    sim->airports_per_rank_row = GRID_ROWS / rank_rows;
    sim->airports_per_rank_col = GRID_COLS / rank_cols;

    /* Build ownership tables */
    sim->num_my_airports = 0;

    for (int r = 0; r < GRID_ROWS; r++) {
        for (int c = 0; c < GRID_COLS; c++) {
            int airport_id = r * GRID_COLS + c;

            int owner_rank_row = r / sim->airports_per_rank_row;
            int owner_rank_col = c / sim->airports_per_rank_col;
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
                    int seed, dstm_darray_t *gates)
{
    sim->rank = rank;
    sim->num_ranks = num_ranks;
    sim->gates = gates;

    /* Validate rank count divides grid evenly */
    if (NUM_AIRPORTS % num_ranks != 0) {
        dstm_log_error("num_ranks (%d) must divide NUM_AIRPORTS (%d)",
                       num_ranks, NUM_AIRPORTS);
        return -1;
    }

    /* Allocate arrays */
    sim->my_airports   = malloc(NUM_AIRPORTS * sizeof(int));
    sim->airport_owner = malloc(NUM_AIRPORTS * sizeof(int));
    sim->airports      = malloc(NUM_AIRPORTS * sizeof(airport_state_t));
    sim->tiers         = malloc(NUM_AIRPORTS * sizeof(airport_tier_t));

    /* Allocate local_inbox as 2D array */
    sim->local_inbox = malloc(NUM_AIRPORTS * sizeof(gate_slot_t *));
    for (int a = 0; a < NUM_AIRPORTS; a++) {
        sim->local_inbox[a] = calloc(NUM_GATES, sizeof(gate_slot_t));
    }

    if (!sim->my_airports || !sim->airport_owner || !sim->airports ||
        !sim->tiers || !sim->local_inbox) {
        dstm_log_error("Failed to allocate simulation memory");
        return -1;
    }

    compute_ownership(sim);

    /* Verify 2D decomposition is valid */
    if (GRID_ROWS % sim->rank_rows != 0 ||
        GRID_COLS % sim->rank_cols != 0) {
        dstm_log_error("Cannot form valid 2D decomposition for %d ranks", num_ranks);
        return -1;
    }

    /* Assign tiers deterministically (same on all ranks) */
    srand(seed);
    for (int i = 0; i < NUM_AIRPORTS; i++) {
        int r = rand() % 100;
        if (r < 15)
            sim->tiers[i] = TIER_HUB;
        else if (r < 50)
            sim->tiers[i] = TIER_REGIONAL;
        else
            sim->tiers[i] = TIER_LOCAL;
    }

    /* Initialize per-rank RNG */
    srand(seed + rank);

    /* Initialize airport states for airports we own */
    for (int i = 0; i < sim->num_my_airports; i++) {
        int airport_id = sim->my_airports[i];
        airport_init(&sim->airports[airport_id], airport_id);
    }

    /* Clear local inbox */
    for (int a = 0; a < NUM_AIRPORTS; a++) {
        for (int g = 0; g < NUM_GATES; g++) {
            sim->local_inbox[a][g] = empty_gate;
        }
    }

    return 0;
}

void simulation_destroy(simulation_state_t *sim)
{
    /* Destroy airports we own */
    for (int i = 0; i < sim->num_my_airports; i++) {
        int airport_id = sim->my_airports[i];
        airport_destroy(&sim->airports[airport_id]);
    }

    /* Free local_inbox */
    for (int a = 0; a < NUM_AIRPORTS; a++) {
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
    (void)quiet;

    /* PHASE 1: Sentinel PUT for each airport we own */
    for (int i = 0; i < sim->num_my_airports; i++) {
        int airport_id = sim->my_airports[i];
        size_t idx = (size_t)airport_id * NUM_GATES;
        dstm_darray_put(sim->gates, idx, &empty_gate, t, 0);
    }

    /* PHASE 2: Clear outboxes */
    for (int i = 0; i < sim->num_my_airports; i++) {
        int airport_id = sim->my_airports[i];
        airport_clear_outbox(&sim->airports[airport_id]);
    }

    /* PHASE 3: Receive from DSTM (remote senders) */
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
                                         &recv_gate, sim->tiers, t);
            }
        }
    }

    /* PHASE 4: Process local inbox (flights from same-rank airports, sent last timestep) */
    for (int i = 0; i < sim->num_my_airports; i++) {
        int airport_id = sim->my_airports[i];

        for (int g = 0; g < NUM_GATES; g++) {
            if (sim->local_inbox[airport_id][g].n_flights > 0) {
                airport_process_incoming(&sim->airports[airport_id],
                                         &sim->local_inbox[airport_id][g],
                                         sim->tiers, t);
                /* Clear after processing */
                sim->local_inbox[airport_id][g] = empty_gate;
            }
        }
    }

    /* PHASE 5: Inject new flights */
    for (int i = 0; i < sim->num_my_airports; i++) {
        int airport_id = sim->my_airports[i];
        airport_inject_flight(&sim->airports[airport_id], sim->tiers, t);
    }

    /* PHASE 6: Send outbox — local vs remote */
    for (int i = 0; i < sim->num_my_airports; i++) {
        int airport_id = sim->my_airports[i];
        airport_state_t *ap = &sim->airports[airport_id];

        for (int dest = 0; dest < NUM_AIRPORTS; dest++) {
            if (dest == airport_id)
                continue;
            if (ap->outbox[dest].n_flights == 0)
                continue;

            int dest_owner = sim->airport_owner[dest];

            if (dest_owner == sim->rank) {
                /* LOCAL: direct copy to local inbox (processed next timestep) */
                int gate = (airport_id * 49) % NUM_GATES;
                merge_into_gate(&sim->local_inbox[dest][gate], &ap->outbox[dest]);

                dstm_log_normal("Airport %d [t=%ld]: LOCAL send %d flights to airport %d\n",
                                airport_id, t, ap->outbox[dest].n_flights, dest);
            } else {
                size_t idx = gate_index(dest, airport_id);
                int delta = airport_distance(airport_id, dest);

                status = dstm_darray_put(sim->gates, idx,
                                         &ap->outbox[dest], t, delta);
                if (status != DSTM_SUCCESS) {
                    dstm_log_error("PUT to airport %d failed", dest);
                }
            }
        }
    }
}
