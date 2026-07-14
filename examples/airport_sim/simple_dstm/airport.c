#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "airport.h"

const gate_slot_t empty_gate = { .n_flights = 0 };

/* Given a global airport id on a `grid` x `grid` torus, return the
 * global id of the neighbor in direction dir (0=N, 1=S, 2=E, 3=W). */
static int neighbor_of(int gid, int grid, int dir)
{
    int row = gid / grid;
    int col = gid % grid;
    switch (dir) {
        case 0: row = (row - 1 + grid) % grid; break;
        case 1: row = (row + 1) % grid;        break;
        case 2: col = (col + 1) % grid;        break;
        case 3: col = (col - 1 + grid) % grid; break;
    }
    return row * grid + col;
}

/* Find the outbox index for a given remote rank. Linear search since
 * n_neigh_ranks is at most ~4. Returns -1 if not found (shouldn't happen
 * for legitimate torus destinations). */
static inline int find_outbox_idx(const airport_state_t *s, int dest_rank)
{
    for (int i = 0; i < s->n_neigh_ranks; i++)
        if (s->neigh_ranks[i] == dest_rank)
            return i;
    return -1;
}

/* Compute the unique set of remote ranks this rank ever needs to send to
 * (or receive from — the torus is symmetric). Walks only this rank's local
 * airports, not all airports globally. */
static void compute_neighbor_ranks(airport_state_t *state)
{
    int rank = state->rank;
    int na   = state->na;
    int grid = state->grid;

    /* Upper bound: 4 neighbors per local airport, at most 4*na uniques.
     * In practice this is <= 4 for na small, and 0 for na >= total/1. */
    int cap = 4 * na;
    int *tmp = malloc(sizeof(int) * cap);
    int n = 0;

    for (int l = 0; l < na; l++) {
        int my_gid = rank * na + l;
        for (int dir = 0; dir < 4; dir++) {
            int n_gid   = neighbor_of(my_gid, grid, dir);
            int n_owner = n_gid / na;
            if (n_owner == rank)
                continue;   /* same-rank neighbor — no DSTM traffic */

            /* dedup */
            int already = 0;
            for (int i = 0; i < n; i++) {
                if (tmp[i] == n_owner) { already = 1; break; }
            }
            if (!already)
                tmp[n++] = n_owner;
        }
    }

    state->neigh_ranks   = malloc(sizeof(int) * (n > 0 ? n : 1));
    state->n_neigh_ranks = n;
    for (int i = 0; i < n; i++)
        state->neigh_ranks[i] = tmp[i];
    free(tmp);
}

void airport_init(airport_state_t *state,
                  int rank,
                  int num_ranks,
                  int na,
                  int grid,
                  int planes_per_airport,
                  dstm_distance_t *topology,
                  dstm_darray_t *gates)
{
    state->rank      = rank;
    state->num_ranks = num_ranks;
    state->na        = na;
    state->grid      = grid;

    state->landings             = calloc(na, sizeof(long long));
    state->planes_on_the_ground = calloc(na, sizeof(int));
    state->pending_landings     = calloc(na, sizeof(int));
    for (int l = 0; l < na; l++)
        state->planes_on_the_ground[l] = planes_per_airport;

    state->departures     = 0;
    state->flight_counter = 0;

    state->topology = topology;
    state->gates    = gates;

    /* Build the neighbor rank cache before allocating outbox. */
    compute_neighbor_ranks(state);

    /* Compact outbox: one slot per unique remote neighbor rank. */
    state->outbox = calloc(
        state->n_neigh_ranks > 0 ? state->n_neigh_ranks : 1,
        sizeof(gate_slot_t));
}

void airport_destroy(airport_state_t *state)
{
    free(state->landings);
    free(state->planes_on_the_ground);
    free(state->outbox);
    free(state->neigh_ranks);
    free(state->pending_landings);
    state->landings = NULL;
    state->planes_on_the_ground = NULL;
    state->outbox = NULL;
    state->neigh_ranks = NULL;
    state->pending_landings = NULL;
}

void airport_timestep(airport_state_t *state, dstm_time_t t)
{
    int rank = state->rank;
    int na   = state->na;
    int grid = state->grid;
    int num_ranks = state->num_ranks;
    int status;

    /* clear outbox */
    for (int i = 0; i < state->n_neigh_ranks; i++)
        state->outbox[i] = empty_gate;

    /* --- STEP 1: receive arrivals from each neighbor rank ---
     * Arrivals go into pending_landings, NOT directly onto the ground.
     * This matches ROSS's runway-serialized landing model. */
    gate_slot_t recv_gate;
    for (int i = 0; i < state->n_neigh_ranks; i++) {
        int g = state->neigh_ranks[i];

        size_t idx = (size_t)rank * num_ranks + g;
        status = dstm_darray_get_or_default(state->gates, idx,
                                            &recv_gate, t, 0,
                                            &empty_gate);
        if (status != DSTM_SUCCESS) {
            dstm_log_error("GET failed on gate %d", g);
            continue;
        }

        for (int f = 0; f < recv_gate.n_flights; f++) {
            flight_t flt = recv_gate.flights[f];
            int local = flt.dest_global - rank * na;
            if (local < 0 || local >= na)
                continue;

            state->pending_landings[local]++;    /* queue, don't land yet */
        }
    }

    /* --- STEP 1b: drain the per-airport landing queue ---
     * Each airport can land at most LANDINGS_PER_STEP planes per timestep.
     * Extras wait for the next step, exactly like ROSS's furthest_flight_landing
     * pushes late arrivals into future virtual time. */
    for (int l = 0; l < na; l++) {
        int budget = LANDINGS_PER_STEP;
        int can_land = state->pending_landings[l] < budget
                     ? state->pending_landings[l]
                     : budget;
        if (can_land == 0) continue;

        state->pending_landings[l]     -= can_land;
        state->planes_on_the_ground[l] += can_land;
        state->landings[l]             += can_land;
    }

    /* --- STEP 2: depart every plane on the ground (unchanged) --- */
    for (int l = 0; l < na; l++) {
        int planes_here = state->planes_on_the_ground[l];
        if (planes_here == 0) continue;

        int my_gid = rank * na + l;

        for (int k = 0; k < planes_here; k++) {
            int dir       = rand() & 3;
            int dest_gid  = neighbor_of(my_gid, grid, dir);
            int dest_rank = airport_owner(dest_gid, na);

            if (dest_rank == rank) {
                int dest_local = dest_gid - rank * na;
                /* Same-rank arrival also has to queue for landing. */
                state->pending_landings[dest_local]++;
                state->planes_on_the_ground[l]--;
                state->departures++;
                continue;
            }

            int ob_idx = find_outbox_idx(state, dest_rank);
            if (ob_idx < 0) continue;

            gate_slot_t *ob = &state->outbox[ob_idx];
            if (ob->n_flights >= GATE_CAPACITY) {
                break;   /* outbox full; plane stays put, retries next step */
            }

            flight_t flt = {
                .dest_global = dest_gid,
                .flight_id   = (long long)my_gid * 1000000LL
                               + state->flight_counter++,
            };
            ob->flights[ob->n_flights++] = flt;

            state->planes_on_the_ground[l]--;
            state->departures++;
        }
    }

    /* --- STEP 3: PUT to each neighbor rank (unchanged) --- */
    for (int i = 0; i < state->n_neigh_ranks; i++) {
        int dest = state->neigh_ranks[i];
        if (state->outbox[i].n_flights == 0)
            continue;

        size_t idx = (size_t)dest * num_ranks + rank;
        status = dstm_darray_put(state->gates, idx,
                                 &state->outbox[i], t, 1);
        if (status != DSTM_SUCCESS)
            dstm_log_error("PUT to rank %d gate failed", dest);
    }
}
