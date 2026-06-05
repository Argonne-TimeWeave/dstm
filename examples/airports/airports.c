#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "airports.h"

const char *tier_name[] = { "HUB", "REGIONAL", "LOCAL" };
const int tier_send_prob[] = { 80, 40, 15 };
const int tier_recv_weight[] = { 8, 3, 1 };
const gate_slot_t empty_gate = { .n_flights = 0 };

void airport_init(airport_state_t *state,
                  int rank,
                  int num_ranks,
                  airport_tier_t *tiers,
                  dstm_distance_t *topology,
                  dstm_darray_t *gates)
{
    state->rank           = rank;
    state->num_ranks      = num_ranks;
    state->tiers          = tiers;
    state->topology       = topology;
    state->gates          = gates;
    state->outbox         = calloc(num_ranks, sizeof(gate_slot_t));
    state->flight_counter = 0;
    state->departures     = 0;
    state->arrivals       = 0;
    state->connections    = 0;
}

void airport_destroy(airport_state_t *state)
{
    free(state->outbox);
    state->outbox = NULL;
}

int airport_pick_dest(airport_state_t *state)
{
    int total_weight = 0;
    for (int i = 0; i < state->num_ranks; i++) {
        if (i == state->rank)
            continue;
        total_weight += tier_recv_weight[state->tiers[i]];
    }

    int r = rand() % total_weight;
    int accum = 0;
    for (int i = 0; i < state->num_ranks; i++) {
        if (i == state->rank)
            continue;
        accum += tier_recv_weight[state->tiers[i]];
        if (r < accum)
            return i;
    }

    return (state->rank + 1) % state->num_ranks;
}

/*
 * Find next hop from 'origin' toward 'dest', respecting LAYOVER_THRESHOLD.
 *
 * Strategy:
 *   1. If dest is within range, go direct.
 *   2. Otherwise, find the best intermediate airport (prefer hubs) that:
 *      - Is within LAYOVER_THRESHOLD distance from origin
 *      - Makes progress toward dest (reduces remaining distance)
 *   3. If no hub found, pick any airport within range that makes progress.
 *
 * Returns the next hop airport, or dest if reachable directly.
 */
int airport_find_next_hop(airport_state_t *state, int origin, int dest)
{
    int dist_to_dest = dstm_distance_get(state->topology, origin, dest);

    /* Direct flight if within range */
    if (dist_to_dest <= LAYOVER_THRESHOLD)
        return dest;

    int best_hop = -1;
    int best_remaining = dist_to_dest;
    int best_is_hub = 0;

    for (int i = 0; i < state->num_ranks; i++) {
        if (i == origin || i == dest)
            continue;

        int dist_to_hop = dstm_distance_get(state->topology, origin, i);
        if (dist_to_hop > LAYOVER_THRESHOLD)
            continue;  /* too far */

        int remaining = dstm_distance_get(state->topology, i, dest);
        if (remaining >= dist_to_dest)
            continue;  /* no progress */

        int is_hub = (state->tiers[i] == TIER_HUB);

        /* Prefer hubs, then prefer closer to dest */
        if (best_hop == -1 ||
            (is_hub && !best_is_hub) ||
            (is_hub == best_is_hub && remaining < best_remaining)) {
            best_hop = i;
            best_remaining = remaining;
            best_is_hub = is_hub;
        }
    }

    /* Fallback: if no good hop found, just go direct (shouldn't happen
     * in a connected mesh, but safety first) */
    if (best_hop == -1)
        return dest;

    return best_hop;
}

void airport_timestep(airport_state_t *state, dstm_time_t t, int quiet)
{
    int rank = state->rank;
    int num_ranks = state->num_ranks;
    int status;

    dstm_log_normal("Airport %d (%s) [t=%ld]: === timestep ===\n",
                    rank, tier_name[state->tiers[rank]], t);

    /* STEP 1 — sentinel PUT */
    {
        size_t idx = (size_t)rank * num_ranks + rank;
        dstm_darray_put(state->gates, idx, &empty_gate, t, 0);
    }

    /* STEP 2 — clear outbox and receive */
    for (int g = 0; g < num_ranks; g++)
        state->outbox[g] = empty_gate;

    gate_slot_t recv_gate;

    for (int g = 0; g < num_ranks; g++) {
        size_t idx = (size_t)rank * num_ranks + g;

        status = dstm_darray_get_or_default(state->gates, idx,
                                            &recv_gate, t, 0,
                                            &empty_gate);
        if (status != DSTM_SUCCESS) {
            dstm_log_error("GET failed on gate %d", g);
            continue;
        }

        if (recv_gate.n_flights == 0)
            continue;

        dstm_log_normal("Airport %d [t=%ld]: gate %d has %d flight(s)\n",
                        rank, t, g, recv_gate.n_flights);

        /* STEP 3 — process flights */
        for (int f = 0; f < recv_gate.n_flights; f++) {
            flight_t flt = recv_gate.flights[f];

            if (flt.dest_airport == rank) {
                /* Final destination reached */
                state->arrivals++;
                dstm_log_normal("Airport %d [t=%ld]: LANDED flight %d — "
                                "origin=%d (%s) dep_t=%ld legs=%d \"%s\"\n",
                                rank, t, flt.flight_id,
                                flt.origin_airport,
                                tier_name[state->tiers[flt.origin_airport]],
                                flt.departure_time,
                                flt.legs_completed + 1,
                                flt.payload);
            } else {
                /* Connecting flight — find next hop toward destination */
                state->connections++;
                flt.legs_completed++;

                int next_hop = airport_find_next_hop(state, rank, flt.dest_airport);
                int delta = dstm_distance_get(state->topology, rank, next_hop);

                gate_slot_t *ob = &state->outbox[next_hop];
                if (ob->n_flights < GATE_CAPACITY) {
                    ob->flights[ob->n_flights++] = flt;

                    if (next_hop == flt.dest_airport) {
                        dstm_log_normal("Airport %d [t=%ld]: CONNECTING flight %d "
                                        "-> final dst=%d delta=%d (leg %d)\n",
                                        rank, t, flt.flight_id,
                                        next_hop, delta, flt.legs_completed);
                    } else {
                        dstm_log_normal("Airport %d [t=%ld]: CONNECTING flight %d "
                                        "-> hop=%d (%s) toward dst=%d delta=%d (leg %d)\n",
                                        rank, t, flt.flight_id,
                                        next_hop, tier_name[state->tiers[next_hop]],
                                        flt.dest_airport, delta, flt.legs_completed);
                    }
                } else {
                    dstm_log_warning("Airport %d [t=%ld]: CONNECTION DELAYED "
                                     "flight %d — gate full\n",
                                     rank, t, flt.flight_id);
                }
            }
        }
    }

    /* STEP 4 — inject new flight */
    if (num_ranks > 1 && (rand() % 100) < tier_send_prob[state->tiers[rank]]) {
        int dest = airport_pick_dest(state);
        int next_hop = airport_find_next_hop(state, rank, dest);
        int delta = dstm_distance_get(state->topology, rank, next_hop);

        gate_slot_t *ob = &state->outbox[next_hop];
        if (ob->n_flights < GATE_CAPACITY) {
            flight_t flt = {
                .valid          = 1,
                .origin_airport = rank,
                .dest_airport   = dest,
                .next_hop       = next_hop,
                .legs_completed = 0,
                .departure_time = t,
                .flight_id      = rank * 10000 + state->flight_counter++,
            };

            int is_direct = (next_hop == dest);
            snprintf(flt.payload, sizeof(flt.payload),
                     "%s(%d)->%s(%d)%s t=%ld",
                     tier_name[state->tiers[rank]], rank,
                     tier_name[state->tiers[dest]], dest,
                     is_direct ? "" : " [LAYOVER]",
                     t);
            ob->flights[ob->n_flights++] = flt;

            state->departures++;

            if (is_direct) {
                dstm_log_normal("Airport %d [t=%ld]: DEPART flight %d "
                                "dst=%d (%s) DIRECT delta=%d\n",
                                rank, t, flt.flight_id,
                                dest, tier_name[state->tiers[dest]], delta);
            } else {
                dstm_log_normal("Airport %d [t=%ld]: DEPART flight %d "
                                "dst=%d via %s(%d) delta=%d\n",
                                rank, t, flt.flight_id,
                                dest, tier_name[state->tiers[next_hop]],
                                next_hop, delta);
            }
        }
    }

    /* STEP 5 — send outbox */
    for (int dest = 0; dest < num_ranks; dest++) {
        if (dest == rank)
            continue;
        if (state->outbox[dest].n_flights == 0)
            continue;

        size_t idx = (size_t)dest * num_ranks + rank;
        int delta = dstm_distance_get(state->topology, rank, dest);

        status = dstm_darray_put(state->gates, idx,
                                 &state->outbox[dest], t, delta);
        if (status != DSTM_SUCCESS)
            dstm_log_error("PUT to airport %d gate %d failed", dest, rank);
    }
}  
