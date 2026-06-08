#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "airport.h"

const char *tier_name[] = { "HUB", "REGIONAL", "LOCAL" };
const int tier_send_prob[] = { 80, 40, 15 };
const int tier_recv_weight[] = { 8, 3, 1 };
const gate_slot_t empty_gate = { .n_flights = 0 };

void airport_init(airport_state_t *state, int airport_id)
{
    state->airport_id     = airport_id;
    state->flight_counter = 0;
    state->departures     = 0;
    state->arrivals       = 0;
    state->connections    = 0;
    state->outbox         = calloc(NUM_AIRPORTS, sizeof(gate_slot_t));
    airport_clear_outbox(state);
}

void airport_destroy(airport_state_t *state) {
    free(state->outbox);
    state->outbox = NULL;
}

void airport_clear_outbox(airport_state_t *state)
{
    for (int i = 0; i < NUM_AIRPORTS; i++) {
        state->outbox[i] = empty_gate;
    }
}

int airport_pick_dest(int origin, int num_airports, airport_tier_t *tiers)
{
    int total_weight = 0;
    for (int i = 0; i < num_airports; i++) {
        if (i == origin)
            continue;
        total_weight += tier_recv_weight[tiers[i]];
    }

    int r = rand() % total_weight;
    int accum = 0;
    for (int i = 0; i < num_airports; i++) {
        if (i == origin)
            continue;
        accum += tier_recv_weight[tiers[i]];
        if (r < accum)
            return i;
    }

    return (origin + 1) % num_airports;
}

int airport_find_next_hop(int origin, int dest, airport_tier_t *tiers)
{
    int dist_to_dest = airport_distance(origin, dest);

    /* Direct flight if within range */
    if (dist_to_dest <= LAYOVER_THRESHOLD)
        return dest;

    int best_hop = -1;
    int best_remaining = dist_to_dest;
    int best_is_hub = 0;

    for (int i = 0; i < NUM_AIRPORTS; i++) {
        if (i == origin || i == dest)
            continue;

        int dist_to_hop = airport_distance(origin, i);
        if (dist_to_hop > LAYOVER_THRESHOLD)
            continue;

        int remaining = airport_distance(i, dest);
        if (remaining >= dist_to_dest)
            continue;

        int is_hub = (tiers[i] == TIER_HUB);

        if (best_hop == -1 ||
            (is_hub && !best_is_hub) ||
            (is_hub == best_is_hub && remaining < best_remaining)) {
            best_hop = i;
            best_remaining = remaining;
            best_is_hub = is_hub;
        }
    }

    if (best_hop == -1)
        return dest;

    return best_hop;
}

void airport_process_incoming(airport_state_t *state,
                              gate_slot_t *recv_gate,
                              airport_tier_t *tiers,
                              dstm_time_t t)
{
    int airport_id = state->airport_id;

    for (int f = 0; f < recv_gate->n_flights; f++) {
        flight_t flt = recv_gate->flights[f];

        if (flt.dest_airport == airport_id) {
            /* Final destination reached */
            state->arrivals++;
            dstm_log_normal("Airport %d [t=%ld]: LANDED flight %d — "
                            "origin=%d (%s) dep_t=%ld legs=%d \"%s\"\n",
                            airport_id, t, flt.flight_id,
                            flt.origin_airport,
                            tier_name[tiers[flt.origin_airport]],
                            flt.departure_time,
                            flt.legs_completed + 1,
                            flt.payload);
        } else {
            /* Connecting flight */
            state->connections++;
            flt.legs_completed++;

            int next_hop = airport_find_next_hop(airport_id, flt.dest_airport, tiers);

            gate_slot_t *ob = &state->outbox[next_hop];
            if (ob->n_flights < GATE_CAPACITY) {
                ob->flights[ob->n_flights++] = flt;

                dstm_log_normal("Airport %d [t=%ld]: CONNECTING flight %d "
                                "-> hop=%d toward dst=%d (leg %d)\n",
                                airport_id, t, flt.flight_id,
                                next_hop, flt.dest_airport, flt.legs_completed);
            } else {
                dstm_log_warning("Airport %d [t=%ld]: CONNECTION DELAYED "
                                 "flight %d — gate full\n",
                                 airport_id, t, flt.flight_id);
            }
        }
    }
}

void airport_inject_flight(airport_state_t *state,
                           airport_tier_t *tiers,
                           dstm_time_t t)
{
    int airport_id = state->airport_id;

    if ((rand() % 100) >= tier_send_prob[tiers[airport_id]])
        return;

    int dest = airport_pick_dest(airport_id, NUM_AIRPORTS, tiers);
    int next_hop = airport_find_next_hop(airport_id, dest, tiers);

    gate_slot_t *ob = &state->outbox[next_hop];
    if (ob->n_flights >= GATE_CAPACITY)
        return;

    flight_t flt = {
        .valid          = 1,
        .origin_airport = airport_id,
        .dest_airport   = dest,
        .next_hop       = next_hop,
        .legs_completed = 0,
        .departure_time = t,
        .flight_id      = airport_id * 10000 + state->flight_counter++,
    };

    int is_direct = (next_hop == dest);
    snprintf(flt.payload, sizeof(flt.payload),
             "%s(%d)->%s(%d)%s t=%ld",
             tier_name[tiers[airport_id]], airport_id,
             tier_name[tiers[dest]], dest,
             is_direct ? "" : " [LAYOVER]",
             t);

    ob->flights[ob->n_flights++] = flt;
    state->departures++;

    dstm_log_normal("Airport %d [t=%ld]: DEPART flight %d dst=%d via %d%s\n",
                    airport_id, t, flt.flight_id, dest, next_hop,
                    is_direct ? " DIRECT" : "");
}
