#ifndef AIRPORT_H
#define AIRPORT_H

#include <dstm.h>

#define FLIGHT_PAYLOAD_SIZE  200
#define GATE_CAPACITY        4
#define LAYOVER_THRESHOLD    3

#define NUM_GATES 24

typedef enum {
    TIER_HUB      = 0,
    TIER_REGIONAL = 1,
    TIER_LOCAL    = 2,
    NUM_TIERS     = 3,
} airport_tier_t;

extern const char *tier_name[];
extern const int tier_send_prob[];
extern const int tier_recv_weight[];

typedef struct {
    int         valid;
    int         origin_airport;
    int         dest_airport;
    int         next_hop;
    int         legs_completed;
    dstm_time_t departure_time;
    int         flight_id;
    char        payload[FLIGHT_PAYLOAD_SIZE];
} flight_t;

typedef struct {
    int      n_flights;
    flight_t flights[GATE_CAPACITY];
} gate_slot_t;

extern const gate_slot_t empty_gate;

static inline size_t gate_index(int dest_airport, int source_airport) {
    int gate = (source_airport* 49) % NUM_GATES;
    return (size_t)dest_airport * NUM_GATES + gate;
}

typedef struct {
    int              rank;
    int              num_ranks;
    airport_tier_t  *tiers;
    dstm_distance_t *topology;
    dstm_darray_t   *gates;
    gate_slot_t     *outbox;
    int              flight_counter;
    int              departures;
    int              arrivals;
    int              connections;
} airport_state_t;

void airport_init(airport_state_t *state,
                  int rank,
                  int num_ranks,
                  airport_tier_t *tiers,
                  dstm_distance_t *topology,
                  dstm_darray_t *gates);

void airport_destroy(airport_state_t *state);

int airport_pick_dest(airport_state_t *state);

/*
 * Find next hop from current position toward dest.
 * Always returns an airport within LAYOVER_THRESHOLD distance.
 * Prefers hubs when making progress toward destination.
 */
int airport_find_next_hop(airport_state_t *state, int origin, int dest);

void airport_timestep(airport_state_t *state, dstm_time_t t, int quiet);

#endif /* AIRPORT_H */
