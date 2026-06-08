#ifndef AIRPORT_H
#define AIRPORT_H

#include <dstm.h>

#define FLIGHT_PAYLOAD_SIZE  200
#define GATE_CAPACITY        4
#define LAYOVER_THRESHOLD    3
#define NUM_GATES            24

#define GRID_ROWS 16
#define GRID_COLS 16
#define NUM_AIRPORTS (GRID_ROWS * GRID_COLS)

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

typedef struct {
    int              airport_id;
    int              flight_counter;
    int              departures;
    int              arrivals;
    int              connections;
    gate_slot_t      *outbox;
} airport_state_t;

/* Compute Manhattan distance between two airports on the grid */
static inline int airport_distance(int airport_a, int airport_b)
{
    int row_a = airport_a / GRID_COLS;
    int col_a = airport_a % GRID_COLS;
    int row_b = airport_b / GRID_COLS;
    int col_b = airport_b % GRID_COLS;
    return abs(row_a - row_b) + abs(col_a - col_b);
}

/* Map source airport to a gate index at destination airport */
static inline size_t gate_index(int dest_airport, int source_airport)
{
    int gate = (source_airport * 49) % NUM_GATES;
    return (size_t)dest_airport * NUM_GATES + gate;
}

void airport_init(airport_state_t *state, int airport_id);
void airport_destroy(airport_state_t *state);
void airport_clear_outbox(airport_state_t *state);

int airport_pick_dest(int origin, int num_airports, airport_tier_t *tiers);
int airport_find_next_hop(int origin, int dest, airport_tier_t *tiers);

void airport_process_incoming(airport_state_t *state,
                              gate_slot_t *recv_gate,
                              airport_tier_t *tiers,
                              dstm_time_t t);

void airport_inject_flight(airport_state_t *state,
                           airport_tier_t *tiers,
                           dstm_time_t t);

#endif /* AIRPORT_H */
