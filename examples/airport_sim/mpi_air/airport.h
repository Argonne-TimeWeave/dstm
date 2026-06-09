#ifndef AIRPORT_H
#define AIRPORT_H

#include <stdio.h>
#include <stdlib.h>

#define FLIGHT_PAYLOAD_SIZE  200
#define GATE_CAPACITY        4
#define LAYOVER_THRESHOLD    3
#define NUM_GATES            24

typedef long dstm_time_t;  /* keep the name for minimal diff */

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
    gate_slot_t      *outbox;       /* indexed by destination airport */
    int              num_airports;
} airport_state_t;

static inline int airport_distance(int a, int b, int grid_cols)
{
    int ra = a / grid_cols, ca = a % grid_cols;
    int rb = b / grid_cols, cb = b % grid_cols;
    return abs(ra - rb) + abs(ca - cb);
}

static inline size_t gate_index(int dest_airport, int source_airport)
{
    int gate = (source_airport * 49) % NUM_GATES;
    return (size_t)dest_airport * NUM_GATES + gate;
}

/* Logging shims so airport.c needs almost no edits */
extern int g_log_quiet;
#define dstm_log_normal(...)  do { if (!g_log_quiet) fprintf(stdout, __VA_ARGS__); } while (0)
#define dstm_log_warning(...) do { if (!g_log_quiet) fprintf(stderr, __VA_ARGS__); } while (0)
#define dstm_log_error(...)   do { fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } while (0)

extern int g_work_iterations;
void do_synthetic_work(void);

void airport_init(airport_state_t *state, int airport_id, int num_airports);
void airport_destroy(airport_state_t *state);
void airport_clear_outbox(airport_state_t *state);

int airport_pick_dest(int origin, int num_airports, airport_tier_t *tiers);
int airport_find_next_hop(int origin, int dest, int num_airports,
                          int grid_cols, airport_tier_t *tiers);

void airport_process_incoming(airport_state_t *state,
                              gate_slot_t *recv_gate,
                              int grid_cols,
                              airport_tier_t *tiers,
                              dstm_time_t t);

void airport_inject_flight(airport_state_t *state,
                           int grid_cols,
                           airport_tier_t *tiers,
                           dstm_time_t t);

#endif
