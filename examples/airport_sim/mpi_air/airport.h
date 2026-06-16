#ifndef AIRPORT_H
#define AIRPORT_H

#include <dstm.h>

#define FLIGHT_PAYLOAD_SIZE  200
#define GATE_CAPACITY        128
#define LAYOVER_THRESHOLD    3

#define NUM_GATES 24

#define LOGGING 1 //lazy

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
    int         origin_rank;
    int         origin_local;
    int         dest_rank;
    int         dest_local;
    int         next_hop_rank;   /* next rank toward dest_rank */
    int         legs_completed;
    dstm_time_t departure_time;
    dstm_time_t arrive_time;
    int         flight_id;
    char        payload[FLIGHT_PAYLOAD_SIZE];
} flight_t;

typedef struct {
    int      n_flights;
    flight_t flights[GATE_CAPACITY];
} gate_slot_t;

extern const gate_slot_t empty_gate;

/*
 * Burn some CPU cycles to simulate per-arrival processing work.
 * `iterations` is taken from the --work CLI flag. Returns a value
 * derived from the computation so the compiler can't optimize it away.
 */
double airport_do_work(int iterations);

/*
 * Per-rank inbox layout: gates indexed by destination rank, with NUM_GATES
 * slots per rank (sender hashes into one of them). Local airport within
 * the rank is resolved on receive (from flight_t.dest_local).
 */
static inline size_t gate_index(int dest_rank, int source_rank) {
    int gate = (source_rank * 49) % NUM_GATES;
    return (size_t)dest_rank * NUM_GATES + gate;
}

typedef struct {
    int              rank;
    int              num_ranks;
    int              na;             /* airports per rank */
    airport_tier_t  *tiers;          /* size: num_ranks * na (global) */
    dstm_distance_t *topology;       /* per-rank distance */
    dstm_darray_t   *gates;
    gate_slot_t     *outbox;         /* size: num_ranks (one slot per dest rank) */
    int              flight_counter;
    int              work;
    /* Per-local-airport counters, each of size na */
    int             *departures;
    int             *arrivals;
    int             *connections;
} airport_state_t;

/* Helpers for global <-> (rank, local) addressing */
static inline int airport_global(int rank, int local, int na) {
    return rank * na + local;
}

void airport_init(airport_state_t *state,
                  int rank,
                  int num_ranks,
                  int na,
                  int work,
                  airport_tier_t *tiers,
                  dstm_distance_t *topology,
                  dstm_darray_t *gates);

void airport_destroy(airport_state_t *state);

/*
 * Pick a destination (dest_rank, dest_local) from the perspective of
 * an airport on `origin_rank`. The destination must NOT live on
 * origin_rank (no local-to-local flights allowed).
 */
void airport_pick_dest(airport_state_t *state,
                       int origin_rank,
                       int *dest_rank,
                       int *dest_local);

/*
 * Find next hop RANK from origin_rank toward dest_rank, respecting
 * LAYOVER_THRESHOLD. Distance is measured per-rank only.
 */
int airport_find_next_hop(airport_state_t *state, int origin_rank, int dest_rank);

void airport_timestep(airport_state_t *state, dstm_time_t t, int quiet);

#endif /* AIRPORT_H */
