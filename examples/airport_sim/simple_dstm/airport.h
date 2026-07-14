#ifndef AIRPORT_H
#define AIRPORT_H

#include <dstm.h>
#include <mpi.h>

#define GATE_CAPACITY 1024

typedef struct {
    int         dest_global;
    long long   flight_id;
} flight_t;

typedef struct {
    int      n_flights;
    flight_t flights[GATE_CAPACITY];
} gate_slot_t;

extern const gate_slot_t empty_gate;

/* Match ROSS's landing-side serialization. In ROSS: arrivals queue
 * behind a runway with mean service time MEAN_LAND=10 virtual units,
 * and one DSTM timestep corresponds to MEAN_DEPARTURE=30 virtual units,
 * giving ~3 landings per airport per DSTM step on average. */
#define LANDINGS_PER_STEP 3

typedef struct {
    int              rank;
    int              num_ranks;
    int              na;
    int              grid;

    long long       *landings;
    int             *planes_on_the_ground;

    /* NEW: planes that arrived this step but haven't landed yet
     * (queued behind a virtual runway to match ROSS's MEAN_LAND). */
    int             *pending_landings;

    long long        departures;
    long long        flight_counter;

    int             *neigh_ranks;
    int              n_neigh_ranks;

    dstm_distance_t *topology;
    dstm_darray_t   *gates;
    gate_slot_t     *outbox;
} airport_state_t;

static inline int airport_owner(int global_id, int na) {
    return global_id / na;
}

void airport_init(airport_state_t *state,
                  int rank,
                  int num_ranks,
                  int na,
                  int grid,
                  int planes_per_airport,
                  dstm_distance_t *topology,
                  dstm_darray_t *gates);

void airport_destroy(airport_state_t *state);

void airport_timestep(airport_state_t *state, dstm_time_t t);

#endif /* AIRPORT_H */
