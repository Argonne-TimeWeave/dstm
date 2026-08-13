#ifndef AIRPORT_H
#define AIRPORT_H

#include <stddef.h>
#include <stdint.h>
#include <dstm.h>

/*
 * Per-edge gate capacity.
 *
 * The darray benchmark aggregated every flight between a *pair of ranks*
 * into one slot, so it needed 1024. Here a gate is per (airport, direction)
 * link, and the landing budget below caps sustained departures at
 * LANDINGS_PER_STEP per airport per tick, so the only real pressure is the
 * t=0 burst of `planes_per_airport`. 32 covers that with slack.
 *
 * This directly sets the region footprint: edge slab bytes =
 * 4 * na * (8 + sizeof(gate_slot_t)), and DSTM keeps a full-size buffer per
 * version. Do not inflate it casually.
 */
#ifndef EDGE_GATE_CAPACITY
#define EDGE_GATE_CAPACITY 32
#endif

/* Match ROSS's landing-side serialization: arrivals queue behind a runway
 * with mean service MEAN_LAND=10 virtual units, one DSTM tick is
 * MEAN_DEPARTURE=30, hence ~3 landings/airport/tick. */
#define LANDINGS_PER_STEP 3

/* 16 bytes, no implicit padding. */
typedef struct {
	int32_t src_global;
	int32_t dst_global;
	int64_t flight_id;
} flight_t;

/* Edge state: planes traversing one directed corridor during one tick. */
typedef struct {
	int32_t  n_flights;
	int32_t  pad;
    int64_t  epoch;
	flight_t flights[EDGE_GATE_CAPACITY];
} gate_slot_t;

/* Vertex state: published every tick, read by peers under --congestion. */
typedef struct {
	int32_t planes_on_ground;
	int32_t pending_landings;
	int64_t landings;
} airport_vertex_t;

extern const gate_slot_t empty_gate;

/*
 * Local CSR slice handed to dstm_dgraph_create(), and reused afterwards to
 * walk our own out-edges without going back through the dgraph API.
 *
 * Torus adjacency is symmetric, so out(v) == in(v); the receive phase walks
 * the same list in reverse orientation.
 */
typedef struct {
	int     rank;
	int     na;
	int     grid;
	size_t  vertex_offset;   /* == rank * na (contiguous rank-order) */
	size_t *out_degrees;     /* [na]              */
	size_t *row_ptr;         /* [na + 1]          */
	size_t *edge_dsts;       /* [edge_count], ascending within each row */
	size_t  edge_count;
} airport_topology_t;

typedef struct {
	int rank;
	int num_ranks;
	int na;
	int grid;
	int congestion;          /* route by neighbor load (remote vertex gets) */
	int publish;             /* publish vertex state each tick */

	const airport_topology_t *topo;
	dstm_dgraph_t            *g;

	gate_slot_t      *outbox;            /* [edge_count] staging */
	gate_slot_t      *recv;              /* scratch, 1 slot      */
	airport_vertex_t *vstate;            /* [na] scratch         */

	int       *planes_on_ground;         /* [na] */
	int       *pending_landings;         /* [na] */
	long long *landings;                 /* [na] */

	long long departures;
	long long arrivals;
	long long overflows;                 /* plane held back, gate full */
	long long misroutes;                 /* sanity: flight on wrong edge */
	long long flight_counter;

	uint64_t rng;
} airport_state_t;

/* Build the local torus CSR slice. Neighbors are deduped and sorted
 * ascending as dstm_dgraph_create() requires. */
int  airport_topology_build(airport_topology_t *topo,
                            int rank, int na, int grid);
void airport_topology_destroy(airport_topology_t *topo);

/* Bytes the dgraph will carve out of the local region, per version. */
size_t airport_topology_region_bytes(const airport_topology_t *topo,
                                     size_t vertex_state_size,
                                     size_t edge_state_size);

int  airport_init(airport_state_t *state,
                  int rank, int num_ranks,
                  int planes_per_airport,
                  int seed, int congestion, int publish,
                  const airport_topology_t *topo,
                  dstm_dgraph_t *g);
void airport_destroy(airport_state_t *state);

/* Create every (edge, t) version up front with an empty gate, so no GET
 * stalls and no tick inherits the previous tick's manifest. */
int  airport_prime(airport_state_t *state, dstm_time_t timesteps);

void airport_timestep(airport_state_t *state, dstm_time_t t);

#endif /* AIRPORT_H */
