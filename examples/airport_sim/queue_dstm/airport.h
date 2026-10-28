#ifndef AIRPORT_H
#define AIRPORT_H

#include <stddef.h>
#include <stdint.h>
#include <dstm.h>
#include <mpi.h>

/*
 * Queue-backed airport torus benchmark.
 *
 * This is the dstm_dqueue twin of simple_dstm (dstm_darray) and graph_dstm
 * (dstm_dgraph). Same toroidal airport grid, same runway-serialized landing
 * model, same conservation invariant, same RNG trajectory -- only the
 * transport substrate differs.
 *
 * ROUTING MODEL.  darray routes by rank-pair gate slot and dgraph routes by
 * directed edge. dqueue routes by RANK: a departing plane is addressed to
 * the OWNING RANK of its destination airport and carries that destination
 * airport's global id in its payload so the consumer delivers it locally.
 * Same-rank planes never enter the queue; they are delivered in-process,
 * exactly like simple_dstm/graph_dstm.
 *
 * COLLISIONS.  darray needed one slot per (dest,src) rank-pair to avoid
 * wholesale-overwrite collisions. dqueue dissolves the problem: every
 * producer APPENDS into the owner's single slot in MPI arrival order, so
 * there is no per-neighbor mailbox to index and no collision to design
 * around.
 */

/*
 * Per-tick landing budget. Byte-identical to the other two backends:
 * ROSS queues arrivals behind a runway with mean service MEAN_LAND=10
 * virtual units, one DSTM tick is MEAN_DEPARTURE=30, hence ~3 landings
 * per airport per tick.
 */
#define LANDINGS_PER_STEP 3

/*
 * dqueue capacity sizing.  Because every producer appends into ONE slot per
 * owner, capacity must cover the SUM over all producers of their worst-case
 * per-tick contribution to a single owner -- not one producer's worth.
 *
 * Worst case: every neighbor rank sends all of its departing planes to this
 * rank in one tick. With a bounded ground population (LANDINGS_PER_STEP caps
 * sustained growth) and the t=0 burst of planes_per_airport, this stays
 * modest. main() computes the exact value from the run parameters and sizes
 * the queue accordingly; this is only a floor used before that calc.
 */
#ifndef QUEUE_CAPACITY_FLOOR
#define QUEUE_CAPACITY_FLOOR 256
#endif

#define PLANE_SENTINEL (-1)

/*
 * One plane in flight through the queue. Fixed-size POD, no embedded
 * pointers (dqueue requirement). It carries its destination airport id so
 * the consumer can route it to the right local airport after pop.
 */
typedef struct {
        int64_t flight_id;
        int32_t dst_global;     /* destination airport global id            */
        int32_t src_global;     /* source airport global id (diagnostics)   */
} plane_msg_t;

typedef struct {
        int rank;
        int num_ranks;
        int na;                 /* airports per rank                        */
        int grid;               /* torus side length                        */

        dstm_distance_t *topology;
        dstm_dqueue_t   *queue;

        /* Host-side per-airport state (not versioned: the queue carries all
         * distributed state, like mini_airport_q). */
        long long *landings;              /* [na] */
        int       *planes_on_the_ground;  /* [na] */
        int       *pending_landings;      /* [na] */

        long long departures;
        long long arrivals;               /* cross-rank arrivals via queue  */
        long long overflows;              /* plane held back, queue full     */
        long long misroutes;              /* plane delivered to wrong rank   */
        long long flight_counter;

        /* Pop scratch buffer, sized to dstm_dqueue_max_incoming() once. */
        plane_msg_t *inbox;
        size_t       inbox_cap;
} airport_state_t;

static inline int airport_owner(int global_id, int na) {
        return global_id / na;
}

/* Global id of the torus neighbor of gid in direction dir (0=N,1=S,2=E,3=W).
 * Direction encoding is byte-for-byte the same as the other two backends. */
int airport_neighbor_of(int gid, int grid, int dir);

void airport_init(airport_state_t *state,
                  int rank,
                  int num_ranks,
                  int na,
                  int grid,
                  int planes_per_airport,
                  dstm_distance_t *topology,
                  dstm_dqueue_t *queue);

void airport_destroy(airport_state_t *state);

void airport_timestep(airport_state_t *state, dstm_time_t t);

/* Final drain after the last tick: pops version `t_last` so any plane still
 * in flight is counted (into pending_landings) for the conservation check.
 * Returns the number of planes drained. */
long long airport_drain_inflight(airport_state_t *state, dstm_time_t t_last);

#endif /* AIRPORT_H */
