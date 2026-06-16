#ifndef ZOMBIE_H
#define ZOMBIE_H

#include <dstm.h>
#include "human.h"

/* Popularity tiers for local cells (decided deterministically from seed,
 * identical on every rank — same pattern as airport tier assignment). */
typedef enum {
    LOC_POPULAR = 0,   /* e.g. stadium, mall    */
    LOC_NORMAL  = 1,   /* e.g. office, school   */
    LOC_QUIET   = 2    /* e.g. park, residence  */
} loc_tier_t;

extern const char *loc_tier_name[];
extern const int   loc_tier_weight[];   /* sampling weights for event picks */

/* Per-rank bundle slot in the global darray.
 *
 * Layout mirrors gate_slot_t in the airport sim:
 *   The darray is indexed by (dest_rank, src_rank), so each rank has
 *   exactly num_ranks bundles addressed to it — one from every potential
 *   sender. Bundles carry humans whose `cur_node == dest_rank` (i.e.
 *   they have landed on dest_rank's territory this hour).
 *
 * The destination *local* cell lives inside each human_t (dest_local /
 * cur_local), exactly like flight_t.dest_local.
 *
 * CELL_CAPACITY is per-bundle, so the total carrying capacity per
 * (dest_rank, src_rank) pair per hour is CELL_CAPACITY humans. */
#define CELL_CAPACITY  32

typedef struct cell_slot_t {
    int      n_humans;
    human_t  humans[CELL_CAPACITY];
} cell_slot_t;

extern cell_slot_t empty_cell;

/* Index helpers ---------------------------------------------------------- */

/* darray index: one bundle per (dest_rank, src_rank). */
static inline size_t bundle_index(int dest_rank, int src_rank, int num_ranks) {
    return (size_t)dest_rank * (size_t)num_ranks + (size_t)src_rank;
}

/* Global cell index for tier lookups. */
static inline size_t cell_index(int node, int local, int nl) {
    return (size_t)node * (size_t)nl + (size_t)local;
}

/* Per-rank state. Mirrors airport_state_t. */
typedef struct zombie_state_t {
    int                rank;
    int                num_ranks;
    int                rows;          /* mesh rows */
    int                cols;          /* mesh cols */
    int                nl;            /* local cells per rank */
    int                nh;            /* humans per rank */
    int                work;          /* synthetic per-step work knob */

    loc_tier_t        *loc_tiers;     /* size = num_ranks * nl, same on all ranks */
    dstm_distance_t   *topology;
    dstm_darray_t     *cells;         /* darray of cell_slot_t */

    /* Authoritative copy of humans this rank owns (home == rank).
     * Owner re-rolls their schedule each morning and applies infection
     * reports that arrive in bundles addressed to home. */
    human_t           *humans;        /* size nh */

    /* Per-remote-rank outbox of humans we're shipping there this hour. */
    cell_slot_t       *outbox;        /* size num_ranks */

    /* Counters gathered at the end (sized nl, indexed by local cell). */
    int               *departures;
    int               *arrivals;
    int               *infections;
} zombie_state_t;

/* Lifecycle */
void zombie_init(zombie_state_t *st,
                 int rank, int num_ranks, int nl, int nh,
                 int work, int seed,
                 loc_tier_t *loc_tiers,
                 dstm_distance_t *topology,
                 dstm_darray_t   *cells);

void zombie_destroy(zombie_state_t *st);

/* Main per-hour step. */
void zombie_timestep(zombie_state_t *st, dstm_time_t t, int quiet);

/* Routing + helpers. */
int    human_find_next_hop(zombie_state_t *st, int from_rank, int to_rank);
void   pick_event_location(zombie_state_t *st, event_slot_t slot,
                           int *out_node, int *out_local);
double zombie_do_work(int work);

#endif /* ZOMBIE_H */
