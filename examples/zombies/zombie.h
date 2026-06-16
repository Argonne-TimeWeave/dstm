#ifndef ZOMBIE_H
#define ZOMBIE_H

#include <dstm.h>
#include "human.h"

typedef enum {
    LOC_POPULAR = 0,
    LOC_NORMAL  = 1,
    LOC_QUIET   = 2
} loc_tier_t;

extern const char *loc_tier_name[];
extern const int   loc_tier_weight[];

#define CELL_CAPACITY  64    /* per-bundle, only for inter-rank traffic */

typedef struct cell_slot_t {
    int      n_humans;
    human_t  humans[CELL_CAPACITY];
} cell_slot_t;

extern cell_slot_t empty_cell;

static inline size_t bundle_index(int dest_rank, int src_rank, int num_ranks) {
    return (size_t)dest_rank * (size_t)num_ranks + (size_t)src_rank;
}

static inline size_t cell_index(int node, int local, int nl) {
    return (size_t)node * (size_t)nl + (size_t)local;
}

typedef struct zombie_state_t {
    int                rank;
    int                num_ranks;
    int                rows;
    int                cols;
    int                nl;
    int                nh;
    int                work;

    loc_tier_t        *loc_tiers;
    dstm_distance_t   *topology;
    dstm_darray_t     *cells;

    human_t           *humans;        /* owner record, size nh */

    cell_slot_t       *outbox;        /* size num_ranks; only inter-rank used */

    /* Humans physically on this rank who DIDN'T arrive via the darray
     * this hour (stayed put, or were home all along).  */
    human_t           *carryover;
    int                n_carryover;
    int                carryover_cap;

    int               *departures;
    int               *arrivals;
    int               *infections;
} zombie_state_t;

void zombie_init(zombie_state_t *st,
                 int rank, int num_ranks, int nl, int nh,
                 int work, int seed,
                 loc_tier_t *loc_tiers,
                 dstm_distance_t *topology,
                 dstm_darray_t   *cells);

void zombie_destroy(zombie_state_t *st);

void zombie_timestep(zombie_state_t *st, dstm_time_t t);

int    human_find_next_hop(zombie_state_t *st, int from_rank, int to_rank);
void   pick_event_location(zombie_state_t *st, event_slot_t slot,
                           int *out_node, int *out_local);
double zombie_do_work(int work);

#endif /* ZOMBIE_H */
