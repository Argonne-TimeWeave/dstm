#ifndef ZOMBIE_H
#define ZOMBIE_H

#include <dstm.h>
#include "human.h"

#include <limits.h>
#include <stdio.h>

#define CELL_CAPACITY    64
#define MAX_CELL_SIDE    32   /* max cell_side per rank; strip length cap */
#define LOGGING 0

typedef struct cell_slot_t {
    int      n_humans;
    human_t  humans[CELL_CAPACITY];
    /* Border strip counters. Layout is direction-agnostic: the sender
     * writes the strip of cells facing this particular receiver, and
     * the receiver reads them back in the same order. Length used per
     * bundle is min(cell_side, MAX_CELL_SIDE). Corner neighbors only
     * use index 0. */
    int      strip_population[MAX_CELL_SIDE];
    int      strip_zombie_count[MAX_CELL_SIDE];
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
    int                rows;         /* rank-grid rows */
    int                cols;         /* rank-grid cols */

    /* Local sub-grid geometry: nl == cell_side * cell_side.
     * cell_side is chosen as the smallest s with s*s >= requested_nl. */
    int                nl;
    int                cell_side;

    int                nh;           /* humans + zombies per rank total */
    int                work;

    dstm_distance_t   *topology;
    dstm_darray_t     *cells;

    human_t           *humans;       /* owner record, size nh */

    cell_slot_t       *outbox;       /* size num_ranks; only inter-rank used */

    /* Agents physically on this rank who DIDN'T arrive via the darray
     * this tick (stayed put or moved locally). */
    human_t           *carryover;
    int                n_carryover;
    int                carryover_cap;

    int               *departures;
    int               *arrivals;
    int               *infections;

    /* Per-local-cell caches used by desired_cell() for movement. */
    int               *last_population;
    int               *last_zombie_count;

    /* Per-rank deterministic RNG (independent of global rand()). */
    unsigned int       rng_state;

    /* Per-tick infection count file, rank 0 only. */
    FILE              *infection_csv;

    /* Neighbor rank list (mesh-adjacent, delta==1) — precomputed for the
     * every-tick synchronization pattern. */
    int               *mesh_neighbors;
    int                n_mesh_neighbors;

    /* Ghost view of each mesh neighbor's sub-grid, updated every tick
     * from the inbound cell_slot_t bundles. Indexed by
     * [neighbor_slot][local_cell_id_on_that_neighbor]. */
    int              **ghost_strip_pop;    /* [n_mesh_neighbors][nl] */
    int              **ghost_strip_zom;  /* [n_mesh_neighbors][nl] */
} zombie_state_t;

void zombie_init(zombie_state_t *st,
                 int rank, int num_ranks, int nl, int nh, int nz,
                 int work, int seed,
                 dstm_distance_t *topology,
                 dstm_darray_t   *cells,
                 FILE            *infection_csv);

void zombie_destroy(zombie_state_t *st);

void zombie_timestep(zombie_state_t *st, dstm_time_t t);

int    human_find_next_hop(zombie_state_t *st, int from_rank, int to_rank);
double zombie_do_work(int work);

#endif /* ZOMBIE_H */
