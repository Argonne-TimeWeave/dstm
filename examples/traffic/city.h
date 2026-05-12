/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef TRAFFIC_CITY_H
#define TRAFFIC_CITY_H 1

#include <stddef.h>

#include "dstm.h"

/*
 * City model
 * ==========
 *
 * The city is a fixed K x K grid of intersections, augmented with
 * two cross-city highways and a dense downtown block.  Intersection
 * IDs are row-major (id = row * GRID_SIDE + col).
 *
 * The graph is sparse outside downtown (degree ~4) and dense inside
 * (every intersection in downtown is connected to every other one).
 * The two highways add long-range edges along row HIGHWAY_ROW and
 * column HIGHWAY_COL.
 *
 * Partitioning (5 ranks):
 *
 *   rank 0 : downtown (4x4 block in the centre)
 *   rank 1 : NW quadrant (rows 0..GRID_SIDE/2-1, cols 0..GRID_SIDE/2-1)
 *   rank 2 : NE quadrant
 *   rank 3 : SW quadrant
 *   rank 4 : SE quadrant
 *
 * The "downtown" intersections are removed from the suburb partitions
 * and reassigned to rank 0.
 */

#define GRID_SIDE         16
#define NUM_INTERSECTIONS (GRID_SIDE * GRID_SIDE)
#define HIGHWAY_ROW       8
#define HIGHWAY_COL       8
#define DOWNTOWN_R0       6
#define DOWNTOWN_R1       9 /* inclusive */
#define DOWNTOWN_C0       6
#define DOWNTOWN_C1       9 /* inclusive */
#define MAX_NEIGHBORS     32
#define EXPECTED_RANKS    5

/* Rank IDs */
#define RANK_DOWNTOWN 0
#define RANK_NW       1
#define RANK_NE       2
#define RANK_SW       3
#define RANK_SE       4

struct intersection {
	int id;
	int row;
	int col;
	int owner_rank;
	int neighbor_count;
	int neighbors[MAX_NEIGHBORS];
	int travel_time[MAX_NEIGHBORS]; /* ticks to reach neighbor */
};

struct city {
	struct intersection nodes[NUM_INTERSECTIONS];
	/* Per-region accounting */
	int rank_node_count[EXPECTED_RANKS];
	int rank_distance[EXPECTED_RANKS * EXPECTED_RANKS];
	/* Translation table mapping intersection id -> congestion darray
	 * global index.  The darray is ordered such that all of rank 0's
	 * intersections come first (contiguous indices), then rank 1's,
	 * etc., so that the default dstm_darray_create distribution
	 * places each intersection's congestion record on its owning
	 * rank. */
	int cong_index[NUM_INTERSECTIONS];
	/* Per-rank precomputed list of owned intersection ids, used by
	 * the per-tick publish loop.  rank_intersections[r] is valid for
	 * indices [0, rank_node_count[r]). */
	int rank_intersections[EXPECTED_RANKS][NUM_INTERSECTIONS];
};

/* Build the city graph (deterministic, identical on every rank). */
void city_build(struct city *c);

/* Returns rank that owns the given intersection. */
static inline int city_owner(const struct city *c, int id)
{
	return c->nodes[id].owner_rank;
}

/* Build a rank-distance matrix expressed in ticks of minimum travel time
 * between regions, suitable for dstm_distance_create_from_matrix.
 * The matrix lives in c->rank_distance after city_build. */
void city_distance_matrix(const struct city *c, int *out_matrix);

/* Pretty-print summary on rank 0 */
void city_print_summary(const struct city *c);

/* Pick a destination intersection for a vehicle starting at origin.
 * Distribution is biased so that ~60% of trips terminate in downtown.
 * rng_state is a per-thread xorshift state. */
int city_pick_destination(const struct city *c, int origin,
                          unsigned int *rng_state);

/* Returns 1 if intersection id is in the downtown block. */
int city_is_downtown(int row, int col);

#endif /* TRAFFIC_CITY_H */
