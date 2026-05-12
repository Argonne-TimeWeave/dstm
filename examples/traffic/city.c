/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <limits.h>
#include <stdio.h>
#include <string.h>

#include "city.h"

/* Simple xorshift32 to avoid pulling in <random> dependencies and to keep
 * the simulation deterministic per (rank, seed). */
static unsigned int xrand(unsigned int *state)
{
	unsigned int x = *state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	*state = x ? x : 1u;
	return *state;
}

int city_is_downtown(int row, int col)
{
	return row >= DOWNTOWN_R0 && row <= DOWNTOWN_R1 &&
	       col >= DOWNTOWN_C0 && col <= DOWNTOWN_C1;
}

static int classify_owner(int row, int col)
{
	if (city_is_downtown(row, col))
		return RANK_DOWNTOWN;
	int north = row < GRID_SIDE / 2;
	int west = col < GRID_SIDE / 2;
	if (north && west)
		return RANK_NW;
	if (north && !west)
		return RANK_NE;
	if (!north && west)
		return RANK_SW;
	return RANK_SE;
}

static void add_edge(struct intersection *n, int neighbor, int travel)
{
	if (n->neighbor_count >= MAX_NEIGHBORS)
		return;
	/* Skip duplicates */
	for (int k = 0; k < n->neighbor_count; k++) {
		if (n->neighbors[k] == neighbor) {
			if (travel < n->travel_time[k])
				n->travel_time[k] = travel;
			return;
		}
	}
	n->neighbors[n->neighbor_count] = neighbor;
	n->travel_time[n->neighbor_count] = travel;
	n->neighbor_count++;
}

static int id_of(int row, int col)
{
	return row * GRID_SIDE + col;
}

void city_build(struct city *c)
{
	memset(c, 0, sizeof(*c));

	/* Initialise nodes */
	for (int r = 0; r < GRID_SIDE; r++) {
		for (int col = 0; col < GRID_SIDE; col++) {
			int id = id_of(r, col);
			c->nodes[id].id = id;
			c->nodes[id].row = r;
			c->nodes[id].col = col;
			c->nodes[id].owner_rank = classify_owner(r, col);
			c->rank_node_count[c->nodes[id].owner_rank]++;
		}
	}

	/* Grid edges (4-connected), 1 tick travel time */
	for (int r = 0; r < GRID_SIDE; r++) {
		for (int col = 0; col < GRID_SIDE; col++) {
			int id = id_of(r, col);
			if (col + 1 < GRID_SIDE) {
				add_edge(&c->nodes[id], id_of(r, col + 1), 1);
				add_edge(&c->nodes[id_of(r, col + 1)], id, 1);
			}
			if (r + 1 < GRID_SIDE) {
				add_edge(&c->nodes[id], id_of(r + 1, col), 1);
				add_edge(&c->nodes[id_of(r + 1, col)], id, 1);
			}
		}
	}

	/* Downtown dense mesh (every pair connected, 1 tick) */
	for (int r1 = DOWNTOWN_R0; r1 <= DOWNTOWN_R1; r1++) {
		for (int c1 = DOWNTOWN_C0; c1 <= DOWNTOWN_C1; c1++) {
			int a = id_of(r1, c1);
			for (int r2 = DOWNTOWN_R0; r2 <= DOWNTOWN_R1; r2++) {
				for (int c2 = DOWNTOWN_C0;
				     c2 <= DOWNTOWN_C1; c2++) {
					if (r1 == r2 && c1 == c2)
						continue;
					int b = id_of(r2, c2);
					add_edge(&c->nodes[a], b, 1);
				}
			}
		}
	}

	/* Highway: every pair on HIGHWAY_ROW connected with 1 tick. */
	for (int c1 = 0; c1 < GRID_SIDE; c1++) {
		int a = id_of(HIGHWAY_ROW, c1);
		for (int c2 = 0; c2 < GRID_SIDE; c2++) {
			if (c1 == c2)
				continue;
			int b = id_of(HIGHWAY_ROW, c2);
			add_edge(&c->nodes[a], b, 1);
		}
	}
	/* Highway: every pair on HIGHWAY_COL connected with 1 tick. */
	for (int r1 = 0; r1 < GRID_SIDE; r1++) {
		int a = id_of(r1, HIGHWAY_COL);
		for (int r2 = 0; r2 < GRID_SIDE; r2++) {
			if (r1 == r2)
				continue;
			int b = id_of(r2, HIGHWAY_COL);
			add_edge(&c->nodes[a], b, 1);
		}
	}

	/* Build region distance matrix.  Use BFS over the city graph,
	 * collapsing to region adjacency: distance(i,j) = minimum number
	 * of intersection-edge traversals between any node of region i
	 * and any node of region j (in ticks).
	 *
	 * For our city, suburbs share grid borders with downtown and
	 * with each other along the highway lines, so:
	 *   - any suburb <-> any suburb: 1 (highway shortcut, both touch
	 *     row 8 or col 8, except diagonal pairs which are 2).
	 *   - any suburb <-> downtown:   1 (downtown is adjacent in grid).
	 *
	 * We compute it from the actual graph rather than hardcoding it,
	 * so the matrix stays correct if constants change. */
	for (int i = 0; i < EXPECTED_RANKS; i++) {
		for (int j = 0; j < EXPECTED_RANKS; j++) {
			c->rank_distance[i * EXPECTED_RANKS + j] =
			        (i == j) ? 0 : INT_MAX;
		}
	}

	/* For every directed edge (u,v) where u in region A and v in
	 * region B != A, set rank_distance[A][B] = min(current, 1).
	 * Then run Floyd-Warshall to fill in multi-hop region distances. */
	for (int u = 0; u < NUM_INTERSECTIONS; u++) {
		int ra = c->nodes[u].owner_rank;
		for (int k = 0; k < c->nodes[u].neighbor_count; k++) {
			int v = c->nodes[u].neighbors[k];
			int rb = c->nodes[v].owner_rank;
			if (ra == rb)
				continue;
			int t = c->nodes[u].travel_time[k];
			int cur = c->rank_distance[ra * EXPECTED_RANKS + rb];
			if (t < cur)
				c->rank_distance[ra * EXPECTED_RANKS + rb] = t;
		}
	}

	/* Floyd-Warshall for region distances */
	for (int k = 0; k < EXPECTED_RANKS; k++) {
		for (int i = 0; i < EXPECTED_RANKS; i++) {
			for (int j = 0; j < EXPECTED_RANKS; j++) {
				int ik = c->rank_distance[i * EXPECTED_RANKS +
				                          k];
				int kj = c->rank_distance[k * EXPECTED_RANKS +
				                          j];
				if (ik == INT_MAX || kj == INT_MAX)
					continue;
				int via = ik + kj;
				if (via <
				    c->rank_distance[i * EXPECTED_RANKS + j])
					c->rank_distance[i * EXPECTED_RANKS +
					                 j] = via;
			}
		}
	}

	/* Sanity: no INT_MAX should remain (graph is connected). */
	for (int i = 0; i < EXPECTED_RANKS; i++) {
		for (int j = 0; j < EXPECTED_RANKS; j++) {
			if (c->rank_distance[i * EXPECTED_RANKS + j] ==
			    INT_MAX)
				c->rank_distance[i * EXPECTED_RANKS + j] = 1;
		}
	}

	/* Build congestion-darray index map: intersections grouped by
	 * owner rank so that dstm_darray_create's contiguous distribution
	 * matches our intersection->rank mapping. */
	int cursor[EXPECTED_RANKS];
	cursor[0] = 0;
	for (int r = 1; r < EXPECTED_RANKS; r++)
		cursor[r] = cursor[r - 1] + c->rank_node_count[r - 1];
	for (int id = 0; id < NUM_INTERSECTIONS; id++) {
		int r = c->nodes[id].owner_rank;
		c->cong_index[id] = cursor[r]++;
	}

	/* Build per-rank lists of owned intersections for the publish
	 * loop.  Reuse cursor[] as the per-rank write index. */
	for (int r = 0; r < EXPECTED_RANKS; r++)
		cursor[r] = 0;
	for (int id = 0; id < NUM_INTERSECTIONS; id++) {
		int r = c->nodes[id].owner_rank;
		c->rank_intersections[r][cursor[r]++] = id;
	}
}

void city_distance_matrix(const struct city *c, int *out_matrix)
{
	memcpy(out_matrix, c->rank_distance,
	       sizeof(int) * EXPECTED_RANKS * EXPECTED_RANKS);
}

void city_print_summary(const struct city *c)
{
	printf("=== City summary ===\n");
	printf("Grid: %d x %d (%d intersections)\n", GRID_SIDE, GRID_SIDE,
	       NUM_INTERSECTIONS);
	printf("Downtown: rows %d-%d, cols %d-%d (%d intersections)\n",
	       DOWNTOWN_R0, DOWNTOWN_R1, DOWNTOWN_C0, DOWNTOWN_C1,
	       (DOWNTOWN_R1 - DOWNTOWN_R0 + 1) *
	               (DOWNTOWN_C1 - DOWNTOWN_C0 + 1));
	printf("Highways: row %d, col %d\n", HIGHWAY_ROW, HIGHWAY_COL);
	printf("Per-rank intersection counts:\n");
	const char *names[EXPECTED_RANKS] = {"downtown", "NW", "NE", "SW",
	                                     "SE"};
	for (int r = 0; r < EXPECTED_RANKS; r++)
		printf("  rank %d (%-8s): %d intersections\n", r, names[r],
		       c->rank_node_count[r]);

	printf("Region distance matrix (ticks of min travel time):\n");
	printf("        ");
	for (int j = 0; j < EXPECTED_RANKS; j++)
		printf("%8s", names[j]);
	printf("\n");
	for (int i = 0; i < EXPECTED_RANKS; i++) {
		printf("%-8s", names[i]);
		for (int j = 0; j < EXPECTED_RANKS; j++)
			printf("%8d",
			       c->rank_distance[i * EXPECTED_RANKS + j]);
		printf("\n");
	}
	printf("====================\n");
}

int city_pick_destination(const struct city *c, int origin,
                          unsigned int *rng_state)
{
	(void)c;
	int dst;
	/* 60% downtown, 40% uniform random non-origin. */
	if ((xrand(rng_state) % 100u) < 60u) {
		int dr = DOWNTOWN_R0 +
		         (int)(xrand(rng_state) %
		               (unsigned int)(DOWNTOWN_R1 - DOWNTOWN_R0 + 1));
		int dc = DOWNTOWN_C0 +
		         (int)(xrand(rng_state) %
		               (unsigned int)(DOWNTOWN_C1 - DOWNTOWN_C0 + 1));
		dst = id_of(dr, dc);
	} else {
		dst = (int)(xrand(rng_state) %
		            (unsigned int)NUM_INTERSECTIONS);
	}
	if (dst == origin)
		dst = (origin + 1) % NUM_INTERSECTIONS;
	return dst;
}
