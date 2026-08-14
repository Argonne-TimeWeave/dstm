/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef GRAPH_TRAFFIC_CITY_H
#define GRAPH_TRAFFIC_CITY_H

#include <stddef.h>
#include <stdint.h>

/*
 * Replicated road-network topology.
 *
 * The city is a rows x cols 4-connected grid. It is small and generated
 * deterministically from (rows, cols), so every rank builds an identical
 * copy of the *whole* topology in host memory. That is deliberate: the
 * congestion replica is indexed by global edge id, and every rank must
 * agree on that numbering without any communication.
 *
 * Global edge id == index into the global CSR dst[] array. Out-edges of
 * vertex u occupy [row_ptr[u], row_ptr[u+1]) and are sorted by ascending
 * destination, which is exactly the ordering dstm_dgraph_create requires
 * of each rank's local slice.
 *
 * Vertices are partitioned in contiguous blocks in rank order, matching
 * dstm_dgraph_create_equal: rank r owns [vtx_off[r], vtx_off[r+1]).
 * Because ids are row-major, a rank owns a horizontal band of the grid
 * and its rank-neighbors are r-1 and r+1. Congestion news from rank 0
 * therefore needs P-1 ticks to reach rank P-1, which is the whole point
 * of the example.
 */
struct city {
        int    rows;
        int    cols;
        size_t nv;              /* rows * cols                          */
        size_t ne;              /* directed edge count                  */
        int    world_size;

        /* Global CSR, replicated. */
        size_t *row_ptr;        /* nv + 1                               */
        size_t *dst;            /* ne, ascending within each row        */
        size_t *esrc;           /* ne, source vertex of each edge id    */

        /* Block partition, identical on every rank. */
        size_t *vtx_off;        /* world_size + 1                       */

        /* Central "downtown" box: destination bias lands here, so the
         * middle of the grid congests and routing has to work around it. */
        int dt_r0, dt_r1, dt_c0, dt_c1;
};

int  city_build(struct city *c, int rows, int cols, int world_size);
void city_free(struct city *c);

static inline int city_row(const struct city *c, size_t v)
{
        return (int)(v / (size_t)c->cols);
}

static inline int city_col(const struct city *c, size_t v)
{
        return (int)(v % (size_t)c->cols);
}

static inline size_t city_degree(const struct city *c, size_t v)
{
        return c->row_ptr[v + 1] - c->row_ptr[v];
}

/* Owning rank of a vertex, by binary search over vtx_off. */
int city_owner(const struct city *c, size_t v);

/* Global edge id of (u, v), or SIZE_MAX if the two are not adjacent. */
size_t city_edge_id(const struct city *c, size_t u, size_t v);

int city_manhattan(const struct city *c, size_t a, size_t b);

int city_is_downtown(const struct city *c, size_t v);

/* 60% downtown, 40% uniform. Never returns `origin`. */
size_t city_pick_destination(const struct city *c, size_t origin,
                             unsigned int *rng);

void city_print_summary(const struct city *c);

#endif /* GRAPH_TRAFFIC_CITY_H */
