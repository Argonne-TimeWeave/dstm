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
 * Ids are row-major, so a rank owns a contiguous run of vertices -- a
 * horizontal band only when the run happens to align with `cols`, which
 * with the base+remainder split it usually does not. A rank owning fewer
 * than `cols` vertices borders more than just r-1 and r+1.
 *
 * Diffusion delay is measured in GRAPH hops, not rank indices: news
 * about an edge reaches a rank after hopdist(owner, here) ticks, so the
 * worst case tracks the grid diameter (rows-1 + cols-1), independent of
 * P. That delay is the whole point of the example.
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

static inline unsigned int xrand(unsigned int *state)
{
        unsigned int x = *state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *state = x ? x : 1u;
        return *state;
}

/* --- Lookups with no caller today ---------------------------------
 *
 * Both are needed by the generic-loader work (see ai_readme.md): under
 * an arbitrary partition, owner lookup stops being a block-range
 * comparison and edge-id lookup stops being gslot_base + local_slot.
 * Until then traffic.c gets both by arithmetic. */

/* Owning rank of a vertex, by binary search over vtx_off. */
int city_owner(const struct city *c, size_t v);

/* Global edge id of (u, v), or SIZE_MAX if the two are not adjacent. */
size_t city_edge_id(const struct city *c, size_t u, size_t v);

/* --- End no-caller block ------------------------------------------ */

/* 60% downtown, 40% uniform. Never returns `origin`. */
size_t city_pick_destination(const struct city *c, size_t origin,
                             unsigned int *rng);

void city_print_summary(const struct city *c);

#endif /* GRAPH_TRAFFIC_CITY_H */
