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

struct city {
        size_t nv;              /* vertex count                         */
        size_t ne;              /* directed edge count                  */
        int    world_size;

        /* Global CSR, replicated. */
        size_t *row_ptr;        /* nv + 1                               */
        size_t *dst;            /* ne, ascending within each row        */
        size_t *esrc;           /* ne, source vertex of each edge id    */
        int    *ff_ticks;       /* ne, free-flow travel time in ticks   */

        /* Block partition, identical on every rank. */
        size_t *vtx_off;        /* world_size + 1                       */

        /* Original (sparse) POLARIS node id per dense vertex 0..nv-1.
         * Dense ids are assigned by ascending original id, so every rank
         * builds an identical map from the same CSV with no comms. */
        long   *dense2orig;     /* nv                                   */

        /* Destination-bias set (dense ids): the top-degree vertices.
         * Replaces the grid's central box so congestion concentrates. */
        size_t *downtown;       /* n_downtown                           */
        size_t  n_downtown;
};

/* Load a POLARIS-style network from <dir>/node.csv and <dir>/link.csv.
 * Ignores every field except node ids and link endpoints; each link
 * becomes a bidirectional edge pair. Self-loops and duplicate directed
 * edges are dropped. Deterministic and identical on every rank. */
int  city_load(struct city *c, const char *dir, int world_size);
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
