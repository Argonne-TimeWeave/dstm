/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef GRAPH_TRAFFIC_CAR_H
#define GRAPH_TRAFFIC_CAR_H

#include <stddef.h>
#include <stdint.h>

/* Host-side agent. Overlaps struct car_rec on id/origin/dest/
 * depart_tick/hops but is NOT a superset: `at` and `ready_tick` are
 * local-only and never go on the wire, and car_rec's `dst_vertex` has no
 * counterpart here -- it becomes `at` when inject_car lands the agent. */
struct car {
        int32_t id;             /* < 0 marks a free slot                */
        int32_t origin;
        int32_t dest;
        int32_t at;             /* current vertex, always locally owned */
        int32_t depart_tick;
        int32_t hops;
        int32_t ready_tick;     /* may depart when tick >= ready_tick   */
};

struct car_pool {
        struct car *v;
        size_t      size;
        size_t      capacity;
        size_t      live;
};

void   car_pool_init(struct car_pool *p);
void   car_pool_free(struct car_pool *p);
size_t car_pool_insert(struct car_pool *p, struct car c);
void   car_pool_remove(struct car_pool *p, size_t idx);

#endif /* GRAPH_TRAFFIC_CAR_H */
