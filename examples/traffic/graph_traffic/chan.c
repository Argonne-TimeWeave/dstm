/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <math.h>
#include <stdlib.h>

#include "chan.h"

#define CHAN_ALIGN ((size_t)8)

static inline size_t align_up(size_t x, size_t a)
{
        return (x + (a - 1)) & ~(a - 1);
}

void car_layout_init(struct car_layout *L, unsigned int cap)
{
        L->cap      = cap;
        L->cars_off = (uint32_t)align_up(sizeof(struct car_header), CHAN_ALIGN);
        L->bytes    = (size_t)L->cars_off + (size_t)cap * sizeof(struct car_rec);
        L->bytes    = align_up(L->bytes, CHAN_ALIGN);
}

void state_layout_init(struct state_layout *L, size_t n_edges)
{
        L->n_state   = (uint32_t)n_edges;
        L->state_off = (uint32_t)align_up(sizeof(struct state_header),
                                          CHAN_ALIGN);
        L->bytes     = (size_t)L->state_off +
                       n_edges * sizeof(struct cong_entry);
        L->bytes     = align_up(L->bytes, CHAN_ALIGN);
}

void car_reset(const struct car_layout *L, void *buf, int32_t arrive_tick,
               size_t src_vertex, size_t dst_vertex)
{
        struct car_header *h = car_hdr(buf);

        /* Zero the whole buffer: the car tail must not leak the previous
         * version's records, and the receiver relies on n_cars alone. */
        memset(buf, 0, L->bytes);

        h->magic      = CHAN_MAGIC_CAR;
        h->tick       = arrive_tick;
        h->n_cars     = 0;
        h->cap        = L->cap;
        h->cars_off   = L->cars_off;
        h->src_vertex = (uint32_t)src_vertex;
        h->dst_vertex = (uint32_t)dst_vertex;
}

int car_push(const struct car_layout *L, void *buf, const struct car_rec *car)
{
        struct car_header *h = car_hdr(buf);

        if (h->n_cars >= L->cap)
                return -1;

        car_cars(L, buf)[h->n_cars] = *car;
        h->n_cars++;
        return 0;
}

int car_validate(const struct car_layout *L, const void *buf)
{
        const struct car_header *h = car_hdr_c(buf);

        if (h->magic != CHAN_MAGIC_CAR)
                return -1;                      /* never written / zeroed */
        if (h->cars_off != L->cars_off)
                return -1;
        if (h->cap != L->cap || h->n_cars > h->cap)
                return -1;
        return 0;
}

void state_reset(const struct state_layout *L, void *buf, int32_t arrive_tick,
                 size_t src_vertex, size_t dst_vertex)
{
        struct state_header *h = state_hdr(buf);

        memset(buf, 0, L->bytes);

        h->magic      = CHAN_MAGIC_STATE;
        h->tick       = arrive_tick;
        h->n_state    = L->n_state;
        h->state_off  = L->state_off;
        h->src_vertex = (uint32_t)src_vertex;
        h->dst_vertex = (uint32_t)dst_vertex;
}

int state_validate(const struct state_layout *L, const void *buf)
{
        const struct state_header *h = state_hdr_c(buf);

        if (h->magic != CHAN_MAGIC_STATE)
                return -1;                      /* never written / zeroed */
        if (h->n_state != L->n_state)
                return -1;
        if (h->state_off != L->state_off)
                return -1;
        return 0;
}

/*******************************************************************************
 * Congestion replica
 ******************************************************************************/

int cong_replica_init(struct cong_replica *R, size_t n_edges)
{
        R->n = n_edges;
        R->e = malloc(n_edges * sizeof(*R->e));
        if (R->e == NULL)
                return -1;

        for (size_t i = 0; i < n_edges; i++) {
                R->e[i].value = 0.0f;
                R->e[i].stamp = -1;             /* never observed */
        }
        return 0;
}

void cong_replica_free(struct cong_replica *R)
{
        free(R->e);
        R->e = NULL;
        R->n = 0;
}

size_t cong_merge(struct cong_replica *R, const struct cong_entry *in)
{
        size_t updated = 0;

        for (size_t i = 0; i < R->n; i++) {
                if (in[i].stamp > R->e[i].stamp) {
                        R->e[i] = in[i];
                        updated++;
                }
        }
        return updated;
}

float cong_effective(const struct cong_entry *e, int32_t now, float tau,
                     float baseline)
{
        float age;
        float w;

        if (e->stamp < 0)
                return baseline;               /* no information at all */

        age = (float)(now - e->stamp);
        if (age < 0.0f)
                age = 0.0f;

        w = expf(-age / tau);
        return w * e->value + (1.0f - w) * baseline;
}
