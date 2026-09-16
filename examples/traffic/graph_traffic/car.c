/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "car.h"

void car_pool_init(struct car_pool *p)
{
        p->capacity = 256;
        p->size     = 0;
        p->live     = 0;
        p->v        = malloc(p->capacity * sizeof(*p->v));
        assert(p->v != NULL);
}

void car_pool_free(struct car_pool *p)
{
        free(p->v);
        p->v        = NULL;
        p->size     = 0;
        p->capacity = 0;
        p->live     = 0;
}

size_t car_pool_insert(struct car_pool *p, struct car c)
{
        /* Reuse a freed slot if any. O(n) scan, but the pool is small and
         * this keeps the array dense so the per-tick walk stays cheap. */
        for (size_t i = 0; i < p->size; i++) {
                if (p->v[i].id < 0) {
                        p->v[i] = c;
                        p->live++;
                        return i;
                }
        }

        if (p->size == p->capacity) {
                p->capacity *= 2;
                p->v = realloc(p->v, p->capacity * sizeof(*p->v));
                assert(p->v != NULL);
        }

        p->v[p->size] = c;
        p->live++;
        return p->size++;
}

void car_pool_remove(struct car_pool *p, size_t idx)
{
        assert(idx < p->size);
        if (p->v[idx].id < 0)
                return;
        p->v[idx].id = -1;
        p->live--;
}
