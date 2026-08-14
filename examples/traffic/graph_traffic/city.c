/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "city.h"

static inline size_t vid(const struct city *c, int r, int col)
{
        return (size_t)r * (size_t)c->cols + (size_t)col;
}

int city_build(struct city *c, int rows, int cols, int world_size)
{
        memset(c, 0, sizeof(*c));

        if (rows < 2 || cols < 2 || world_size < 1)
                return -1;

        c->rows       = rows;
        c->cols       = cols;
        c->nv         = (size_t)rows * (size_t)cols;
        c->world_size = world_size;

        if ((size_t)world_size > c->nv)
                return -1;

        c->row_ptr = malloc((c->nv + 1) * sizeof(*c->row_ptr));
        c->vtx_off = malloc(((size_t)world_size + 1) * sizeof(*c->vtx_off));
        if (c->row_ptr == NULL || c->vtx_off == NULL)
                goto oom;

        /* --- Degrees and row pointers --------------------------------- */
        {
                size_t acc = 0;
                for (int r = 0; r < rows; r++) {
                        for (int col = 0; col < cols; col++) {
                                size_t v   = vid(c, r, col);
                                size_t deg = 0;

                                if (r > 0)
                                        deg++;
                                if (col > 0)
                                        deg++;
                                if (col + 1 < cols)
                                        deg++;
                                if (r + 1 < rows)
                                        deg++;

                                c->row_ptr[v] = acc;
                                acc += deg;
                        }
                }
                c->row_ptr[c->nv] = acc;
                c->ne             = acc;
        }

        c->dst  = malloc(c->ne * sizeof(*c->dst));
        c->esrc = malloc(c->ne * sizeof(*c->esrc));
        if (c->dst == NULL || c->esrc == NULL)
                goto oom;

        /* --- Adjacency, ascending by destination ----------------------
         *
         * Row-major ids make the ascending order exactly:
         *   up (v - cols), left (v - 1), right (v + 1), down (v + cols).
         * dstm_dgraph_create validates this ordering, so emit it directly
         * rather than sorting afterwards. */
        {
                size_t w = 0;
                for (int r = 0; r < rows; r++) {
                        for (int col = 0; col < cols; col++) {
                                size_t v = vid(c, r, col);

                                if (r > 0) {
                                        c->esrc[w] = v;
                                        c->dst[w++] = vid(c, r - 1, col);
                                }
                                if (col > 0) {
                                        c->esrc[w] = v;
                                        c->dst[w++] = vid(c, r, col - 1);
                                }
                                if (col + 1 < cols) {
                                        c->esrc[w] = v;
                                        c->dst[w++] = vid(c, r, col + 1);
                                }
                                if (r + 1 < rows) {
                                        c->esrc[w] = v;
                                        c->dst[w++] = vid(c, r + 1, col);
                                }
                        }
                }
        }

        /* --- Block partition, matching dstm_dgraph_create_equal -------- */
        {
                size_t base = c->nv / (size_t)world_size;
                size_t rem  = c->nv % (size_t)world_size;
                size_t acc  = 0;

                for (int r = 0; r < world_size; r++) {
                        c->vtx_off[r] = acc;
                        acc += base + ((size_t)r < rem ? 1u : 0u);
                }
                c->vtx_off[world_size] = acc;
        }

        /* --- Downtown box: central quarter of the grid ----------------- */
        c->dt_r0 = rows / 4;
        c->dt_r1 = rows - rows / 4 - 1;
        c->dt_c0 = cols / 4;
        c->dt_c1 = cols - cols / 4 - 1;

        return 0;

oom:
        city_free(c);
        return -1;
}

void city_free(struct city *c)
{
        if (c == NULL)
                return;
        free(c->row_ptr);
        free(c->dst);
        free(c->esrc);
        free(c->vtx_off);
        memset(c, 0, sizeof(*c));
}

int city_owner(const struct city *c, size_t v)
{
        int lo = 0;
        int hi = c->world_size - 1;

        while (lo < hi) {
                int mid = lo + ((hi - lo + 1) / 2);
                if (c->vtx_off[mid] <= v)
                        lo = mid;
                else
                        hi = mid - 1;
        }
        return lo;
}

size_t city_edge_id(const struct city *c, size_t u, size_t v)
{
        size_t lo = c->row_ptr[u];
        size_t hi = c->row_ptr[u + 1];

        while (lo < hi) {
                size_t mid = lo + (hi - lo) / 2;
                if (c->dst[mid] == v)
                        return mid;
                if (c->dst[mid] < v)
                        lo = mid + 1;
                else
                        hi = mid;
        }
        return (size_t)-1;
}

size_t city_pick_destination(const struct city *c, size_t origin,
                             unsigned int *rng)
{
        size_t dst;

        if ((xrand(rng) % 100u) < 60u) {
                int nr = c->dt_r1 - c->dt_r0 + 1;
                int nc = c->dt_c1 - c->dt_c0 + 1;
                int r  = c->dt_r0 + (int)(xrand(rng) % (unsigned int)nr);
                int k  = c->dt_c0 + (int)(xrand(rng) % (unsigned int)nc);

                dst = vid(c, r, k);
        } else {
                dst = (size_t)(xrand(rng) % (unsigned int)c->nv);
        }

        if (dst == origin)
                dst = (origin + 1) % c->nv;
        return dst;
}

void city_print_summary(const struct city *c)
{
        printf("=== City summary ===\n");
        printf("Grid            : %d x %d  (%zu vertices, %zu directed edges)\n",
               c->rows, c->cols, c->nv, c->ne);
        printf("Downtown box    : rows %d-%d, cols %d-%d\n",
               c->dt_r0, c->dt_r1, c->dt_c0, c->dt_c1);
        printf("Ranks           : %d (contiguous row-major vertex ranges)\n",
               c->world_size);
        for (int r = 0; r < c->world_size; r++) {
                size_t lo = c->vtx_off[r];
                size_t hi = c->vtx_off[r + 1];
                size_t le = c->row_ptr[hi] - c->row_ptr[lo];

                printf("  rank %-3d: vertices [%zu, %zu)  %zu vtx, %zu out-edges\n",
                       r, lo, hi, hi - lo, le);
        }
        printf("====================\n");
}
