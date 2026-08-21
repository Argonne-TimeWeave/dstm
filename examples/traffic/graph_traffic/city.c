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

/* --- comparators --------------------------------------------------- */

static int cmp_long(const void *a, const void *b)
{
        long x = *(const long *)a, y = *(const long *)b;
        return (x > y) - (x < y);
}

static int cmp_szt(const void *a, const void *b)
{
        size_t x = *(const size_t *)a, y = *(const size_t *)b;
        return (x > y) - (x < y);
}

/* Original id -> dense id.  dense2orig is no longer sorted (dense ids are
 * assigned by a spatial key), so we search a separate index sorted by
 * original id.  idx[k] holds (orig, dense) pairs in ascending orig. */
struct orig_index { long orig; size_t dense; };

static int cmp_orig_index(const void *a, const void *b)
{
        long x = ((const struct orig_index *)a)->orig;
        long y = ((const struct orig_index *)b)->orig;
        return (x > y) - (x < y);
}

static size_t dense_of(const struct orig_index *idx, size_t nv, long orig)
{
        size_t lo = 0, hi = nv;
        while (lo < hi) {
                size_t mid = lo + (hi - lo) / 2;
                if (idx[mid].orig == orig)
                        return idx[mid].dense;
                if (idx[mid].orig < orig)
                        lo = mid + 1;
                else
                        hi = mid;
        }
        return (size_t)-1;
}

/* First three CSV columns (node id, x, y) of every non-header line.
 * ids_out gets the id; xy_out gets 2 doubles per node (x then y).
 * Returns count or -1. */
static long read_nodes_xy(const char *path, long **ids_out, double **xy_out)
{
        FILE *f = fopen(path, "r");
        if (!f) { perror(path); return -1; }

        size_t cap = 1024, n = 0;
        long   *ids = malloc(cap * sizeof(*ids));
        double *xy  = malloc(cap * 2 * sizeof(*xy));
        if (!ids || !xy) { free(ids); free(xy); fclose(f); return -1; }

        char line[8192];
        int  first = 1;
        while (fgets(line, sizeof(line), f)) {
                if (first) { first = 0; continue; }
                long   v;
                double x, y;
                /* node,x,y,... */
                if (sscanf(line, "%ld,%lf,%lf,", &v, &x, &y) != 3)
                        continue;
                if (n == cap) {
                        cap *= 2;
                        long   *ti = realloc(ids, cap * sizeof(*ids));
                        double *tx = realloc(xy, cap * 2 * sizeof(*xy));
                        if (!ti || !tx) {
                                free(ti ? ti : ids);
                                free(tx ? tx : xy);
                                fclose(f);
                                return -1;
                        }
                        ids = ti; xy = tx;
                }
                ids[n] = v;
                xy[2 * n] = x;
                xy[2 * n + 1] = y;
                n++;
        }
        fclose(f);
        *ids_out = ids;
        *xy_out  = xy;
        return (long)n;
}

/* node_a (col 3) and node_b (col 4) of every link line into two parallel
 * arrays. Returns count or -1.
 *
 * The `name` column is TEXT and unquoted in this export, but a name could
 * in principle contain a comma. We locate node_a/node_b by counting
 * commas from the LEFT is unsafe; instead we parse: link is col0 (int),
 * then skip exactly one field (name), then two ints. sscanf with
 * %*[^,] consumes the name up to the next comma -- correct as long as the
 * name is unquoted and comma-free, which holds for this dataset. If that
 * ever breaks, switch to a real quoted-CSV split. */
static long read_link_pairs(const char *path, long **a_out, long **b_out)
{
        FILE *f = fopen(path, "r");
        if (!f) { perror(path); return -1; }

        size_t cap = 1024, n = 0;
        long  *a = malloc(cap * sizeof(*a));
        long  *b = malloc(cap * sizeof(*b));
        if (!a || !b) { free(a); free(b); fclose(f); return -1; }

        char line[8192];
        int  first = 1;
        while (fgets(line, sizeof(line), f)) {
                if (first) { first = 0; continue; }
                long link, na, nb;
                /* name present: link,name,na,nb,... */
                if (sscanf(line, "%ld,%*[^,],%ld,%ld,", &link, &na, &nb) == 3)
                        ;
                /* name empty: link,,na,nb,... */
                else if (sscanf(line, "%ld,,%ld,%ld,", &link, &na, &nb) == 3)
                        ;
                else
                        continue;

                if (n == cap) {
                        cap *= 2;
                        long *ta = realloc(a, cap * sizeof(*a));
                        long *tb = realloc(b, cap * sizeof(*b));
                        if (!ta || !tb) {
                                free(ta ? ta : a);
                                free(tb ? tb : b);
                                fclose(f);
                                return -1;
                        }
                        a = ta; b = tb;
                }
                a[n] = na; b[n] = nb; n++;
        }
        fclose(f);
        *a_out = a; *b_out = b;
        return (long)n;
}

/* --- Spatial ordering (recursive coordinate bisection) -------------
 *
 * Produces a permutation of vertex indices such that spatially-close
 * vertices are contiguous.  The block split over vtx_off then yields
 * spatially-compact partitions with no change to gslot_base addressing.
 *
 * We sort an array of point structs in place; each carries its x,y and a
 * payload (the original id).  rcb() alternates split axis by depth and
 * recurses until the segment maps to a single output rank's share is not
 * needed here -- we just want a full spatial order, so we recurse down to
 * segments of size 1. */
struct rcb_pt {
        double x, y;
        long   orig;    /* original node id, carried through the sort */
};

static int cmp_rcb_x(const void *a, const void *b)
{
        double xa = ((const struct rcb_pt *)a)->x;
        double xb = ((const struct rcb_pt *)b)->x;
        if (xa < xb) return -1;
        if (xa > xb) return 1;
        /* stable tie-break by orig id so the order is deterministic */
        long oa = ((const struct rcb_pt *)a)->orig;
        long ob = ((const struct rcb_pt *)b)->orig;
        return (oa > ob) - (oa < ob);
}

static int cmp_rcb_y(const void *a, const void *b)
{
        double ya = ((const struct rcb_pt *)a)->y;
        double yb = ((const struct rcb_pt *)b)->y;
        if (ya < yb) return -1;
        if (ya > yb) return 1;
        long oa = ((const struct rcb_pt *)a)->orig;
        long ob = ((const struct rcb_pt *)b)->orig;
        return (oa > ob) - (oa < ob);
}

/* Sort [lo,hi) by the longer extent (x or y), then recurse on halves.
 * depth is unused for axis choice (we pick the wider spread each level),
 * which keeps cells roughly square. */
static void rcb(struct rcb_pt *p, size_t lo, size_t hi)
{
        if (hi - lo <= 1)
                return;

        double xmin = p[lo].x, xmax = p[lo].x;
        double ymin = p[lo].y, ymax = p[lo].y;
        for (size_t i = lo + 1; i < hi; i++) {
                if (p[i].x < xmin) xmin = p[i].x;
                if (p[i].x > xmax) xmax = p[i].x;
                if (p[i].y < ymin) ymin = p[i].y;
                if (p[i].y > ymax) ymax = p[i].y;
        }

        if ((xmax - xmin) >= (ymax - ymin))
                qsort(p + lo, hi - lo, sizeof(*p), cmp_rcb_x);
        else
                qsort(p + lo, hi - lo, sizeof(*p), cmp_rcb_y);

        size_t mid = lo + (hi - lo) / 2;
        rcb(p, lo, mid);
        rcb(p, mid, hi);
}
/* --- End spatial ordering ------------------------------------------ */


struct deg_pair { size_t deg, id; };

static int cmp_deg_desc(const void *a, const void *b)
{
        const struct deg_pair *x = a, *y = b;
        if (x->deg != y->deg) return (x->deg < y->deg) - (x->deg > y->deg);
        return (x->id > y->id) - (x->id < y->id);
}

/* Top-N vertices by degree -> c->downtown. N = max(1, 5% of nv). */
static int city_pick_downtown(struct city *c)
{
        size_t nv = c->nv;
        size_t n  = nv / 20;
        if (n == 0) n = (nv < 1 ? 0 : 1);

        /* selection of top-n by degree via a full sort keyed on degree.
         * Small networks; a full qsort with a captured base is awkward in
         * C, so we sort (degree,id) pairs. */
        struct { size_t deg, id; } *pv =
                malloc(nv * sizeof(*pv));
        for (size_t i = 0; i < nv; i++) {
                pv[i].deg = c->row_ptr[i + 1] - c->row_ptr[i];
                pv[i].id  = i;
        }
        /* descending degree, ascending id tie-break */
        qsort(pv, nv, sizeof(*pv), cmp_deg_desc);

        c->downtown = malloc((n ? n : 1) * sizeof(*c->downtown));
        for (size_t i = 0; i < n; i++) c->downtown[i] = pv[i].id;
        c->n_downtown = n;

        free(pv);
        return 0;
}

int city_load(struct city *c, const char *dir, int world_size)
{
        memset(c, 0, sizeof(*c));
        if (world_size < 1)
                return -1;

        char npath[4096], lpath[4096];
        snprintf(npath, sizeof(npath), "%s/node.csv", dir);
        snprintf(lpath, sizeof(lpath), "%s/link.csv", dir);

        long              *node_ids = NULL, *la = NULL, *lb = NULL;
        double            *node_xy  = NULL;
        long              *uni      = NULL;
        struct orig_index *nidx     = NULL;
        struct orig_index *oidx     = NULL;
        struct rcb_pt     *pts      = NULL;
        size_t            *eu = NULL, *ev = NULL, *seg = NULL, *cur = NULL;

        long n_nodes = read_nodes_xy(npath, &node_ids, &node_xy);
        long n_links = read_link_pairs(lpath, &la, &lb);
        if (n_nodes < 0 || n_links < 0) {
                free(node_ids); free(node_xy); free(la); free(lb);
                return -1;
        }

        /* --- Vertex set: union of node.csv and all link endpoints -----
         *
         * We first find the unique original ids (sorted), then attach a
         * coordinate to each: from node.csv where available, otherwise a
         * sentinel that sorts every coordinate-less vertex (link-only
         * endpoints) together and last. */
        size_t upcap = (size_t)n_nodes + 2 * (size_t)n_links;
        uni   = malloc((upcap ? upcap : 1) * sizeof(*uni));
        if (!uni) goto oom;

        size_t un = 0;
        for (long i = 0; i < n_nodes; i++) uni[un++] = node_ids[i];
        for (long i = 0; i < n_links; i++) {
                uni[un++] = la[i];
                uni[un++] = lb[i];
        }
        qsort(uni, un, sizeof(*uni), cmp_long);

        size_t nv = 0;
        for (size_t i = 0; i < un; i++)
                if (i == 0 || uni[i] != uni[i - 1])
                        uni[nv++] = uni[i];   /* compact unique ids in place */
        c->nv = nv;

        /* Index of node.csv coordinates by original id, for O(log n)
         * lookup while building the spatial point array. */
        nidx = malloc(((size_t)n_nodes ? (size_t)n_nodes : 1) * sizeof(*nidx));
        if (!nidx) goto oom;
        for (long i = 0; i < n_nodes; i++) {
                nidx[i].orig  = node_ids[i];
                nidx[i].dense = (size_t)i;   /* row into node_xy */
        }
        qsort(nidx, (size_t)n_nodes, sizeof(*nidx), cmp_orig_index);

        /* Build the spatial point array over the unique vertices. */
        pts = malloc((nv ? nv : 1) * sizeof(*pts));
        if (!pts) { free(nidx); goto oom; }

        /* Sentinel for coordinate-less vertices: sort them after every
         * real point.  Using the data's own max plus a margin keeps them
         * clustered without a magic absolute value. */
        double xmax = 0.0, ymax = 0.0;
        int have_any = 0;
        for (long i = 0; i < n_nodes; i++) {
                if (!have_any) {
                        xmax = node_xy[2 * i];
                        ymax = node_xy[2 * i + 1];
                        have_any = 1;
                } else {
                        if (node_xy[2 * i]     > xmax) xmax = node_xy[2 * i];
                        if (node_xy[2 * i + 1] > ymax) ymax = node_xy[2 * i + 1];
                }
        }
        double sentinel_x = xmax + 1.0e9;
        double sentinel_y = ymax + 1.0e9;

        for (size_t i = 0; i < nv; i++) {
                long orig = uni[i];
                size_t row = dense_of(nidx, (size_t)n_nodes, orig);
                pts[i].orig = orig;
                if (row != (size_t)-1) {
                        pts[i].x = node_xy[2 * row];
                        pts[i].y = node_xy[2 * row + 1];
                } else {
                        pts[i].x = sentinel_x;
                        pts[i].y = sentinel_y;
                }
        }
        free(nidx); nidx = NULL;
        free(uni); uni = NULL;

        /* Recursive coordinate bisection: reorders pts so that dense id i
         * (its position after this call) is spatially compact. */
        rcb(pts, 0, nv);

        /* Emit dense2orig in the spatial order, and build the sorted
         * (orig -> dense) index the edge remap needs. */
        c->dense2orig = malloc((nv ? nv : 1) * sizeof(*c->dense2orig));
        oidx = malloc((nv ? nv : 1) * sizeof(*oidx));
        if (!c->dense2orig || !oidx) { free(oidx); free(pts); goto oom; }
        for (size_t i = 0; i < nv; i++) {
                c->dense2orig[i] = pts[i].orig;
                oidx[i].orig  = pts[i].orig;
                oidx[i].dense = i;
        }
        free(pts); pts = NULL;
        qsort(oidx, nv, sizeof(*oidx), cmp_orig_index);        

        if ((size_t)world_size > nv) {  /* same guard as the grid path */
                fprintf(stderr, "world_size %d > %zu vertices\n",
                        world_size, nv);
                goto fail;
        }

        /* --- Directed edges (dense), drop self-loops ------------------ */
        size_t ecap = 2 * (size_t)n_links, en = 0;
        eu = malloc((ecap ? ecap : 1) * sizeof(*eu));
        ev = malloc((ecap ? ecap : 1) * sizeof(*ev));
        if (!eu || !ev) goto oom;

        size_t dropped_self = 0;
        for (long i = 0; i < n_links; i++) {
                size_t u = dense_of(oidx, nv, la[i]);
                size_t v = dense_of(oidx, nv, lb[i]);
                if (u == (size_t)-1 || v == (size_t)-1) continue;
                if (u == v) { dropped_self++; continue; }
                eu[en] = u; ev[en] = v; en++;
                eu[en] = v; ev[en] = u; en++;
        }
        free(la); free(lb); la = lb = NULL;
        free(oidx); oidx = NULL;

        /* --- Counting sort into per-row segments (pre-dedup) ---------- */
        c->row_ptr = calloc(nv + 1, sizeof(*c->row_ptr));
        if (!c->row_ptr) goto oom;
        for (size_t e = 0; e < en; e++) c->row_ptr[eu[e] + 1]++;
        for (size_t i = 0; i < nv; i++) c->row_ptr[i + 1] += c->row_ptr[i];

        seg = malloc((en ? en : 1) * sizeof(*seg));  /* dst by row */
        cur = malloc((nv ? nv : 1) * sizeof(*cur));
        if (!seg || !cur) goto oom;
        for (size_t i = 0; i < nv; i++) cur[i] = c->row_ptr[i];
        for (size_t e = 0; e < en; e++) seg[cur[eu[e]]++] = ev[e];
        free(eu); free(ev); eu = ev = NULL;

        /* --- Sort + dedup each row into fresh CSR --------------------- */
        c->dst  = malloc((en ? en : 1) * sizeof(*c->dst));
        c->esrc = malloc((en ? en : 1) * sizeof(*c->esrc));
        size_t *new_row = malloc((nv + 1) * sizeof(*new_row));
        if (!c->dst || !c->esrc || !new_row) goto oom;

        size_t w = 0, dropped_dup = 0;
        for (size_t u = 0; u < nv; u++) {
                size_t lo = c->row_ptr[u], hi = c->row_ptr[u + 1];
                new_row[u] = w;
                qsort(seg + lo, hi - lo, sizeof(*seg), cmp_szt);
                for (size_t k = lo; k < hi; k++) {
                        if (k > lo && seg[k] == seg[k - 1]) {
                                dropped_dup++;
                                continue;
                        }
                        c->dst[w]  = seg[k];
                        c->esrc[w] = u;
                        w++;
                }
        }
        new_row[nv] = w;
        free(seg); free(cur); cur = NULL; seg = NULL;
        free(c->row_ptr);
        c->row_ptr = new_row;   /* compacted offsets */
        c->ne = w;

        /* --- Block partition (identical to old city_build) ------------ */
        c->vtx_off = malloc(((size_t)world_size + 1) * sizeof(*c->vtx_off));
        if (!c->vtx_off) goto oom;
        {
                size_t base = nv / (size_t)world_size;
                size_t rem  = nv % (size_t)world_size;
                size_t acc  = 0;
                for (int r = 0; r < world_size; r++) {
                        c->vtx_off[r] = acc;
                        acc += base + ((size_t)r < rem ? 1u : 0u);
                }
                c->vtx_off[world_size] = acc;
        }
        c->world_size = world_size;

        if (city_pick_downtown(c) != 0) goto oom;

        printf("Loaded %s: %zu vtx, %zu directed edges "
               "(dropped %zu self-loop, %zu duplicate)\n",
               dir, c->nv, c->ne, dropped_self, dropped_dup);

        free(node_ids);
        free(node_xy);
        return 0;

oom:
        fprintf(stderr, "city_load: out of memory\n");
fail:
        free(node_ids); free(node_xy); free(uni); free(la); free(lb);
        free(nidx); free(oidx); free(pts);
        free(eu); free(ev); free(seg); free(cur);
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
        free(c->dense2orig);
        free(c->downtown);
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

        if (c->n_downtown > 0 && (xrand(rng) % 100u) < 60u) {
                size_t k = (size_t)(xrand(rng) % (unsigned int)c->n_downtown);
                dst = c->downtown[k];
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
        printf("Network         : %zu vertices, %zu directed edges\n",
               c->nv, c->ne);
        printf("Downtown        : %zu highest-degree vertices\n",
               c->n_downtown);
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
