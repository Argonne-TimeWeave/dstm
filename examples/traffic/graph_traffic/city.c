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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "city.h"

/* One simulation tick, in seconds. free-flow travel time of an edge is
 * ceil((length / fspd) / SECONDS_PER_TICK), clamped to >= 1. */
#define SECONDS_PER_TICK 5.0

/* Convert length[m] and free-flow speed[m/s] to integer ticks (>= 1).
 * Missing/zero/garbage data (fspd <= 0 or length <= 0 or non-finite)
 * degrades to 1 tick -- the pre-change behavior. */
static int ff_ticks_of(double length_m, double fspd_ms)
{
        if (!(length_m > 0.0) || !(fspd_ms > 0.0))
                return 1;

        double secs = length_m / fspd_ms;
        double t    = ceil(secs / SECONDS_PER_TICK);

        if (!(t >= 1.0))        /* catches NaN and sub-1 */
                return 1;
        if (t > (double)INT32_MAX)
                return INT32_MAX;
        return (int)t;
}

/* Fallback road capacity, veh/hour, for links whose cap_ab/cap_ba is
 * empty or non-positive. 323 is the value pervasive in this dataset's
 * LOCAL links; using it keeps load/capacity ratios sane for bad rows. */
#define CAP_FALLBACK_VPH 323.0

/* Convert a lane capacity in vehicles/hour to vehicles/tick, so it shares
 * the same 10s time base as the per-tick load fed into the BPR ratio.
 * Missing/zero/garbage falls back to CAP_FALLBACK_VPH. Result is always
 * > 0, so load/capacity never divides by zero. */
static float cap_tick_of(double cap_vph)
{
        if (!(cap_vph > 0.0))
                cap_vph = CAP_FALLBACK_VPH;
        return (float)(cap_vph * (SECONDS_PER_TICK / 3600.0));
}

/* --- comparators --------------------------------------------------- */

static int cmp_long(const void *a, const void *b)
{
        long x = *(const long *)a, y = *(const long *)b;
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

/* node_a (col 2), node_b (col 3), length (col 4), fspd_ab (col 14) and
 * fspd_ba (col 17) of every link line, into five parallel arrays.
 * Returns count or -1.
 *
 * We now need several columns past the free-text `name` field, some of
 * which may be empty (the schema allows NULL length / fspd). sscanf
 * field-skipping does not compose cleanly across many optional columns,
 * so we tokenize on commas by hand. This still assumes the `name` field
 * is unquoted and comma-free, exactly as the previous sscanf version did;
 * if that ever breaks, switch to a real quoted-CSV split. Empty numeric
 * fields parse as 0.0, which ff_ticks_of() treats as "missing" -> 1 tick. */
static long read_link_pairs(const char *path, long **a_out, long **b_out,
                            double **len_out, double **fab_out,
                            double **fba_out, double **cab_out,
                            double **cba_out)
{
        FILE *f = fopen(path, "r");
        if (!f) { perror(path); return -1; }

        size_t  cap = 1024, n = 0;
        long   *a   = malloc(cap * sizeof(*a));
        long   *b   = malloc(cap * sizeof(*b));
        double *len = malloc(cap * sizeof(*len));
        double *fab = malloc(cap * sizeof(*fab));
        double *fba = malloc(cap * sizeof(*fba));
        double *cab = malloc(cap * sizeof(*cab));
        double *cba = malloc(cap * sizeof(*cba));
        if (!a || !b || !len || !fab || !fba || !cab || !cba) {
                free(a); free(b); free(len); free(fab); free(fba);
                free(cab); free(cba);
                fclose(f);
                return -1;
        }

        char line[8192];
        int  first = 1;
        while (fgets(line, sizeof(line), f)) {
                if (first) { first = 0; continue; }

                /* Split the line into up to 19 comma-delimited fields.
                 * We read indices 2,3,4,14,15,17,18, so we must walk every
                 * comma up to 18. A quoted geo_wkt with commas appears only
                 * in the LAST column (index 20+), well past 18, so it can
                 * never corrupt the fields we read. */
                enum { WANT = 19 };
                const char *fld[WANT];
                size_t      fln[WANT];
                for (int k = 0; k < WANT; k++) { fld[k] = ""; fln[k] = 0; }

                const char *p    = line;
                int         col  = 0;
                while (col < WANT) {
                        const char *comma = strchr(p, ',');
                        size_t      flen  = comma ? (size_t)(comma - p)
                                                  : strlen(p);
                        fld[col] = p;
                        fln[col] = flen;
                        col++;
                        if (!comma)
                                break;
                        p = comma + 1;
                }

                /* Need at least through node_a/node_b to be a real row. */
                if (col < 4)
                        continue;

                /* Local NUL-terminated copies for strtod/strtol. Fields
                 * are short; a small stack buffer is plenty. */
                char buf[64];
                #define FIELD_STR(idx) \
                        (fln[idx] < sizeof(buf) ? \
                          (memcpy(buf, fld[idx], fln[idx]), \
                           buf[fln[idx]] = '\0', buf) : "")

                char  *end;
                long   na = strtol(FIELD_STR(2), &end, 10);
                if (end == buf) continue;          /* node_a not numeric */
                long   nb = strtol(FIELD_STR(3), &end, 10);
                if (end == buf) continue;          /* node_b not numeric */

                double length  = strtod(FIELD_STR(4),  &end);
                double fspd_ab = strtod(FIELD_STR(14), &end);
                double cap_ab  = strtod(FIELD_STR(15), &end);
                double fspd_ba = strtod(FIELD_STR(17), &end);
                double cap_ba  = strtod(FIELD_STR(18), &end);
                #undef FIELD_STR

                if (n == cap) {
                        cap *= 2;
                        long   *ta = realloc(a,   cap * sizeof(*a));
                        long   *tb = realloc(b,   cap * sizeof(*b));
                        double *tl = realloc(len, cap * sizeof(*len));
                        double *tp = realloc(fab, cap * sizeof(*fab));
                        double *tq = realloc(fba, cap * sizeof(*fba));
                        double *tc = realloc(cab, cap * sizeof(*cab));
                        double *td = realloc(cba, cap * sizeof(*cba));
                        if (!ta || !tb || !tl || !tp || !tq || !tc || !td) {
                                free(ta ? ta : a);   free(tb ? tb : b);
                                free(tl ? tl : len); free(tp ? tp : fab);
                                free(tq ? tq : fba); free(tc ? tc : cab);
                                free(td ? td : cba);
                                fclose(f);
                                return -1;
                        }
                        a = ta; b = tb; len = tl; fab = tp; fba = tq;
                        cab = tc; cba = td;
                }
                a[n]   = na;  b[n]   = nb;
                len[n] = length;
                fab[n] = fspd_ab;
                fba[n] = fspd_ba;
                cab[n] = cap_ab;
                cba[n] = cap_ba;
                n++;
        }
        fclose(f);
        *a_out   = a;   *b_out   = b;
        *len_out = len; *fab_out = fab; *fba_out = fba;
        *cab_out = cab; *cba_out = cba;
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
        double            *llen = NULL, *lfab = NULL, *lfba = NULL;
        double            *lcab = NULL, *lcba = NULL;
        long              *uni      = NULL;
        struct orig_index *nidx     = NULL;
        struct orig_index *oidx     = NULL;
        struct rcb_pt     *pts      = NULL;
        size_t            *eu = NULL, *ev = NULL, *seg = NULL, *cur = NULL;
        int               *ew = NULL, *segw = NULL;
        float             *ec = NULL, *segc = NULL;

        long n_nodes = read_nodes_xy(npath, &node_ids, &node_xy);
        long n_links = read_link_pairs(lpath, &la, &lb, &llen, &lfab, &lfba,
                                       &lcab, &lcba);
        if (n_nodes < 0 || n_links < 0) {
                free(node_ids); free(node_xy);
                free(la); free(lb); free(llen); free(lfab); free(lfba);
                free(lcab); free(lcba);
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
        ew = malloc((ecap ? ecap : 1) * sizeof(*ew));
        ec = malloc((ecap ? ecap : 1) * sizeof(*ec));
        if (!eu || !ev || !ew || !ec) goto oom;

        size_t dropped_self = 0;
        for (long i = 0; i < n_links; i++) {
                size_t u = dense_of(oidx, nv, la[i]);
                size_t v = dense_of(oidx, nv, lb[i]);
                if (u == (size_t)-1 || v == (size_t)-1) continue;
                if (u == v) { dropped_self++; continue; }
                /* A->B uses fspd_ab/cap_ab, B->A uses fspd_ba/cap_ba;
                 * both share length. */
                eu[en] = u; ev[en] = v;
                ew[en] = ff_ticks_of(llen[i], lfab[i]);
                ec[en] = cap_tick_of(lcab[i]);
                en++;
                eu[en] = v; ev[en] = u;
                ew[en] = ff_ticks_of(llen[i], lfba[i]);
                ec[en] = cap_tick_of(lcba[i]);
                en++;
        }
        free(la); free(lb); la = lb = NULL;
        free(llen); free(lfab); free(lfba); llen = lfab = lfba = NULL;
        free(lcab); free(lcba); lcab = lcba = NULL;
        free(oidx); oidx = NULL;

        /* --- Counting sort into per-row segments (pre-dedup) ---------- */
        c->row_ptr = calloc(nv + 1, sizeof(*c->row_ptr));
        if (!c->row_ptr) goto oom;
        for (size_t e = 0; e < en; e++) c->row_ptr[eu[e] + 1]++;
        for (size_t i = 0; i < nv; i++) c->row_ptr[i + 1] += c->row_ptr[i];

        seg  = malloc((en ? en : 1) * sizeof(*seg));  /* dst by row */
        segw = malloc((en ? en : 1) * sizeof(*segw)); /* ff ticks by row */
        segc = malloc((en ? en : 1) * sizeof(*segc)); /* cap/tick by row */
        cur  = malloc((nv ? nv : 1) * sizeof(*cur));
        if (!seg || !segw || !segc || !cur) goto oom;
        for (size_t i = 0; i < nv; i++) cur[i] = c->row_ptr[i];
        /* Scatter dst and its per-edge attributes together so segw/segc
         * stay aligned to seg before the per-row sort re-permutes them all
         * in lockstep. */
        for (size_t e = 0; e < en; e++) {
                size_t pos = cur[eu[e]]++;
                seg[pos]  = ev[e];
                segw[pos] = ew[e];
                segc[pos] = ec[e];
        }
        free(eu); free(ev); free(ew); free(ec);
        eu = ev = NULL; ew = NULL; ec = NULL;

        /* --- Sort + dedup each row into fresh CSR --------------------- */
        c->dst      = malloc((en ? en : 1) * sizeof(*c->dst));
        c->esrc     = malloc((en ? en : 1) * sizeof(*c->esrc));
        c->ff_ticks = malloc((en ? en : 1) * sizeof(*c->ff_ticks));
        c->cap_tick = malloc((en ? en : 1) * sizeof(*c->cap_tick));
        size_t *new_row = malloc((nv + 1) * sizeof(*new_row));
        if (!c->dst || !c->esrc || !c->ff_ticks || !c->cap_tick || !new_row)
                goto oom;

        size_t w = 0, dropped_dup = 0;
        for (size_t u = 0; u < nv; u++) {
                size_t lo = c->row_ptr[u], hi = c->row_ptr[u + 1];
                new_row[u] = w;

                /* Insertion co-sort of (dst, ff, cap) triples by dst.
                 * Rows are tiny (road-network out-degrees), so this is
                 * cheap and avoids allocating a struct array. */
                for (size_t i = lo + 1; i < hi; i++) {
                        size_t dv = seg[i];
                        int    wv = segw[i];
                        float  cv = segc[i];
                        size_t j  = i;
                        while (j > lo && seg[j - 1] > dv) {
                                seg[j]  = seg[j - 1];
                                segw[j] = segw[j - 1];
                                segc[j] = segc[j - 1];
                                j--;
                        }
                        seg[j]  = dv;
                        segw[j] = wv;
                        segc[j] = cv;
                }

                for (size_t k = lo; k < hi; k++) {
                        if (k > lo && seg[k] == seg[k - 1]) {
                                /* Duplicate directed edge (e.g. two links
                                 * between the same pair). Keep the SMALLER
                                 * free-flow time (more constraining, matches
                                 * the distance-matrix min reduction), and
                                 * the LARGER capacity: two parallel roads
                                 * carry more traffic than one. */
                                dropped_dup++;
                                if (segw[k] < c->ff_ticks[w - 1])
                                        c->ff_ticks[w - 1] = segw[k];
                                if (segc[k] > c->cap_tick[w - 1])
                                        c->cap_tick[w - 1] = segc[k];
                                continue;
                        }
                        c->dst[w]      = seg[k];
                        c->esrc[w]     = u;
                        c->ff_ticks[w] = segw[k];
                        c->cap_tick[w] = segc[k];
                        w++;
                }
        }
        new_row[nv] = w;
        free(seg); free(segw); free(segc); free(cur);
        cur = NULL; seg = NULL; segw = NULL; segc = NULL;
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
        free(node_ids); free(node_xy); free(uni);
        free(la); free(lb); free(llen); free(lfab); free(lfba);
        free(lcab); free(lcba);
        free(nidx); free(oidx); free(pts);
        free(eu); free(ev); free(ew); free(ec);
        free(seg); free(segw); free(segc); free(cur);
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
        free(c->ff_ticks);
        free(c->cap_tick);
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
