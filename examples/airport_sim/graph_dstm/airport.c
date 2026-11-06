#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "airport.h"

const gate_slot_t empty_gate = { .n_flights = 0, .pad = 0, .epoch = -1 };

/* ------------------------------------------------------------------ */
/* RNG                                                                 */
/*                                                                     */
/* FAIRNESS: simple_dstm draws with libc rand() after srand(seed+rank) */
/* and uses `rand() & 3` as a *direction* index. We do exactly the     */
/* same here -- same generator, same seeding, same consumption order   */
/* (airport l ascending, plane p ascending, one draw per plane) -- so  */
/* both backends fly the identical set of flights for a given --seed.  */
/* The previous xorshift64* / `% deg` scheme produced a different      */
/* trajectory set and therefore a different queue occupancy, which     */
/* made head-to-head timings meaningless.                              */
/*                                                                     */
/* srand() lives in airport_init() rather than main() so this file is  */
/* self-contained; make sure graph_dstm/main.c does not call srand()   */
/* after airport_init(), or it will clobber the seeding.               */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* Topology                                                            */
/* ------------------------------------------------------------------ */

static inline size_t align8(size_t x) { return (x + 7u) & ~(size_t)7u; }

/* Torus neighbor of gid in direction dir (0=N 1=S 2=E 3=W).
 * Direction encoding is byte-for-byte the same as simple_dstm's. */
static size_t neighbor_of(size_t gid, int grid, int dir)
{
        size_t row = gid / (size_t)grid;
        size_t col = gid % (size_t)grid;
        switch (dir) {
        case 0: row = (row + (size_t)grid - 1) % (size_t)grid; break;
        case 1: row = (row + 1) % (size_t)grid;                break;
        case 2: col = (col + 1) % (size_t)grid;                break;
        default: col = (col + (size_t)grid - 1) % (size_t)grid; break;
        }
        return row * (size_t)grid + col;
}

/*
 * dstm_dgraph_create() requires each vertex's destination slice to be
 * strictly ascending, so we sort and dedupe. On a grid of 2 the N/S and
 * E/W wraps collide and the vertex legitimately ends up with degree 2;
 * self-loops (grid == 1) are dropped, and main() rejects grid < 2 anyway.
 */
static size_t torus_neighbors(size_t gid, int grid, size_t *out /* [4] */)
{
        size_t tmp[4];
        size_t n = 0;

        for (int d = 0; d < 4; d++) {
                size_t v = neighbor_of(gid, grid, d);
                if (v == gid)
                        continue;             /* no self-loops */
                int dup = 0;
                for (size_t i = 0; i < n; i++)
                        if (tmp[i] == v) { dup = 1; break; }
                if (!dup)
                        tmp[n++] = v;
        }

        /* insertion sort, n <= 4 */
        for (size_t i = 1; i < n; i++) {
                size_t key = tmp[i];
                size_t j   = i;
                while (j > 0 && tmp[j - 1] > key) { tmp[j] = tmp[j - 1]; j--; }
                tmp[j] = key;
        }
        for (size_t i = 0; i < n; i++)
                out[i] = tmp[i];
        return n;
}

int airport_topology_build(airport_topology_t *topo,
                           int rank, int na, int grid)
{
        memset(topo, 0, sizeof(*topo));
        topo->rank          = rank;
        topo->na            = na;
        topo->grid          = grid;
        topo->vertex_offset = (size_t)rank * (size_t)na;

        topo->out_degrees = calloc((size_t)na, sizeof(size_t));
        topo->row_ptr     = calloc((size_t)na + 1, sizeof(size_t));
        if (!topo->out_degrees || !topo->row_ptr)
                goto oom;

        /* pass 1: degrees */
        for (int l = 0; l < na; l++) {
                size_t nb[4];
                topo->out_degrees[l] =
                        torus_neighbors(topo->vertex_offset + (size_t)l, grid, nb);
        }

        /* prefix sum */
        {
                size_t acc = 0;
                for (int l = 0; l < na; l++) {
                        topo->row_ptr[l] = acc;
                        acc += topo->out_degrees[l];
                }
                topo->row_ptr[na] = acc;
                topo->edge_count  = acc;
        }

        topo->edge_dsts = calloc(topo->edge_count ? topo->edge_count : 1,
                                 sizeof(size_t));
        if (!topo->edge_dsts)
                goto oom;

        /* pass 2: packed destinations */
        for (int l = 0; l < na; l++) {
                size_t nb[4];
                size_t deg = torus_neighbors(topo->vertex_offset + (size_t)l,
                                             grid, nb);
                for (size_t k = 0; k < deg; k++)
                        topo->edge_dsts[topo->row_ptr[l] + k] = nb[k];
        }
        return 0;

oom:
        airport_topology_destroy(topo);
        return -1;
}

void airport_topology_destroy(airport_topology_t *topo)
{
        if (!topo)
                return;
        free(topo->out_degrees);
        free(topo->row_ptr);
        free(topo->edge_dsts);
        topo->out_degrees = NULL;
        topo->row_ptr     = NULL;
        topo->edge_dsts   = NULL;
        topo->edge_count  = 0;
}

/* Mirrors dgraph's internal slab math: [vertex state | row ptrs | edge slots],
 * each slab 8-aligned, each edge slot = 8-byte dst + padded user state. */
size_t airport_topology_region_bytes(const airport_topology_t *topo,
                                     size_t vertex_state_size,
                                     size_t edge_state_size)
{
        size_t vs     = align8((size_t)topo->na * vertex_state_size);
        size_t rp     = align8(((size_t)topo->na + 1) * sizeof(uint64_t));
        size_t stride = align8(sizeof(uint64_t) + edge_state_size);
        size_t es     = align8(topo->edge_count * stride);
        return vs + rp + es;
}

/* ------------------------------------------------------------------ */
/* State                                                               */
/* ------------------------------------------------------------------ */

int airport_init(airport_state_t *s,
                 int rank, int num_ranks,
                 int planes_per_airport,
                 int seed, int congestion, int publish,
                 const airport_topology_t *topo,
                 dstm_dgraph_t *g)
{
        memset(s, 0, sizeof(*s));
        s->rank       = rank;
        s->num_ranks  = num_ranks;
        s->na         = topo->na;
        s->grid       = topo->grid;
        s->congestion = congestion;
        s->publish    = publish;
        s->topo       = topo;
        s->g          = g;

        /* FAIRNESS: byte-identical to simple_dstm/main.c's `srand(seed + rank)`.
         * s->rng is retained only so airport.h needs no edit; it is unused. */
        srand((unsigned int)(seed + rank));
        s->rng = 0;

        s->outbox = calloc(topo->edge_count ? topo->edge_count : 1,
                           sizeof(gate_slot_t));
        s->recv   = calloc(1, sizeof(gate_slot_t));
        s->vstate = calloc((size_t)s->na, sizeof(airport_vertex_t));

        s->planes_on_ground = calloc((size_t)s->na, sizeof(int));
        s->pending_landings = calloc((size_t)s->na, sizeof(int));
        s->landings         = calloc((size_t)s->na, sizeof(long long));

        if (!s->outbox || !s->recv || !s->vstate ||
            !s->planes_on_ground || !s->pending_landings || !s->landings) {
                airport_destroy(s);
                return -1;
        }

        for (int l = 0; l < s->na; l++)
                s->planes_on_ground[l] = planes_per_airport;
        return 0;
}

void airport_destroy(airport_state_t *s)
{
        if (!s)
                return;
        free(s->outbox);
        free(s->recv);
        free(s->vstate);
        free(s->planes_on_ground);
        free(s->pending_landings);
        free(s->landings);
        s->outbox = NULL;
        s->recv = NULL;
        s->vstate = NULL;
        s->planes_on_ground = NULL;
        s->pending_landings = NULL;
        s->landings = NULL;
}

/*
 * Exact destination lookup, the dgraph analogue of simple_dstm's
 * find_outbox_idx(): linear scan over a slice of at most 4, returning -1
 * when the destination has no edge (only reachable at grid == 1, which
 * main() rejects).
 *
 * FAIRNESS: this deliberately does NOT fall back to some other edge with
 * room, as the old pick_edge() did. simple_dstm cannot reroute -- the RNG
 * draw fixes the destination, and a full gate just stalls the airport --
 * so rerouting here would have given graph_dstm a strictly better traffic
 * distribution under pressure. Destination is now RNG-determined in both.
 */
static inline size_t edge_index_of(const airport_state_t *s,
                                   size_t rs, size_t re, size_t dst)
{
        for (size_t k = rs; k < re; k++)
                if (s->topo->edge_dsts[k] == dst)
                        return k;
        return (size_t)-1;
}

void airport_timestep(airport_state_t *s, dstm_time_t t)
{
        const airport_topology_t *topo = s->topo;
        const size_t vo = topo->vertex_offset;
        const int    na = s->na;
        int rc;

        /* --- STEP 0: publish load / beacon the tick ----------------------
         * Must come before any local delta-0 get at time t.
         *
         * Under --no-vstate we still publish vertex 0. A rank whose first op
         * of the tick is a blocking get may not register originating time t
         * with its neighbors before it parks, and if every rank does that
         * simultaneously nobody ever advances. One cheap local put removes
         * the question entirely.
         *
         * simple_dstm has no analogue, so run the comparison with
         * --no-vstate and report the full-publish cost as an ablation. */
        {
                int n_pub = s->publish ? na : (na > 0 ? 1 : 0);
                for (int l = 0; l < n_pub; l++) {
                        s->vstate[l].planes_on_ground = s->planes_on_ground[l];
                        s->vstate[l].pending_landings = s->pending_landings[l];
                        s->vstate[l].landings         = s->landings[l];
                        rc = dstm_dgraph_vertex_put(s->g, vo + (size_t)l,
                                                    &s->vstate[l], t, 0);
                        if (rc != DSTM_SUCCESS)
                                dstm_log_error("vertex_put %zu t=%lld: %d",
                                               vo + (size_t)l, (long long)t, rc);
                }
        }

        for (size_t e = 0; e < topo->edge_count; e++) {
                s->outbox[e].n_flights = 0;
                s->outbox[e].epoch     = t + 1;
        }

        /* --- STEP 1: drain our inbox edges -------------------------------
         * Local, delta 0. A manifest whose epoch is not t is a version we
         * inherited from an earlier tick that nobody overwrote -- drop it,
         * it was already delivered. */
        for (int l = 0; l < na; l++) {
                size_t u = vo + (size_t)l;
                for (size_t k = topo->row_ptr[l];
                     k < topo->row_ptr[l + 1]; k++) {
                        size_t w = topo->edge_dsts[k];

                        rc = dstm_dgraph_edge_get(s->g, u, w, s->recv, t, 0);
                        if (rc != DSTM_SUCCESS) {
                                dstm_log_error("edge_get (%zu,%zu) t=%lld: %d",
                                               u, w, (long long)t, rc);
                                continue;
                        }
                        if (s->recv->epoch != (int64_t)t)
                                continue;               /* stale, already landed */

                        for (int f = 0; f < s->recv->n_flights; f++) {
                                if (s->recv->flights[f].dst_global
                                    != (int32_t)u) {
                                        s->misroutes++;
                                        continue;
                                }
                                s->pending_landings[l]++;
                                s->arrivals++;
                        }
                }
        }

        /* --- STEP 1b: drain the runway queue, LANDINGS_PER_STEP max ------ */
        for (int l = 0; l < na; l++) {
                int can = s->pending_landings[l] < LANDINGS_PER_STEP
                        ? s->pending_landings[l] : LANDINGS_PER_STEP;
                if (!can)
                        continue;
                s->pending_landings[l] -= can;
                s->planes_on_ground[l] += can;
                s->landings[l]         += can;
        }

        /* --- STEP 2: everything on the ground departs -------------------- */
        for (int l = 0; l < na; l++) {
                size_t rs  = topo->row_ptr[l];
                size_t re  = topo->row_ptr[l + 1];
                size_t deg = re - rs;
                int    planes = s->planes_on_ground[l];

                if (!deg || !planes)
                        continue;

                size_t u = vo + (size_t)l;

                /* One remote vertex read per neighbor per tick, at t-1: the
                 * only legal remote read under a lookahead-1 matrix. Steers a
                 * heuristic, never affects conservation. Off by default;
                 * simple_dstm has no equivalent. */
                size_t hint = rs;
                if (s->congestion && t > 0) {
                        int best = INT_MAX;
                        for (size_t k = rs; k < re; k++) {
                                airport_vertex_t nv;
                                if (dstm_dgraph_vertex_get(s->g,
                                                           topo->edge_dsts[k],
                                                           &nv, t, -1)
                                    != DSTM_SUCCESS)
                                        continue;
                                int load = nv.planes_on_ground
                                         + nv.pending_landings;
                                if (load < best) { best = load; hint = k; }
                        }
                }

                for (int p = 0; p < planes; p++) {
                        /* FAIRNESS: draw unconditionally, exactly one rand()
                         * per plane, so the stream position is identical to
                         * simple_dstm's and so congestion on/off consume the
                         * generator identically. */
                        int    dir = rand() & 3;
                        size_t dst = neighbor_of(u, s->grid, dir);
                        size_t k;

                        if (s->congestion && t > 0) {
                                k = hint;
                        } else {
                                k = edge_index_of(s, rs, re, dst);
                                if (k == (size_t)-1)
                                        continue;   /* no such edge; mirrors
                                                     * simple's ob_idx < 0 */
                        }

                        gate_slot_t *ob = &s->outbox[k];
                        if (ob->n_flights >= EDGE_GATE_CAPACITY) {
                                /* Gate full: this plane and every plane behind
                                 * it stay on the ground and retry next tick.
                                 * Identical to simple_dstm's `break`. */
                                s->overflows += planes - p;
                                break;
                        }

                        flight_t *f = &ob->flights[ob->n_flights++];
                        f->src_global = (int32_t)u;
                        f->dst_global = (int32_t)topo->edge_dsts[k];
                        f->flight_id  = (int64_t)u * 1000000LL
                                      + s->flight_counter++;

                        s->planes_on_ground[l]--;
                        s->departures++;
                }
        }

        /* --- STEP 3: push manifests into receiver-owned reverse edges ----
         * Remote, delta +1, effective t+1 -- still writable, since we are
         * only now at t and no neighbor has passed t. */
        for (int l = 0; l < na; l++) {
                size_t u = vo + (size_t)l;
                for (size_t k = topo->row_ptr[l];
                     k < topo->row_ptr[l + 1]; k++) {
                        if (s->outbox[k].n_flights == 0)
                                continue;
                        size_t v = topo->edge_dsts[k];
                        rc = dstm_dgraph_edge_put(s->g, v, u,
                                                  &s->outbox[k], t, 1);
                        if (rc != DSTM_SUCCESS)
                                dstm_log_error("edge_put (%zu,%zu) t=%lld: %d",
                                               v, u, (long long)t, rc);
                }
        }
}
