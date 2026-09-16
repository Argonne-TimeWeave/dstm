/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

/*
 * Graph traffic: routing under diffused, delayed congestion information.
 *
 * Every rank replicates the ENTIRE graph's congestion state. Each tick a
 * rank refreshes the entries it owns, then ships the whole replica -- plus
 * crossing agents -- along every outgoing edge. A distant edge's state
 * reaches a rank only after as many ticks as hops between them, so routing
 * runs against that stale view.
 *
 * Mapping onto DSTM
 * =================
 * The channel for directed edge u->v is the edge slot (v,u), owned by
 * rank(v), the RECEIVER:
 *   sender    dstm_dgraph_edge_put(g, v, u, buf, t, delta)  remote write
 *   receiver  dstm_dgraph_edge_get(g, v, u, buf, t, 0)      local read
 * One writer, one reader per slot, so no version is ever clobbered.
 *
 * delta is the edge's physical crossing delay (edge_delay): free-flow
 * ticks plus a capped congestion penalty, so a jam slows a car, not just
 * its routing cost. Replica diffusion rides the same puts.
 *
 * The merge is highest-stamp-wins with one authoritative writer per edge:
 * commutative, associative, idempotent, so MPI delivery order cannot
 * change the result. STALE folds neighbour state every S.stale_delay
 * ticks; oracle overwrites the replica globally; blind never reads it.
 *
 * Per-tick discipline
 * ===================
 *   1. publish owned vertex state at (t, 0) -- creates version t so the
 *      self-region get at delta 0 cannot block; emits heartbeats
 *   2. receive: read slot (v,u) at (t, 0); merge state, inject cars
 *   3. route: each eligible car picks its next edge, staged subject to the
 *      per-edge cap; denied cars stay put and count as load
 *   4. measure: fold this tick's load into owned entries, stamp with t
 *   5. send: pack [header][state][cars] per edge, put at (t, edge_delay)
 *
 * Step 4 follows step 2 so a relayed copy of our own edge cannot overwrite
 * the authoritative value.
 *
 * Run with:  mpirun -n P ./graph_traffic --help
 */

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "dstm.h"

#include "car.h"
#include "chan.h"
#include "city.h"

/*******************************************************************************
 * Tunables
 ******************************************************************************/

/* Ticks an agent dwells at a vertex before it may depart again. 1 gives
 * intersections a meaningful queue length; 0 makes agents move every
 * tick. */
#define DWELL_TICKS 1

/* Congestion smoothing. Raw per-tick edge load is spiky; feeding it
 * straight to the router produces the classic traffic-assignment
 * flip-flop (everyone avoids A, B congests, everyone avoids B). An EWMA
 * plus the age decay in cong_effective damps it. */
#define EWMA_ALPHA 0.25f

/* A denied crossing costs more than a successful one: a blocked agent is
 * a queue forming at the tail of the edge, which is exactly the signal
 * downstream routers should react to. */
#define BLOCK_PENALTY 1.0f

/* Confidence half-life, in ticks, for age-weighted routing costs. */
#define CONG_TAU 6.0f

/* Volume-delay (BPR-style, linear) congestion model:
 *
 *     travel_time = free_flow_ticks + (load / capacity) * DELAY_FACTOR
 *
 * load     = age-decayed EWMA of per-tick crossings (cong_effective)
 * capacity = edge capacity in veh/tick (city.cap_tick[ge])
 * The penalty is additive and UNCAPPED: an edge at 2x capacity costs
 * free_flow + 2*DELAY_FACTOR, matching the classic linear volume-delay
 * function. There is no hard cap -- capacity makes an edge progressively
 * more expensive, it does not block additional demand. */
#define ROUTE_DELAY_FACTOR 1.0f

#define MAX_CONGESTION_MULTIPLIER 8

#define STALE_DELAY_DEFAULT 2

/*******************************************************************************
 * Replica modes
 *
 * Three ways of populating the congestion replica, so the cost of
 * staleness can be measured rather than assumed. All three ship
 * byte-identical channel messages and run the identical tick sequence --
 * only the contents of the replica consulted by route_pick differ. That
 * keeps communication volume out of the comparison.
 *
 *   STALE   diffusion by relay. What the design is actually about.
 *   ORACLE  replica synchronised globally every tick. Upper bound: the
 *           best any router could do with this heuristic.
 *   BLIND   congestion ignored entirely, pure distance routing. Lower
 *           bound.
 ******************************************************************************/
enum replica_mode {
        REPLICA_STALE = 0,
        REPLICA_ORACLE,
        REPLICA_BLIND,
};

static const char *replica_mode_name(enum replica_mode m)
{
        switch (m) {
        case REPLICA_ORACLE: return "oracle";
        case REPLICA_BLIND:  return "blind";
        default:             return "stale";
        }
}

/*******************************************************************************
 * Binary min-heap for Dijkstra (lazy deletion, so pushes may exceed |V|)
 ******************************************************************************/

struct pqheap {
        size_t *v;
        float  *k;
        size_t  n;
        size_t  cap;
};

static void heap_clear(struct pqheap *h)
{
        h->n = 0;
}

static void heap_push(struct pqheap *h, size_t v, float k)
{
        size_t i;

        if (h->n == h->cap) {
                h->cap = h->cap ? h->cap * 2 : 256;
                h->v = realloc(h->v, h->cap * sizeof(*h->v));
                h->k = realloc(h->k, h->cap * sizeof(*h->k));
                assert(h->v != NULL && h->k != NULL);
        }

        i = h->n++;
        h->v[i] = v;
        h->k[i] = k;

        while (i > 0) {
                size_t p = (i - 1) / 2;

                if (h->k[i] >= h->k[p])
                        break;

                size_t tv = h->v[i]; float tk = h->k[i];
                h->v[i] = h->v[p];   h->k[i] = h->k[p];
                h->v[p] = tv;        h->k[p] = tk;
                i = p;
        }
}

static int heap_pop(struct pqheap *h, size_t *out_v, float *out_k)
{
        size_t i = 0;

        if (h->n == 0)
                return 0;

        *out_v = h->v[0];
        *out_k = h->k[0];

        h->n--;
        h->v[0] = h->v[h->n];
        h->k[0] = h->k[h->n];

        for (;;) {
                size_t l = 2 * i + 1, r = 2 * i + 2, s = i;

                if (l < h->n && h->k[l] < h->k[s])
                        s = l;
                if (r < h->n && h->k[r] < h->k[s])
                        s = r;
                if (s == i)
                        break;

                size_t tv = h->v[i]; float tk = h->k[i];
                h->v[i] = h->v[s];   h->k[i] = h->k[s];
                h->v[s] = tv;        h->k[s] = tk;
                i = s;
        }
        return 1;
}

/* Deterministic tie-break, independent of the per-rank RNG stream, so
 * results do not depend on how many routing lookups happen locally. */
static inline float tie_jitter(int32_t id, int32_t t, size_t ge)
{
        uint32_t h = (uint32_t)id * 2654435761u ^ (uint32_t)t * 40503u
                     ^ (uint32_t)ge * 2246822519u;

        h ^= h >> 15;
        h *= 2246822519u;
        h ^= h >> 13;
        return (float)(h & 63u) * (1.0f / 4096.0f);
}

/*******************************************************************************
 * Simulation state
 ******************************************************************************/

struct rank_stats {
        uint64_t spawned;
        uint64_t arrived;
        uint64_t live;
        uint64_t crossings;
        uint64_t blocked;
        uint64_t sent_msgs;
        uint64_t recv_msgs;
        uint64_t merge_updates;
        uint64_t route_lookups;
        uint64_t route_age_sum;

        uint64_t metric_spawned;
        uint64_t metric_arrived;
        uint64_t metric_hop_sum;
        uint64_t metric_trip_time_sum;

        uint64_t bytes_out;
};

enum trip_event_kind {
        TRIP_EVENT_SPAWN = 1,
        TRIP_EVENT_ROUTE,
        TRIP_EVENT_ARRIVE,
};

/* Events are retained locally and gathered to rank zero after the run.
 * This avoids adding MPI collectives or point-to-point operations to the
 * simulation tick. */
struct trip_event {
        int32_t kind;
        int32_t trip_id;
        int32_t origin;
        int32_t dest;
        int32_t depart_tick;
        int32_t arrival_tick;
        int32_t hops;
        int32_t reserved;
        float   free_flow_tt;
        float   routed_tt;
        float   distance_m;
};

struct trip_row {
        int32_t trip_id;
        int32_t origin;
        int32_t dest;
        int32_t depart_tick;
        int32_t arrival_tick;
        int32_t hops;
        int     spawned;
        int     routed;
        int     arrived;
        float   free_flow_tt;
        float   routed_tt;
        float   distance_m;
};

struct sim {
        int rank;
        int world_size;

        struct city         city;
        struct car_layout    Lcar;
        struct state_layout  Lstate;
        struct cong_replica rep;

        dstm_distance_t *dist;
        dstm_dgraph_t   *g_cars;
        dstm_dgraph_t   *g_state;

        /* Local vertex ownership. */
        size_t v_lo;            /* first owned vertex, global id        */
        size_t v_hi;            /* one past last owned                  */
        size_t nlocal;          /* v_hi - v_lo                          */

        /* Local CSR slice. lrow_ptr[i] is the first local slot of local
         * vertex i; local slot ls corresponds to global edge id
         * city.row_ptr[v_lo] + ls. The two numbering schemes differ only
         * by that constant offset. */
        size_t *lrow_ptr;       /* nlocal + 1                           */
        size_t *ldeg;           /* nlocal                               */
        size_t *ldst;           /* nedge_local                          */
        size_t  nedge_local;
        size_t  gslot_base;     /* city.row_ptr[v_lo]                   */

        /* Outgoing staging, one bucket of `cap` cars per local edge. */
        struct car_rec *stage;          /* nedge_local * cap            */
        uint32_t       *stage_count;    /* nedge_local                  */
        float          *load_cross;     /* nedge_local, this tick       */
        float          *load_block;     /* nedge_local, this tick       */

        void *car_sendbuf;
        void *car_recvbuf;
        void *state_sendbuf;
        void *state_recvbuf;

        struct car_pool pool;
        int32_t        *vqueue;         /* nlocal, cars dwelling        */

        dstm_time_t t;
        dstm_time_t t_end;
        int         ff_max;//max free-flow ticks over ALL edges;
        unsigned int rng;
        int seed;
        int verbose;
        int stale_delay;

        enum replica_mode mode;

        /* Reverse CSR of the replicated topology: in-edges of each
         * vertex, as global edge ids. Built once, never mutated. */
        size_t *rrow;           /* nv + 1                               */
        size_t *redge;          /* ne                                   */

        /* Per-destination cost-to-go trees, cached for one tick.
         * sp[d] is lazily allocated the first time destination d is
         * routed to. sp_tick[d] == -1 means "not yet computed". */
        float        **sp;      /* nv pointers, each nv floats          */
        int32_t       *sp_tick; /* nv                                   */

        /* Free-flow cost-to-go trees. Unlike sp, these never expire
         * because topology and ff_ticks are immutable. */
        float        **ff_sp;
        unsigned char *ff_sp_valid;

        struct pqheap  h;

        /* Oracle mode only. Owned edges are the contiguous global range
         * [gslot_base, gslot_base + nedge_local), so an allgatherv of
         * owned slices rebuilds the entire replica exactly. Byte counts
         * and byte displacements, one per rank. */
        int *ag_counts;
        int *ag_displs;

        int spawn_rate;         /* global cars spawned per tick */
        int next_car_id;        /* persistent across calls */

        /* Trip-level output. */
        struct trip_event *events;
        size_t             nevents;
        size_t             events_cap;
        const char        *trip_output;

        /* Number of initial timestamps included in the routing-metric
         * cohort. A car is included when:
         *
         *     0 <= depart_tick < metric_depart_limit
         *
         * Since timestamps are 0..t_end inclusive, this is
         * ceil((t_end + 1) / 2). */
        int32_t metric_depart_limit; 

        struct rank_stats st;
};

static struct sim S;

/* Return true when a car belongs to the fixed routing-metric cohort.
 *
 * Restricting the metric to cars that depart near the beginning gives
 * every measured car a long observation window. The percentage-arrived
 * metric reports how much of the cohort was completed, so slow unfinished
 * trips are not silently omitted from the result. */
static int routing_metric_member(int32_t depart_tick)
{
        return depart_tick >= 0 &&
               depart_tick < S.metric_depart_limit;
}

static void trip_event_push(struct trip_event event)
{
        /* Do not retain trip events outside the routing-metric cohort. */
        //if (!routing_metric_member(event.depart_tick))
         //       return;

        if (S.nevents == S.events_cap) {
                size_t new_cap = S.events_cap ? S.events_cap * 2 : 1024;
                struct trip_event *new_events =
                        realloc(S.events, new_cap * sizeof(*new_events));

                assert(new_events != NULL);
                S.events = new_events;
                S.events_cap = new_cap;
        }

        S.events[S.nevents++] = event;
}

static void record_trip_spawn(const struct car *c)
{
        trip_event_push((struct trip_event) {
                .kind         = TRIP_EVENT_SPAWN,
                .trip_id      = c->id,
                .origin       = c->origin,
                .dest         = c->dest,
                .depart_tick  = c->depart_tick,
                .arrival_tick = -1,
                .hops         = 0,
                .free_flow_tt = NAN,
                .routed_tt    = NAN,
                .distance_m   = 0.0f,
        });
}

static void record_trip_route(const struct car *c, float free_flow_tt)
{
        trip_event_push((struct trip_event) {
                .kind         = TRIP_EVENT_ROUTE,
                .trip_id      = c->id,
                .origin       = c->origin,
                .dest         = c->dest,
                .depart_tick  = c->depart_tick,
                .arrival_tick = -1,
                .hops         = c->hops,
                .free_flow_tt = free_flow_tt,
                .routed_tt    = c->routed_tt,
                .distance_m   = c->distance_m,
        });
}

static void record_trip_arrival(const struct car_rec *r, int32_t t)
{
        trip_event_push((struct trip_event) {
                .kind         = TRIP_EVENT_ARRIVE,
                .trip_id      = r->id,
                .origin       = r->origin,
                .dest         = r->dest,
                .depart_tick  = r->depart_tick,
                .arrival_tick = t,
                .hops         = r->hops + 1,
                .free_flow_tt = NAN,
                .routed_tt    = r->routed_tt,
                .distance_m   = r->distance_m,
        });
}

/* Per-vertex state published every tick. Its only job today is to create
 * the local region version at (t, 0) and emit heartbeats -- nothing reads
 * it back. queue_len rides along so a future consumer (a neighbour-aware
 * spawn policy, say) does not force a layout change. */
struct vertex_state {
        int32_t queue_len;
        int32_t stamp;
};

/*******************************************************************************
 * Step 1: publish owned vertex state
 ******************************************************************************/

static void publish_vertices(int32_t t)
{
        /* g_state only needs a local delta-0 version on the ticks STALE
         * will read it. BLIND and ORACLE never read g_state. */
        int state_pub = (S.mode == REPLICA_STALE) &&
                        (t % S.stale_delay == 0);

        for (size_t i = 0; i < S.nlocal; i++) {
                struct vertex_state vs = { .queue_len = S.vqueue[i], .stamp = t };
                int rc = dstm_dgraph_vertex_put(S.g_cars, S.v_lo + i, &vs,
                                                (dstm_time_t)t, 0);
                assert(rc == DSTM_SUCCESS);

                if (state_pub) {
                        rc = dstm_dgraph_vertex_put(S.g_state, S.v_lo + i, &vs,
                                                    (dstm_time_t)t, 0);
                        assert(rc == DSTM_SUCCESS);
                }
        }
}

/*******************************************************************************
 * Step 2: receive on every incoming channel
 *
 * Slot (v,u) with v local is our inbox from u. The grid is symmetric, so
 * iterating our own out-edges enumerates exactly our in-edges.
 ******************************************************************************/

static void inject_car(const struct car_rec *r, int32_t t)
{
        size_t at = (size_t)r->dst_vertex;

        S.st.crossings++;

        if ((size_t)r->dest == at) {
                S.st.arrived++;
                record_trip_arrival(r, t);

                if (routing_metric_member(r->depart_tick)) {
                        S.st.metric_arrived++;
                        S.st.metric_hop_sum +=
                                (uint64_t)(r->hops + 1);
                        S.st.metric_trip_time_sum +=
                                (uint64_t)(t - r->depart_tick);
                }
                return;
        }

        struct car c = {
                .id          = r->id,
                .origin      = r->origin,
                .dest        = r->dest,
                .at          = (int32_t)at,
                .depart_tick = r->depart_tick,
                .hops        = r->hops + 1,
                .ready_tick  = t + DWELL_TICKS,
                .routed_tt   = r->routed_tt,
                .distance_m  = r->distance_m,
        };
        car_pool_insert(&S.pool, c);
        S.vqueue[at - S.v_lo]++;
}

static void recv_cars(int32_t t)
{
        for (size_t i = 0; i < S.nlocal; i++) {
                size_t v = S.v_lo + i;

                for (size_t ls = S.lrow_ptr[i]; ls < S.lrow_ptr[i + 1]; ls++) {
                        size_t u = S.ldst[ls];
                        int    rc;

                        rc = dstm_dgraph_edge_get(S.g_cars, v, u,
                                                  S.car_recvbuf,
                                                  (dstm_time_t)t, 0);
                        if (rc != DSTM_SUCCESS)
                                continue;
                        if (car_validate(&S.Lcar, S.car_recvbuf) != 0)
                                continue;       /* never written yet */

                        const struct car_header *h = car_hdr_c(S.car_recvbuf);

                        S.st.recv_msgs++;

                        /* Car half is NOT idempotent; the arrival tick in
                         * the header guards against a stale version. */
                        if (h->tick != t)
                                continue;

                        const struct car_rec *cars =
                                car_cars_c(&S.Lcar, S.car_recvbuf);
                        for (uint32_t k = 0; k < h->n_cars; k++)
                                inject_car(&cars[k], t);
                }
        }
}

/* Called only when (t % stale_delay == 0). The state buffer was put at
 * tick (t - stale_delay) with delta = stale_delay, so it lands at
 * effective time t and is read here at (t, 0). On non-state ticks the
 * version would be zero and state_validate would reject it, so we simply
 * do not read on those ticks. */
static void recv_state(int32_t t)
{
        for (size_t i = 0; i < S.nlocal; i++) {
                size_t v = S.v_lo + i;

                for (size_t ls = S.lrow_ptr[i]; ls < S.lrow_ptr[i + 1]; ls++) {
                        size_t u = S.ldst[ls];
                        int    rc;

                        rc = dstm_dgraph_edge_get(S.g_state, v, u,
                                                  S.state_recvbuf,
                                                  (dstm_time_t)t, 0);
                        if (rc != DSTM_SUCCESS)
                                continue;
                        if (state_validate(&S.Lstate, S.state_recvbuf) != 0)
                                continue;       /* never written yet */

                        if (S.mode == REPLICA_STALE)
                                S.st.merge_updates += cong_merge(&S.rep,
                                        state_entries_c(&S.Lstate,
                                                        S.state_recvbuf));
                        S.st.recv_msgs++;
                }
        }
}

/*******************************************************************************
 * Step 3: route over the replica and stage crossings
 *
 * Backward Dijkstra on the reverse CSR gives cost-to-go to a destination
 * from every vertex; the agent takes the first hop. Trees are cached per
 * (destination, tick), which pays off heavily under the downtown bias.
 ******************************************************************************/

/* In-edges per vertex, as global edge ids. Replicated, built once. */
static int build_reverse(void)
{
        size_t nv = S.city.nv, ne = S.city.ne;

        S.rrow  = calloc(nv + 1, sizeof(*S.rrow));
        S.redge = malloc(ne * sizeof(*S.redge));
        if (!S.rrow || !S.redge)
                return -1;

        for (size_t e = 0; e < ne; e++)
                S.rrow[S.city.dst[e] + 1]++;
        for (size_t v = 0; v < nv; v++)
                S.rrow[v + 1] += S.rrow[v];

        size_t *cur = malloc(nv * sizeof(*cur));
        if (cur == NULL)
            return -1;
        memcpy(cur, S.rrow, nv * sizeof(*cur));
        for (size_t e = 0; e < ne; e++)
                S.redge[cur[S.city.dst[e]]++] = e;
        free(cur);
        return 0;
}

static float edge_cost(size_t ge, int32_t t)
{
        /* Free-flow travel time (ticks) is the base cost of the hop. */
        float ff = (float)S.city.ff_ticks[ge];

        if (S.mode == REPLICA_BLIND)
                return ff;

        const struct cong_entry *e = &S.rep.e[ge];

        /* Only count entries that carry real information. */
        if (e->stamp >= 0) {
                S.st.route_lookups++;
                S.st.route_age_sum += (uint64_t)(t - e->stamp);
        }

        /* load = age-decayed smoothed per-tick crossings on this edge.
         * cong_effective returns 0 when the entry has never been observed,
         * so an unseen edge costs pure free-flow. */
        float load = cong_effective(e, t, CONG_TAU, 0.0f);

        /* capacity in veh/tick; city.c guarantees it is > 0, so no
         * division-by-zero guard is needed here. */
        float cap = S.city.cap_tick[ge];

        /* Linear volume-delay: additive, uncapped congestion penalty. */
        return ff + (load / cap) * ROUTE_DELAY_FACTOR;
}

/* Physical crossing delay (ticks) for edge `ge`, as seen by the SENDER at
 * tick `t`. This is the DSTM put delta that decides when the car arrives,
 * distinct from edge_cost()'s uncapped routing belief. It must be bounded
 * so dstm_fini's drain to last_time cannot be outrun by a car.
 *
 * Model: free-flow + linear volume-delay, congestion capped so the total
 * is at most MAX_CONGESTION_MULTIPLIER * ff -- the bound the drain math in
 * main() relies on. Congestion is read from the sender's replica, so under
 * STALE the delay inherits the same staleness the example studies.
 */
static int edge_delay(size_t ge, int32_t t)
{
        int ff = S.city.ff_ticks[ge];          /* >= 1 */

        const struct cong_entry *e = &S.rep.e[ge];

        float load = cong_effective(e, t, CONG_TAU, 0.0f);
        float cap  = S.city.cap_tick[ge];      /* > 0 by construction */

        int cong = (int)ceilf((load / cap) * ROUTE_DELAY_FACTOR);
        if (cong < 0)
                cong = 0;

        /* Cap so total <= MAX_CONGESTION_MULTIPLIER * ff (drain bound). */
        if (cong > (MAX_CONGESTION_MULTIPLIER - 1) * ff)
                cong = (MAX_CONGESTION_MULTIPLIER - 1) * ff;

        int actual = ff + cong;
        if (actual < 1)
                actual = 1;
        return actual;
}

/* sp[dest] = cost-to-go from every vertex to dest, valid for sp_tick[dest]. */
static const float *shortest_path_tree(size_t dest, int32_t t)
{
        if (S.sp_tick[dest] == t)
                return S.sp[dest];

        float *d = S.sp[dest];
        if (d == NULL)
                d = S.sp[dest] = malloc(S.city.nv * sizeof(float));

        for (size_t v = 0; v < S.city.nv; v++)
                d[v] = INFINITY;
        d[dest] = 0.0f;

        heap_clear(&S.h);
        heap_push(&S.h, dest, 0.0f);

        size_t v; float dv;
        while (heap_pop(&S.h, &v, &dv)) {
                if (dv > d[v])
                        continue;
                /* Relax in-edges: e is u->v, so reaching dest from u
                 * costs cost(e) + d[v]. */
                for (size_t k = S.rrow[v]; k < S.rrow[v + 1]; k++) {
                        size_t e  = S.redge[k];
                        size_t u  = S.city.esrc[e];
                        float  nd = dv + edge_cost(e, t);

                        if (nd < d[u]) {
                                d[u] = nd;
                                heap_push(&S.h, u, nd);
                        }
                }
        }
        S.sp_tick[dest] = t;
        return d;
}

/* Free-flow cost-to-go from every vertex to dest. The result is immutable
 * and can be reused for the remainder of the run. */
static const float *free_flow_shortest_path_tree(size_t dest)
{
        if (S.ff_sp_valid[dest])
                return S.ff_sp[dest];

        float *d = S.ff_sp[dest];
        if (d == NULL) {
                d = malloc(S.city.nv * sizeof(*d));
                assert(d != NULL);
                S.ff_sp[dest] = d;
        }

        for (size_t v = 0; v < S.city.nv; v++)
                d[v] = INFINITY;
        d[dest] = 0.0f;

        heap_clear(&S.h);
        heap_push(&S.h, dest, 0.0f);

        size_t v;
        float dv;

        while (heap_pop(&S.h, &v, &dv)) {
                if (dv > d[v])
                        continue;

                for (size_t k = S.rrow[v]; k < S.rrow[v + 1]; k++) {
                        size_t e = S.redge[k];
                        size_t u = S.city.esrc[e];
                        float nd = dv + (float)S.city.ff_ticks[e];

                        if (nd < d[u]) {
                                d[u] = nd;
                                heap_push(&S.h, u, nd);
                        }
                }
        }

        S.ff_sp_valid[dest] = 1;
        return d;
}

static size_t route_pick(size_t i_local, int32_t dest, int32_t t,
                         int32_t car_id, float *routed_tt)
{
        const float *d = shortest_path_tree((size_t)dest, t);
        size_t best = (size_t)-1;
        float  best_score = 0.0f;
        float  best_prediction = INFINITY;

        for (size_t ls = S.lrow_ptr[i_local];
             ls < S.lrow_ptr[i_local + 1]; ls++) {
                size_t v  = S.ldst[ls];
                size_t ge = S.gslot_base + ls;

                if (!isfinite(d[v]))
                        continue;

                float prediction = edge_cost(ge, t) + d[v];
                float score = prediction + tie_jitter(car_id, t, ge);

                if (best == (size_t)-1 || score < best_score) {
                        best_score = score;
                        best_prediction = prediction;
                        best = ls;
                }
        }

        if (routed_tt != NULL)
                *routed_tt = best_prediction;
        return best;
}

/* Walk every eligible local agent, pick its next edge from the stale
 * replica, and stage it into that edge's outgoing bucket.
 *
 * An agent denied a crossing because the edge is at its cap does NOT
 * vanish and does not get rerouted this tick: it stays where it is and
 * is counted as queueing load on the edge it wanted. */
static void route_and_stage(int32_t t)
{
        memset(S.stage_count, 0, S.nedge_local * sizeof(*S.stage_count));
        memset(S.load_cross,  0, S.nedge_local * sizeof(*S.load_cross));
        memset(S.load_block,  0, S.nedge_local * sizeof(*S.load_block));

        for (size_t idx = 0; idx < S.pool.size; idx++) {
                struct car *c = &S.pool.v[idx];

                if (c->id < 0)
                        continue;               /* free slot */
                if (t < c->ready_tick)
                        continue;               /* still dwelling */

                                size_t i_local = (size_t)c->at - S.v_lo;
                float prediction = INFINITY;
                size_t ls = route_pick(i_local, c->dest, t, c->id,
                                       &prediction);

                if (ls == (size_t)-1)
                        continue;               /* isolated vertex */

                /* The first routing decision defines routed_tt. Record it
                 * even if the car is denied by the edge cap this tick. */
                if (isnan(c->routed_tt)) {
                        const float *ff =
                                free_flow_shortest_path_tree((size_t)c->dest);

                        c->routed_tt = prediction;
                        record_trip_route(c, ff[(size_t)c->at]);
                }

                if (S.stage_count[ls] >= S.Lcar.cap) {
                        S.load_block[ls] += 1.0f;
                        S.st.blocked++;
                        continue;
                }

                size_t ge = S.gslot_base + ls;

                struct car_rec r = {
                        .id          = c->id,
                        .origin      = c->origin,
                        .dest        = c->dest,
                        .dst_vertex  = (int32_t)S.ldst[ls],
                        .depart_tick = c->depart_tick,
                        .hops        = c->hops,
                        .routed_tt   = c->routed_tt,
                        .distance_m  = c->distance_m +
                                       S.city.length_m[ge],
                };

                S.stage[ls * S.Lcar.cap + S.stage_count[ls]] = r;
                S.stage_count[ls]++;
                S.load_cross[ls] += 1.0f;

                /* The agent is now the wire's responsibility. Its local
                 * slot is released here and recreated by the receiver in
                 * inject_car; exactly one copy exists at all times. */
                S.vqueue[i_local]--;
                car_pool_remove(&S.pool, idx);
        }
}

/*******************************************************************************
 * Step 4: measure owned edges and stamp them
 *
 * This is the ONLY place a rank writes an entry it owns, and no rank ever
 * writes an entry it does not own. That single-writer property is what
 * makes the last-writer-wins merge conflict-free.
 *
 * Raw per-tick load is spiky, so it is folded in through an EWMA. The
 * stamp is always the current tick even when the value barely moves:
 * "measured now and it is quiet" is information, and downstream routers
 * need to be able to distinguish it from "no news since tick 3".
 ******************************************************************************/

static void measure_local(int32_t t)
{
        for (size_t ls = 0; ls < S.nedge_local; ls++) {
                size_t             ge = S.gslot_base + ls;
                struct cong_entry *e  = &S.rep.e[ge];

                float raw  = S.load_cross[ls] +
                             BLOCK_PENALTY * S.load_block[ls];
                float prev = (e->stamp < 0) ? 0.0f : e->value;

                e->value = (1.0f - EWMA_ALPHA) * prev + EWMA_ALPHA * raw;
                e->stamp = t;
        }
}

/*******************************************************************************
 * Oracle synchronisation (control mode only)
 *
 * Vertices are block-partitioned, so rank r owns the contiguous edge range
 * [gslot_base, gslot_base + nedge_local); those ranges tile [0, |E|)
 * (asserted at setup), so an allgatherv of owned slices IS the full
 * replica. Since only the owner writes an entry and its stamp is t after
 * measure_local, "take the owner's copy" equals a max-by-stamp allreduce.
 * MPI_IN_PLACE contributes each rank's own bytes at its own displacement.
 * Collective: all ranks parse the same argv, so the mode always agrees.
 ******************************************************************************/
static void replica_sync_oracle(void)
{
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_BYTE,
                       S.rep.e, S.ag_counts, S.ag_displs, MPI_BYTE,
                       MPI_COMM_WORLD);

        /* This rank contributed its own slice (ag_counts[rank] bytes) to
         * the collective. Count it so MiB/tick reflects oracle's actual
         * state movement, comparable to the per-edge diffusion bytes the
         * other modes report. */
        S.st.bytes_out += (uint64_t)S.ag_counts[S.rank];
        S.st.sent_msgs++;
}

/*******************************************************************************
 * Step 5: send on every outgoing channel
 *
 * Every edge is written every tick, even when nothing crosses it: the
 * buffer carries the whole replica, so skipping a quiet edge would stall
 * diffusion through it.
 *
 * delta is edge_delay(ge, t) -- free-flow plus capped congestion. For a
 * cross-rank edge this is >= the distance the matrix enforces (a larger
 * delta is always legal); for an intra-rank edge it is what the model
 * needs so a hop's cost is independent of the partition boundary.
 ******************************************************************************/
static void send_cars(int32_t t)
{
        for (size_t i = 0; i < S.nlocal; i++) {
                size_t u = S.v_lo + i;

                for (size_t ls = S.lrow_ptr[i]; ls < S.lrow_ptr[i + 1]; ls++) {
                        size_t   v      = S.ldst[ls];
                        uint32_t n      = S.stage_count[ls];
                        size_t   ge     = S.gslot_base + ls;
                        int      actual = edge_delay(ge, t);
                        int      rc;

                        car_reset(&S.Lcar, S.car_sendbuf, t + actual, u, v);

                        for (uint32_t k = 0; k < n; k++) {
                                int prc = car_push(&S.Lcar, S.car_sendbuf,
                                        &S.stage[ls * S.Lcar.cap + k]);
                                assert(prc == 0);
                        }

                        rc = dstm_dgraph_edge_put(S.g_cars, v, u,
                                                  S.car_sendbuf,
                                                  (dstm_time_t)t, actual);
                        assert(rc == DSTM_SUCCESS);

                        S.st.sent_msgs++;
                        S.st.bytes_out += S.Lcar.bytes;
                }
        }
}

/* Called only when (t % stale_delay == 0). Uses a fixed delta of
 * stale_delay so it lands exactly on the next state tick. This delta
 * must be >= the boundary distance, which the matrix floor guarantees
 * (see setup()). */
static void send_state(int32_t t)
{
        int delta = S.stale_delay;

        for (size_t i = 0; i < S.nlocal; i++) {
                size_t u = S.v_lo + i;

                for (size_t ls = S.lrow_ptr[i]; ls < S.lrow_ptr[i + 1]; ls++) {
                        size_t v = S.ldst[ls];
                        int    rc;

                        state_reset(&S.Lstate, S.state_sendbuf,
                                    t + delta, u, v);

                        memcpy(state_entries(&S.Lstate, S.state_sendbuf),
                               S.rep.e, S.rep.n * sizeof(struct cong_entry));

                        rc = dstm_dgraph_edge_put(S.g_state, v, u,
                                                  S.state_sendbuf,
                                                  (dstm_time_t)t, delta);
                        assert(rc == DSTM_SUCCESS);

                        S.st.sent_msgs++;
                        S.st.bytes_out += S.Lstate.bytes;
                }
        }
}

/*******************************************************************************
 * Spawn
 ******************************************************************************/

static unsigned int trip_rng_seed(int seed, int32_t trip_id)
{
        uint32_t x = (uint32_t)seed ^ ((uint32_t)trip_id * 0x9e3779b9u);

        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;
        x *= 0x846ca68bu;
        x ^= x >> 16;

        return x ? x : 1u;
}

static void spawn_cars(int n)
{
        if (n <= 0)
                return;

        int32_t first_id = S.next_car_id;
        S.next_car_id += n;

        /* Every rank generates the same global demand records. Only the
         * owner of the generated origin creates the local car. This makes
         * trip_id, origin, destination, and departure tick independent of
         * rank count and replica mode. */
        for (int k = 0; k < n; k++) {
                int32_t trip_id = first_id + k;
                unsigned int rng = trip_rng_seed(S.seed, trip_id);

                size_t origin =
                        (size_t)(xrand(&rng) % (unsigned int)S.city.nv);
                size_t dest =
                        city_pick_destination(&S.city, origin, &rng);

                if (city_owner(&S.city, origin) != S.rank)
                        continue;

                struct car c = {
                        .id          = trip_id,
                        .origin      = (int32_t)origin,
                        .dest        = (int32_t)dest,
                        .at          = (int32_t)origin,
                        .depart_tick = (int32_t)S.t,
                        .hops        = 0,
                        .ready_tick  = 0,
                        .routed_tt   = NAN,
                        .distance_m  = 0.0f,
                };

                car_pool_insert(&S.pool, c);
                S.vqueue[origin - S.v_lo]++;
                S.st.spawned++;
                record_trip_spawn(&c);

                if (routing_metric_member(c.depart_tick))
                        S.st.metric_spawned++;
        }
}

/*******************************************************************************
 * Tick
 ******************************************************************************/

static void run_step(int32_t t)
{
        /* Only STALE diffuses congestion edge-by-edge, and only on its
         * cadence. BLIND never touches the replica. ORACLE replaces
         * diffusion with a collective, so it ships no per-edge state. */
        int diffuse_state = (S.mode == REPLICA_STALE) &&
                            (t % S.stale_delay == 0);

        if (S.spawn_rate > 0)
            spawn_cars(S.spawn_rate);
        publish_vertices(t);    /* creates local version t (both graphs) */

        recv_cars(t);           /* every tick, every mode */
        if (diffuse_state)
                recv_state(t);

        route_and_stage(t);
        measure_local(t);

        if (S.mode == REPLICA_ORACLE)
                replica_sync_oracle();  /* every tick: always-fresh */

        send_cars(t);           /* every tick, every mode */
        if (diffuse_state)
                send_state(t);

        if (S.verbose && S.rank == 0) {
                printf("[t=%4d] live=%-6zu arrived=%-6" PRIu64
                       " crossings=%-7" PRIu64 " blocked=%-6" PRIu64
                       " merged=%" PRIu64 "\n",
                       t, S.pool.live, S.st.arrived, S.st.crossings,
                       S.st.blocked, S.st.merge_updates);
                fflush(stdout);
        }
}

/*******************************************************************************
 * Setup / teardown
 ******************************************************************************/

/* Carve this rank's CSR slice out of the replicated global topology.
 *
 * The slice is what dstm_dgraph_create consumes, and because vertices are
 * partitioned in contiguous blocks the local slot numbering differs from
 * the global edge id by the single constant gslot_base. That is the whole
 * reason the block partition was chosen: converting a local slot to the
 * replica index it addresses is one addition, on every routing lookup. */
static int build_local_slice(void)
{
        S.v_lo   = S.city.vtx_off[S.rank];
        S.v_hi   = S.city.vtx_off[S.rank + 1];
        S.nlocal = S.v_hi - S.v_lo;

        if (S.nlocal == 0) {
                fprintf(stderr,
                        "rank %d: empty partition; use fewer ranks or a "
                        "larger network\n", S.rank);
                return -1;
        }

        S.gslot_base  = S.city.row_ptr[S.v_lo];
        S.nedge_local = S.city.row_ptr[S.v_hi] - S.gslot_base;

        S.lrow_ptr = malloc((S.nlocal + 1) * sizeof(*S.lrow_ptr));
        S.ldeg     = malloc(S.nlocal * sizeof(*S.ldeg));
        S.ldst     = malloc(S.nedge_local * sizeof(*S.ldst));
        if (S.lrow_ptr == NULL || S.ldeg == NULL || S.ldst == NULL) 
                return -1;

        for (size_t i = 0; i <= S.nlocal; i++)
                S.lrow_ptr[i] = S.city.row_ptr[S.v_lo + i] - S.gslot_base;
        for (size_t i = 0; i < S.nlocal; i++)
                S.ldeg[i] = S.lrow_ptr[i + 1] - S.lrow_ptr[i];

        memcpy(S.ldst, &S.city.dst[S.gslot_base],
               S.nedge_local * sizeof(*S.ldst));

        return 0;
}

static int setup(const char *dir, int steps, unsigned int cap)
{
        int rc;

        MPI_Comm_rank(MPI_COMM_WORLD, &S.rank);
        MPI_Comm_size(MPI_COMM_WORLD, &S.world_size);

        if (city_load(&S.city, dir, S.world_size) != 0) {
                if (S.rank == 0)
                        fprintf(stderr, "city_load failed (missing %s/"
                                        "{node,link}.csv, or fewer than %d "
                                        "vertices?)\n", dir, S.world_size);
                return -1;
        }

        if (build_local_slice() != 0)
                return -1;

        car_layout_init(&S.Lcar, cap);
        state_layout_init(&S.Lstate, S.city.ne);

        if (S.rank == 0) {
                city_print_summary(&S.city);
                printf("=== Channel layout ===\n");
                printf("congestion entries : %u  (%zu B state buf)\n",
                       S.Lstate.n_state, S.Lstate.bytes);
                printf("car cap per edge   : %u  (%zu B car buf)\n",
                       S.Lcar.cap, S.Lcar.bytes);
                printf("state cadence      : every %d ticks (delta %d)\n",
                       S.stale_delay, S.stale_delay);
                printf("car edge_state_size   : %zu B\n", S.Lcar.bytes);
                printf("state edge_state_size : %zu B\n", S.Lstate.bytes);
                printf("replica mode       : %s\n", replica_mode_name(S.mode));
                printf("per-tick out/rank  : cars %.2f MiB + state %.2f MiB "
                       "every %d ticks (%zu edges)\n",
                       (double)(S.nedge_local * S.Lcar.bytes)/(1024.0*1024.0),
                       (double)(S.nedge_local * S.Lstate.bytes)/(1024.0*1024.0),
                       S.stale_delay, S.nedge_local);
                printf("======================\n");
                fflush(stdout);
        }

        /* Distance matrix must exist before dstm_start; the graph is created
         * after, since dstm_dgraph_create allocates from the local region.
         *
         * Lookahead per rank<->rank boundary is the MIN free-flow time of
         * any crossing edge, so every crossing put satisfies delta >=
         * distance. The topology is replicated and deterministic, so each
         * rank builds a byte-identical matrix with no communication:
         * diagonal 0, unreachable -1, reachable = min crossing ff. ff_max
         * (the largest put delta component) is computed here for the drain
         * math in main(). */
        {
                int ws = S.world_size;
                int *m = malloc((size_t)ws * (size_t)ws * sizeof(int));
                if (m == NULL) {
                        fprintf(stderr, "rank %d: matrix OOM\n", S.rank);
                        return -1;
                }
                for (int i = 0; i < ws; i++)
                        for (int j = 0; j < ws; j++)
                                m[(size_t)i * ws + j] = (i == j) ? 0 : -1;

                S.ff_max = 1;
                for (size_t u = 0; u < S.city.nv; u++) {
                        int ru = city_owner(&S.city, u);
                        for (size_t k = S.city.row_ptr[u];
                             k < S.city.row_ptr[u + 1]; k++) {
                                size_t v  = S.city.dst[k];
                                int    rv = city_owner(&S.city, v);
                                int    w  = S.city.ff_ticks[k];

                                if (w > S.ff_max)
                                        S.ff_max = w;
                                if (ru == rv)
                                        continue;   /* intra-rank: no matrix */

                                /* Floor the boundary distance at
                                 * min(min_crossing_ff, stale_delay): the
                                 * state stream uses delta = stale_delay,
                                 * so the matrix must not exceed it. Cars
                                 * still use delta = edge_delay >= this. */
                                int wf = (w < S.stale_delay) ? w
                                                             : S.stale_delay;
                                int *a = &m[(size_t)ru * ws + rv];
                                int *b = &m[(size_t)rv * ws + ru];
                                if (*a < 0 || wf < *a) *a = wf;
                                if (*b < 0 || wf < *b) *b = wf;
                        }
                }

                /* Diagnostic: histogram of per-edge free-flow ticks plus
                 * the min-crossing matrix DSTM enforces. If deltas were
                 * uniform, ff_max would be 1 and only one bucket non-empty. */
                if (S.rank == 0) {
                        size_t hist[8] = {0};   /* 1,2,3,4,5,6-9,10-19,20+ */
                        size_t ff_sum = 0, ff_cnt = 0;
                        int    ff_min = INT32_MAX;

                        for (size_t e = 0; e < S.city.ne; e++) {
                                int w = S.city.ff_ticks[e];
                                ff_sum += (size_t)w;
                                ff_cnt++;
                                if (w < ff_min) ff_min = w;
                                if      (w <= 1)  hist[0]++;
                                else if (w == 2)  hist[1]++;
                                else if (w == 3)  hist[2]++;
                                else if (w == 4)  hist[3]++;
                                else if (w == 5)  hist[4]++;
                                else if (w <= 9)  hist[5]++;
                                else if (w <= 19) hist[6]++;
                                else              hist[7]++;
                        }

                        printf("=== Free-flow delta profile (all edges) ===\n");
                        printf("edges=%zu  min=%d  max=%d  mean=%.2f ticks\n",
                               ff_cnt, ff_min, S.ff_max,
                               ff_cnt ? (double)ff_sum / (double)ff_cnt : 0.0);
                        printf("delta histogram: "
                               "[1]=%zu [2]=%zu [3]=%zu [4]=%zu [5]=%zu "
                               "[6-9]=%zu [10-19]=%zu [20+]=%zu\n",
                               hist[0], hist[1], hist[2], hist[3], hist[4],
                               hist[5], hist[6], hist[7]);
                        if (S.ff_max == 1)
                                printf("WARNING: all deltas are 1 -- variable "
                                       "delta is NOT active (check length/"
                                       "fspd parsing)\n");

                        printf("=== Rank-boundary lookahead matrix "
                               "(min crossing ff, -1 = no edge) ===\n");
                        for (int i = 0; i < ws; i++) {
                                printf("  r%-3d:", i);
                                for (int j = 0; j < ws; j++)
                                        printf(" %3d",
                                               m[(size_t)i * ws + j]);
                                printf("\n");
                        }
                        printf("============================================\n");
                        fflush(stdout);
                }

                rc = dstm_distance_create_from_matrix(&S.dist, (size_t)ws, m);
                free(m);
                if (rc != DSTM_SUCCESS) {
                        fprintf(stderr,
                                "rank %d: distance_create_from_matrix "
                                "failed: %d\n", S.rank, rc);
                        return -1;
                }
        }

        rc = dstm_start(S.dist);
        if (rc != DSTM_SUCCESS) {
                fprintf(stderr, "rank %d: dstm_start failed: %d\n",
                        S.rank, rc);
                return -1;
        }

        rc = dstm_dgraph_create(&S.g_cars, sizeof(struct vertex_state),
                                S.Lcar.bytes, S.nlocal, S.ldeg, S.ldst,
                                NULL, NULL);
        if (rc != DSTM_SUCCESS) {
                fprintf(stderr, "rank %d: dgraph_create(cars) failed: %d\n",
                        S.rank, rc);
                return -1;
        }

        rc = dstm_dgraph_create(&S.g_state, sizeof(struct vertex_state),
                                S.Lstate.bytes, S.nlocal, S.ldeg, S.ldst,
                                NULL, NULL);
        if (rc != DSTM_SUCCESS) {
                fprintf(stderr, "rank %d: dgraph_create(state) failed: %d\n",
                        S.rank, rc);
                return -1;
        }

        if (cong_replica_init(&S.rep, S.city.ne) != 0)
                return -1;

        if (build_reverse() != 0) {
                fprintf(stderr, "rank %d: build_reverse failed\n", S.rank);
                return -1;
        }

        /* sp_tick MUST start at -1, not 0: tick 0 is a legitimate tick,
         * so a calloc'd 0 would falsely report a cache hit on the very
         * first routing call and hand out an uninitialised tree. */
        S.sp          = calloc(S.city.nv, sizeof(*S.sp));
        S.sp_tick     = malloc(S.city.nv * sizeof(*S.sp_tick));
        S.ff_sp       = calloc(S.city.nv, sizeof(*S.ff_sp));
        S.ff_sp_valid = calloc(S.city.nv, sizeof(*S.ff_sp_valid));

        if (S.sp == NULL || S.sp_tick == NULL ||
            S.ff_sp == NULL || S.ff_sp_valid == NULL) {
                fprintf(stderr, "rank %d: out of memory in setup\n", S.rank);
                return -1;
        }

        for (size_t i = 0; i < S.city.nv; i++)
                S.sp_tick[i] = -1;
        
        /* Oracle-mode allgatherv descriptors, in BYTES. Built
         * unconditionally: two int arrays of length world_size cost
         * nothing and keeping them mode-independent avoids a second code
         * path that only ever runs in one configuration. */
        S.ag_counts = malloc((size_t)S.world_size * sizeof(*S.ag_counts));
        S.ag_displs = malloc((size_t)S.world_size * sizeof(*S.ag_displs));
        if (S.ag_counts == NULL || S.ag_displs == NULL) {
                fprintf(stderr, "rank %d: out of memory in setup\n", S.rank);
                return -1;
        }
        {
                int mine = (int)(S.nedge_local * sizeof(struct cong_entry));
                int acc  = 0;

                MPI_Allgather(&mine, 1, MPI_INT, S.ag_counts, 1, MPI_INT,
                              MPI_COMM_WORLD);

                for (int r = 0; r < S.world_size; r++) {
                        S.ag_displs[r] = acc;
                        acc += S.ag_counts[r];
                }

                /* If this fires, owned edge ranges do not tile the global
                 * edge id space contiguously and the whole gslot_base
                 * addressing scheme is invalid -- not just oracle mode. */
                assert((size_t)acc == S.rep.n * sizeof(struct cong_entry));

                /* Our own slice must sit at our own displacement, or
                 * MPI_IN_PLACE contributes the wrong bytes. */
                assert(S.ag_displs[S.rank] ==
                       (int)(S.gslot_base * sizeof(struct cong_entry)));
        }

        S.stage       = malloc(S.nedge_local * (size_t)cap * sizeof(*S.stage));
        S.stage_count = malloc(S.nedge_local * sizeof(*S.stage_count));
        S.load_cross  = malloc(S.nedge_local * sizeof(*S.load_cross));
        S.load_block  = malloc(S.nedge_local * sizeof(*S.load_block));
        S.vqueue      = calloc(S.nlocal, sizeof(*S.vqueue));
        S.car_sendbuf   = malloc(S.Lcar.bytes);
        S.car_recvbuf   = malloc(S.Lcar.bytes);
        S.state_sendbuf = malloc(S.Lstate.bytes);
        S.state_recvbuf = malloc(S.Lstate.bytes);

        if (!S.stage || !S.stage_count || !S.load_cross || !S.load_block ||
            !S.vqueue || !S.car_sendbuf || !S.car_recvbuf ||
            !S.state_sendbuf || !S.state_recvbuf) {
                fprintf(stderr, "rank %d: out of memory in setup\n", S.rank);
                return -1;
        }

        car_pool_init(&S.pool);

        S.t     = 0;
        S.t_end = steps;

        // first 2%
        S.metric_depart_limit = 1 + steps / 50;

        S.rng   = (unsigned int)(S.seed * 7919u + (unsigned int)S.rank * 31u
                                 + 1u);

        /* Every rank generates the same deterministic global demand
         * records and only instantiates records whose origin it owns.
         * Therefore every rank must begin from the same global trip ID.
         *
         * Do not use rank-prefixed ID ranges here: doing so causes each
         * rank to generate a different demand set and prevents trip rows
         * from being paired across ranks, modes, and rank counts. */
        S.next_car_id = 0;

        return 0;
}

/* DSTM-owned objects must die before dstm_fini, which destroys the region
 * their allocations live in. */
static void teardown_dstm(void)
{
        if (S.g_cars != NULL)
                dstm_dgraph_destroy(&S.g_cars);
        if (S.g_state != NULL)
                dstm_dgraph_destroy(&S.g_state);
}

static void teardown_local(void)
{
        car_pool_free(&S.pool);
        cong_replica_free(&S.rep);

        free(S.lrow_ptr);
        free(S.ldeg);
        free(S.ldst);
        free(S.stage);
        free(S.stage_count);
        free(S.load_cross);
        free(S.load_block);
        free(S.vqueue);
        free(S.car_sendbuf);
        free(S.car_recvbuf);
        free(S.state_sendbuf);
        free(S.state_recvbuf);
        free(S.ag_counts);
        free(S.ag_displs);
        free(S.rrow);
        free(S.redge);
        if (S.sp != NULL) {
                for (size_t i = 0; i < S.city.nv; i++)
                        free(S.sp[i]);
                free(S.sp);
        }

        if (S.ff_sp != NULL) {
                for (size_t i = 0; i < S.city.nv; i++)
                        free(S.ff_sp[i]);
                free(S.ff_sp);
        }

        free(S.sp_tick);
        free(S.ff_sp_valid);
        free(S.events);
        free(S.h.v);
        free(S.h.k);

        city_free(&S.city);

        if (S.dist != NULL)
                dstm_distance_destroy(&S.dist);
}

/*******************************************************************************
 * Stats
 ******************************************************************************/
static void write_trip_table(void)
{
        int local_bytes;
        int *recv_counts = NULL;
        int *recv_displs = NULL;
        struct trip_event *all_events = NULL;

        assert(S.nevents <= (size_t)INT32_MAX /
                            sizeof(struct trip_event));
        local_bytes = (int)(S.nevents * sizeof(struct trip_event));

        if (S.rank == 0) {
                recv_counts = malloc((size_t)S.world_size *
                                     sizeof(*recv_counts));
                recv_displs = malloc((size_t)S.world_size *
                                     sizeof(*recv_displs));
                assert(recv_counts != NULL && recv_displs != NULL);
        }

        MPI_Gather(&local_bytes, 1, MPI_INT,
                   recv_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int total_bytes = 0;
        if (S.rank == 0) {
                for (int r = 0; r < S.world_size; r++) {
                        recv_displs[r] = total_bytes;
                        total_bytes += recv_counts[r];
                }

                all_events = malloc(total_bytes > 0
                        ? (size_t)total_bytes : 1);
                assert(all_events != NULL);
        }

        MPI_Gatherv(S.events, local_bytes, MPI_BYTE,
                    all_events, recv_counts, recv_displs, MPI_BYTE,
                    0, MPI_COMM_WORLD);

        if (S.rank != 0)
                return;

        size_t ntrips = (size_t)S.next_car_id;
        struct trip_row *rows =
                calloc(ntrips ? ntrips : 1, sizeof(*rows));
        assert(rows != NULL);

        for (size_t i = 0; i < ntrips; i++) {
                rows[i].trip_id = (int32_t)i;
                rows[i].origin = -1;
                rows[i].dest = -1;
                rows[i].depart_tick = -1;
                rows[i].arrival_tick = -1;
                rows[i].hops = -1;
                rows[i].free_flow_tt = NAN;
                rows[i].routed_tt = NAN;
                rows[i].distance_m = NAN;
        }

        size_t nevents = (size_t)total_bytes / sizeof(*all_events);

        for (size_t i = 0; i < nevents; i++) {
                const struct trip_event *e = &all_events[i];

                if (e->trip_id < 0 ||
                    (size_t)e->trip_id >= ntrips)
                        continue;

                struct trip_row *row = &rows[e->trip_id];

                if (e->kind == TRIP_EVENT_SPAWN) {
                        row->spawned = 1;
                        row->origin = e->origin;
                        row->dest = e->dest;
                        row->depart_tick = e->depart_tick;
                } else if (e->kind == TRIP_EVENT_ROUTE) {
                        row->routed = 1;
                        row->free_flow_tt = e->free_flow_tt;
                        row->routed_tt = e->routed_tt;
                } else if (e->kind == TRIP_EVENT_ARRIVE) {
                        row->arrived = 1;
                        row->arrival_tick = e->arrival_tick;
                        row->hops = e->hops;
                        row->distance_m = e->distance_m;

                        /* The routed estimate also travels with the car,
                         * so retain it as a fallback if the route event is
                         * unexpectedly absent. */
                        if (!isfinite(row->routed_tt))
                                row->routed_tt = e->routed_tt;
                }
        }

        char automatic_path[4096];
        const char *path = S.trip_output;

        if (path == NULL) {
                snprintf(automatic_path, sizeof(automatic_path),
                         "trips-%s-d%d-seed%d-pid%ld.csv",
                         replica_mode_name(S.mode),
                         S.stale_delay, S.seed, (long)getpid());
                path = automatic_path;
        }

        FILE *f = fopen(path, "w");
        if (f == NULL) {
                perror(path);
                free(rows);
                free(all_events);
                free(recv_counts);
                free(recv_displs);
                return;
        }

        fprintf(f,
                "trip_id,mode,stale_delay,seed,origin,destination,"
                "depart_tick,arrival_tick,arrived,free_flow_tt,"
                "routed_tt,experienced_tt,prediction_gap,"
                "absolute_gap,hops,distance_m,vmt,vht\n");

        const double metres_per_mile = 1609.344;
        const double hours_per_tick = SECONDS_PER_TICK / 3600.0;

        for (size_t i = 0; i < ntrips; i++) {
                const struct trip_row *row = &rows[i];

                if (!row->spawned /*||
                    !routing_metric_member(row->depart_tick)*/)
                        continue;

                double experienced_tt = NAN;
                double prediction_gap = NAN;
                double absolute_gap = NAN;
                double distance_m = NAN;
                double vmt = NAN;
                double vht = NAN;

                if (row->arrived) {
                        experienced_tt =
                                (double)(row->arrival_tick -
                                         row->depart_tick);
                        distance_m = row->distance_m;
                        vmt = distance_m / metres_per_mile;
                        vht = experienced_tt * hours_per_tick;

                        if (isfinite(row->routed_tt)) {
                                prediction_gap =
                                        experienced_tt - row->routed_tt;
                                absolute_gap = fabs(prediction_gap);
                        }
                }

                fprintf(f,
                        "%d,%s,%d,%d,%d,%d,%d,%d,%d,"
                        "%.9g,%.9g,%.9g,%.9g,%.9g,%d,"
                        "%.9g,%.9g,%.9g\n",
                        row->trip_id,
                        replica_mode_name(S.mode),
                        S.stale_delay,
                        S.seed,
                        row->origin,
                        row->dest,
                        row->depart_tick,
                        row->arrival_tick,
                        row->arrived,
                        row->free_flow_tt,
                        row->routed_tt,
                        experienced_tt,
                        prediction_gap,
                        absolute_gap,
                        row->hops,
                        distance_m,
                        vmt,
                        vht);
        }

        fclose(f);
        printf("trip table: %s\n", path);

        free(rows);
        free(all_events);
        free(recv_counts);
        free(recv_displs);
}

static void collect_and_print_stats(void)
{
        struct rank_stats mine = S.st;
        struct rank_stats *all = NULL;

        mine.live = (uint64_t)S.pool.live;

        /* Final replica staleness -- the number this design exists to
         * produce -- reduced separately from the per-rank table. */
        double age_sum = 0.0;
        double age_cnt = 0.0;
        double age_max = 0.0;

        for (size_t i = 0; i < S.rep.n; i++) {
                if (S.rep.e[i].stamp < 0)
                        continue;
                double a = (double)((int32_t)S.t_end - S.rep.e[i].stamp);
                age_sum += a;
                age_cnt += 1.0;
                if (a > age_max)
                        age_max = a;
        }

        double loc[3]  = { age_sum, age_cnt, age_max };
        double glob[3] = { 0.0, 0.0, 0.0 };

        MPI_Reduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&loc[2], &glob[2], 1, MPI_DOUBLE, MPI_MAX, 0,
                   MPI_COMM_WORLD);

        if (S.rank == 0) {
                all = malloc(sizeof(*all) * (size_t)S.world_size);
                assert(all != NULL);
        }
        MPI_Gather(&mine, sizeof(mine), MPI_BYTE, all, sizeof(mine), MPI_BYTE,
                   0, MPI_COMM_WORLD);

        if (S.rank != 0)
                return;

        uint64_t t_spawn = 0, t_arr = 0, t_live = 0, t_cross = 0, t_blk = 0;
        uint64_t t_send = 0, t_recv = 0, t_merge = 0, t_look = 0, t_age = 0;
        uint64_t t_bytes = 0;

        uint64_t m_spawn = 0;
        uint64_t m_arr   = 0;
        uint64_t m_hops  = 0;
        uint64_t m_trip  = 0;

        printf("\n=== Per-rank stats (replica mode: %s) ===\n",
               replica_mode_name(S.mode));
        printf("%-5s %8s %8s %6s %9s %8s %9s %9s %12s\n",
               "rank", "spawned", "arrived", "live", "crossings", "blocked",
               "sent", "recv", "merge_upd");
        for (int r = 0; r < S.world_size; r++) {
                printf("%-5d %8" PRIu64 " %8" PRIu64 " %6" PRIu64
                       " %9" PRIu64 " %8" PRIu64 " %9" PRIu64 " %9" PRIu64
                       " %12" PRIu64 "\n",
                       r, all[r].spawned, all[r].arrived, all[r].live,
                       all[r].crossings, all[r].blocked, all[r].sent_msgs,
                       all[r].recv_msgs, all[r].merge_updates);

                t_spawn += all[r].spawned;
                t_arr   += all[r].arrived;
                t_live  += all[r].live;
                t_cross += all[r].crossings;
                t_blk   += all[r].blocked;
                t_send  += all[r].sent_msgs;
                t_recv  += all[r].recv_msgs;
                t_merge += all[r].merge_updates;
                t_look  += all[r].route_lookups;
                t_age   += all[r].route_age_sum;
                t_bytes += all[r].bytes_out;

                m_spawn += all[r].metric_spawned;
                m_arr   += all[r].metric_arrived;
                m_hops  += all[r].metric_hop_sum;
                m_trip  += all[r].metric_trip_time_sum;
        }

        printf("--- totals ---\n");
        printf("spawned=%" PRIu64 " arrived=%" PRIu64 " live=%" PRIu64
               " lost=%" PRIu64 "\n",
               t_spawn, t_arr, t_live,
               t_spawn - t_arr - t_live);
        printf("crossings=%" PRIu64 " blocked=%" PRIu64 " (%.1f%% of "
               "attempts denied by per-edge cap)\n",
               t_cross, t_blk,
               100.0 * (double)t_blk /
                       (double)((t_cross + t_blk) ? (t_cross + t_blk) : 1));
        printf("channel msgs: sent=%" PRIu64 " recv=%" PRIu64
               "  merge updates=%" PRIu64 "\n", t_send, t_recv, t_merge);
        printf("bytes shipped=%.2f MiB  (%.2f MiB/tick across all ranks)\n",
               (double)t_bytes / (1024.0 * 1024.0),
               (double)t_bytes / (1024.0 * 1024.0) /
                       (double)(S.t_end + 1));

        {
                double arrived_pct =
                        100.0 * (double)m_arr /
                        (double)(m_spawn ? m_spawn : 1);
                double mean_dur =
                        m_arr ? (double)m_trip / (double)m_arr : NAN;
                double mean_hops =
                        m_arr ? (double)m_hops / (double)m_arr : NAN;

                printf("routing metric cohort: depart ticks [0, %d), "
                       "%d of %d timestamps\n",
                       (int)S.metric_depart_limit,
                       (int)S.metric_depart_limit,
                       (int)S.t_end + 1);

                printf("routing metric: cohort_spawned=%" PRIu64
                       " cohort_arrived=%" PRIu64
                       " arrived_pct=%.2f"
                       " mean_dur=%.2f"
                       " mean_hops=%.2f\n",
                       m_spawn, m_arr, arrived_pct, mean_dur, mean_hops);
        }

        if (S.mode == REPLICA_BLIND)
                printf("routing decisions: not counted in blind mode "
                       "(edge_cost returns free_flow ticks without reading the "
                       "replica)\n");

        else
                printf("routing decisions=%" PRIu64 "  mean age of consulted "
                       "entry=%.2f ticks%s\n",
                       t_look,
                       (double)t_age / (double)(t_look ? t_look : 1),
                       S.mode == REPLICA_ORACLE
                               ? "  (must be exactly 1.00)" : "");    
        
        printf("replica staleness at t=%d: mean=%.2f ticks  max=%.0f ticks\n",
               (int)S.t_end,
               glob[1] > 0.0 ? glob[0] / glob[1] : 0.0, glob[2]);
        printf("  (max should track the graph diameter in hops -- that is\n"
               "   the propagation delay this example exists to model)\n");

        free(all);
}

/*******************************************************************************
 * main
 ******************************************************************************/

static void usage(const char *prog)
{
        fprintf(stderr,
                "usage: %s --data DIR [--steps N] [--cars M] [--rate R]\n"
                "          [--cap C] [--seed S] [--replica-mode M]\n"
                "          [--trip-output FILE] [--verbose] [--quiet]\n"
                "\n"
                "  --data  DIR path to a dir with node.csv and link.csv\n"
                "  --steps N   simulation ticks, t=0..N   (default 60)\n"
                "  --cars  M   global initial trip count    (default 64)\n"
                "  --rate  R   global trips spawned/tick   (default 0)\n"
                "  --cap   C   max cars per edge per tick (default 8)\n"
                "  --seed  S   RNG seed                   (default 42)\n"
                "  --stale-delay D merge replica every D ticks (default 1)\n"
                "  --quiet     silence DSTM logging (default level: debug)\n"
                "  --trip-output FILE  per-trip CSV output path\n"
                "  --replica-mode stale|oracle|blind      (default stale)\n"
                "        stale   diffusion by relay (the real model)\n"
                "        oracle  replica synced globally each tick\n"
                "        blind   congestion ignored, distance only\n"
                "      Run all three at one seed: trip duration must obey\n"
                "      blind >= stale >= oracle, both strict.\n",
                prog);
}

int main(int argc, char *argv[])
{
        int steps = 60;
        int cars  = 64;
        const char *data = NULL;
        int cap   = 8;
        int quiet = 1;
        S.stale_delay = STALE_DELAY_DEFAULT;

        S.seed = 42;

        MPI_Init(&argc, &argv);
        dstm_init(&argc, &argv);

        int r;
        MPI_Comm_rank(MPI_COMM_WORLD, &r);

        for (int i = 1; i < argc; i++) {
                if (!strcmp(argv[i], "--steps") && i + 1 < argc)
                        steps = atoi(argv[++i]);
                else if (!strcmp(argv[i], "--cars") && i + 1 < argc)
                        cars = atoi(argv[++i]);
                else if (!strcmp(argv[i], "--data") && i + 1 < argc)
                        data = argv[++i];
                else if (!strcmp(argv[i], "--cap") && i + 1 < argc)
                        cap = atoi(argv[++i]);
                else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
                        S.seed = atoi(argv[++i]);
                else if (!strcmp(argv[i], "--stale-delay") && i + 1 < argc)
                        S.stale_delay = atoi(argv[++i]);
                else if (!strcmp(argv[i], "--trip-output") && i + 1 < argc)
                        S.trip_output = argv[++i];
                else if (!strcmp(argv[i], "--rate") && i + 1 < argc)
                        S.spawn_rate = atoi(argv[++i]);
                else if (!strcmp(argv[i], "--replica-mode") && i + 1 < argc) {
                        const char *m = argv[++i];

                        if (!strcmp(m, "stale"))
                                S.mode = REPLICA_STALE;
                        else if (!strcmp(m, "oracle"))
                                S.mode = REPLICA_ORACLE;
                        else if (!strcmp(m, "blind"))
                                S.mode = REPLICA_BLIND;
                        else {
                                if (r == 0)
                                        fprintf(stderr,
                                                "unknown --replica-mode "
                                                "'%s' (want stale|oracle|"
                                                "blind)\n", m);
                                dstm_finalize();
                                MPI_Finalize();
                                return 1;
                        }
                }
                else if (!strcmp(argv[i], "--verbose"))
                        S.verbose = 1;
                else if (!strcmp(argv[i], "--quiet"))
                        quiet = 1;
                else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
                        if (r == 0)
                                usage(argv[0]);
                        dstm_finalize();
                        MPI_Finalize();
                        return 0;
                }
        }

        if (steps < 0)
            steps = 0;
        if (cap < 1)
                cap = 1;
        if (S.stale_delay < 1)
                S.stale_delay = 1;

        if (data == NULL) {
                if (r == 0) {
                        fprintf(stderr, "error: --data DIR is required\n\n");
                        usage(argv[0]);
                }
                dstm_finalize();
                MPI_Finalize();
                return 1;
        }

        dstm_log_setlevel(quiet ? DSTM_LOG_QUIET : DSTM_LOG_DEBUG);

        if (setup(data, steps, (unsigned int)cap) != 0) {
                dstm_finalize();
                MPI_Finalize();
                return 1;
        }
    
        spawn_cars(cars);

        double start = MPI_Wtime();

        for (S.t = 0; S.t <= S.t_end; S.t++)
                run_step((int32_t)S.t);

        /* Cars: last put at t_end with delta up to MAX*ff_max. State: last
         * put at the final state tick with delta stale_delay. Drain past
         * whichever reaches further. */
        dstm_time_t last_cars  = S.t_end + MAX_CONGESTION_MULTIPLIER * S.ff_max;
        dstm_time_t last_state = S.t_end + S.stale_delay;
        dstm_time_t last_time  =
                (last_cars > last_state ? last_cars : last_state) + 1;

        teardown_dstm();
        dstm_fini(last_time);
    
        double end = MPI_Wtime();
        if (r == 0) 
            printf("total run time: %.5f\n\n", end - start);

        write_trip_table();
        collect_and_print_stats();

        teardown_local();
        dstm_finalize();
        MPI_Finalize();
        return 0;
}
