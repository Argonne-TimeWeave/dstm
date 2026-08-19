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
 * Every rank holds a replica of the congestion state of the ENTIRE graph.
 * Each tick a rank refreshes the entries it owns from local measurement,
 * then ships the whole replica -- plus the agents physically crossing --
 * along every one of its outgoing edges. Neighbours merge what they
 * receive into their own replica. Information about a distant edge
 * therefore reaches a rank only after as many ticks as there are hops
 * between them, and routing decisions are made against that stale view.
 *
 * Mapping onto DSTM
 * =================
 *
 * The channel for directed edge u->v is the dstm_dgraph edge slot (v,u),
 * which is owned by rank(v) -- the RECEIVER. So:
 *
 *   sender    dstm_dgraph_edge_put(g, v, u, buf, t, +1)   remote write
 *   receiver  dstm_dgraph_edge_get(g, v, u, buf, t,  0)   local read
 *
 * This is the arr_traffic inbox discipline (push into the receiver's
 * future version, drain at delta 0) with an edge slot standing in for an
 * inbox cell. Each slot has exactly one writer and one reader, so no
 * version can be clobbered.
 *
 * delta is 1 on EVERY channel, including edges whose endpoints happen to
 * live on the same rank. Information and agents therefore both advance
 * exactly one hop per tick no matter how the graph is partitioned, so
 * diffusion timing does not depend on where the partition boundary falls.
 *
 * This is NOT the same as reproducibility across rank counts: spawn_cars
 * draws origins from the local vertex range using a per-rank RNG stream,
 * so changing -n changes the demand itself. Only the propagation model is
 * partition-independent, not the run.
 *
 * The merge is highest-stamp-wins on (value, stamp) with exactly one
 * authoritative writer per edge. "Last" is deliberately not what decides
 * it: arrival order carries no weight, only the stamp does.
 * Entries with equal stamps are
 * bit-identical, so the join is commutative, associative and idempotent:
 * MPI delivery order cannot change the answer, and re-reading an
 * unchanged version is harmless.
 *
 * Per-tick discipline
 * ===================
 *
 *   1. publish owned vertex state at (t, 0)
 *        - creates local region version t, so the self-region edge_get
 *          at delta 0 below cannot block on a missing version
 *        - emits heartbeats so the distributed horizon advances
 *   2. receive: for every local vertex v and every neighbour u, read slot
 *      (v,u) at (t, 0); merge the state half, inject the car half
 *   3. route: every eligible local car picks its next edge from the
 *      replica and is staged into that edge's outgoing buffer, subject to
 *      the per-edge cap. Denied cars stay put and are counted as load.
 *   4. measure: fold this tick's per-edge load into the owned entries of
 *      the replica and stamp them with t
 *   5. send: pack [header][state][cars] for every outgoing edge and put
 *      it at (t, +1)
 *
 * Step 4 follows step 2 so a relayed copy of one of our own edges can
 * never overwrite the authoritative value. The stamp rule already
 * guarantees that; the ordering makes it obvious.
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

/* Free-flow hop costs 1. BETA is the congestion premium in units of
 * hops -- BETA=1.0 means a fully-saturated edge is worth detouring one
 * extra hop to avoid. Clamped so no single edge can dominate a path. */
#define ROUTE_BETA     1.0f
#define ROUTE_CONG_MAX 4.0f

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
 *
 * The expected ordering on mean trip duration is
 *   BLIND >= STALE >= ORACLE
 * with BOTH inequalities strict. If STALE ~= BLIND the congestion signal
 * never reaches the routing decision (ROUTE_BETA too small, or a load
 * level where there is nothing to route around); if STALE ~= ORACLE the
 * staleness costs nothing and there is no phenomenon to study. Either
 * collapse means the configuration is wrong, not the tuning.
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
        uint64_t hop_sum;
        uint64_t trip_time_sum;
        uint64_t bytes_out;
};

struct sim {
        int rank;
        int world_size;

        struct city         city;
        struct chan_layout  L;
        struct cong_replica rep;

        dstm_distance_t *dist;
        dstm_dgraph_t   *g;

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

        /* One reusable send buffer and one receive buffer. Both are
         * exactly edge_state_size, allocated once, never resized. */
        void *sendbuf;
        void *recvbuf;

        struct car_pool pool;
        int32_t        *vqueue;         /* nlocal, cars dwelling        */

        dstm_time_t t;
        dstm_time_t t_end;
        unsigned int rng;
        int seed;
        int verbose;

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
        struct pqheap  h;

        /* Oracle mode only. Owned edges are the contiguous global range
         * [gslot_base, gslot_base + nedge_local), so an allgatherv of
         * owned slices rebuilds the entire replica exactly. Byte counts
         * and byte displacements, one per rank. */
        int *ag_counts;
        int *ag_displs;

        int spawn_rate;         /* cars per rank per tick */
        int next_car_id;        /* persistent across calls */

        struct rank_stats st;
};

static struct sim S;

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
        for (size_t i = 0; i < S.nlocal; i++) {
                struct vertex_state vs = {
                        .queue_len = S.vqueue[i],
                        .stamp     = t,
                };
                int rc = dstm_dgraph_vertex_put(S.g, S.v_lo + i, &vs,
                                                (dstm_time_t)t, 0);
                assert(rc == DSTM_SUCCESS);
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
                S.st.hop_sum       += (uint64_t)(r->hops + 1);
                S.st.trip_time_sum += (uint64_t)(t - r->depart_tick);
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
        };
        car_pool_insert(&S.pool, c);
        S.vqueue[at - S.v_lo]++;
}

static void recv_channels(int32_t t)
{
        for (size_t i = 0; i < S.nlocal; i++) {
                size_t v = S.v_lo + i;

                for (size_t ls = S.lrow_ptr[i]; ls < S.lrow_ptr[i + 1]; ls++) {
                        size_t u = S.ldst[ls];
                        int    rc;

                        /* Slot (v,u) lives in our own region: rank(v)
                         * == self. delta 0 is safe because version t was
                         * created by publish_vertices above. */
                        rc = dstm_dgraph_edge_get(S.g, v, u, S.recvbuf,
                                                  (dstm_time_t)t, 0);
                        if (rc != DSTM_SUCCESS)
                                continue;

                        if (chan_validate(&S.L, S.recvbuf) != 0)
                                continue;       /* never written yet */

                        const struct chan_header *h = chan_hdr_c(S.recvbuf);

                        /* The state half is idempotent under merge, so
                         * apply it unconditionally -- even a stale copy
                         * can only be ignored, never regress us.
                         *
                         * Oracle overwrites the whole replica later in the
                         * tick and blind never reads it, so both skip the
                         * merge pass. The message was still sent and
                         * received: only this fold is elided, so the
                         * comparison is not confounded by differing
                         * communication volume. */
                        if (S.mode == REPLICA_STALE)
                                S.st.merge_updates +=
                                        cong_merge(&S.rep,
                                                   chan_state_c(&S.L,
                                                                S.recvbuf));
                        S.st.recv_msgs++;

                        /* The car half is NOT idempotent. DSTM versions
                         * copy forward, so a version nobody wrote this
                         * tick still holds last tick's buffer; injecting
                         * from it would duplicate agents. The arrival
                         * tick in the header is the guard. */
                        if (h->tick != t)
                                continue;

                        const struct car_rec *cars = chan_cars_c(&S.L,
                                                                 S.recvbuf);
                        for (uint32_t k = 0; k < h->n_cars; k++)
                                inject_car(&cars[k], t);
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
        if (S.mode == REPLICA_BLIND)
                return 1.0f;

        const struct cong_entry *e = &S.rep.e[ge];

        /* Only count entries that carry real information. An unstamped
         * entry has no age, so including it in the denominator would
         * silently bias the mean toward zero. */
        if (e->stamp >= 0) {
                S.st.route_lookups++;
                S.st.route_age_sum += (uint64_t)(t - e->stamp);
        }

        float c = cong_effective(e, t, CONG_TAU, 0.0f);

        c = c / (float)S.L.cap;                  /* normalise by capacity */
        if (c > ROUTE_CONG_MAX)
                c = ROUTE_CONG_MAX;
        return 1.0f + ROUTE_BETA * c;
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

static size_t route_pick(size_t i_local, int32_t dest, int32_t t, int32_t car_id)
{
        const float *d = shortest_path_tree((size_t)dest, t);
        size_t best = (size_t)-1;
        float  best_score = 0.0f;

        for (size_t ls = S.lrow_ptr[i_local]; ls < S.lrow_ptr[i_local+1]; ls++) {
                size_t v  = S.ldst[ls];
                size_t ge = S.gslot_base + ls;

                if (v == (size_t)dest)
                        return ls;
                if (!isfinite(d[v]))
                        continue;

                float score = edge_cost(ge, t) + d[v]
                            + tie_jitter(car_id, t, ge);

                if (best == (size_t)-1 || score < best_score) {
                        best_score = score;
                        best = ls;
                }
        }
        return best;
}

/* Walk every eligible local agent, pick its next edge from the stale
 * replica, and stage it into that edge's outgoing bucket.
 *
 * An agent denied a crossing because the edge is at its cap does NOT
 * vanish and does not get rerouted this tick: it stays where it is and
 * is counted as queueing load on the edge it wanted. That feedback is
 * what makes the cap a model parameter rather than just a buffer size --
 * saturated edges report higher load, which propagates outward and steers
 * later agents away. */
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
                size_t ls      = route_pick(i_local, c->dest, t, c->id);

                if (ls == (size_t)-1)
                        continue;               /* isolated vertex */

                if (S.stage_count[ls] >= S.L.cap) {
                        S.load_block[ls] += 1.0f;
                        S.st.blocked++;
                        continue;
                }

                struct car_rec r = {
                        .id          = c->id,
                        .origin      = c->origin,
                        .dest        = c->dest,
                        .dst_vertex  = (int32_t)S.ldst[ls],
                        .depart_tick = c->depart_tick,
                        .hops        = c->hops,
                };

                S.stage[ls * S.L.cap + S.stage_count[ls]] = r;
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
 * Rebuilds the replica from every rank's owned slice. Because vertices
 * are block-partitioned, rank r owns exactly the contiguous global edge
 * range [gslot_base(r), gslot_base(r) + nedge_local(r)), and those ranges
 * tile [0, |E|) without gap or overlap -- asserted at setup. So an
 * allgatherv of owned slices IS the full replica.
 *
 * That makes this provably identical to a max-by-stamp allreduce, without
 * needing MPI_Op_create: after measure_local(t) the owner's stamp for any
 * entry is t and no other rank can hold anything newer, so "take the
 * owner's copy" and "take the max stamp" coincide by the single-writer
 * property.
 *
 * MPI_IN_PLACE means each rank contributes from S.rep.e at its own
 * displacement -- exactly the bytes measure_local just wrote.
 *
 * Collective: every rank must agree on the mode or this deadlocks. All
 * ranks parse the same argv, so that holds.
 ******************************************************************************/
static void replica_sync_oracle(void)
{
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_BYTE,
                       S.rep.e, S.ag_counts, S.ag_displs, MPI_BYTE,
                       MPI_COMM_WORLD);
}

/*******************************************************************************
 * Step 5: send on every outgoing channel
 *
 * Every edge is written every tick, even when no agent crosses it and
 * nothing local changed. The buffer is the vehicle for the whole replica,
 * so skipping a quiet edge would stall diffusion through it.
 *
 * delta is 1 unconditionally. For a cross-rank edge that satisfies DSTM's
 * conservative rule because the distance matrix was built from this same
 * topology with default_delta 1, so adjacent ranks are exactly 1 apart.
 * For an intra-rank edge it is not required by DSTM at all -- it is
 * required by the MODEL, so that one hop always costs one tick regardless
 * of where the partition boundary happens to fall.
 ******************************************************************************/

static void send_channels(int32_t t)
{
        for (size_t i = 0; i < S.nlocal; i++) {
                size_t u = S.v_lo + i;

                for (size_t ls = S.lrow_ptr[i]; ls < S.lrow_ptr[i + 1]; ls++) {
                        size_t   v = S.ldst[ls];
                        uint32_t n = S.stage_count[ls];
                        int      rc;

                        /* chan_reset zeroes the whole buffer including the
                         * state half we are about to overwrite. The double
                         * write is deliberate: it guarantees no stale car
                         * record or padding byte can survive in the tail,
                         * and the memset is trivially vectorised. */
                        chan_reset(&S.L, S.sendbuf, t + 1, u, v);

                        memcpy(chan_state(&S.L, S.sendbuf), S.rep.e,
                               S.rep.n * sizeof(struct cong_entry));

                        for (uint32_t k = 0; k < n; k++) {
                                int prc = chan_push_car(&S.L, S.sendbuf,
                                                &S.stage[ls * S.L.cap + k]);
                                assert(prc == 0);
                        }

                        /* Channel u->v is the slot (v,u), owned by the
                         * RECEIVER. Cross-rank this is a remote put into
                         * rank(v)'s version t+1; intra-rank it is a local
                         * write to our own future version. */
                        rc = dstm_dgraph_edge_put(S.g, v, u, S.sendbuf,
                                                  (dstm_time_t)t, 1);
                        assert(rc == DSTM_SUCCESS);

                        S.st.sent_msgs++;
                        S.st.bytes_out += S.L.bytes;
                }
        }
}

/*******************************************************************************
 * Spawn
 ******************************************************************************/

static void spawn_cars(int n)
{
        if (S.nlocal == 0)
                return;

        for (int k = 0; k < n; k++) {
                size_t origin = S.v_lo + (size_t)(xrand(&S.rng) %
                                                  (unsigned int)S.nlocal);
                size_t dest   = city_pick_destination(&S.city, origin, &S.rng);

                struct car c = {
                        .id          = S.next_car_id++,
                        .origin      = (int32_t)origin,
                        .dest        = (int32_t)dest,
                        .at          = (int32_t)origin,
                        .depart_tick = (int32_t)S.t,
                        .hops        = 0,
                        .ready_tick  = 0,
                };

                car_pool_insert(&S.pool, c);
                S.vqueue[origin - S.v_lo]++;
                S.st.spawned++;
        }
}

/*******************************************************************************
 * Tick
 ******************************************************************************/

static void run_step(int32_t t)
{
        if (S.spawn_rate > 0)
            spawn_cars(S.spawn_rate);
        publish_vertices(t);    /* creates local version t, heartbeats  */
        recv_channels(t);       /* merge state, inject arriving agents  */
        route_and_stage(t);     /* decide, subject to per-edge cap      */
        measure_local(t);       /* stamp owned entries with tick t      */

        /* Placed after measure_local, so at routing time next tick every
         * entry carries stamp t -- age exactly 1 everywhere. Note this is
         * age 1, not age 0: routing precedes measurement within a tick,
         * so even a rank's own edges are one tick stale when consulted.
         * That makes the contrast exact rather than approximate:
         *   oracle  age == 1
         *   stale   age == max(1, hopdist(owner -> here))
         * i.e. the two modes differ by precisely the diffusion delay and
         * nothing else. */
        if (S.mode == REPLICA_ORACLE)
                replica_sync_oracle();

        send_channels(t);       /* [header][state][cars] at (t, +1)     */

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

        chan_layout_init(&S.L, S.city.ne, cap);

        if (S.rank == 0) {
                city_print_summary(&S.city);
                printf("=== Channel layout ===\n");
                printf("congestion entries : %u  (%zu B)\n",
                       S.L.n_state,
                       (size_t)S.L.n_state * sizeof(struct cong_entry));
                printf("car cap per edge   : %u  (%zu B)\n",
                       S.L.cap, (size_t)S.L.cap * sizeof(struct car_rec));
                printf("state_off/cars_off : %u / %u\n",
                       S.L.state_off, S.L.cars_off);
                printf("edge_state_size    : %zu B\n", S.L.bytes);
                printf("replica mode       : %s\n", replica_mode_name(S.mode));
                printf("per-tick out/rank  : %.2f MiB (%zu edges)\n",
                       (double)(S.nedge_local * S.L.bytes) / (1024.0 * 1024.0),
                       S.nedge_local);
                printf("======================\n");
                fflush(stdout);
        }

        /* Distance matrix must exist BEFORE dstm_start, and the graph can
         * only be created AFTER it, because dstm_dgraph_create allocates
         * from the session's local region. Hence the topology-only
         * variant here: it needs nothing but dstm_init. Every edge weighs
         * 1, which is exactly the lookahead our delta=1 channels rely
         * on. */
        rc = dstm_dgraph_distance_from_topology(&S.dist, S.nlocal, S.ldeg,
                                                S.ldst, 1);
        if (rc != DSTM_SUCCESS) {
                fprintf(stderr, "rank %d: distance_from_topology failed: %d\n",
                        S.rank, rc);
                return -1;
        }

        rc = dstm_start(S.dist);
        if (rc != DSTM_SUCCESS) {
                fprintf(stderr, "rank %d: dstm_start failed: %d\n",
                        S.rank, rc);
                return -1;
        }

        /* NULL initial states: every edge slot is seeded to zero, so its
         * header magic is 0 and chan_validate rejects it until someone
         * actually writes. That is the "no message yet" case, handled
         * without a separate valid flag. */
        rc = dstm_dgraph_create(&S.g, sizeof(struct vertex_state), S.L.bytes,
                                S.nlocal, S.ldeg, S.ldst, NULL, NULL);
        if (rc != DSTM_SUCCESS) {
                fprintf(stderr, "rank %d: dstm_dgraph_create failed: %d\n",
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
        S.sp      = calloc(S.city.nv, sizeof(*S.sp));
        S.sp_tick = malloc(S.city.nv * sizeof(*S.sp_tick));
        if (S.sp == NULL || S.sp_tick == NULL) {
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
        S.sendbuf     = malloc(S.L.bytes);
        S.recvbuf     = malloc(S.L.bytes);

        if (!S.stage || !S.stage_count || !S.load_cross || !S.load_block ||
            !S.vqueue || !S.sendbuf || !S.recvbuf) {
                fprintf(stderr, "rank %d: out of memory in setup\n", S.rank);
                return -1;
        }

        car_pool_init(&S.pool);

        S.t     = 0;
        S.t_end = steps;
        S.rng   = (unsigned int)(S.seed * 7919u + (unsigned int)S.rank * 31u
                                 + 1u);

        /* Disjoint id ranges per rank so ids stay unique on the wire. */
        S.next_car_id = S.rank * 1000000;

        return 0;
}

/* DSTM-owned objects must die before dstm_fini, which destroys the region
 * their allocations live in. */
static void teardown_dstm(void)
{
        if (S.g != NULL)
                dstm_dgraph_destroy(&S.g);
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
        free(S.sendbuf);
        free(S.recvbuf);
        free(S.ag_counts);
        free(S.ag_displs);
        free(S.rrow);
        free(S.redge);
        if (S.sp != NULL) {
                for (size_t i = 0; i < S.city.nv; i++)
                        free(S.sp[i]);
                free(S.sp);
        }
        free(S.sp_tick);
        free(S.h.v);
        free(S.h.k);

        city_free(&S.city);

        if (S.dist != NULL)
                dstm_distance_destroy(&S.dist);
}

/*******************************************************************************
 * Stats
 ******************************************************************************/

static void collect_and_print_stats(void)
{
        struct rank_stats mine = S.st;
        struct rank_stats *all = NULL;

        mine.live = (uint64_t)S.pool.live;

        /* Final staleness of the replica: how out of date is each rank's
         * view of the graph once the dust settles? This is the number the
         * whole design exists to produce, so it gets reduced separately
         * rather than squeezed into the per-rank table. */
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
        uint64_t t_hops = 0, t_trip = 0, t_bytes = 0;

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
                t_hops  += all[r].hop_sum;
                t_trip  += all[r].trip_time_sum;
                t_bytes += all[r].bytes_out;
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

        if (t_arr > 0)
                printf("completed trips: mean hops=%.2f mean duration=%.2f "
                       "ticks\n",
                       (double)t_hops / (double)t_arr,
                       (double)t_trip / (double)t_arr);

        if (S.mode == REPLICA_BLIND)
                printf("routing decisions: not counted in blind mode "
                       "(edge_cost returns 1.0 without reading the "
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
                "          [--verbose] [--quiet]\n"
                "\n"
                "  --data  DIR path to a dir with node.csv and link.csv\n"
                "  --steps N   simulation ticks, t=0..N   (default 60)\n"
                "  --cars  M   agents spawned per rank    (default 64)\n"
                "  --rate  R   agents per rank per tick   (default 0)\n"
                "  --cap   C   max cars per edge per tick (default 8)\n"
                "  --seed  S   RNG seed                   (default 42)\n"
                "  --quiet     silence DSTM logging (default level: debug)\n"
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
        int quiet = 0;

        S.seed = 42;

        MPI_Init(&argc, &argv);
        dstm_init(&argc, &argv);

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
                                int r;
                                MPI_Comm_rank(MPI_COMM_WORLD, &r);
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
                        int r;
                        MPI_Comm_rank(MPI_COMM_WORLD, &r);
                        if (r == 0)
                                usage(argv[0]);
                        dstm_finalize();
                        MPI_Finalize();
                        return 0;
                }
        }

        if (cap < 1)
                cap = 1;

        if (data == NULL) {
                int r;
                MPI_Comm_rank(MPI_COMM_WORLD, &r);
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

        for (S.t = 0; S.t <= S.t_end; S.t++)
                run_step((int32_t)S.t);

        /* Last write was at t_end with delta 1. Drain one tick past it. */
        dstm_time_t last_time = S.t_end + 2;

        teardown_dstm();
        dstm_fini(last_time);

        collect_and_print_stats();

        teardown_local();
        dstm_finalize();
        MPI_Finalize();
        return 0;
}
