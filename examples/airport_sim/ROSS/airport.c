#include <math.h>
#include <stdlib.h>
#include "airport.h"

/* --- Globals --- */
unsigned int nairports          = 0;
unsigned int mesh_rows          = 16;
unsigned int mesh_cols          = 16;
unsigned int airports_per_rank  = 1;
unsigned int layover_threshold  = 4;
unsigned int work_iters         = 0;
double       mean_interdeparture = 1.0;
double       mean_flight_time    = 0.5;

/* --- Tier assignment: deterministic function of gid so all ranks agree --- */
static inline int tier_of(tw_lpid gid)
{
    /* Simple hash → tier distribution 15/35/50 */
    unsigned h = (unsigned)gid * 2654435761u;
    unsigned r = h % 100u;
    if (r < 15) return TIER_HUB;
    if (r < 50) return TIER_REGIONAL;
    return TIER_LOCAL;
}

/* An airport's owning "rank" is derived from its gid.
 * This is purely bookkeeping — one LP per airport still. */
static inline unsigned rank_of(tw_lpid g)
{
    return (unsigned)(g / airports_per_rank);
}

/* Rank-mesh coordinates */
static inline int rank_row(unsigned r) { return (int)(r / mesh_cols); }
static inline int rank_col(unsigned r) { return (int)(r % mesh_cols); }

static inline int rank_manh(unsigned a, unsigned b)
{
    int dr = rank_row(a) - rank_row(b);
    int dc = rank_col(a) - rank_col(b);
    if (dr < 0) dr = -dr;
    if (dc < 0) dc = -dc;
    return dr + dc;
}

/* Direct reachability is between the OWNING RANKS.
 * Same rank => always reachable (distance 0). */
static inline int reachable_direct(tw_lpid a, tw_lpid b)
{
    if (a == b) return 0;
    unsigned ra = rank_of(a), rb = rank_of(b);
    if (ra == rb) return 1;   /* same rank: always direct */
    return rank_manh(ra, rb) <= (int)layover_threshold;
}

static inline tw_stime clamp_ts(tw_stime ts) {
    return (ts < g_tw_lookahead) ? g_tw_lookahead : ts;
}

/* --- Synthetic per-departure work (identical intent to airport_do_work) --- */
static double do_work(unsigned iters)
{
    if (iters == 0) return 0.0;
    volatile double acc = 1.0;
    for (unsigned i = 1; i <= iters; i++) {
        acc = acc + sqrt((double)i) * 1.0000001;
        acc = acc - (acc * 0.5);
    }
    return acc;
}

/* --- Weighted destination pick (uses ROSS RNG; count calls for reverse) ---
 * Deterministic weight table means we do NOT consume RNG on the loop that
 * builds total_weight — only one rng call for the pick itself.
 */
static tw_lpid pick_destination(tw_lp *lp, unsigned *rng_calls)
{
    unsigned my_rank = rank_of(lp->gid);
    unsigned long total = 0;
    for (tw_lpid g = 0; g < nairports; g++) {
        if (rank_of(g) == my_rank) continue;   /* no same-rank flights */
        total += tier_recv_weight[tier_of(g)];
    }
    if (total == 0) return (lp->gid + 1) % nairports;

    long r = tw_rand_integer(lp->rng, 0, (long)total - 1);
    (*rng_calls)++;

    unsigned long accum = 0;
    for (tw_lpid g = 0; g < nairports; g++) {
        if (rank_of(g) == my_rank) continue;
        accum += tier_recv_weight[tier_of(g)];
        if ((unsigned long)r < accum) return g;
    }
    return (lp->gid + 1) % nairports;
}

/* Pick canonical LP for a rank: airport 0 on that rank */
static inline tw_lpid rank_representative(unsigned r)
{
    return (tw_lpid)r * airports_per_rank;
}

/* Route by ranks. Returns the LP-gid to send the next event to. */
static tw_lpid next_hop(tw_lpid origin, tw_lpid dest)
{
    unsigned o_rank = rank_of(origin);
    unsigned d_rank = rank_of(dest);

    /* Same rank => deliver straight to final airport */
    if (o_rank == d_rank) return dest;

    /* Direct rank-hop => go straight to final airport */
    if (rank_manh(o_rank, d_rank) <= (int)layover_threshold) return dest;

    /* Otherwise search over RANKS for a good intermediate rank */
    int o_r = rank_row(o_rank), o_c = rank_col(o_rank);
    int d_r = rank_row(d_rank), d_c = rank_col(d_rank);
    int cur_manh = rank_manh(o_rank, d_rank);

    unsigned nranks = mesh_rows * mesh_cols;
    unsigned best_rank = d_rank;   /* fallback: pretend direct */
    int best_manh = cur_manh;
    int best_is_hub = 0;
    int best_cardinal = 0;
    int best_dist = INT_MAX;
    int found = 0;

    for (unsigned r = 0; r < nranks; r++) {
        if (r == o_rank || r == d_rank) continue;
        if (rank_manh(o_rank, r) > (int)layover_threshold) continue; /* not reachable in one rank-hop */

        int h_r = rank_row(r), h_c = rank_col(r);
        int hop_manh = rank_manh(r, d_rank);
        if (hop_manh >= cur_manh) continue;

        /* "Is this rank a HUB rank?" = does any airport on it have tier HUB?
         * Cheap approximation: check its representative airport only. */
        int is_hub = (tier_of(rank_representative(r)) == TIER_HUB);
        int is_cardinal = (h_r == o_r) || (h_c == o_c);
        int dist = rank_manh(o_rank, r);

        int take = 0;
        if (!found)                            take = 1;
        else if (hop_manh != best_manh)        take = (hop_manh < best_manh);
        else if (is_hub != best_is_hub)        take = is_hub;
        else if (is_cardinal != best_cardinal) take = is_cardinal;
        else                                   take = (dist < best_dist);

        if (take) {
            best_rank = r;
            best_manh = hop_manh;
            best_is_hub = is_hub;
            best_cardinal = is_cardinal;
            best_dist = dist;
            found = 1;
        }
    }

    /* Hand off to the representative airport of the next rank */
    return rank_representative(best_rank);
}

/* ================== ROSS callbacks ================== */

/* Linear mapping: LP gid → owning PE (rank) */
static tw_peid airport_map(tw_lpid gid)
{
    return (tw_peid)(gid / g_tw_nlp);
}

static void airport_init(airport_state *s, tw_lp *lp)
{
    s->tier = tier_of(lp->gid);
    s->departures = s->arrivals = s->connections = 0;

    /* Kick off with a self TICK event */
    tw_stime ts = tw_rand_exponential(lp->rng, mean_interdeparture);
    tw_event *e = tw_event_new(lp->gid, clamp_ts(ts), lp);
    airport_message *m = tw_event_data(e);
    m->type = EV_TICK;
    m->origin = lp->gid;
    m->final_dest = lp->gid;
    m->legs = 0;
    m->rc_did_depart = 0;
    m->rc_rng_calls = 1;   /* the exponential above */
    tw_event_send(e);
}

static void airport_event(airport_state *s, tw_bf *bf,
                          airport_message *msg, tw_lp *lp)
{
    (void)bf;
    msg->rc_rng_calls = 0;
    msg->rc_did_depart = 0;

    switch (msg->type) {

    case EV_TICK: {
        /* Schedule next tick regardless */
        tw_stime next_ts = tw_rand_exponential(lp->rng, mean_interdeparture);
        msg->rc_rng_calls++;
        tw_event *nt = tw_event_new(lp->gid, clamp_ts(next_ts), lp);
        airport_message *nm = tw_event_data(nt);
        nm->type = EV_TICK;
        nm->origin = lp->gid;
        nm->final_dest = lp->gid;
        nm->legs = 0;
        tw_event_send(nt);

        /* Roll for departure this tick */
        long roll = tw_rand_integer(lp->rng, 0, 99);
        msg->rc_rng_calls++;
        if (roll >= tier_send_prob[s->tier]) break;

        /* Pick destination */
        tw_lpid dest = pick_destination(lp, &msg->rc_rng_calls);
        if (dest == lp->gid) break;

        /* Route */
        tw_lpid hop = next_hop(lp->gid, dest);
        tw_stime fly_ts = tw_rand_exponential(lp->rng, mean_flight_time);
        msg->rc_rng_calls++;

        tw_event *fe = tw_event_new(hop, clamp_ts(fly_ts), lp);
        airport_message *fm = tw_event_data(fe);
        fm->type = (hop == dest) ? EV_ARRIVAL : EV_CONNECT;
        fm->origin = lp->gid;
        fm->final_dest = dest;
        fm->legs = 1;
        tw_event_send(fe);

        /* Synthetic workload */
        volatile double sink = do_work(work_iters);
        (void)sink;

        s->departures++;
        msg->rc_did_depart = 1;
        break;
    }

    case EV_CONNECT: {
        s->connections++;
        tw_lpid hop = next_hop(lp->gid, msg->final_dest);
        tw_stime fly_ts = tw_rand_exponential(lp->rng, mean_flight_time);
        msg->rc_rng_calls++;

        tw_event *fe = tw_event_new(hop, clamp_ts(fly_ts), lp);
        airport_message *fm = tw_event_data(fe);
        fm->type = (hop == msg->final_dest) ? EV_ARRIVAL : EV_CONNECT;
        fm->origin = msg->origin;
        fm->final_dest = msg->final_dest;
        fm->legs = msg->legs + 1;
        tw_event_send(fe);
        break;
    }

    case EV_ARRIVAL:
        s->arrivals++;
        break;
    }
}

static void airport_revent(airport_state *s, tw_bf *bf,
                           airport_message *msg, tw_lp *lp)
{
    (void)bf; (void)lp;
    /* Undo counter bumps */
    switch (msg->type) {
    case EV_TICK:    if (msg->rc_did_depart) s->departures--; break;
    case EV_CONNECT: s->connections--; break;
    case EV_ARRIVAL: s->arrivals--;    break;
    }
    /* Undo RNG calls in reverse order */
    for (unsigned i = 0; i < msg->rc_rng_calls; i++)
        tw_rand_reverse_unif(lp->rng);
}

static void airport_final(airport_state *s, tw_lp *lp)
{
    (void)lp; (void)s;
    /* Per-LP printing is noisy; totals are aggregated below via stats. */
}

/* --- LP type table --- */
tw_lptype airport_lps[] = {
    {
        (init_f)    airport_init,
        (pre_run_f) NULL,
        (event_f)   airport_event,
        (revent_f)  airport_revent,
        (commit_f)  NULL,
        (final_f)   airport_final,
        (map_f)     airport_map,
        sizeof(airport_state),
    },
    {0},
};

/* --- Command-line options --- */
const tw_optdef app_opt[] = {
    TWOPT_GROUP("Airport Model"),
    TWOPT_UINT("rows",       mesh_rows,          "mesh rows"),
    TWOPT_UINT("cols",       mesh_cols,          "mesh cols (rows*cols must equal nairports)"),
    TWOPT_UINT("na",     airports_per_rank,  "airports per rank (DSTM analog)"),
    TWOPT_UINT("layover",    layover_threshold,  "max Manhattan distance for a direct edge"),
    TWOPT_UINT("work",       work_iters,         "synthetic FP iterations per departure"),
    TWOPT_STIME("interdep",  mean_interdeparture,"mean inter-departure time"),
    TWOPT_STIME("flight",    mean_flight_time,   "mean per-hop flight time"),
    TWOPT_END()
};

int main(int argc, char **argv, char **env)
{
    (void)env;

    tw_opt_add(app_opt);
    tw_init(&argc, &argv);

    if (airports_per_rank == 0) airports_per_rank = 1;
    if (mesh_rows == 0 || mesh_cols == 0) {
        if (tw_ismaster())
            fprintf(stderr, "error: --rows and --cols must both be > 0\n");
        tw_end();
        return 1;
    }
    nairports = mesh_rows * mesh_cols * airports_per_rank;

    /* Sanity: rank count in the mesh must match MPI world size */
    unsigned nranks_mesh = mesh_rows * mesh_cols;
    if (nranks_mesh != (unsigned)tw_nnodes()) {
        if (tw_ismaster())
            fprintf(stderr,
                    "warning: rank-mesh (%u ranks) != MPI world size (%d).\n"
                    "         Routing distances are in terms of the abstract mesh, not real MPI ranks.\n",
                    nranks_mesh, tw_nnodes());
    }

    /* Divide LPs evenly across MPI ranks */
    if (nairports % tw_nnodes() != 0) {
        if (tw_ismaster())
            fprintf(stderr, "error: nairports (%u) not divisible by MPI ranks (%d)\n",
                    nairports, tw_nnodes());
        tw_end();
        return 1;
    }

    /* Divide LPs evenly across ranks */
    unsigned nlp_per_pe = nairports / tw_nnodes();
    g_tw_events_per_pe = 2048 + nlp_per_pe * 8;

    tw_define_lps(nlp_per_pe, sizeof(airport_message));
    for (tw_lpid i = 0; i < g_tw_nlp; i++)
        tw_lp_settype(i, &airport_lps[0]);

    tw_run();

    /* Aggregate & print totals on rank 0 */
    unsigned long local[3]  = {0,0,0};
    unsigned long global[3] = {0,0,0};
    for (tw_lpid i = 0; i < g_tw_nlp; i++) {
        airport_state *s = (airport_state *)tw_getlp(i)->cur_state;
        local[0] += s->departures;
        local[1] += s->arrivals;
        local[2] += s->connections;
    }
    MPI_Reduce(local, global, 3, MPI_UNSIGNED_LONG, MPI_SUM,
               g_tw_masternode, MPI_COMM_ROSS);

    if (tw_ismaster()) {
        printf("\n=============== FLIGHT SUMMARY ===============\n");
        printf("Airports:            %u  (mesh %ux%u, layover<=%u)\n",
               nairports, mesh_rows, mesh_cols, layover_threshold);
        printf("Total departures:    %lu\n", global[0]);
        printf("Total arrivals:      %lu\n", global[1]);
        printf("Total connections:   %lu\n", global[2]);
        printf("Work iters/depart:   %u\n",  work_iters);
        printf("==============================================\n");
    }

    tw_end();
    return 0;
}
