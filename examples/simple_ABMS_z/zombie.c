#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <mpi.h>
#include <dstm.h>

#include "zombie.h"

cell_slot_t empty_cell = { .n_humans = 0 };

/* ---------- generic helpers ---------- */

double zombie_do_work(int work)
{
    double s = 0.0;
    for (int i = 0; i < work; i++) s += sin((double)i);
    return s;
}

int human_find_next_hop(zombie_state_t *st, int from_rank, int to_rank)
{
    if (from_rank == to_rank) return to_rank;

    int cols = st->cols;
    int fr = from_rank / cols, fc = from_rank % cols;
    int tr = to_rank   / cols, tc = to_rank   % cols;

    if (fc != tc) {
        int nc = fc + ((tc > fc) ? 1 : -1);
        return fr * cols + nc;
    } else {
        int nr = fr + ((tr > fr) ? 1 : -1);
        return nr * cols + fc;
    }
}

/* Deterministic per-rank RNG (no reliance on global rand()). */
static inline int rr(zombie_state_t *st, int mod)
{
    if (mod <= 0) return 0;
    return (int)(rand_r(&st->rng_state) % (unsigned)mod);
}

/* Local-cell geometry helpers.
 * A rank owns a cell_side x cell_side sub-grid, row-major, so local id
 * l decomposes as (row, col) = (l / cell_side, l % cell_side). */
static inline int local_row(zombie_state_t *st, int l) { return l / st->cell_side; }
static inline int local_col(zombie_state_t *st, int l) { return l % st->cell_side; }
static inline int local_id (zombie_state_t *st, int r, int c) { return r * st->cell_side + c; }
static inline int local_in_bounds(zombie_state_t *st, int r, int c) {
    return (r >= 0 && r < st->cell_side && c >= 0 && c < st->cell_side);
}

/* ---------- lifecycle ---------- */

void zombie_init(zombie_state_t *st,
                 int rank, int num_ranks, int nl_req, int nh, int nz,
                 int work, int seed,
                 dstm_distance_t *topology,
                 dstm_darray_t   *cells,
                 FILE            *infection_csv)
{
    st->rank      = rank;
    st->num_ranks = num_ranks;

    /* Snap nl up to the next perfect square so local cells form a
     * cell_side x cell_side grid. This preserves the caller's intent
     * ("at least this many local cells") without needing them to pass
     * a perfect square. */
    int side = 1;
    while (side * side < nl_req) side++;
    st->cell_side = side;
    st->nl        = side * side;

    st->nh        = nh + nz;   /* one owner slot per agent */
    st->work      = work;
    st->topology  = topology;
    st->cells     = cells;

    int cols = num_ranks;
    for (int r = 1; r * r <= num_ranks; r++)
        if (num_ranks % r == 0) cols = num_ranks / r;
    st->cols = cols;
    st->rows = num_ranks / cols;

    st->humans     = calloc(st->nh,    sizeof(human_t));
    st->outbox     = calloc(num_ranks, sizeof(cell_slot_t));
    st->departures = calloc(st->nl,    sizeof(int));
    st->arrivals   = calloc(st->nl,    sizeof(int));
    st->infections = calloc(st->nl,    sizeof(int));

    /* Carryover sized for the worst case. */
    st->carryover_cap = num_ranks * st->nh;
    st->carryover     = calloc(st->carryover_cap, sizeof(human_t));
    st->n_carryover   = 0;

    st->rng_state = (unsigned int)(seed + 1);

    /* Seed nz zombies first, then nh humans. Matches Repast's
     * zombie.count / human.count per-process configuration. */
    for (int i = 0; i < st->nh; i++) {
        human_t *h = &st->humans[i];
        h->valid           = 1;
        h->human_id        = rank * st->nh + i;
        h->state           = (i < nz) ? HSTATE_ZOMBIE : HSTATE_HEALTHY;
        h->infection_time  = 0;
        h->home_node       = rank;
        h->home_local      = rr(st, st->nl);
        h->cur_node        = h->home_node;
        h->cur_local       = h->home_local;
        h->dest_node       = h->home_node;
        h->dest_local      = h->home_local;
        h->next_hop_rank   = rank;
        h->legs_completed  = 0;
        snprintf(h->payload, PAYLOAD_MAX, "%c%d@%d.%d",
                 h->state == HSTATE_ZOMBIE ? 'Z' : 'H',
                 h->human_id, h->home_node, h->home_local);
    }

    st->last_population   = calloc(st->nl, sizeof(int));
    st->last_zombie_count = calloc(st->nl, sizeof(int));

    st->infection_csv = infection_csv;

    /* Precompute mesh-adjacent neighbors (delta == 1). Used to drive the
     * every-tick sync pattern regardless of whether any agent actually
     * has traffic for a given neighbor this tick. */
    st->mesh_neighbors = calloc(num_ranks, sizeof(int));
    st->n_mesh_neighbors = 0;
    for (int r = 0; r < num_ranks; r++) {
        if (r == rank) continue;
        int d = dstm_distance_get(topology, rank, r);
        if (d == 1) {
            st->mesh_neighbors[st->n_mesh_neighbors++] = r;
        }
    }
}

void zombie_destroy(zombie_state_t *st)
{
    free(st->humans);
    free(st->outbox);
    free(st->carryover);
    free(st->departures);
    free(st->arrivals);
    free(st->infections);
    free(st->last_population);
    free(st->last_zombie_count);
    free(st->mesh_neighbors);
}

/* ---------- per-agent movement (Repast-style with Moore neighborhood) ---
 *
 * Zombie: face the neighboring local cell with the most humans.
 * Human:  face the neighboring local cell with the fewest zombies.
 * "Neighboring" = the 8 Moore neighbors of the human's current cell
 * inside the rank's cell_side x cell_side sub-grid, plus the current
 * cell itself (the "stay put" option, matching Repast where staying is
 * always a valid winner).
 *
 * Both stay on the current rank (Repast movement is a short step, not
 * a rank hop). The inter-rank codepath below is preserved but is only
 * driven by the always-on sync described in the timestep. */
static void desired_cell(zombie_state_t *st, human_t *h,
                         int *want_node, int *want_local)
{
    int cr = local_row(st, h->cur_local);
    int cc = local_col(st, h->cur_local);

    if (h->state == HSTATE_ZOMBIE) {
        int best = h->cur_local;
        int humans_here_best =
            st->last_population[h->cur_local] - st->last_zombie_count[h->cur_local];

        for (int dr = -1; dr <= 1; dr++) {
            for (int dc = -1; dc <= 1; dc++) {
                if (dr == 0 && dc == 0) continue;
                int nr = cr + dr, nc = cc + dc;
                if (!local_in_bounds(st, nr, nc)) continue;
                int nl = local_id(st, nr, nc);
                int humans_here = st->last_population[nl] - st->last_zombie_count[nl];
                if (humans_here > humans_here_best) {
                    humans_here_best = humans_here;
                    best = nl;
                }
            }
        }
        *want_node  = st->rank;
        *want_local = best;
    } else {
        int best = h->cur_local;
        int zombies_here_best = st->last_zombie_count[h->cur_local];

        for (int dr = -1; dr <= 1; dr++) {
            for (int dc = -1; dc <= 1; dc++) {
                if (dr == 0 && dc == 0) continue;
                int nr = cr + dr, nc = cc + dc;
                if (!local_in_bounds(st, nr, nc)) continue;
                int nl = local_id(st, nr, nc);
                if (st->last_zombie_count[nl] < zombies_here_best) {
                    zombies_here_best = st->last_zombie_count[nl];
                    best = nl;
                }
            }
        }
        *want_node  = st->rank;
        *want_local = best;
    }
}

/* Push an agent into the local carryover list, growing as needed. */
static void carryover_push(zombie_state_t *st, const human_t *h)
{
    if (st->n_carryover >= st->carryover_cap) {
        st->carryover_cap *= 2;
        st->carryover = realloc(st->carryover,
                                st->carryover_cap * sizeof(human_t));
    }
    st->carryover[st->n_carryover++] = *h;
}

/* ---------- main timestep (one TICK) ----------
 *
 * Two streams feed into each tick's roster:
 *   (A) inbound bundles from the darray — agents that traveled here
 *   (B) carryover from last tick — agents physically here that
 *       stayed put or moved locally. NEVER crosses MPI.
 *
 * The darray only ever carries inter-rank traffic. Every tick we
 * PUT to every mesh-adjacent neighbor (with an empty bundle if we
 * have nothing to send) and GET from every mesh-adjacent neighbor,
 * mirroring Repast's per-tick synchronize<AgentPackage>. This keeps
 * the DSTM communication path on the critical path of every tick
 * even in the common case where no agent physically crosses a rank
 * boundary in the minimal model.
 *
 * The self-bundle is sentineled (empty) but never used to ship agents.
 */

void zombie_timestep(zombie_state_t *st, dstm_time_t t)
{
    int rank      = st->rank;
    int num_ranks = st->num_ranks;
    int nl        = st->nl;
    int status;

    double start = 0;
    double end = 0;
    double sum = 0;

    dstm_log_normal("Rank %d [t=%ld]: === step ===\n", rank, t);

    /* STEP 0 — age every infected agent by one tick.
     * When infection_time reaches INFECT_TICKS_TO_ZOMBIE the human
     * becomes a zombie (Repast semantics: hatch<Zombie>(this); die()). */
    for (int i = 0; i < st->n_carryover; i++) {
        human_t *h = &st->carryover[i];
        if (h->state == HSTATE_INFECTED) {
            h->infection_time++;
            if (h->infection_time >= INFECT_TICKS_TO_ZOMBIE) {
                h->state = HSTATE_ZOMBIE;
                dstm_log_normal("Rank %d [t=%ld]: agent %d TURNED ZOMBIE "
                                "at %d.%d\n",
                                rank, t, h->human_id,
                                h->cur_node, h->cur_local);
            }
        }
    }

    /* STEP 2 — clear outbox. */
    for (int r = 0; r < num_ranks; r++)
        st->outbox[r] = empty_cell;

    /* STEP 3 — build roster from BOTH streams.
     * Roster cells 0..nl-1 are local cells; nl is LIMBO (in-transit). */
    int LIMBO = nl;
    int n_buckets = nl + 1;

    human_t **bucket      = calloc(n_buckets, sizeof(human_t *));
    int      *bucket_n    = calloc(n_buckets, sizeof(int));
    int      *bucket_cap  = calloc(n_buckets, sizeof(int));
    int      *zombies_at  = calloc(n_buckets, sizeof(int));

    #define BUCKET_PUSH(b, hp) do {                                    \
        int _b = (b);                                                  \
        if (bucket_n[_b] >= bucket_cap[_b]) {                          \
            bucket_cap[_b] = bucket_cap[_b] ? bucket_cap[_b] * 2 : 16; \
            bucket[_b] = realloc(bucket[_b],                           \
                                 bucket_cap[_b] * sizeof(human_t));    \
        }                                                              \
        bucket[_b][bucket_n[_b]++] = *(hp);                            \
    } while (0)

    /* 3a — inbound bundles from EVERY mesh-adjacent neighbor.
     * We drain the neighbor set unconditionally so the darray GET
     * path fires on every tick, matching Repast's per-tick ghost
     * sync. Empty bundles are cheap and expected. */
    for (int ni = 0; ni < st->n_mesh_neighbors; ni++) {
        int src = st->mesh_neighbors[ni];

        size_t idx = bundle_index(rank, src, num_ranks);
        cell_slot_t recv;

        if (LOGGING) start = MPI_Wtime();

        status = dstm_darray_get_or_default(st->cells, idx,
                                            &recv, t, 0, &empty_cell);

        if (LOGGING) { end = MPI_Wtime(); sum += end - start; }

        if (status != DSTM_SUCCESS) {
            dstm_log_error("GET failed on bundle (dest=%d,src=%d)", rank, src);
            continue;
        }
        if (recv.n_humans == 0) continue;

        for (int i = 0; i < recv.n_humans; i++) {
            human_t h = recv.humans[i];

            /* Tick infection for agents in transit so they don't skip
             * their aging step. */
            if (h.state == HSTATE_INFECTED) {
                h.infection_time++;
                if (h.infection_time >= INFECT_TICKS_TO_ZOMBIE)
                    h.state = HSTATE_ZOMBIE;
            }

            int b;
            if (h.cur_node == rank && h.dest_node == rank &&
                h.cur_local >= 0 && h.cur_local < nl) {
                b = h.cur_local;
                st->arrivals[b]++;
                if (h.state == HSTATE_ZOMBIE) zombies_at[b]++;
            } else {
                b = LIMBO;
            }
            BUCKET_PUSH(b, &h);
        }
    }

    if (LOGGING) { printf("_GET: %f\n", sum); sum = 0; }

    /* 3b — carryover. */
    for (int i = 0; i < st->n_carryover; i++) {
        human_t h = st->carryover[i];
        int b;
        if (h.cur_node == rank && h.dest_node == rank &&
            h.cur_local >= 0 && h.cur_local < nl) {
            b = h.cur_local;
            if (h.state == HSTATE_ZOMBIE) zombies_at[b]++;
        } else {
            b = LIMBO;
        }
        BUCKET_PUSH(b, &h);
    }
    st->n_carryover = 0;

    /* 3c — bootstrap at t=0 from owner records. */
    if (t == 0) {
        for (int i = 0; i < st->nh; i++) {
            human_t *h = &st->humans[i];
            if (!h->valid) continue;
            int b;
            if (h->cur_node == rank && h->dest_node == rank &&
                h->cur_local >= 0 && h->cur_local < nl) {
                b = h->cur_local;
                if (h->state == HSTATE_ZOMBIE) zombies_at[b]++;
            } else {
                b = LIMBO;
            }
            BUCKET_PUSH(b, h);
        }
    }

    /* STEP 4 — INFECTION (Repast rule).
     * For each local cell with >=1 zombie, pick ONE random healthy
     * human currently in that cell and infect them. Matches
     * Zombie::step: humans.oneOf(); infect(human). */
    for (int l = 0; l < nl; l++) {
        if (bucket_n[l] == 0 || zombies_at[l] == 0) continue;

        /* Count zombies and collect indices of healthy humans in this cell. */
        int n_healthy = 0;
        for (int i = 0; i < bucket_n[l]; i++) {
            if (bucket[l][i].state == HSTATE_HEALTHY) n_healthy++;
        }
        if (n_healthy == 0) continue;

        /* Each zombie in the cell infects up to one healthy human.
         * We cap total infections at n_healthy so we don't double-infect. */
        int n_bites = zombies_at[l];
        if (n_bites > n_healthy) n_bites = n_healthy;

        for (int bite = 0; bite < n_bites; bite++) {
            /* Pick the k-th still-healthy human, k in [0, remaining). */
            int remaining = n_healthy - bite;
            int k = rr(st, remaining);

            int seen = 0;
            for (int i = 0; i < bucket_n[l]; i++) {
                human_t *h = &bucket[l][i];
                if (h->state != HSTATE_HEALTHY) continue;
                if (seen == k) {
                    h->state          = HSTATE_INFECTED;
                    h->infection_time = 0;
                    st->infections[l]++;
                    dstm_log_normal("Rank %d [t=%ld]: INFECTED agent %d at %d.%d\n",
                                    rank, t, h->human_id, rank, l);
                    break;
                }
                seen++;
            }
        }
    }

    /* Cache per-cell counts for desired_cell() below. */
    for (int l = 0; l < nl; l++) {
        st->last_population[l]   = bucket_n[l];
        st->last_zombie_count[l] = zombies_at[l];
    }

    /* STEP 5 — MOVE & route every roster agent.
     * Zombies take up to ZOMBIE_STEP local hops; humans take up to
     * HUMAN_STEP. Each hop consults desired_cell() again using the
     * per-cell caches computed above, so multi-step movement still
     * chases/flees greedily but never crosses ranks. */
    int total_present = 0;
    for (int b = 0; b < n_buckets; b++) total_present += bucket_n[b];

    dstm_log_normal("Rank %d [t=%ld]: ROSTER total=%d (limbo=%d)\n",
                    rank, t, total_present, bucket_n[LIMBO]);

    for (int b = 0; b < n_buckets; b++) {
        for (int j = 0; j < bucket_n[b]; j++) {
            human_t *h = &bucket[b][j];

            int from_local = h->cur_local;
            int max_hops = (h->state == HSTATE_ZOMBIE) ? ZOMBIE_STEP : HUMAN_STEP;

            /* Multi-hop local movement (Repast: zombies 0.5, humans 1.5). */
            for (int hop = 0; hop < max_hops; hop++) {
                int want_node, want_local;
                desired_cell(st, h, &want_node, &want_local);
                h->dest_node  = want_node;
                h->dest_local = want_local;

                if (h->cur_node == want_node) {
                    if (want_local == h->cur_local) break; /* nowhere better */
                    h->cur_local = want_local;
                } else {
                    /* Inter-rank branch retained for the sync path; it
                     * shouldn't fire from desired_cell() but is preserved
                     * for the day someone adds cross-rank movement. */
                    int dest_rank = human_find_next_hop(st, rank, want_node);
                    h->cur_node = dest_rank;
                    if (dest_rank == want_node)
                        h->cur_local = want_local;
                    break;
                }
            }

            int dest_rank;
            if (h->cur_node == rank) {
                dest_rank = rank;
            } else {
                dest_rank = human_find_next_hop(st, rank, h->cur_node);
            }
            h->next_hop_rank = dest_rank;
            h->legs_completed++;

            if (dest_rank == rank) {
                carryover_push(st, h);
            } else {
                cell_slot_t *ob = &st->outbox[dest_rank];
                if (ob->n_humans >= CELL_CAPACITY) {
                    /* Held over — bounce back to carryover. */
                    h->cur_node  = rank;
                    h->cur_local = from_local;
                    carryover_push(st, h);
                    dstm_log_warning("Rank %d [t=%ld]: outbox->%d FULL, "
                                     "agent %d held over at %d.%d\n",
                                     rank, t, dest_rank,
                                     h->human_id, rank, from_local);
                } else {
                    ob->humans[ob->n_humans++] = *h;
                    if (b != LIMBO) st->departures[from_local]++;
                }
            }

            /* Mirror to owner record so the final summary is accurate. */
            if (h->home_node == rank) {
                int li = h->human_id - rank * st->nh;
                if (li >= 0 && li < st->nh)
                    st->humans[li] = *h;
            }
        }
    }

    volatile double sink = zombie_do_work(st->work);
    (void)sink;

    /* Cleanup roster. */
    for (int b = 0; b < n_buckets; b++) free(bucket[b]);
    free(bucket);
    free(bucket_n);
    free(bucket_cap);
    free(zombies_at);

    /* STEP 6 — PUT to EVERY mesh-adjacent neighbor.
     * We always PUT, even if the bundle is empty, to mirror Repast's
     * per-tick synchronize<AgentPackage>: the framework's ghost buffers
     * exchange messages every tick regardless of whether any agent
     * physically crossed a boundary. This exercises the DSTM darray
     * PUT path on every tick without changing its shape. */
    for (int ni = 0; ni < st->n_mesh_neighbors; ni++) {
        int dest = st->mesh_neighbors[ni];
        size_t idx = bundle_index(dest, rank, num_ranks);
        int delta = dstm_distance_get(st->topology, rank, dest);
        if (delta == -1) continue;

        if (LOGGING) start = MPI_Wtime();

        status = dstm_darray_put(st->cells, idx,
                                 &st->outbox[dest], t, delta);

        if (LOGGING) { end = MPI_Wtime(); sum += end - start; }

        if (status != DSTM_SUCCESS)
            dstm_log_error("PUT to rank %d failed", dest);
    }

    if (LOGGING) { printf("_PUT: %f\n", sum); sum = 0; }

    /* STEP 7 — Per-tick infection reduction, mirroring Repast's
     * InfectionSum data set (std::plus<int>). Rank 0 writes one CSV row
     * per tick. Only ranks with a non-null csv handle write. */
    int local_infected_now = 0;
    for (int i = 0; i < st->nh; i++) {
        if (st->humans[i].state == HSTATE_INFECTED) local_infected_now++;
    }

    if (t % 10 == 11) {
        int total_infected_now = 0;
        MPI_Reduce(&local_infected_now, &total_infected_now, 1,
                   MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0 && st->infection_csv) {
            fprintf(st->infection_csv, "%ld,%d\n", (long)t, total_infected_now);
        }
    }

    #undef BUCKET_PUSH
}
