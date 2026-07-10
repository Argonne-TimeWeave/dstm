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

static inline int rr(zombie_state_t *st, int mod)
{
    if (mod <= 0) return 0;
    return (int)(rand_r(&st->rng_state) % (unsigned)mod);
}

/* Local-cell geometry helpers. */
static inline int local_row(zombie_state_t *st, int l) { return l / st->cell_side; }
static inline int local_col(zombie_state_t *st, int l) { return l % st->cell_side; }
static inline int local_id (zombie_state_t *st, int r, int c) { return r * st->cell_side + c; }
static inline int local_in_bounds(zombie_state_t *st, int r, int c) {
    return (r >= 0 && r < st->cell_side && c >= 0 && c < st->cell_side);
}

/* Rank-grid geometry helper. */
static inline void neighbor_offset(zombie_state_t *st, int neigh_rank,
                                   int *drank_r, int *drank_c)
{
    int my_r = st->rank    / st->cols, my_c = st->rank    % st->cols;
    int nr   = neigh_rank  / st->cols, nc   = neigh_rank  % st->cols;
    *drank_r = nr - my_r;
    *drank_c = nc - my_c;
}

/* Cross-rank cell mapping (same as before). */
static inline int neighbor_across(zombie_state_t *st,
                                  int cur_local,
                                  int dr, int dc,
                                  int *out_rank, int *out_local)
{
    int cr = local_row(st, cur_local);
    int cc = local_col(st, cur_local);
    int nr = cr + dr;
    int nc = cc + dc;

    int my_rr = st->rank / st->cols;
    int my_rc = st->rank % st->cols;

    int drank_r = 0, drank_c = 0;
    int wrapped_r = nr, wrapped_c = nc;

    if (nr < 0)                     { drank_r = -1; wrapped_r = st->cell_side - 1; }
    else if (nr >= st->cell_side)   { drank_r = +1; wrapped_r = 0; }

    if (nc < 0)                     { drank_c = -1; wrapped_c = st->cell_side - 1; }
    else if (nc >= st->cell_side)   { drank_c = +1; wrapped_c = 0; }

    int nrank_r = my_rr + drank_r;
    int nrank_c = my_rc + drank_c;

    if (nrank_r < 0 || nrank_r >= st->rows) return 0;
    if (nrank_c < 0 || nrank_c >= st->cols) return 0;

    *out_rank  = nrank_r * st->cols + nrank_c;
    *out_local = local_id(st, wrapped_r, wrapped_c);
    return 1;
}

static inline int neighbor_slot_for_rank(zombie_state_t *st, int neigh_rank)
{
    for (int ni = 0; ni < st->n_mesh_neighbors; ni++)
        if (st->mesh_neighbors[ni] == neigh_rank) return ni;
    return -1;
}

/* ---------- border-strip encode / decode ----------
 *
 * When rank A sends to neighbor rank B (offset (dr, dc) in the rank
 * grid, dr,dc in {-1,0,+1}), the "border strip" is A's boundary of
 * cells facing B:
 *
 *   dr = 0, |dc| = 1  -> W or E edge: a full column, indexed by row
 *   |dr| = 1, dc = 0  -> N or S edge: a full row, indexed by col
 *   |dr| = 1, |dc| = 1 -> corner: a single cell at strip index 0
 *
 * B's inverse map: given a cell (wrapped_r, wrapped_c) on B that A
 * cares about (i.e., A's out-of-bounds step landed there), pick the
 * strip index B used when stamping its bundle to A. That index is the
 * row for E/W edges, the col for N/S edges, and 0 for corners.
 */

/* Sender side: fill strip_pop/strip_zom for the outgoing bundle to
 * `neigh_rank`. Returns strip length used (1 for corner, cell_side for
 * edge). */
static int fill_border_strip(zombie_state_t *st, int neigh_rank,
                             int *out_pop, int *out_zom)
{
    int dr, dc;
    neighbor_offset(st, neigh_rank, &dr, &dc);
    int s = st->cell_side;

    if (dr != 0 && dc != 0) {
        int row = (dr < 0) ? 0 : s - 1;
        int col = (dc < 0) ? 0 : s - 1;
        int l = local_id(st, row, col);
        out_pop[0] = st->last_population[l];
        out_zom[0] = st->last_zombie_count[l];
        return 1;
    }
    if (dr != 0) {
        int row = (dr < 0) ? 0 : s - 1;
        for (int c = 0; c < s; c++) {
            int l = local_id(st, row, c);
            out_pop[c] = st->last_population[l];
            out_zom[c] = st->last_zombie_count[l];
        }
        return s;
    }
    /* dc != 0 */
    int col = (dc < 0) ? 0 : s - 1;
    for (int r = 0; r < s; r++) {
        int l = local_id(st, r, col);
        out_pop[r] = st->last_population[l];
        out_zom[r] = st->last_zombie_count[l];
    }
    return s;
}

/* Receiver side: given a candidate cell (wrapped_r, wrapped_c) on the
 * neighbor rank `neigh_rank`, return the strip index that neighbor
 * used for that cell. Returns -1 if the (wrapped_r, wrapped_c) doesn't
 * actually lie on the strip that neighbor sent us (shouldn't happen
 * with correctly-generated calls from neighbor_across()). */
static int strip_index_from_receiver(zombie_state_t *st, int neigh_rank,
                                     int wrapped_r, int wrapped_c)
{
    /* Offset from *me* to neighbor: */
    int dr_me_to_n, dc_me_to_n;
    neighbor_offset(st, neigh_rank, &dr_me_to_n, &dc_me_to_n);

    /* Offset from *neighbor* to *me* is the negation. That's the (dr,dc)
     * the neighbor used when it filled its strip destined for me. */
    int dr = -dr_me_to_n;
    int dc = -dc_me_to_n;
    int s  = st->cell_side;

    if (dr != 0 && dc != 0) {
        /* Corner: only cell (row_expected, col_expected) matters. */
        int row_expected = (dr < 0) ? 0 : s - 1;
        int col_expected = (dc < 0) ? 0 : s - 1;
        if (wrapped_r == row_expected && wrapped_c == col_expected) return 0;
        return -1;
    }
    if (dr != 0) {
        /* N/S edge: neighbor's strip is a row indexed by col. */
        int row_expected = (dr < 0) ? 0 : s - 1;
        if (wrapped_r != row_expected) return -1;
        return wrapped_c;
    }
    /* E/W edge: neighbor's strip is a column indexed by row. */
    int col_expected = (dc < 0) ? 0 : s - 1;
    if (wrapped_c != col_expected) return -1;
    return wrapped_r;
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

    int side = 1;
    while (side * side < nl_req) side++;
    st->cell_side = side;
    st->nl        = side * side;

    if (st->cell_side > MAX_CELL_SIDE) {
        fprintf(stderr,
                "zombie_init: cell_side=%d exceeds MAX_CELL_SIDE=%d "
                "(bump MAX_CELL_SIDE in zombie.h)\n",
                st->cell_side, MAX_CELL_SIDE);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    st->nh        = nh + nz;
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

    st->carryover_cap = num_ranks * st->nh;
    st->carryover     = calloc(st->carryover_cap, sizeof(human_t));
    st->n_carryover   = 0;

    st->rng_state = (unsigned int)(seed + 1);

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

    /* Precompute mesh-adjacent neighbors (delta == 1). */
    st->mesh_neighbors = calloc(num_ranks, sizeof(int));
    st->n_mesh_neighbors = 0;
    for (int r = 0; r < num_ranks; r++) {
        if (r == rank) continue;
        int d = dstm_distance_get(topology, rank, r);
        if (d != -1) {
            st->mesh_neighbors[st->n_mesh_neighbors++] = r;
        }
    }

    /* Ghost strips: one small strip per mesh neighbor. */
    st->ghost_strip_pop = calloc(st->n_mesh_neighbors, sizeof(int *));
    st->ghost_strip_zom = calloc(st->n_mesh_neighbors, sizeof(int *));
    for (int ni = 0; ni < st->n_mesh_neighbors; ni++) {
        st->ghost_strip_pop[ni] = calloc(MAX_CELL_SIDE, sizeof(int));
        st->ghost_strip_zom[ni] = calloc(MAX_CELL_SIDE, sizeof(int));
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

    if (st->ghost_strip_pop) {
        for (int ni = 0; ni < st->n_mesh_neighbors; ni++)
            free(st->ghost_strip_pop[ni]);
        free(st->ghost_strip_pop);
    }
    if (st->ghost_strip_zom) {
        for (int ni = 0; ni < st->n_mesh_neighbors; ni++)
            free(st->ghost_strip_zom[ni]);
        free(st->ghost_strip_zom);
    }
}

/* ---------- per-agent movement (strip-informed) ---------- */
static void desired_cell(zombie_state_t *st, human_t *h,
                         int *want_node, int *want_local)
{
    int cr = local_row(st, h->cur_local);
    int cc = local_col(st, h->cur_local);

    int best_node  = st->rank;
    int best_local = h->cur_local;

    if (h->state == HSTATE_ZOMBIE) {
        int best_score = st->last_population[h->cur_local]
                       - st->last_zombie_count[h->cur_local];

        for (int dr = -1; dr <= 1; dr++) {
            for (int dc = -1; dc <= 1; dc++) {
                if (dr == 0 && dc == 0) continue;
                int nr = cr + dr, nc = cc + dc;

                int cand_node, cand_local, score;
                if (local_in_bounds(st, nr, nc)) {
                    cand_node  = st->rank;
                    cand_local = local_id(st, nr, nc);
                    score = st->last_population[cand_local]
                          - st->last_zombie_count[cand_local];
                } else {
                    if (!neighbor_across(st, h->cur_local, dr, dc,
                                         &cand_node, &cand_local))
                        continue;
                    int slot = neighbor_slot_for_rank(st, cand_node);
                    if (slot < 0) continue;

                    int wr = local_row(st, cand_local);
                    int wc = local_col(st, cand_local);
                    int si = strip_index_from_receiver(st, cand_node, wr, wc);
                    if (si < 0) continue;

                    int pop = st->ghost_strip_pop[slot][si];
                    int zom = st->ghost_strip_zom[slot][si];
                    score = pop - zom;
                }

                if (score > best_score) {
                    best_score = score;
                    best_node  = cand_node;
                    best_local = cand_local;
                }
            }
        }
    } else {
        int best_score = st->last_zombie_count[h->cur_local];

        for (int dr = -1; dr <= 1; dr++) {
            for (int dc = -1; dc <= 1; dc++) {
                if (dr == 0 && dc == 0) continue;
                int nr = cr + dr, nc = cc + dc;

                int cand_node, cand_local, score;
                if (local_in_bounds(st, nr, nc)) {
                    cand_node  = st->rank;
                    cand_local = local_id(st, nr, nc);
                    score = st->last_zombie_count[cand_local];
                } else {
                    if (!neighbor_across(st, h->cur_local, dr, dc,
                                         &cand_node, &cand_local))
                        continue;
                    int slot = neighbor_slot_for_rank(st, cand_node);
                    if (slot < 0) continue;

                    int wr = local_row(st, cand_local);
                    int wc = local_col(st, cand_local);
                    int si = strip_index_from_receiver(st, cand_node, wr, wc);
                    if (si < 0) continue;

                    score = st->ghost_strip_zom[slot][si];
                }

                if (score < best_score) {
                    best_score = score;
                    best_node  = cand_node;
                    best_local = cand_local;
                }
            }
        }
    }

    *want_node  = best_node;
    *want_local = best_local;
}

static void carryover_push(zombie_state_t *st, const human_t *h)
{
    if (st->n_carryover >= st->carryover_cap) {
        st->carryover_cap *= 2;
        st->carryover = realloc(st->carryover,
                                st->carryover_cap * sizeof(human_t));
    }
    st->carryover[st->n_carryover++] = *h;
}

/* ---------- main timestep ---------- */

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

    /* STEP 0 — age every infected agent. */
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
    for (int r = 0; r < num_ranks; r++) {
        memset(&st->outbox[r], 0, sizeof(cell_slot_t));
    }

    /* STEP 3 — build roster. */
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

    /* 3a — inbound bundles + refresh ghost strips. */
    for (int ni = 0; ni < st->n_mesh_neighbors; ni++) {
        int src = st->mesh_neighbors[ni];

        int delta = dstm_distance_get(st->topology, rank, src);
        if (delta == -1) continue;

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

        /* Refresh ghost strip for this neighbor. Copy the whole
         * MAX_CELL_SIDE slice; unused entries stay at whatever the
         * sender left there (they won't be indexed by strip_index_
         * from_receiver, so it's harmless). */
        for (int k = 0; k < MAX_CELL_SIDE; k++) {
            st->ghost_strip_pop[ni][k] = recv.strip_population[k];
            st->ghost_strip_zom[ni][k] = recv.strip_zombie_count[k];
        }

        if (recv.n_humans == 0) continue;

        for (int i = 0; i < recv.n_humans; i++) {
            human_t h = recv.humans[i];

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

    /* 3c — bootstrap at t=0. */
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

    /* STEP 4 — INFECTION. */
    for (int l = 0; l < nl; l++) {
        if (bucket_n[l] == 0 || zombies_at[l] == 0) continue;

        int n_healthy = 0;
        for (int i = 0; i < bucket_n[l]; i++) {
            if (bucket[l][i].state == HSTATE_HEALTHY) n_healthy++;
        }
        if (n_healthy == 0) continue;

        int n_bites = zombies_at[l];
        if (n_bites > n_healthy) n_bites = n_healthy;

        for (int bite = 0; bite < n_bites; bite++) {
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

    /* Cache per-cell counts for desired_cell(). */
    for (int l = 0; l < nl; l++) {
        st->last_population[l]   = bucket_n[l];
        st->last_zombie_count[l] = zombies_at[l];
    }

    /* Stamp per-neighbor border strip into each outbox slot. */
    for (int ni = 0; ni < st->n_mesh_neighbors; ni++) {
        int dest = st->mesh_neighbors[ni];
        fill_border_strip(st, dest,
                          st->outbox[dest].strip_population,
                          st->outbox[dest].strip_zombie_count);
    }

    /* STEP 5 — MOVE & route every roster agent. */
    int total_present = 0;
    for (int b = 0; b < n_buckets; b++) total_present += bucket_n[b];

    dstm_log_normal("Rank %d [t=%ld]: ROSTER total=%d (limbo=%d)\n",
                    rank, t, total_present, bucket_n[LIMBO]);

    for (int b = 0; b < n_buckets; b++) {
        for (int j = 0; j < bucket_n[b]; j++) {
            human_t *h = &bucket[b][j];

            int from_local = h->cur_local;
            int max_hops = (h->state == HSTATE_ZOMBIE) ? ZOMBIE_STEP : HUMAN_STEP;

            for (int hop = 0; hop < max_hops; hop++) {
                int want_node, want_local;
                desired_cell(st, h, &want_node, &want_local);
                h->dest_node  = want_node;
                h->dest_local = want_local;

                if (h->cur_node == want_node) {
                    if (want_local == h->cur_local) break;
                    h->cur_local = want_local;
                } else {
                    h->cur_node  = want_node;
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

            if (h->home_node == rank) {
                int li = h->human_id - rank * st->nh;
                if (li >= 0 && li < st->nh)
                    st->humans[li] = *h;
            }
        }
    }

    volatile double sink = zombie_do_work(st->work);
    (void)sink;

    for (int b = 0; b < n_buckets; b++) free(bucket[b]);
    free(bucket);
    free(bucket_n);
    free(bucket_cap);
    free(zombies_at);

    /* STEP 6 — PUT to every mesh-adjacent neighbor.
     *
     * IMPORTANT: the outbox for each neighbor already has:
     *   - n_humans + humans[]         (from Step 5's routing)
     *   - strip_population[]           (from the fill_border_strip call
     *   - strip_zombie_count[]          just before Step 5)
     * So this loop only performs the PUT.
     */
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

    /* STEP 7 — Per-tick infection reduction. */
    int local_infected_now = 0;
    for (int i = 0; i < st->nh; i++) {
        if (st->humans[i].state == HSTATE_INFECTED) local_infected_now++;
    }

    if (t % 10 == 0) {
        int total_infected_now = 0;
        MPI_Reduce(&local_infected_now, &total_infected_now, 1,
                   MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0 && st->infection_csv) {
            fprintf(st->infection_csv, "%ld,%d\n", (long)t, total_infected_now);
        }
    }

    #undef BUCKET_PUSH
}
