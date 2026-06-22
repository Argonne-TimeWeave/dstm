/*
 * Variable-topology network simulation with DSTM.
 *
 * Generalizes ring_mini/mini.c to an arbitrary (undirected, weighted) graph.
 *
 * Key differences from the ring version:
 *
 *   - Each rank has one mailbox slot per INCOMING edge (variable, not 2).
 *   - Slot ownership is decided by sorting each node's in-neighbors by rank.
 *     The k-th in-neighbor of node v writes into v's slot k.  Sender and
 *     receiver compute the same ordering independently from the static
 *     edge list, so there is still exactly one writer per slot.
 *   - The distributed array is laid out ragged: slot_offset[r] gives the
 *     base index of rank r's mailbox block.  Total slots = sum of degrees.
 *   - Routing uses a next-hop matrix derived from Floyd-Warshall, not
 *     ring arithmetic.  Forwarding looks up next[rank][dest] and finds
 *     which of rank's outgoing neighbors that is.
 *   - PUT delta = edge weight to the immediate neighbor (queried from the
 *     DSTM topology), not a hardcoded 1, so packets take the right amount
 *     of simulated time to traverse each link.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <inttypes.h>
#include <mpi.h>
#include "dstm.h"

#define PACKET_SIZE   256
#define SLOT_CAPACITY 8

/* ---------- graph definition ---------- */

#define N 4
static const int edges[][3] = {
    {0, 1, 2},
    {0, 3, 2},
    {1, 2, 3},
    {1, 3, 1},
};
#define NUM_EDGES (sizeof(edges) / sizeof(edges[0]))

/* ---------- packet / slot types (unchanged from ring) ---------- */

typedef struct {
    int         valid;
    int         source_rank;
    int         dest_rank;
    int         hops_taken;
    dstm_time_t origin_time;
    uint64_t    id;
    char        payload[PACKET_SIZE - 32];
} mailbox_entry_t;

typedef struct {
    int             n_packets;
    mailbox_entry_t packets[SLOT_CAPACITY];
} mailbox_slot_t;

static const mailbox_slot_t empty_slot = { .n_packets = 0 };

/* ---------- graph derived data, computed by every rank ---------- */

typedef struct {
    int  n;
    int *adj;
    int *dist;
    int *next;

    int  *in_deg;
    int  *out_deg;
    int **in_neigh;
    int **out_neigh;

    size_t *slot_offset;
    size_t  total_slots;
} graph_t;

/* ---------- helpers ---------- */

static const char *pkt_tag(const mailbox_entry_t *p)
{
    static char buf[32];
    snprintf(buf, sizeof(buf), "R%d#%" PRIu64, p->source_rank, p->id);
    return buf;
}

static int cmp_int(const void *a, const void *b)
{
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

static void floyd_warshall(int *dist, int *next, int n)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            next[i * n + j] = (dist[i * n + j] != INT_MAX && i != j) ? j : -1;

    for (int k = 0; k < n; k++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                int ik = dist[i * n + k];
                int kj = dist[k * n + j];
                if (ik == INT_MAX || kj == INT_MAX) continue;
                int via = ik + kj;
                if (via < dist[i * n + j]) {
                    dist[i * n + j] = via;
                    next[i * n + j] = next[i * n + k];
                }
            }
}

static void graph_build(graph_t *g)
{
    g->n = N;
    g->adj  = malloc(sizeof(int) * N * N);
    g->dist = malloc(sizeof(int) * N * N);
    g->next = malloc(sizeof(int) * N * N);

    for (int i = 0; i < N * N; i++) g->adj[i] = INT_MAX;
    for (int i = 0; i < N; i++)     g->adj[i * N + i] = 0;

    for (size_t e = 0; e < NUM_EDGES; e++) {
        int r = edges[e][0], c = edges[e][1], w = edges[e][2];
        g->adj[r * N + c] = w;
        g->adj[c * N + r] = w;
    }

    memcpy(g->dist, g->adj, sizeof(int) * N * N);
    floyd_warshall(g->dist, g->next, N);

    g->in_deg    = calloc(N, sizeof(int));
    g->out_deg   = calloc(N, sizeof(int));
    g->in_neigh  = calloc(N, sizeof(int *));
    g->out_neigh = calloc(N, sizeof(int *));

    for (int v = 0; v < N; v++) {
        for (int u = 0; u < N; u++) {
            if (u == v) continue;
            if (g->adj[u * N + v] != INT_MAX) g->in_deg[v]++;
            if (g->adj[v * N + u] != INT_MAX) g->out_deg[v]++;
        }
        g->in_neigh[v]  = malloc(sizeof(int) * g->in_deg[v]);
        g->out_neigh[v] = malloc(sizeof(int) * g->out_deg[v]);

        int ki = 0, ko = 0;
        for (int u = 0; u < N; u++) {
            if (u == v) continue;
            if (g->adj[u * N + v] != INT_MAX) g->in_neigh[v][ki++]  = u;
            if (g->adj[v * N + u] != INT_MAX) g->out_neigh[v][ko++] = u;
        }
        qsort(g->in_neigh[v],  g->in_deg[v],  sizeof(int), cmp_int);
        qsort(g->out_neigh[v], g->out_deg[v], sizeof(int), cmp_int);
    }

    g->slot_offset = malloc(sizeof(size_t) * (N + 1));
    g->slot_offset[0] = 0;
    for (int v = 0; v < N; v++)
        g->slot_offset[v + 1] = g->slot_offset[v] + g->in_deg[v];
    g->total_slots = g->slot_offset[N];
}

static void graph_free(graph_t *g)
{
    for (int v = 0; v < N; v++) {
        free(g->in_neigh[v]);
        free(g->out_neigh[v]);
    }
    free(g->in_neigh);  free(g->out_neigh);
    free(g->in_deg);    free(g->out_deg);
    free(g->slot_offset);
    free(g->adj); free(g->dist); free(g->next);
}

static int slot_index_on(const graph_t *g, int dest_v, int src_u)
{
    for (int k = 0; k < g->in_deg[dest_v]; k++)
        if (g->in_neigh[dest_v][k] == src_u) return k;
    return -1;
}

static size_t global_slot_index(const graph_t *g, int dest_v, int src_u)
{
    return g->slot_offset[dest_v] + (size_t)slot_index_on(g, dest_v, src_u);
}

static void print_matrix(const char *label, const int *m, int n)
{
    printf("%s:\n    ", label);
    for (int j = 0; j < n; j++) printf("%4d", j);
    printf("\n");
    for (int i = 0; i < n; i++) {
        printf("%2d [", i);
        for (int j = 0; j < n; j++) {
            if (m[i * n + j] == INT_MAX) printf(" inf");
            else                          printf("%4d", m[i * n + j]);
        }
        printf(" ]\n");
    }
}

/* ---------- main ---------- */

static void usage(const char *prog)
{
    fprintf(stderr,
        "usage: %s [--seed S] [--time T] [--prob P] [--quiet]\n", prog);
}

int main(int argc, char *argv[])
{
    int seed = 42, quiet = 0, timesteps = 16, send_prob = 60;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--seed") && i + 1 < argc) seed      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--time") && i + 1 < argc) timesteps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prob") && i + 1 < argc) send_prob = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--quiet"))                quiet     = 1;
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]); return 0;
        }
    }

    int rank, num_ranks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    if (num_ranks != N) {
        if (rank == 0)
            fprintf(stderr, "This example requires exactly %d ranks (got %d)\n",
                    N, num_ranks);
        MPI_Finalize();
        return -1;
    }

    dstm_init(&argc, &argv);
    dstm_log_init(stdout, "variable_net_sim");
    dstm_log_setlevel(quiet ? DSTM_LOG_QUIET : DSTM_LOG_NORMAL);

    graph_t g;
    graph_build(&g);

    if (rank == 0) {
        print_matrix("Distance matrix", g.dist, N);
        print_matrix("Next-hop matrix", g.next, N);
        printf("Total mailbox slots: %zu\n", g.total_slots);
        for (int v = 0; v < N; v++) {
            printf("  node %d: in_deg=%d  in_neigh=[", v, g.in_deg[v]);
            for (int k = 0; k < g.in_deg[v]; k++)
                printf("%s%d", k ? "," : "", g.in_neigh[v][k]);
            printf("]  slot_offset=%zu\n", g.slot_offset[v]);
        }
    }

    dstm_distance_t *topology;
    int status = dstm_distance_create_from_matrix(&topology, N, g.dist);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Failed to create distance matrix");
        MPI_Finalize();
        return -1;
    }

    status = dstm_start(topology);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Failed to start DSTM");
        return -1;
    }

    dstm_darray_t *mailbox;
    status = dstm_darray_create(&mailbox,
                                sizeof(mailbox_slot_t),
                                (size_t)g.in_deg[rank]);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Failed to create mailbox");
        return -1;
    }

    srand(seed + rank);

    int out_deg = g.out_deg[rank];
    mailbox_slot_t *outbox = malloc(sizeof(mailbox_slot_t) * out_deg);

    mailbox_slot_t recv_slot;

    uint64_t next_packet_id = 0;

    for (dstm_time_t t = 0; t < timesteps; t++) {
        dstm_log_normal("Rank %d: time step %ld\n", rank, t);

        for (int k = 0; k < g.in_deg[rank]; k++) {
            int    src_u = g.in_neigh[rank][k];
            int    delta = dstm_distance_get(topology, src_u, rank);
            if (t < delta) {
                size_t idx = g.slot_offset[rank] + k;
                dstm_darray_put(mailbox, idx, &empty_slot, t, 0);
            }
        }

        for (int k = 0; k < out_deg; k++) outbox[k] = empty_slot;

        /* STEP 2 — RECEIVE from each incoming slot. */
        for (int k = 0; k < g.in_deg[rank]; k++) {
            int    src_u = g.in_neigh[rank][k];
            size_t idx   = g.slot_offset[rank] + k;

            status = dstm_darray_get_or_default(mailbox, idx,
                                                &recv_slot, t, 0,
                                                &empty_slot);
            if (status != DSTM_SUCCESS) {
                int link_delta = dstm_distance_get(topology, src_u, rank);
                dstm_log_error("GET failed on slot from %d at t=%ld (link delta=%d, status=%d)",
                               src_u, t, link_delta, status);
                continue;
            }

            if (!quiet) {
                dstm_log_normal("Rank %d [t=%ld]: slot from %d has %d packet(s)\n",
                                rank, t, src_u, recv_slot.n_packets);
                for (int p = 0; p < recv_slot.n_packets; p++) {
                    const mailbox_entry_t *q = &recv_slot.packets[p];
                    dstm_log_normal("  -> %s dst=%d hops=%d\n",
                                    pkt_tag(q), q->dest_rank, q->hops_taken);
                }
            }

            /* STEP 3 — process packets. */
            for (int p = 0; p < recv_slot.n_packets; p++) {
                mailbox_entry_t pkt = recv_slot.packets[p];

                if (pkt.dest_rank == rank) {
                    dstm_log_normal("Rank %d [t=%ld]: DELIVERED %s — "
                        "src=%d origin_t=%ld hops=%d latency=%ld payload=\"%s\"\n",
                        rank, t, pkt_tag(&pkt),
                        pkt.source_rank, pkt.origin_time,
                        pkt.hops_taken, (long)(t - pkt.origin_time),
                        pkt.payload);
                    continue;
                }

                int next_hop = g.next[rank * N + pkt.dest_rank];
                if (next_hop < 0) {
                    dstm_log_warning("Rank %d [t=%ld]: no route %s -> %d — DROP\n",
                             rank, t, pkt_tag(&pkt), pkt.dest_rank);
                    continue;
                }

                int fwd_dir = -1;
                for (int j = 0; j < out_deg; j++)
                    if (g.out_neigh[rank][j] == next_hop) { fwd_dir = j; break; }
                if (fwd_dir < 0) {
                    dstm_log_error("Rank %d: next_hop %d not in out_neigh",
                                   rank, next_hop);
                    continue;
                }

                pkt.hops_taken++;
                mailbox_slot_t *ob = &outbox[fwd_dir];
                if (ob->n_packets < SLOT_CAPACITY) {
                    ob->packets[ob->n_packets++] = pkt;
                    dstm_log_normal("Rank %d [t=%ld]: FORWARD %s src=%d dst=%d "
                            "via %d (hop %d)\n",
                            rank, t, pkt_tag(&pkt),
                            pkt.source_rank, pkt.dest_rank,
                            next_hop, pkt.hops_taken);
                } else {
                    dstm_log_warning("Rank %d [t=%ld]: DROP %s src=%d dst=%d "
                             "— outbox to %d full\n",
                             rank, t, pkt_tag(&pkt),
                             pkt.source_rank, pkt.dest_rank, next_hop);
                }
            }
        }

        /* STEP 4 — INJECT a new packet to a random other rank. */
        if ((rand() % 100) < send_prob) {
            int dest = rand() % (N - 1);
            if (dest >= rank) dest++;

            int next_hop = g.next[rank * N + dest];
            if (next_hop >= 0) {
                int fwd_dir = -1;
                for (int j = 0; j < out_deg; j++)
                    if (g.out_neigh[rank][j] == next_hop) { fwd_dir = j; break; }

                if (fwd_dir >= 0 && outbox[fwd_dir].n_packets < SLOT_CAPACITY) {
                    mailbox_entry_t pkt = {
                        .valid       = 1,
                        .source_rank = rank,
                        .dest_rank   = dest,
                        .hops_taken  = 0,
                        .origin_time = t,
                        .id          = next_packet_id++,
                    };
                    snprintf(pkt.payload, sizeof(pkt.payload),
                             "src=%d dst=%d t=%ld", rank, dest, t);
                    outbox[fwd_dir].packets[outbox[fwd_dir].n_packets++] = pkt;

                    dstm_log_normal("Rank %d [t=%ld]: INJECT %s dst=%d via %d\n",
                                    rank, t, pkt_tag(&pkt), dest, next_hop);
                }
            }
        }

        /* STEP 5 — SEND each outbox to the corresponding neighbor.
         *
         * delta = edge weight to that neighbor, queried from the DSTM
         * topology.  This is the one-hop link latency: a packet PUT at
         * timestep t arrives at the neighbor at timestep t + delta. */
        for (int j = 0; j < out_deg; j++) {
            int    neighbor_v = g.out_neigh[rank][j];
            size_t idx        = global_slot_index(&g, neighbor_v, rank);
            int    delta      = dstm_distance_get(topology, rank, neighbor_v);

            mailbox_slot_t *payload = (outbox[j].n_packets > 0)
                                      ? &outbox[j]
                                      : (mailbox_slot_t *)&empty_slot;

            status = dstm_darray_put(mailbox, idx, payload, t, delta);
            if (status != DSTM_SUCCESS)
                dstm_log_error("PUT to rank %d (delta=%d) failed",
                               neighbor_v, delta);
        }
    }

    dstm_time_t last_time = timesteps + 1;
    status = dstm_barrier(last_time);
    if (status != DSTM_SUCCESS) dstm_log_error("Barrier failed");

    free(outbox);
    dstm_darray_destroy(&mailbox);

    status = dstm_fini(last_time);
    if (status != DSTM_SUCCESS) dstm_log_error("Failed to finalize DSTM");

    dstm_distance_destroy(&topology);
    graph_free(&g);
    dstm_finalize();
    MPI_Finalize();
    return status;
}
