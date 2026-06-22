/*
 * two_rings_mini.c — two independent rings, one per DSTM region.
 *
 * Goal: exercise the use of two distinct dstm_region_t objects within a
 * single DSTM job.  Ranks split into two groups:
 *
 *   Ring A: ranks [0, RING_SIZE)              uses region_a
 *   Ring B: ranks [RING_SIZE, 2*RING_SIZE)    uses region_b
 *
 * The two rings do NOT communicate (no bridge).  Within each ring the
 * forwarding logic is identical to the original ring_mini example.
 *
 * Important caveat: DSTM's public API (as of this writing) has no
 * function to create a darray in a non-default region.  This file
 * supplies its own create/destroy helpers that mirror dstm_darray_create
 * but take an explicit region pointer.  Whether DSTM's get/put wire
 * protocol can actually distinguish two regions on the same rank is the
 * experimental question.  If it cannot, this example will misbehave or
 * crash and that itself is a useful result for the advisor's question.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <mpi.h>
#include "dstm.h"

#include "internal/region.h"

/* DSTM internal headers used here ONLY for the custom darray helper.  In a
 * real fix this functionality belongs upstream as
 * dstm_darray_create_in_region().  We rely on the public dstm.h pulling in
 * enough of the dstm_darray_t / dstm_ptr_t definitions to compile. */

#define RING_SIZE      4
#define NUM_RINGS      2
#define PACKET_SIZE    256
#define SLOT_CAPACITY  8

#define MAILBOX_SLOTS  2
#define SLOT_LEFT      0
#define SLOT_RIGHT     1

/* Region size needs to be big enough for the darray storage plus all the
 * version snapshots DSTM creates.  Each region holds one ring's mailbox:
 *   RING_SIZE * MAILBOX_SLOTS * sizeof(mailbox_slot_t)
 * times some factor for version history.  Be generous; the region size
 * must also be a power of 2 (dstm_region_create rounds up). */
#define REGION_SIZE    (1u << 20)   /* 1 MiB per region */

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

static const char *pkt_tag(const mailbox_entry_t *p)
{
    static char buf[40];
    snprintf(buf, sizeof(buf), "R%d#%" PRIu64, p->source_rank, p->id);
    return buf;
}

/* -----------------------------------------------------------------------
 * Custom darray-in-region helper.
 *
 * This is a near-verbatim copy of dstm_darray_create from the DSTM source
 * you pasted, with one change: the region used for allocation is passed in
 * rather than fetched from dstm_get_local_region().  If/when DSTM exposes
 * dstm_darray_create_in_region() upstream, delete this and use that.
 *
 * NOTE: this requires that dstm.h exposes the dstm_darray_t and dstm_ptr_t
 * struct layouts.  If they are opaque you'll need to either move this
 * helper into the DSTM source tree, or have your advisor expose a setter.
 * ----------------------------------------------------------------------- */

extern int dstm_global_world_size(void);   /* if exposed; else use MPI directly */
extern int dstm_global_rank(void);

/* Forward declarations of the bits of DSTM we reach into. */
extern dstm_ptr_t dstm_region_malloc(dstm_region_t *r, size_t size);
extern void       dstm_region_free  (dstm_region_t *r, dstm_ptr_t p);
extern int        dstm_ptr_isnull   (dstm_ptr_t ptr);

/* If dstm_darray_t is opaque in dstm.h, you'll need to either
 *   (a) #include the internal header that defines it, or
 *   (b) put this helper in a .c file inside the DSTM source tree.
 * For the example to compile standalone, we declare the fields we touch:
 */
struct dstm_darray_internal {
    size_t        element_size;
    size_t        local_elements;
    size_t        total_elements;
    size_t       *rank_elements;
    size_t       *rank_offsets;
    dstm_ptr_t   *rank_ptrs;
    dstm_region_t *region;
};

static int my_darray_create_in_region(dstm_darray_t **ret,
                                      dstm_region_t *region,
                                      size_t element_size,
                                      size_t local_elements)
{
    if (!ret || !region || element_size == 0 || local_elements == 0)
        return -1;

    int world_size, self;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &self);

    struct dstm_darray_internal *d = calloc(1, sizeof(*d));
    if (!d) return -1;

    d->element_size   = element_size;
    d->local_elements = local_elements;
    d->region         = region;
    d->rank_elements  = calloc(world_size, sizeof(size_t));
    d->rank_offsets   = calloc(world_size, sizeof(size_t));
    d->rank_ptrs      = calloc(world_size, sizeof(dstm_ptr_t));
    if (!d->rank_elements || !d->rank_offsets || !d->rank_ptrs)
        goto err;

    dstm_ptr_t local_ptr =
        dstm_region_malloc(region, element_size * local_elements);
    if (dstm_ptr_isnull(local_ptr))
        goto err;

    if (MPI_Allgather(&local_elements, 1, MPI_UNSIGNED_LONG_LONG,
                      d->rank_elements, 1, MPI_UNSIGNED_LONG_LONG,
                      MPI_COMM_WORLD) != MPI_SUCCESS)
        goto err_ptr;

    if (MPI_Allgather(&local_ptr, 2, MPI_UNSIGNED_LONG_LONG,
                      d->rank_ptrs, 2, MPI_UNSIGNED_LONG_LONG,
                      MPI_COMM_WORLD) != MPI_SUCCESS)
        goto err_ptr;

    size_t total = 0;
    for (int i = 0; i < world_size; i++)
        total += d->rank_elements[i];
    d->total_elements = total;
    for (int i = 0; i < world_size; i++) {
        d->rank_offsets[i] = 0;
        for (int j = 0; j < i; j++)
            d->rank_offsets[i] += d->rank_elements[j];
    }

    *ret = (dstm_darray_t *)d;
    return 0;

err_ptr:
    dstm_region_free(region, local_ptr);
err:
    free(d->rank_ptrs); free(d->rank_offsets); free(d->rank_elements);
    free(d);
    return -1;
}

static void my_darray_destroy_in_region(dstm_darray_t **darray,
                                        dstm_region_t *region)
{
    if (!darray || !*darray) return;
    struct dstm_darray_internal *d = (struct dstm_darray_internal *)*darray;
    int self;
    MPI_Comm_rank(MPI_COMM_WORLD, &self);
    if (!dstm_ptr_isnull(d->rank_ptrs[self]))
        dstm_region_free(region, d->rank_ptrs[self]);
    free(d->rank_ptrs); free(d->rank_offsets); free(d->rank_elements);
    free(d);
    *darray = NULL;
}

/* ----------------------------------------------------------------------- */

static void usage(const char *prog)
{
    fprintf(stderr,
        "usage: %s [--seed S] [--time T] [--prob P] [--quiet]\n"
        "  Requires exactly %d MPI ranks (two rings of %d).\n",
        prog, RING_SIZE * NUM_RINGS, RING_SIZE);
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

    if (num_ranks != RING_SIZE * NUM_RINGS) {
        if (rank == 0) usage(argv[0]);
        MPI_Finalize();
        return -1;
    }

    dstm_init(&argc, &argv);
    dstm_log_init(stdout, "two_rings");
    dstm_log_setlevel(quiet ? DSTM_LOG_QUIET : DSTM_LOG_NORMAL);

    /* Ring membership: ring 0 = [0, RING_SIZE), ring 1 = [RING_SIZE, 2R). */
    int ring_id    = rank / RING_SIZE;
    int ring_rank  = rank % RING_SIZE;    /* position within the ring */
    int ring_base  = ring_id * RING_SIZE; /* lowest global rank in this ring */

    int left_ring_rank  = (ring_rank - 1 + RING_SIZE) % RING_SIZE;
    int right_ring_rank = (ring_rank + 1) % RING_SIZE;
    int left_global     = ring_base + left_ring_rank;
    int right_global    = ring_base + right_ring_rank;

    /* Build a topology that says "every rank is at distance 1 from every
     * other rank" — DSTM only needs *something*, and since the two rings
     * never PUT outside their own ring this matrix never gets exercised
     * across rings.  Use a ring-of-all-ranks for simplicity. */
    dstm_distance_t *topology;
    int status = dstm_distance_create_ring(&topology, num_ranks);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Failed to create topology");
        MPI_Finalize();
        return -1;
    }

    status = dstm_start(topology);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Failed to start DSTM");
        return -1;
    }

    /* ----- Create the two regions on every rank. ----- */
    dstm_region_t *region_a = NULL, *region_b = NULL;
    if (dstm_region_create(&region_a, REGION_SIZE) != DSTM_SUCCESS) {
        dstm_log_error("region_a create failed"); return -1;
    }
    if (dstm_region_create(&region_b, REGION_SIZE) != DSTM_SUCCESS) {
        dstm_log_error("region_b create failed"); return -1;
    }

    /* Each rank picks the region for its ring. */
    dstm_region_t *my_region = (ring_id == 0) ? region_a : region_b;

    /* ----- Create the per-ring darray, one per region. -----
     *
     * Layout per ring: RING_SIZE * MAILBOX_SLOTS elements, same as
     * ring_mini, but living entirely inside this ring's region.  Ranks in
     * the OTHER ring still need to participate in the darray's create
     * (Allgather over MPI_COMM_WORLD) but they pass local_elements=0... NO,
     * that fails the local_elements==0 check.  Easiest workaround: every
     * rank allocates MAILBOX_SLOTS in its own region, but the "other"
     * ring's darray on this rank lives in a region we never read from.
     *
     * In other words, each rank holds BOTH darrays (one per ring), but
     * only ever touches the one for its own ring.  The other one is dead
     * weight kept alive purely to satisfy the collective create. */

    dstm_darray_t *mailbox_a = NULL, *mailbox_b = NULL;
    if (my_darray_create_in_region(&mailbox_a, region_a,
                                   sizeof(mailbox_slot_t), MAILBOX_SLOTS) != 0) {
        dstm_log_error("mailbox_a create failed"); return -1;
    }
    if (my_darray_create_in_region(&mailbox_b, region_b,
                                   sizeof(mailbox_slot_t), MAILBOX_SLOTS) != 0) {
        dstm_log_error("mailbox_b create failed"); return -1;
    }

    dstm_darray_t *mailbox = (ring_id == 0) ? mailbox_a : mailbox_b;

    if (rank == 0)
        dstm_log_normal("Two rings of %d, total %d ranks, %d packets/slot, "
                        "region size %u bytes each\n",
                        RING_SIZE, num_ranks, SLOT_CAPACITY, REGION_SIZE);

    srand(seed + rank);

    mailbox_slot_t outbox[MAILBOX_SLOTS];
    mailbox_slot_t recv_slot;
    uint64_t       next_packet_id = 0;

    for (dstm_time_t t = 0; t < timesteps; t++) {
        dstm_log_normal("Rank %d [ring %d]: time step %ld\n", rank, ring_id, t);

        /* STEP 1 — sentinel PUT into our own two slots. */
        for (int s = 0; s < MAILBOX_SLOTS; s++) {
            size_t idx = (size_t)ring_rank * MAILBOX_SLOTS + s;
            dstm_darray_put(mailbox, idx, (void *)&empty_slot, t, 0);
        }

        outbox[SLOT_LEFT]  = empty_slot;
        outbox[SLOT_RIGHT] = empty_slot;

        /* STEP 2 — RECEIVE both inbox slots. */
        for (int s = 0; s < MAILBOX_SLOTS; s++) {
            size_t idx = (size_t)ring_rank * MAILBOX_SLOTS + s;

            status = dstm_darray_get_or_default(mailbox, idx,
                                                &recv_slot, t, 0,
                                                &empty_slot);
            if (status != DSTM_SUCCESS) {
                dstm_log_error("Rank %d ring %d: GET failed slot %d t=%ld "
                               "(status=%d)", rank, ring_id, s, t, status);
                continue;
            }

            if (!quiet)
                dstm_log_normal("Rank %d [ring %d, t=%ld]: %s slot has %d "
                                "packet(s)\n", rank, ring_id, t,
                                s == SLOT_LEFT ? "left" : "right",
                                recv_slot.n_packets);

            /* STEP 3 — process. */
            for (int p = 0; p < recv_slot.n_packets; p++) {
                mailbox_entry_t pkt = recv_slot.packets[p];

                if (pkt.dest_rank == rank) {
                    dstm_log_normal("Rank %d [ring %d, t=%ld]: DELIVERED %s "
                        "— src=%d origin_t=%ld hops=%d payload=\"%s\"\n",
                        rank, ring_id, t, pkt_tag(&pkt),
                        pkt.source_rank, pkt.origin_time, pkt.hops_taken,
                        pkt.payload);
                    continue;
                }

                /* Same ring-arithmetic routing as the original. */
                int dist_right = (pkt.dest_rank - rank + num_ranks) % num_ranks;
                int go_right   = (dist_right <= num_ranks - dist_right);
                int fwd_dir    = go_right ? SLOT_RIGHT : SLOT_LEFT;

                pkt.hops_taken++;
                mailbox_slot_t *ob = &outbox[fwd_dir];
                if (ob->n_packets < SLOT_CAPACITY) {
                    ob->packets[ob->n_packets++] = pkt;
                    dstm_log_normal("Rank %d [ring %d, t=%ld]: FORWARD %s "
                                    "src=%d dst=%d via %s (hop %d)\n",
                                    rank, ring_id, t, pkt_tag(&pkt),
                                    pkt.source_rank, pkt.dest_rank,
                                    go_right ? "right" : "left",
                                    pkt.hops_taken);
                } else {
                    dstm_log_warning("Rank %d [ring %d, t=%ld]: DROP %s — "
                                     "outbox full\n",
                                     rank, ring_id, t, pkt_tag(&pkt));
                }
            }
        }

        /* STEP 4 — INJECT.  Destination MUST stay within this ring so the
         * two rings remain isolated. */
        if ((rand() % 100) < send_prob) {
            int local_dest = rand() % (RING_SIZE - 1);
            if (local_dest >= ring_rank) local_dest++;
            int dest = ring_base + local_dest;

            int dist_right = (dest - rank + num_ranks) % num_ranks;
            int go_right   = (dist_right <= num_ranks - dist_right);
            int fwd_dir    = go_right ? SLOT_RIGHT : SLOT_LEFT;

            mailbox_slot_t *ob = &outbox[fwd_dir];
            if (ob->n_packets < SLOT_CAPACITY) {
                mailbox_entry_t pkt = {
                    .valid       = 1,
                    .source_rank = rank,
                    .dest_rank   = dest,
                    .hops_taken  = 0,
                    .origin_time = t,
                    .id          = next_packet_id++,
                };
                snprintf(pkt.payload, sizeof(pkt.payload),
                         "ring=%d src=%d dst=%d t=%ld",
                         ring_id, rank, dest, t);
                ob->packets[ob->n_packets++] = pkt;
                dstm_log_normal("Rank %d [ring %d, t=%ld]: INJECT %s dst=%d "
                                "via %s\n",
                                rank, ring_id, t, pkt_tag(&pkt), dest,
                                go_right ? "right" : "left");
            }
        }

        /* STEP 5 — SEND outboxes to ring-local neighbors (delta=1). */
        if (outbox[SLOT_RIGHT].n_packets > 0) {
            size_t idx = (size_t)right_ring_rank * MAILBOX_SLOTS + SLOT_LEFT;
            status = dstm_darray_put(mailbox, idx,
                                     &outbox[SLOT_RIGHT], t, 1);
            if (status != DSTM_SUCCESS)
                dstm_log_error("PUT right→left failed (status=%d)", status);
        }
        if (outbox[SLOT_LEFT].n_packets > 0) {
            size_t idx = (size_t)left_ring_rank * MAILBOX_SLOTS + SLOT_RIGHT;
            status = dstm_darray_put(mailbox, idx,
                                     &outbox[SLOT_LEFT], t, 1);
            if (status != DSTM_SUCCESS)
                dstm_log_error("PUT left→right failed (status=%d)", status);
        }
    }

    dstm_time_t last_time = timesteps + 1;
    status = dstm_barrier(last_time);
    if (status != DSTM_SUCCESS)
        dstm_log_error("Barrier failed (status=%d)", status);

    my_darray_destroy_in_region(&mailbox_a, region_a);
    my_darray_destroy_in_region(&mailbox_b, region_b);

    status = dstm_fini(last_time);
    if (status != DSTM_SUCCESS)
        dstm_log_error("Failed to finalize DSTM");

    dstm_region_destroy(&region_a);
    dstm_region_destroy(&region_b);
    dstm_distance_destroy(&topology);
    dstm_finalize();
    MPI_Finalize();
    return status;
}
