/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "config.h"

#include <mpi.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "dstm.h"

#include "internal/globals.h"

/*******************************************************************************
 * Internal layout (sparse, neighbor-only)
 *
 * A dqueue gives each rank one logical incoming queue, assembled from one
 * physical block per incoming neighbor so that distinct remote producers
 * never write the same version slot (a put overwrites a slot wholesale, so
 * two producers sharing a slot would clobber each other). "Neighbor" is
 * taken from the distance matrix: rank s is a neighbor of self iff
 * dstm_distance_get(dist, self, s) != -1. Distances are symmetric, so the
 * incoming set and outgoing set are the same.
 *
 * Storage. Each rank owns exactly its incoming blocks (one per neighbor,
 * including self), packed into a darray partition in ascending neighbor-rank
 * order. The darray is built with dstm_darray_create (NOT _equal) so
 * partitions may be uneven: edge ranks in a torus have fewer neighbors than
 * interior ranks.
 *
 *   global darray layout:  [ rank 0's blocks | rank 1's blocks | ... ]
 *   within a rank:         blocks sorted by source neighbor rank ascending
 *
 * To push to a remote rank d, a producer must address d's block whose source
 * is self. That is  rank_block_base[d] + (position of self within d's
 * sorted incoming-neighbor list). Both the per-rank base offsets and each
 * rank's sorted neighbor list are exchanged once at create time with two
 * Allgathers, mirroring dgraph.
 *
 * Each block on the wire is a POD:
 *
 *   [ uint64_t count | element_size * capacity bytes of payload ]
 *
 * copied verbatim by put/get, so elements must not contain pointers.
 *
 * PUSH stages into a local outbox block per destination neighbor; FLUSH
 * publishes each non-empty outbox as one darray_put. POP_INCOMING issues one
 * darray_get per incoming neighbor and concatenates every block's live
 * elements into the caller's buffer -- the single-queue surface.
 ******************************************************************************/

typedef struct {
        uint64_t count;
        /* payload[element_size * capacity] follows */
} dqueue_block_hdr_t;

struct dstm_dqueue_s {
        dstm_darray_t *table;         /* sum over ranks of their neighbor count */
        int            region_id;

        size_t element_size;
        size_t capacity;
        size_t block_size;            /* sizeof(hdr) + element_size*capacity */

        int world_size;
        int self;

        /* --- Local neighbor set (sorted ascending, includes self) --- */
        int  *neighbors;              /* [n_neighbors] sorted rank ids */
        int   n_neighbors;

        /* Map any global rank -> local slot in neighbors[], or -1.
         * Length world_size. Lets push()/pop() go rank -> slot in O(1). */
        int  *rank_to_slot;

        /* --- Global addressing, from Allgather --- */
        /* rank_block_base[r] = global darray index of r's first block. */
        size_t *rank_block_base;      /* [world_size] */
        /* rank_nbr_count[r] = number of incoming neighbors of rank r. */
        int    *rank_nbr_count;       /* [world_size] */
        /* rank_nbrs[r] = r's sorted incoming-neighbor rank list,
         * length rank_nbr_count[r]. Needed to find self's position in a
         * remote rank's block layout. Non-neighbor entries may be NULL. */
        int   **rank_nbrs;            /* [world_size][rank_nbr_count[r]] */

        /* --- Staging --- */
        char *outbox;                 /* n_neighbors * block_size */
        char *scratch;                /* block_size, reused by pop */
};

/* ---------------------------------------------------------------------------
 * Layout helpers
 * ------------------------------------------------------------------------- */

static inline dqueue_block_hdr_t *block_hdr(char *b)
{
        return (dqueue_block_hdr_t *)b;
}

static inline char *block_payload(char *b)
{
        return b + sizeof(dqueue_block_hdr_t);
}

static inline char *outbox_block(dstm_dqueue_t *q, int slot)
{
        return q->outbox + (size_t)slot * q->block_size;
}

/* Binary search rank `r` in a sorted ascending int array of length n.
 * Returns index or -1. */
static int find_sorted(const int *arr, int n, int r)
{
        int lo = 0, hi = n - 1;
        while (lo <= hi) {
                int mid = lo + (hi - lo) / 2;
                if (arr[mid] == r)
                        return mid;
                if (arr[mid] < r)
                        lo = mid + 1;
                else
                        hi = mid - 1;
        }
        return -1;
}

/* Global darray index of the block (dest=d, src=s), where s must be an
 * incoming neighbor of d. Returns (size_t)-1 if s is not d's neighbor. */
static size_t block_index(const dstm_dqueue_t *q, int d, int s)
{
        int pos = find_sorted(q->rank_nbrs[d], q->rank_nbr_count[d], s);
        if (pos < 0)
                return (size_t)-1;
        return q->rank_block_base[d] + (size_t)pos;
}

/* ---------------------------------------------------------------------------
 * Teardown (safe on partially built q)
 * ------------------------------------------------------------------------- */

static void dqueue_free_all(dstm_dqueue_t *q)
{
        if (q->rank_nbrs) {
                for (int r = 0; r < q->world_size; r++)
                        free(q->rank_nbrs[r]);
                free(q->rank_nbrs);
        }
        free(q->rank_block_base);
        free(q->rank_nbr_count);
        free(q->neighbors);
        free(q->rank_to_slot);
        free(q->outbox);
        free(q->scratch);
        if (q->table)
                dstm_darray_destroy(&q->table);
        free(q);
}

/* ---------------------------------------------------------------------------
 * Creation
 * ------------------------------------------------------------------------- */

int dstm_dqueue_create_on_region(dstm_dqueue_t **ret,
                                 size_t element_size,
                                 size_t capacity,
                                 dstm_distance_t *dist,
                                 int region_id)
{
        if (ret == NULL || element_size == 0 || capacity == 0 || dist == NULL)
                return -DSTM_EINVAL;
        *ret = NULL;

        int world_size = dstm_global_context.world_size;
        int self       = dstm_global_context.rank;
        MPI_Comm comm  = dstm_global_context.comm;

        if (dstm_distance_size(dist) != (size_t)world_size)
                return -DSTM_EINVAL;

        dstm_dqueue_t *q = calloc(1, sizeof(*q));
        if (q == NULL)
                return -DSTM_ENOMEM;

        q->region_id    = region_id;
        q->element_size = element_size;
        q->capacity     = capacity;
        q->block_size   = sizeof(dqueue_block_hdr_t) + element_size * capacity;
        q->world_size   = world_size;
        q->self         = self;

        q->rank_to_slot   = malloc((size_t)world_size * sizeof(int));
        q->rank_block_base = calloc((size_t)world_size, sizeof(size_t));
        q->rank_nbr_count  = calloc((size_t)world_size, sizeof(int));
        q->rank_nbrs       = calloc((size_t)world_size, sizeof(int *));
        if (!q->rank_to_slot || !q->rank_block_base ||
            !q->rank_nbr_count || !q->rank_nbrs) {
                dqueue_free_all(q);
                return -DSTM_ENOMEM;
        }
        for (int r = 0; r < world_size; r++)
                q->rank_to_slot[r] = -1;

        /* --- Local incoming-neighbor set from the distance matrix. ---
         * s is a neighbor of self iff reachable; self always included. */
        {
                int nc = 0;
                for (int s = 0; s < world_size; s++) {
                        if (s == self ||
                            dstm_distance_get(dist, self, s) != -1)
                                nc++;
                }
                q->neighbors   = malloc((size_t)nc * sizeof(int));
                if (q->neighbors == NULL) {
                        dqueue_free_all(q);
                        return -DSTM_ENOMEM;
                }
                int j = 0;
                for (int s = 0; s < world_size; s++) {
                        if (s == self ||
                            dstm_distance_get(dist, self, s) != -1)
                                q->neighbors[j++] = s;   /* ascending */
                }
                q->n_neighbors = nc;
                for (int i = 0; i < nc; i++)
                        q->rank_to_slot[q->neighbors[i]] = i;
        }

        /* --- Exchange each rank's neighbor count, build base offsets. --- */
        {
                int mine = q->n_neighbors;
                if (MPI_Allgather(&mine, 1, MPI_INT,
                                  q->rank_nbr_count, 1, MPI_INT, comm)
                    != MPI_SUCCESS) {
                        dqueue_free_all(q);
                        return -DSTM_FAILURE;
                }
                size_t acc = 0;
                for (int r = 0; r < world_size; r++) {
                        q->rank_block_base[r] = acc;
                        acc += (size_t)q->rank_nbr_count[r];
                }
                /* acc == total blocks across all ranks. */
        }

        /* --- Exchange each rank's sorted neighbor list (variable length). ---
         * Use Allgatherv: displacements are the block bases, counts the
         * per-rank neighbor counts. */
        {
                int *displs = malloc((size_t)world_size * sizeof(int));
                int  total  = (int)(q->rank_block_base[world_size - 1] +
                                    q->rank_nbr_count[world_size - 1]);
                int *flat   = malloc((size_t)(total > 0 ? total : 1) *
                                     sizeof(int));
                if (displs == NULL || flat == NULL) {
                        free(displs);
                        free(flat);
                        dqueue_free_all(q);
                        return -DSTM_ENOMEM;
                }
                for (int r = 0; r < world_size; r++)
                        displs[r] = (int)q->rank_block_base[r];

                if (MPI_Allgatherv(q->neighbors, q->n_neighbors, MPI_INT,
                                   flat, q->rank_nbr_count, displs, MPI_INT,
                                   comm) != MPI_SUCCESS) {
                        free(displs);
                        free(flat);
                        dqueue_free_all(q);
                        return -DSTM_FAILURE;
                }

                for (int r = 0; r < world_size; r++) {
                        int c = q->rank_nbr_count[r];
                        if (c == 0)
                                continue;
                        q->rank_nbrs[r] = malloc((size_t)c * sizeof(int));
                        if (q->rank_nbrs[r] == NULL) {
                                free(displs);
                                free(flat);
                                dqueue_free_all(q);
                                return -DSTM_ENOMEM;
                        }
                        memcpy(q->rank_nbrs[r], flat + displs[r],
                               (size_t)c * sizeof(int));
                }
                free(displs);
                free(flat);
        }

        /* --- Staging buffers. --- */
        q->outbox  = calloc((size_t)q->n_neighbors, q->block_size);
        q->scratch = calloc(1, q->block_size);
        if (q->outbox == NULL || q->scratch == NULL) {
                dqueue_free_all(q);
                return -DSTM_ENOMEM;
        }

        /* --- Backing darray: this rank owns exactly its own blocks. ---
         * Uneven partition, so create (not create_equal). Collective. */
        {
                int rc = dstm_darray_create_on_region(
                        &q->table, q->block_size,
                        (size_t)q->n_neighbors, region_id);
                if (rc != DSTM_SUCCESS) {
                        dqueue_free_all(q);
                        return rc;
                }
        }

        *ret = q;
        return DSTM_SUCCESS;
}

int dstm_dqueue_create(dstm_dqueue_t **ret,
                       size_t element_size,
                       size_t capacity,
                       dstm_distance_t *dist)
{
        return dstm_dqueue_create_on_region(ret, element_size, capacity,
                                            dist, dstm_get_local_region_id());
}

void dstm_dqueue_destroy(dstm_dqueue_t **dqueue)
{
        if (dqueue == NULL || *dqueue == NULL)
                return;
        dqueue_free_all(*dqueue);
        *dqueue = NULL;
}

/* ---------------------------------------------------------------------------
 * Introspection
 * ------------------------------------------------------------------------- */

size_t dstm_dqueue_element_size(dstm_dqueue_t *q)
{
        return q ? q->element_size : 0;
}

size_t dstm_dqueue_capacity(dstm_dqueue_t *q)
{
        return q ? q->capacity : 0;
}

size_t dstm_dqueue_max_incoming(dstm_dqueue_t *q)
{
        return q ? (size_t)q->n_neighbors * q->capacity : 0;
}

size_t dstm_dqueue_pending(dstm_dqueue_t *q, int dest_rank)
{
        if (q == NULL || dest_rank < 0 || dest_rank >= q->world_size)
                return 0;
        int slot = q->rank_to_slot[dest_rank];
        if (slot < 0)
                return 0;
        return (size_t)block_hdr(outbox_block(q, slot))->count;
}

/* ---------------------------------------------------------------------------
 * Push / flush  (== put)
 * ------------------------------------------------------------------------- */

int dstm_dqueue_push(dstm_dqueue_t *q, int dest_rank, const void *element)
{
        if (q == NULL || element == NULL)
                return -DSTM_EINVAL;
        if (dest_rank < 0 || dest_rank >= q->world_size)
                return -DSTM_EINVAL;

        int slot = q->rank_to_slot[dest_rank];
        if (slot < 0)
                return -DSTM_EINVAL;    /* not a neighbor: unreachable dest */

        char               *blk = outbox_block(q, slot);
        dqueue_block_hdr_t *hdr = block_hdr(blk);

        if (hdr->count >= q->capacity)
                return -DSTM_ENOMEM;    /* full; flush and retry later */

        memcpy(block_payload(blk) + (size_t)hdr->count * q->element_size,
               element, q->element_size);
        hdr->count++;
        return DSTM_SUCCESS;
}

static int dqueue_flush_slot(dstm_dqueue_t *q, int slot,
                             dstm_time_t time, dstm_time_t delta)
{
        int    dest = q->neighbors[slot];
        char  *blk  = outbox_block(q, slot);
        size_t idx  = block_index(q, dest, q->self);

        if (idx == (size_t)-1)
                return -DSTM_EINVAL;   /* self not in dest's neighbor list */

        int rc = dstm_darray_put(q->table, idx, blk, time, delta);
        if (rc != DSTM_SUCCESS)
                return rc;

        block_hdr(blk)->count = 0;     /* clear staged block */
        return DSTM_SUCCESS;
}

int dstm_dqueue_flush(dstm_dqueue_t *q, dstm_time_t time, dstm_time_t delta)
{
        if (q == NULL)
                return -DSTM_EINVAL;

        /* Publish EVERY outgoing neighbor block, empty or not.
         *
         * A consumer's pop_incoming issues one get per incoming neighbor
         * at a given version. If we skipped empty outboxes, a neighbor we
         * had nothing to send would issue no put, that version would never
         * exist, and the consumer's get would block forever. */
        for (int slot = 0; slot < q->n_neighbors; slot++) {
                int rc = dqueue_flush_slot(q, slot, time, delta);
                if (rc != DSTM_SUCCESS)
                        return rc;
        }
        return DSTM_SUCCESS;
}

int dstm_dqueue_push_flush(dstm_dqueue_t *q, int dest_rank,
                           const void *element,
                           dstm_time_t time, dstm_time_t delta)
{
        int rc = dstm_dqueue_push(q, dest_rank, element);
        if (rc != DSTM_SUCCESS)
                return rc;
        return dqueue_flush_slot(q, q->rank_to_slot[dest_rank], time, delta);
}

/* ---------------------------------------------------------------------------
 * Pop (== get) : single-queue surface, merges every incoming block
 * ------------------------------------------------------------------------- */

int dstm_dqueue_pop_incoming(dstm_dqueue_t *q, void *out_elements,
                             size_t *io_count,
                             dstm_time_t time, dstm_time_t delta)
{
        if (q == NULL || io_count == NULL)
                return -DSTM_EINVAL;

        /* Early-tick fallback: version predates history -> empty. */
        if (time + delta < 0) {
                *io_count = 0;
                return DSTM_SUCCESS;
        }

        size_t cap = *io_count;    /* caller buffer capacity, in elements */
        size_t written = 0;

        for (int slot = 0; slot < q->n_neighbors; slot++) {
                int    src = q->neighbors[slot];
                size_t idx = block_index(q, q->self, src);
                if (idx == (size_t)-1)
                        continue;   /* defensive; slot is self's neighbor */

                int rc = dstm_darray_get(q->table, idx, q->scratch,
                                         time, delta);
                if (rc != DSTM_SUCCESS)
                        return rc;

                uint64_t count = block_hdr(q->scratch)->count;
                if (count > q->capacity)
                        count = q->capacity;   /* corrupt-block guard */
                if (count == 0)
                        continue;

                /* Capacity check: if the caller's buffer is too small,
                 * report the exact total needed via the two-call pattern.
                 * We finish counting the rest so io_count is accurate. */
                if (out_elements == NULL || written + count > cap) {
                        /* switch to count-only for the remainder */
                        written += (size_t)count;
                        for (int s2 = slot + 1; s2 < q->n_neighbors; s2++) {
                                size_t i2 = block_index(q, q->self,
                                                        q->neighbors[s2]);
                                if (i2 == (size_t)-1)
                                        continue;
                                int rc2 = dstm_darray_get(q->table, i2,
                                                          q->scratch,
                                                          time, delta);
                                if (rc2 != DSTM_SUCCESS)
                                        return rc2;
                                uint64_t c2 = block_hdr(q->scratch)->count;
                                if (c2 > q->capacity)
                                        c2 = q->capacity;
                                written += (size_t)c2;
                        }
                        *io_count = written;
                        return -DSTM_ENOMEM;
                }

                memcpy((char *)out_elements + written * q->element_size,
                       block_payload(q->scratch),
                       (size_t)count * q->element_size);
                written += (size_t)count;
        }

        *io_count = written;
        return DSTM_SUCCESS;
}
