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
#include "internal/region.h"

/*******************************************************************************
 * Internal layout (single append-based queue per (rank, timestamp))
 *
 * Each rank owns ONE region holding ONE slot.  Every (rank, timestamp)
 * is a distinct version of that slot, managed by the region version list.
 * The slot is:
 *
 *     [ uint64_t byte_count | element_size * capacity bytes ]
 *
 * Producers APPEND into the slot via dstm_append: all producers targeting
 * the same (owner, timestamp) accumulate into the same version buffer in
 * MPI arrival order.  The owning rank is the concatenation point, so there
 * are no per-neighbor blocks and no wholesale-overwrite collisions.
 *
 *   push(dest, elem)      -> stage into the local outbox for dest
 *   flush(time, delta)    -> one dstm_append per destination with staged
 *                            data (delta >= 0; remote requires delta > 0)
 *   pop_incoming(t, 0)    -> one local dstm_get of this rank's slot, then
 *                            read byte_count/element_size elements
 *
 * CAPACITY SIZING RULE.  Because every producer appends into the SAME
 * slot, `capacity` must be sized for the SUM over all producers of their
 * maximum per-tick contribution, NOT one producer's worth.  push() and
 * flush() bound-check against this and return -DSTM_ENOMEM on overflow.
 *
 * ORDERING.  Concatenation order is MPI arrival order and is therefore
 * nondeterministic.  Elements must be order-independent, or carry their
 * own key and be sorted by the consumer after pop.  This dqueue does NOT
 * sort.
 *
 * VERSION EXISTENCE.  A consumer's pop reads one version of the slot.  To
 * guarantee that version exists even when no producer sent anything this
 * tick, flush() always performs a local self-append of ZERO elements to
 * the owner's own slot, which creates (and keeps WRITABLE until the phase
 * rule fires) the (self, time+delta) version.  Combined with the
 * early-tick fallback (effective time < 0 -> empty), pop never blocks on a
 * missing version.
 ******************************************************************************/

/* The slot's count field counts BYTES (what dstm_append maintains). */
typedef struct {
        uint64_t byte_count;
        /* payload[element_size * capacity] follows */
} dqueue_slot_hdr_t;

struct dstm_dqueue_s {
        int    region_id;        /* this rank's local region id */
        dstm_ptr_t slot;         /* ptr to the slot base (count field) */

        size_t element_size;
        size_t capacity;         /* max elements in the aggregate slot */
        size_t slot_bytes;       /* sizeof(hdr) + element_size*capacity */

        int world_size;
        int self;

        /* Per-rank region id of every owner's queue region, so we can
         * address a remote owner's slot (region ids are a per-rank
         * namespace, unrelated to MPI rank). */
        int *rank_region_id;     /* [world_size] */

        /* Per-rank byte offset of the slot inside that rank's region. */
        intptr_t *rank_slot_off; /* [world_size] */

        /* --- Staging: one outbox per destination rank. ---
         * We keep a count of staged elements per destination.  Memory is
         * element_size * capacity per destination; for a dense world this
         * is world_size * capacity bytes of host staging, which is fine
         * for the inter-rank traffic regime dqueue targets. */
        char   *outbox;          /* world_size * (element_size*capacity) */
        size_t *outbox_count;    /* [world_size] staged elements per dest */
};

/* ---------------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------------- */

static inline char *outbox_for(dstm_dqueue_t *q, int dest)
{
        return q->outbox + (size_t)dest * q->element_size * q->capacity;
}

static inline dstm_ptr_t slot_ptr_of(const dstm_dqueue_t *q, int rank)
{
        return dstm_ptr_make(q->rank_region_id[rank], q->rank_slot_off[rank]);
}

/* ---------------------------------------------------------------------------
 * Teardown
 * ------------------------------------------------------------------------- */

static void dqueue_free_all(dstm_dqueue_t *q)
{
        free(q->rank_region_id);
        free(q->rank_slot_off);
        free(q->outbox);
        free(q->outbox_count);
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

        dstm_region_t *region = dstm_get_local_region(region_id);
        if (region == NULL)
                return -DSTM_EINVAL;

        dstm_dqueue_t *q = calloc(1, sizeof(*q));
        if (q == NULL)
                return -DSTM_ENOMEM;

        q->region_id    = region_id;
        q->element_size = element_size;
        q->capacity     = capacity;
        q->slot_bytes   = sizeof(dqueue_slot_hdr_t) +
                          element_size * capacity;
        q->world_size   = world_size;
        q->self         = self;

        q->rank_region_id = calloc((size_t)world_size, sizeof(int));
        q->rank_slot_off  = calloc((size_t)world_size, sizeof(intptr_t));
        q->outbox_count   = calloc((size_t)world_size, sizeof(size_t));
        q->outbox         = calloc((size_t)world_size,
                                   element_size * capacity);
        if (!q->rank_region_id || !q->rank_slot_off ||
            !q->outbox_count || !q->outbox) {
                dqueue_free_all(q);
                return -DSTM_ENOMEM;
        }

        /* Allocate this rank's single slot in its region. */
        dstm_ptr_t slot = dstm_region_malloc(region, q->slot_bytes);
        if (dstm_ptr_isnull(slot)) {
                dqueue_free_all(q);
                return -DSTM_ENOMEM;
        }
        q->slot = slot;

        /* Exchange every rank's region id and slot offset so remote
         * appends can address the right owner's slot.  slot.offset is an
         * intptr_t; exchange as int64_t for width independence. */
        {
                int rc = MPI_Allgather(&region_id, 1, MPI_INT,
                                       q->rank_region_id, 1, MPI_INT, comm);
                if (rc != MPI_SUCCESS) {
                        dstm_region_free(region, slot);
                        dqueue_free_all(q);
                        return -DSTM_FAILURE;
                }

                int64_t  mine = (int64_t)slot.offset;
                int64_t *offs = calloc((size_t)world_size, sizeof(int64_t));
                if (offs == NULL) {
                        dstm_region_free(region, slot);
                        dqueue_free_all(q);
                        return -DSTM_ENOMEM;
                }
                rc = MPI_Allgather(&mine, 1, MPI_INT64_T,
                                   offs, 1, MPI_INT64_T, comm);
                if (rc != MPI_SUCCESS) {
                        free(offs);
                        dstm_region_free(region, slot);
                        dqueue_free_all(q);
                        return -DSTM_FAILURE;
                }
                for (int r = 0; r < world_size; r++)
                        q->rank_slot_off[r] = (intptr_t)offs[r];
                free(offs);
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
        dstm_dqueue_t *q = *dqueue;
        dstm_region_t *region = dstm_get_local_region(q->region_id);
        if (region != NULL && !dstm_ptr_isnull(q->slot))
                dstm_region_free(region, q->slot);
        dqueue_free_all(q);
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

/* One queue, one slot: the whole queue holds at most `capacity`
 * elements, so max_incoming collapses to capacity. */
size_t dstm_dqueue_max_incoming(dstm_dqueue_t *q)
{
        return q ? q->capacity : 0;
}

size_t dstm_dqueue_pending(dstm_dqueue_t *q, int dest_rank)
{
        if (q == NULL || dest_rank < 0 || dest_rank >= q->world_size)
                return 0;
        return q->outbox_count[dest_rank];
}

/* ---------------------------------------------------------------------------
 * Push / flush  (== append)
 * ------------------------------------------------------------------------- */

int dstm_dqueue_push(dstm_dqueue_t *q, int dest_rank, const void *element)
{
        if (q == NULL || element == NULL)
                return -DSTM_EINVAL;
        if (dest_rank < 0 || dest_rank >= q->world_size)
                return -DSTM_EINVAL;

        /* Bound-check against the aggregate slot capacity.  This is a
         * conservative per-destination check: the receiver's slot holds
         * the SUM of all producers, so even if this one producer fits,
         * aggregate overflow is caught at apply time by the append
         * handler.  Here we simply prevent a single producer from
         * staging more than the slot could ever hold. */
        if (q->outbox_count[dest_rank] >= q->capacity)
                return -DSTM_ENOMEM;

        char *ob = outbox_for(q, dest_rank);
        memcpy(ob + q->outbox_count[dest_rank] * q->element_size,
               element, q->element_size);
        q->outbox_count[dest_rank]++;
        return DSTM_SUCCESS;
}

/* Append one destination's staged elements to that owner's slot.  The
 * owner's slot offset is the base (count field); dstm_append maintains
 * the byte count there and concatenates our payload bytes. */
static int dqueue_flush_dest(dstm_dqueue_t *q, int dest,
                             dstm_time_t time, dstm_time_t delta)
{
        size_t n = q->outbox_count[dest];
        if (n == 0)
                return DSTM_SUCCESS;   /* nothing staged; appends compose */

        dstm_ptr_t dst = slot_ptr_of(q, dest);
        int rc = dstm_append(dest, q->rank_region_id[dest], time, delta,
                             dst, outbox_for(q, dest),
                             n * q->element_size);
        if (rc != DSTM_SUCCESS)
                return rc;

        q->outbox_count[dest] = 0;
        return DSTM_SUCCESS;
}

int dstm_dqueue_flush(dstm_dqueue_t *q, dstm_time_t time, dstm_time_t delta)
{
        if (q == NULL)
                return -DSTM_EINVAL;

        /* Ensure the (self, time+delta) version exists even if no producer
         * (including us) sends anything, so a consumer's pop at this
         * version never blocks.  A zero-byte local append creates the
         * version and leaves it WRITABLE until the phase rule promotes it.
         *
         * delta applies only to remote destinations; the owner always
         * seeds its own slot at the SAME effective time the consumer will
         * read.  We seed at the local rank with delta so effective time
         * matches producers' (time + delta). */
        {
                dstm_ptr_t dst = slot_ptr_of(q, q->self);
                int rc = dstm_append(q->self, q->region_id, time, delta,
                                     dst, q->outbox /* unused: size 0 */, 0);
                if (rc != DSTM_SUCCESS)
                        return rc;
        }

        /* Append only where we actually have data; appends compose, so no
         * empty-block-to-everyone requirement remains. */
        for (int d = 0; d < q->world_size; d++) {
                if (d == q->self)
                        continue;   /* self handled below via local append */
                int rc = dqueue_flush_dest(q, d, time, delta);
                if (rc != DSTM_SUCCESS)
                        return rc;
        }
        /* Self-destined staged elements: local append at the same
         * effective time. */
        {
                int rc = dqueue_flush_dest(q, q->self, time, delta);
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
        /* With append semantics, flushing a single destination is safe
         * and composes with any later flush at the same version. */
        return dqueue_flush_dest(q, dest_rank, time, delta);
}

/* ---------------------------------------------------------------------------
 * Pop (== get) : single local get of the slot
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

        /* One local get of this rank's slot at (time + delta).  The owner
         * always seeds this version in flush(), so it exists. */
        dqueue_slot_hdr_t *hdr = calloc(1, q->slot_bytes);
        if (hdr == NULL)
                return -DSTM_ENOMEM;

        dstm_ptr_t src = slot_ptr_of(q, q->self);
        int rc = dstm_get(q->self, q->region_id, time, delta,
                          hdr, src, q->slot_bytes);
        if (rc != DSTM_SUCCESS) {
                free(hdr);
                return rc;
        }

        uint64_t bytes = hdr->byte_count;
        size_t   count = (size_t)(bytes / q->element_size);
        if (count > q->capacity)
                count = q->capacity;   /* corrupt-slot guard */

        if (out_elements == NULL || count > cap) {
                *io_count = count;
                free(hdr);
                return (out_elements == NULL && cap == 0)
                               ? -DSTM_ENOMEM   /* count-only query */
                               : (count > cap ? -DSTM_ENOMEM : DSTM_SUCCESS);
        }

        memcpy(out_elements,
               (char *)hdr + sizeof(dqueue_slot_hdr_t),
               count * q->element_size);
        *io_count = count;
        free(hdr);
        return DSTM_SUCCESS;
}
