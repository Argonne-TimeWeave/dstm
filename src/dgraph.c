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
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "dstm.h"

#include "internal/globals.h"
#include "internal/region.h"
#include "internal/msg.h"

/*******************************************************************************
 * Internal layout (static topology)
 *
 * One dstm_region_t partitioned into three fixed-size slabs at create time:
 *
 *   [ vertex_state_slab | edge_row_ptr_slab | edge_slot_slab ]
 *
 * vertex_state_slab
 *   local_vertices entries of vertex_state_size bytes, indexed by
 *   local id (global_id - local_vertex_offset). Empty if size == 0.
 *   Versioned; mutated by dstm_dgraph_vertex_put.
 *
 * edge_row_ptr_slab
 *   (local_vertices + 1) uint64_t CSR row pointers. Entry i is the
 *   first outgoing edge slot for local vertex i; entry local_vertices
 *   is a sentinel == local_edge_count. Outgoing edges of vertex i
 *   occupy slots [row_ptr[i], row_ptr[i+1]).
 *
 *   Written once at time 0 and never mutated (static topology). Kept
 *   inside the region so a future implementation can serve remote
 *   topology reads through the ordinary dstm_get path; today all
 *   topology queries are answered from the replicated host cache.
 *
 * edge_slot_slab
 *   local_edge_count packed entries of edge_slot_stride bytes:
 *   (dst_vertex_id, user_state[edge_state_size]). Within each
 *   vertex's window, slots are sorted by ascending dst, enabling
 *   binary search in dstm_dgraph_edge_{get,put}. dst is immutable
 *   after construction; only user_state is mutated by edge_put.
 *
 * Slab base offsets are cached in vertex_state_off / edge_row_ptr_off
 * / edge_slot_off so dstm_ptr_t values can be built by addition.
 *
 * REGION IDS VS RANKS
 *   A dstm_ptr_t names a (region, offset) pair, and dstm_get/dstm_put
 *   take the owning rank and the region id as separate arguments.
 *   These are distinct namespaces: nothing guarantees that rank r's
 *   local region id equals r. We therefore Allgather every rank's
 *   region id at create time and resolve it through
 *   dgraph_region_of(). See the note on that function if your build's
 *   dstm_get/dstm_put expect a rank in that position instead.
 ******************************************************************************/

/* Fixed-width type for CSR row pointers and destination ids in the
 * region, so on-region layout is independent of host size_t width. */
typedef uint64_t dgraph_vid_t;

/* Per-rank layout descriptor, exchanged with a single Allgather.
 * Fixed-width fields only: this struct travels as raw bytes, so it
 * must not depend on host size_t / intptr_t widths. */
typedef struct {
        uint64_t local_vertices;
        uint64_t local_edge_count;
        uint64_t edge_slot_stride;
        int64_t  vertex_state_off;
        int64_t  edge_row_ptr_off;
        int64_t  edge_slot_off;
        int32_t  region_id;
        int32_t  pad;           /* explicit, keeps sizeof a multiple of 8 */
} dgraph_rank_info_t;

struct dstm_dgraph_s {
        /* Backing storage */
        dstm_region_t *region;
        int            region_id;

        /* Element sizes */
        size_t vertex_state_size;
        size_t edge_state_size;
        size_t edge_slot_stride;

        /* Global partitioning */
        int      world_size;
        int      local_rank;
        size_t   total_vertices;
        size_t   local_vertices;
        size_t   local_vertex_offset;
        size_t  *rank_vertex_offsets;   /* [world_size + 1] prefix sum */

        /* Edge counts */
        size_t total_edges;
        size_t local_edge_count;

        /* Host-cached CSR row pointers, length local_vertices + 1. */
        dgraph_vid_t *row_ptrs;

        /* Slab byte offsets within the local region. */
        intptr_t vertex_state_off;
        intptr_t edge_row_ptr_off;
        intptr_t edge_slot_off;

        /* Slab byte sizes. */
        size_t vertex_state_bytes;
        size_t edge_row_ptr_bytes;
        size_t edge_slot_bytes;
        size_t region_bytes;

        /* Per-rank layout, length world_size. Needed so we can build
         * dstm_ptr_t values into remote ranks' slabs. */
        size_t   *rank_local_vertices;
        size_t   *rank_local_edge_count;
        intptr_t *rank_vertex_state_off;
        intptr_t *rank_edge_row_ptr_off;
        intptr_t *rank_edge_slot_off;
        size_t   *rank_edge_slot_stride;
        int      *rank_region_id;

        /* Sparse replicated topology cache: entries are non-NULL only
         * for ranks in the local neighborhood (self and any rank that
         * shares an edge with self, in either direction). Non-neighbor
         * entries are NULL. Indexed by global rank id. */
        dgraph_vid_t **rank_edge_dsts;
        dgraph_vid_t **rank_row_ptrs;

        /* Neighbor-rank membership. is_neighbor_rank[r] == 1 iff rank r
         * is self or shares at least one edge with self. Always includes
         * self. Length world_size. */
        uint8_t *is_neighbor_rank;

        /* Convenience: sorted list of neighbor ranks (including self)
         * and its length, for cheap iteration. */
        int *neighbor_ranks;
        int  neighbor_rank_count;
};

/* ---------------------------------------------------------------------------
 * Layout helpers
 * ------------------------------------------------------------------------- */

/* Alignment used for slab boundaries and per-slot padding. 8 bytes is
 * sufficient for dgraph_vid_t (uint64_t) and any typical user state. */
#define DGRAPH_ALIGN ((size_t)8)

/* Align x up to the next multiple of a (a must be a power of 2). */
static inline size_t align_up(size_t x, size_t a)
{
        return (x + (a - 1)) & ~(a - 1);
}

/* Compute the padded stride of a single edge slot:
 *   [ dgraph_vid_t dst | user_state[edge_state_size] | pad ]
 * padded up to DGRAPH_ALIGN so the next slot's dst stays aligned. */
static inline size_t compute_edge_slot_stride(size_t edge_state_size)
{
        size_t raw = sizeof(dgraph_vid_t) + edge_state_size;
        return align_up(raw, DGRAPH_ALIGN);
}

/* Returns the region identifier to use when addressing memory owned by
 * `rank`, both for the dstm_ptr_t .region field and for the region
 * argument of dstm_get / dstm_put.
 */
static inline int dgraph_region_of(const dstm_dgraph_t *g, int rank)
{
        return g->rank_region_id[rank];
}

/* Free every heap array hanging off g, plus g itself. Safe on partially
 * constructed g: all outer arrays are calloc'd, inner entries NULL. */
static void dgraph_free_all(dstm_dgraph_t *g)
{
        int ws = g->world_size;   /* set early in create, valid in both paths */

        if (g->rank_row_ptrs) {
                for (int r = 0; r < ws; r++)
                        free(g->rank_row_ptrs[r]);
                free(g->rank_row_ptrs);
        }
        if (g->rank_edge_dsts) {
                for (int r = 0; r < ws; r++)
                        free(g->rank_edge_dsts[r]);
                free(g->rank_edge_dsts);
        }
        free(g->row_ptrs);
        free(g->rank_local_vertices);
        free(g->rank_local_edge_count);
        free(g->rank_vertex_state_off);
        free(g->rank_edge_row_ptr_off);
        free(g->rank_edge_slot_off);
        free(g->rank_edge_slot_stride);
        free(g->rank_region_id);
        free(g->rank_vertex_offsets);
        free(g->is_neighbor_rank);
        free(g->neighbor_ranks);
        free(g);
}

/* ---------------------------------------------------------------------------
 * Initial-state seeding
 *
 * Reads and writes go through per-version buffers (see
 * dstm_region_version_s), each a full-size copy of the region, created
 * *uninitialized*. Initial graph state must therefore be stamped into
 * version 0's buffer explicitly; writing region->buffer alone would be
 * invisible to every dstm_get.
 *
 * We do this with the internal region API rather than dstm_put because
 * this is pre-simulation state: it predates the timeline instead of
 * being an event on it, so routing it through the put/get visibility
 * contract only created an effective time we then had to reason about.
 * ------------------------------------------------------------------------- */

/* Write all three slabs into one region-sized buffer. `region_buffer` is
 * the base of the buffer; slab offsets are region-relative and index it
 * directly. */
static void dgraph_seed_slabs(const dstm_dgraph_t *g, char *region_buffer,
                              const size_t *local_edge_dsts,
                              const void *local_vertex_states,
                              const void *local_edge_states)
{
        char *vs = region_buffer + g->vertex_state_off;
        char *rp = region_buffer + g->edge_row_ptr_off;
        char *es = region_buffer + g->edge_slot_off;

        /* Version buffers are malloc'd, not calloc'd. Clear our whole
         * allocation first: this covers inter-slab padding, intra-slot
         * padding, and the NULL-means-zero-init contract for both state
         * arrays. Only our own bytes are touched. */
        memset(region_buffer + g->vertex_state_off, 0, g->region_bytes);

        /* CSR row pointers. Immutable for the graph's lifetime; kept
         * in-region so a future revision can serve remote topology reads
         * without the host cache. */
        memcpy(rp, g->row_ptrs, g->edge_row_ptr_bytes);

        /* Vertex state (already zeroed if the caller passed NULL). */
        if (g->vertex_state_bytes > 0 && local_vertex_states != NULL)
                memcpy(vs, local_vertex_states, g->vertex_state_bytes);

        /* Edge slots: [dst | user_state | pad] per slot, built in place. */
        for (size_t slot = 0; slot < g->local_edge_count; slot++) {
                char        *entry = es + slot * g->edge_slot_stride;
                dgraph_vid_t d     = (dgraph_vid_t)local_edge_dsts[slot];

                memcpy(entry, &d, sizeof(dgraph_vid_t));

                if (g->edge_state_size > 0 && local_edge_states != NULL)
                        memcpy(entry + sizeof(dgraph_vid_t),
                               (const char *)local_edge_states
                               + slot * g->edge_state_size,
                               g->edge_state_size);
        }
}

/* Stamp initial state into version 0 and into the region's own buffer.
 * Version 0 is what dstm_get at effective time 0 will read; region->buffer
 * is seeded too so that any code path deriving a fresh version from it
 * inherits the same bytes. */
static int dgraph_seed_initial_state(dstm_dgraph_t *g,
                                     const size_t *local_edge_dsts,
                                     const void *local_vertex_states,
                                     const void *local_edge_states)
{
        dstm_region_version_t *v0 =
                dstm_region_version_find_or_create(g->region, 0);

        if (v0 == NULL || v0->buffer == NULL)
                return -DSTM_ENOMEM;

        dgraph_seed_slabs(g, (char *)v0->buffer, local_edge_dsts,
                          local_vertex_states, local_edge_states);

        if (g->region->buffer != NULL)
                dgraph_seed_slabs(g, (char *)g->region->buffer,
                                  local_edge_dsts, local_vertex_states,
                                  local_edge_states);

        return DSTM_SUCCESS;
}

/* ---------------------------------------------------------------------------
 * Ownership resolution
 * ------------------------------------------------------------------------- */

/* Binary search over a [world_size + 1] prefix-sum array: return rank r
 * such that offsets[r] <= vertex_id < offsets[r+1]. */
static int find_rank_in_offsets(const size_t *offsets, int world_size,
                                size_t vertex_id)
{
        int lo = 0;
        int hi = world_size - 1;

        while (lo < hi) {
                int mid = lo + ((hi - lo + 1) / 2);
                if (offsets[mid] <= vertex_id)
                        lo = mid;
                else
                        hi = mid - 1;
        }
        return lo;
}

static inline int dgraph_find_rank(const dstm_dgraph_t *g, size_t vertex_id)
{
        return find_rank_in_offsets(g->rank_vertex_offsets, g->world_size,
                                    vertex_id);
}

int dstm_dgraph_vertex_owner(dstm_dgraph_t *g, size_t vertex_id)
{
        if (g == NULL || vertex_id >= g->total_vertices)
                return -DSTM_EINVAL;
        return dgraph_find_rank(g, vertex_id);
}

/* ---------------------------------------------------------------------------
 * Introspection
 * ------------------------------------------------------------------------- */

size_t dstm_dgraph_vertex_count(dstm_dgraph_t *g)
{
        return g ? g->total_vertices : 0;
}

size_t dstm_dgraph_local_vertex_count(dstm_dgraph_t *g)
{
        return g ? g->local_vertices : 0;
}

size_t dstm_dgraph_local_vertex_offset(dstm_dgraph_t *g)
{
        return g ? g->local_vertex_offset : 0;
}

size_t dstm_dgraph_edge_count(dstm_dgraph_t *g)
{
        return g ? g->total_edges : 0;
}

size_t dstm_dgraph_local_edge_count(dstm_dgraph_t *g)
{
        return g ? g->local_edge_count : 0;
}

int dstm_dgraph_is_neighbor_rank(dstm_dgraph_t *g, int rank)
{
        if (g == NULL || rank < 0 || rank >= g->world_size)
                return -DSTM_EINVAL;
        return g->is_neighbor_rank[rank] ? 1 : 0;
}

int dstm_dgraph_neighbor_ranks_get(dstm_dgraph_t *g, int *out_ranks,
                                   size_t *io_count)
{
        if (g == NULL || io_count == NULL)
                return -DSTM_EINVAL;

        size_t need = (size_t)g->neighbor_rank_count;

        if (out_ranks == NULL || *io_count < need) {
                *io_count = need;
                return -DSTM_ENOMEM;
        }
        for (int i = 0; i < g->neighbor_rank_count; i++)
                out_ranks[i] = g->neighbor_ranks[i];
        *io_count = need;
        return DSTM_SUCCESS;
}

/* ---------------------------------------------------------------------------
 * Creation
 * ------------------------------------------------------------------------- */

/* Validate per-vertex neighbor slice: strictly ascending, all in
 * [0, total_vertices). Returns 0 on success, -DSTM_EINVAL on violation. */
static int validate_neighbors(const size_t *dsts, size_t deg,
                              size_t total_vertices)
{
        for (size_t i = 0; i < deg; i++) {
                if (dsts[i] >= total_vertices)
                        return -DSTM_EINVAL;
                if (i > 0 && dsts[i] <= dsts[i - 1])
                        return -DSTM_EINVAL;
        }
        return 0;
}

int dstm_dgraph_create(dstm_dgraph_t **ret,
                       size_t vertex_state_size,
                       size_t edge_state_size,
                       size_t local_vertices,
                       const size_t *local_out_degrees,
                       const size_t *local_edge_dsts,
                       const void *local_vertex_states,
                       const void *local_edge_states)
{
        if (ret == NULL)
                return -DSTM_EINVAL;
        *ret = NULL;
        if (local_vertices == 0 || local_out_degrees == NULL)
                return -DSTM_EINVAL;

        int err        = 0;
        int world_size = dstm_global_context.world_size;
        int self       = dstm_global_context.rank;
        MPI_Comm comm  = dstm_global_context.comm;

        /* Tracks whether the backing region has been successfully
         * allocated. Every other resource is either NULL (never
         * allocated) or a heap pointer safe to free unconditionally. */
        bool region_allocated = false;

        dgraph_rank_info_t  self_info;
        dgraph_rank_info_t *all_info = NULL;

        dstm_dgraph_t *g = calloc(1, sizeof(*g));
        if (g == NULL)
                return -DSTM_ENOMEM;

        g->region_id         = dstm_get_local_region_id();
        g->region            = dstm_get_local_region(g->region_id);
        g->world_size        = world_size;
        g->local_rank        = self;
        g->vertex_state_size = vertex_state_size;
        g->edge_state_size   = edge_state_size;
        g->edge_slot_stride  = compute_edge_slot_stride(edge_state_size);
        g->local_vertices    = local_vertices;

        if (g->region == NULL) {
                free(g);
                return -DSTM_EINVAL;
        }

        /* --- Compute local_edge_count ------------------------------------ */
        {
                size_t local_edge_count = 0;
                for (size_t i = 0; i < local_vertices; i++)
                        local_edge_count += local_out_degrees[i];
                g->local_edge_count = local_edge_count;
        }

        if (g->local_edge_count > 0 && local_edge_dsts == NULL) {
                err = -DSTM_EINVAL;
                goto consensus_setup;
        }

        /* --- Per-rank arrays --------------------------------------------- */
        g->rank_local_vertices   = calloc((size_t)world_size, sizeof(size_t));
        g->rank_local_edge_count = calloc((size_t)world_size, sizeof(size_t));
        g->rank_vertex_state_off = calloc((size_t)world_size, sizeof(intptr_t));
        g->rank_edge_row_ptr_off = calloc((size_t)world_size, sizeof(intptr_t));
        g->rank_edge_slot_off    = calloc((size_t)world_size, sizeof(intptr_t));
        g->rank_edge_slot_stride = calloc((size_t)world_size, sizeof(size_t));
        g->rank_region_id        = calloc((size_t)world_size, sizeof(int));
        g->rank_vertex_offsets   = calloc((size_t)world_size + 1,
                                          sizeof(size_t));
        g->is_neighbor_rank      = calloc((size_t)world_size, sizeof(uint8_t));
        g->rank_row_ptrs         = calloc((size_t)world_size,
                                          sizeof(dgraph_vid_t *));
        g->rank_edge_dsts        = calloc((size_t)world_size,
                                          sizeof(dgraph_vid_t *));
        g->row_ptrs              = calloc(local_vertices + 1,
                                          sizeof(dgraph_vid_t));

        if (!g->rank_local_vertices || !g->rank_local_edge_count ||
            !g->rank_vertex_state_off || !g->rank_edge_row_ptr_off ||
            !g->rank_edge_slot_off   || !g->rank_edge_slot_stride  ||
            !g->rank_region_id       || !g->rank_vertex_offsets    ||
            !g->is_neighbor_rank     || !g->rank_row_ptrs          ||
            !g->rank_edge_dsts       || !g->row_ptrs) {
                err = -DSTM_ENOMEM;
                goto consensus_setup;
        }

        /* --- Host-side cached row_ptrs ----------------------------------- */
        {
                dgraph_vid_t acc = 0;
                for (size_t i = 0; i < local_vertices; i++) {
                        g->row_ptrs[i] = acc;
                        acc += (dgraph_vid_t)local_out_degrees[i];
                }
                g->row_ptrs[local_vertices] = acc;
        }

        /* --- Compute slab sizes on this rank ----------------------------- */
        g->vertex_state_bytes = local_vertices * vertex_state_size;
        g->edge_row_ptr_bytes = (local_vertices + 1) * sizeof(dgraph_vid_t);
        g->edge_slot_bytes    = g->local_edge_count * g->edge_slot_stride;

        /* Pad each slab up so the next one starts aligned. */
        {
                size_t vs_padded = align_up(g->vertex_state_bytes,
                                            DGRAPH_ALIGN);
                size_t rp_padded = align_up(g->edge_row_ptr_bytes,
                                            DGRAPH_ALIGN);
                size_t es_padded = align_up(g->edge_slot_bytes, DGRAPH_ALIGN);

                g->region_bytes = vs_padded + rp_padded + es_padded;
                if (g->region_bytes == 0)
                        g->region_bytes = DGRAPH_ALIGN;

                /* --- Single region allocation ----------------------------
                 *
                 * region_bytes is kept in sync with what was actually
                 * requested: dgraph_seed_slabs memsets exactly this many
                 * bytes, so the two must not diverge. */
                dstm_ptr_t base = dstm_region_malloc(g->region,
                                                     g->region_bytes);
                if (dstm_ptr_isnull(base)) {
                        err = -DSTM_ENOMEM;
                        goto consensus_setup;
                }
                region_allocated = true;

                g->vertex_state_off = base.offset;
                g->edge_row_ptr_off = base.offset + (intptr_t)vs_padded;
                g->edge_slot_off    = g->edge_row_ptr_off + (intptr_t)rp_padded;
        }

consensus_setup:
        /* Everything above is rank-local. Agree on success before the
         * first collective, so a lone failure can't deadlock the job. */
        err = dstm_msg_consensus(comm, err);
        if (err != 0)
                goto err_cleanup;

        /* --- Single layout exchange --------------------------------------
         *
         * Fixed-width fields only, so we never have to claim that size_t
         * is MPI_UNSIGNED_LONG_LONG or that intptr_t is MPI_AINT. */
        memset(&self_info, 0, sizeof(self_info));
        self_info.local_vertices   = (uint64_t)g->local_vertices;
        self_info.local_edge_count = (uint64_t)g->local_edge_count;
        self_info.edge_slot_stride = (uint64_t)g->edge_slot_stride;
        self_info.vertex_state_off = (int64_t)g->vertex_state_off;
        self_info.edge_row_ptr_off = (int64_t)g->edge_row_ptr_off;
        self_info.edge_slot_off    = (int64_t)g->edge_slot_off;
        self_info.region_id        = (int32_t)g->region_id;

        all_info = calloc((size_t)world_size, sizeof(dgraph_rank_info_t));
        if (all_info == NULL) {
                /* Collective already entered by peers; report and let the
                 * post-exchange consensus tear everyone down. */
                err = -DSTM_ENOMEM;
        } else if (MPI_Allgather(&self_info, (int)sizeof(self_info), MPI_BYTE,
                                 all_info, (int)sizeof(self_info), MPI_BYTE,
                                 comm) != MPI_SUCCESS) {
                err = -DSTM_FAILURE;
        }

        if (err == 0) {
                size_t total_v = 0;
                size_t total_e = 0;

                for (int r = 0; r < world_size; r++) {
                        g->rank_local_vertices[r] =
                                (size_t)all_info[r].local_vertices;
                        g->rank_local_edge_count[r] =
                                (size_t)all_info[r].local_edge_count;
                        g->rank_edge_slot_stride[r] =
                                (size_t)all_info[r].edge_slot_stride;
                        g->rank_vertex_state_off[r] =
                                (intptr_t)all_info[r].vertex_state_off;
                        g->rank_edge_row_ptr_off[r] =
                                (intptr_t)all_info[r].edge_row_ptr_off;
                        g->rank_edge_slot_off[r] =
                                (intptr_t)all_info[r].edge_slot_off;
                        g->rank_region_id[r] = (int)all_info[r].region_id;

                        g->rank_vertex_offsets[r] = total_v;
                        total_v += g->rank_local_vertices[r];
                        total_e += g->rank_local_edge_count[r];
                }
                g->rank_vertex_offsets[world_size] = total_v;
                g->total_vertices      = total_v;
                g->total_edges         = total_e;
                g->local_vertex_offset = g->rank_vertex_offsets[self];

                /* --- Validate local topology, now that total_vertices
                 *     is known. ------------------------------------------ */
                const size_t *cursor = local_edge_dsts;
                for (size_t i = 0; i < local_vertices && err == 0; i++) {
                        size_t deg = local_out_degrees[i];
                        if (deg > 0) {
                                int rc = validate_neighbors(cursor, deg,
                                                            total_v);
                                if (rc != 0)
                                        err = rc;
                        }
                        cursor += deg;
                }
        }

        free(all_info);
        all_info = NULL;

        err = dstm_msg_consensus(comm, err);
        if (err != 0)
                goto err_cleanup;

        /* --- Discover neighbor ranks -------------------------------------
         *
         * Phase 1: local_touched[r] = 1 iff any of self's outgoing edges
         * lands on a vertex owned by rank r. Purely local. */
        {
                uint8_t *local_touched = calloc((size_t)world_size,
                                                sizeof(uint8_t));
                uint8_t *touch_matrix =
                        calloc((size_t)world_size * (size_t)world_size,
                               sizeof(uint8_t));

                if (local_touched == NULL || touch_matrix == NULL) {
                        err = -DSTM_ENOMEM;
                } else {
                        const size_t *cursor = local_edge_dsts;
                        for (size_t i = 0; i < local_vertices; i++) {
                                size_t deg = local_out_degrees[i];
                                for (size_t k = 0; k < deg; k++) {
                                        int r = find_rank_in_offsets(
                                                g->rank_vertex_offsets,
                                                world_size, cursor[k]);
                                        local_touched[r] = 1;
                                }
                                cursor += deg;
                        }
                        local_touched[self] = 1; /* self is always a neighbor */
                }

                /* Phase 2: symmetrize. We need to know who touches *us*,
                 * not just who we touch. Gather every rank's row, then OR
                 * row `self` with column `self`. O(world_size^2) bytes at
                 * setup only. Every rank must enter this Allgather even if
                 * it has already failed, hence the throwaway fallback. */
                if (local_touched != NULL && touch_matrix != NULL) {
                        if (MPI_Allgather(local_touched, world_size,
                                          MPI_UINT8_T, touch_matrix,
                                          world_size, MPI_UINT8_T, comm)
                            != MPI_SUCCESS)
                                err = -DSTM_FAILURE;
                } else {
                        uint8_t *tmp_s = calloc((size_t)world_size, 1);
                        uint8_t *tmp_r = calloc((size_t)world_size *
                                                (size_t)world_size, 1);
                        if (tmp_s && tmp_r)
                                MPI_Allgather(tmp_s, world_size, MPI_UINT8_T,
                                              tmp_r, world_size, MPI_UINT8_T,
                                              comm);
                        free(tmp_s);
                        free(tmp_r);
                }

                if (err == 0) {
                        for (int r = 0; r < world_size; r++) {
                                if (touch_matrix[(size_t)self *
                                                 (size_t)world_size + r] ||
                                    touch_matrix[(size_t)r *
                                                 (size_t)world_size + self])
                                        g->is_neighbor_rank[r] = 1;
                        }
                        g->is_neighbor_rank[self] = 1;
                }

                free(local_touched);
                free(touch_matrix);
        }

        if (err == 0) {
                /* Build the compact neighbor-ranks list. */
                int nc = 0;
                for (int r = 0; r < world_size; r++)
                        if (g->is_neighbor_rank[r])
                                nc++;
                g->neighbor_rank_count = nc;
                g->neighbor_ranks = malloc((size_t)nc * sizeof(int));
                if (g->neighbor_ranks == NULL) {
                        err = -DSTM_ENOMEM;
                } else {
                        int j = 0;
                        for (int r = 0; r < world_size; r++)
                                if (g->is_neighbor_rank[r])
                                        g->neighbor_ranks[j++] = r;
                }
        }

        /* --- Populate self's topology cache entries locally --------------- */
        if (err == 0) {
                size_t n_rp = local_vertices + 1;

                g->rank_row_ptrs[self] = malloc(n_rp * sizeof(dgraph_vid_t));
                if (g->rank_row_ptrs[self] == NULL) {
                        err = -DSTM_ENOMEM;
                } else {
                        memcpy(g->rank_row_ptrs[self], g->row_ptrs,
                               n_rp * sizeof(dgraph_vid_t));
                }

                if (err == 0 && g->local_edge_count > 0) {
                        g->rank_edge_dsts[self] =
                                malloc(g->local_edge_count *
                                       sizeof(dgraph_vid_t));
                        if (g->rank_edge_dsts[self] == NULL) {
                                err = -DSTM_ENOMEM;
                        } else {
                                for (size_t s = 0; s < g->local_edge_count; s++)
                                        g->rank_edge_dsts[self][s] =
                                                (dgraph_vid_t)
                                                local_edge_dsts[s];
                        }
                }
        }

        /* --- Pre-allocate all neighbor receive buffers -------------------
         *
         * Done before posting anything so an OOM cannot strand in-flight
         * requests. Sizes come from the layout exchange above. */
        if (err == 0) {
                for (int i = 0; i < g->neighbor_rank_count && err == 0; i++) {
                        int r = g->neighbor_ranks[i];
                        if (r == self)
                                continue;

                        size_t n_rp = g->rank_local_vertices[r] + 1;
                        size_t n_ed = g->rank_local_edge_count[r];

                        g->rank_row_ptrs[r] =
                                malloc(n_rp * sizeof(dgraph_vid_t));
                        if (g->rank_row_ptrs[r] == NULL) {
                                err = -DSTM_ENOMEM;
                                break;
                        }
                        if (n_ed > 0) {
                                g->rank_edge_dsts[r] =
                                        malloc(n_ed * sizeof(dgraph_vid_t));
                                if (g->rank_edge_dsts[r] == NULL) {
                                        err = -DSTM_ENOMEM;
                                        break;
                                }
                        }
                }
        }

        err = dstm_msg_consensus(comm, err);
        if (err != 0)
                goto err_cleanup;

        /* --- Neighbor-only topology exchange -----------------------------
         *
         * Plain MPI: this is setup-time metadata, not simulation state, so
         * it has no place on the versioned timeline. "Neighbor" is
         * symmetric by construction (OR of touched rows), so every peer we
         * send to is issuing the matching receive. */
        {
                enum { TAG_ROW_PTRS = 0x100, TAG_EDGE_DSTS = 0x101 };

                int peer_count = g->neighbor_rank_count - 1;
                MPI_Request *reqs = NULL;
                int req_n = 0;

                if (peer_count > 0) {
                        /* Up to 4 requests per peer: rp send, ed send,
                         * rp recv, ed recv. */
                        reqs = malloc((size_t)peer_count * 4 *
                                      sizeof(MPI_Request));
                        if (reqs == NULL)
                                err = -DSTM_ENOMEM;
                }

                if (err == 0 && peer_count > 0) {
                        for (int i = 0; i < g->neighbor_rank_count; i++) {
                                int r = g->neighbor_ranks[i];
                                if (r == self)
                                        continue;

                                /* Receives first: gives the runtime a
                                 * matching buffer before the sends land. */
                                size_t n_rp = g->rank_local_vertices[r] + 1;
                                size_t n_ed = g->rank_local_edge_count[r];

                                MPI_Irecv(g->rank_row_ptrs[r], (int)n_rp,
                                          MPI_UINT64_T, r, TAG_ROW_PTRS,
                                          comm, &reqs[req_n++]);
                                if (n_ed > 0)
                                        MPI_Irecv(g->rank_edge_dsts[r],
                                                  (int)n_ed, MPI_UINT64_T, r,
                                                  TAG_EDGE_DSTS, comm,
                                                  &reqs[req_n++]);

                                MPI_Isend(g->rank_row_ptrs[self],
                                          (int)(local_vertices + 1),
                                          MPI_UINT64_T, r, TAG_ROW_PTRS,
                                          comm, &reqs[req_n++]);
                                if (g->local_edge_count > 0)
                                        MPI_Isend(g->rank_edge_dsts[self],
                                                  (int)g->local_edge_count,
                                                  MPI_UINT64_T, r,
                                                  TAG_EDGE_DSTS, comm,
                                                  &reqs[req_n++]);
                        }

                        if (MPI_Waitall(req_n, reqs, MPI_STATUSES_IGNORE)
                            != MPI_SUCCESS)
                                err = -DSTM_FAILURE;
                }
                free(reqs);
        }

        err = dstm_msg_consensus(comm, err);
        if (err != 0)
                goto err_cleanup;

        err = dgraph_seed_initial_state(g, local_edge_dsts,
                                        local_vertex_states,
                                        local_edge_states);

        err = dstm_msg_consensus(comm, err);
        if (err != 0)
                goto err_cleanup;

        *ret = g;
        return DSTM_SUCCESS;

err_cleanup:
        free(all_info);

        /* Backing region, only if we got as far as allocating it. */
        if (region_allocated) {
                dstm_ptr_t base_p = dstm_ptr_make(g->region_id,
                                             g->vertex_state_off);
                dstm_region_free(g->region, base_p);
        }

        dgraph_free_all(g);

        *ret = NULL;
        return err;
}

int dstm_dgraph_create_equal(dstm_dgraph_t **ret,
                             size_t vertex_state_size,
                             size_t edge_state_size,
                             size_t total_vertices,
                             const size_t *local_out_degrees,
                             const size_t *local_edge_dsts,
                             const void *local_vertex_states,
                             const void *local_edge_states)
{
        int world_size = dstm_global_context.world_size;
        int self       = dstm_global_context.rank;

        size_t local_vertices = total_vertices / (size_t)world_size;
        size_t remainder      = total_vertices % (size_t)world_size;

        if (self < (int)remainder)
                local_vertices++;

        return dstm_dgraph_create(ret, vertex_state_size, edge_state_size,
                                  local_vertices, local_out_degrees,
                                  local_edge_dsts, local_vertex_states,
                                  local_edge_states);
}

void dstm_dgraph_destroy(dstm_dgraph_t **dgraph)
{
        if (dgraph == NULL || *dgraph == NULL)
                return;

        dstm_dgraph_t *g = *dgraph;

        dstm_ptr_t base_p = dstm_ptr_make(g->region_id, g->vertex_state_off);
        dstm_region_free(g->region, base_p);

        dgraph_free_all(g);

        *dgraph = NULL;
}

/* ---------------------------------------------------------------------------
 * Vertex state operations
 * ------------------------------------------------------------------------- */

/* Resolve a vertex to its owning rank, region id, and the dstm_ptr_t
 * addressing its versioned state. Mirror of resolve_edge_state. */
static int resolve_vertex_state(const dstm_dgraph_t *g, size_t vertex_id,
                                int *out_rank, int *out_region,
                                dstm_ptr_t *out_ptr)
{
        if (vertex_id >= g->total_vertices)
                return -DSTM_EINVAL;

        int    rank      = dgraph_find_rank(g, vertex_id);
        size_t local_idx = vertex_id - g->rank_vertex_offsets[rank];
        intptr_t off = g->rank_vertex_state_off[rank] +
                       (intptr_t)(local_idx * g->vertex_state_size);

        *out_rank   = rank;
        *out_region = dgraph_region_of(g, rank);
        *out_ptr    = dstm_ptr_make(*out_region, off);
        return DSTM_SUCCESS;
}

int dstm_dgraph_vertex_get(dstm_dgraph_t *g, size_t vertex_id, void *dst,
                           dstm_time_t time, dstm_time_t delta)
{
        if (g == NULL || dst == NULL)
                return -DSTM_EINVAL;
        if (g->vertex_state_size == 0)
                return (vertex_id < g->total_vertices)
                               ? DSTM_SUCCESS : -DSTM_EINVAL;

        int rank, region;
        dstm_ptr_t src;
        int rc = resolve_vertex_state(g, vertex_id, &rank, &region, &src);
        if (rc != DSTM_SUCCESS)
                return rc;

        return dstm_get(rank, region, time, delta, dst, src,
                        g->vertex_state_size);
}

int dstm_dgraph_vertex_put(dstm_dgraph_t *g, size_t vertex_id, void *src,
                           dstm_time_t time, dstm_time_t delta)
{
        if (g == NULL || src == NULL)
                return -DSTM_EINVAL;
        if (g->vertex_state_size == 0)
                return (vertex_id < g->total_vertices)
                               ? DSTM_SUCCESS : -DSTM_EINVAL;

        int rank, region;
        dstm_ptr_t dst;
        int rc = resolve_vertex_state(g, vertex_id, &rank, &region, &dst);
        if (rc != DSTM_SUCCESS)
                return rc;

        return dstm_put(rank, region, time, delta, dst, src,
                        g->vertex_state_size);
}

int dstm_dgraph_vertex_get_or_default(dstm_dgraph_t *g, size_t vertex_id,
                                      void *dst, dstm_time_t time,
                                      dstm_time_t delta,
                                      const void *default_value)
{
        if (g == NULL || dst == NULL || default_value == NULL)
                return -DSTM_EINVAL;

        if (time + delta < 0) {
                if (g->vertex_state_size > 0)
                        memcpy(dst, default_value, g->vertex_state_size);
                return DSTM_SUCCESS;
        }
        return dstm_dgraph_vertex_get(g, vertex_id, dst, time, delta);
}

/* ---------------------------------------------------------------------------
 * Edge lookup helpers
 *
 * Topology (row pointers and per-slot dst fields) is immutable and
 * fully replicated for every neighbor rank, so these lookups are pure
 * host-memory operations: no (time, delta) argument is meaningful and
 * no MPI traffic is generated. Only the user_state half of an edge slot
 * is versioned and reached through dstm_get / dstm_put.
 * ------------------------------------------------------------------------- */

/* Return [row_start, row_end) for local vertex `local_idx` on `rank`.
 * Requires `rank` to be a neighbor of self. */
static int fetch_row_range(const dstm_dgraph_t *g, int rank, size_t local_idx,
                           dgraph_vid_t *out_start, dgraph_vid_t *out_end)
{
        if (rank < 0 || rank >= g->world_size)
                return -DSTM_EINVAL;
        if (local_idx >= g->rank_local_vertices[rank])
                return -DSTM_EINVAL;
        if (!g->is_neighbor_rank[rank] || g->rank_row_ptrs[rank] == NULL)
                return -DSTM_ENOTLOCAL;

        *out_start = g->rank_row_ptrs[rank][local_idx];
        *out_end   = g->rank_row_ptrs[rank][local_idx + 1];
        return DSTM_SUCCESS;
}

/* Binary search for destination `v` within slots [row_start, row_end)
 * of `rank`'s edge array. Slots are sorted by ascending dst. */
static int find_edge_slot(const dstm_dgraph_t *g, int rank,
                          dgraph_vid_t row_start, dgraph_vid_t row_end,
                          size_t v, dgraph_vid_t *out_slot)
{
        if (!g->is_neighbor_rank[rank])
                return -DSTM_ENOTLOCAL;
        if (g->rank_edge_dsts[rank] == NULL)
                return -DSTM_ENOTFOUND;

        const dgraph_vid_t *dsts   = g->rank_edge_dsts[rank];
        dgraph_vid_t        lo     = row_start;
        dgraph_vid_t        hi     = row_end;
        dgraph_vid_t        target = (dgraph_vid_t)v;

        while (lo < hi) {
                dgraph_vid_t mid     = lo + (hi - lo) / 2;
                dgraph_vid_t mid_dst = dsts[mid];

                if (mid_dst == target) {
                        *out_slot = mid;
                        return DSTM_SUCCESS;
                }
                if (mid_dst < target)
                        lo = mid + 1;
                else
                        hi = mid;
        }
        return -DSTM_ENOTFOUND;
}

/* Resolve (u, v) to the byte offset of the edge's user_state and the
 * owning rank. Shared by edge_get and edge_put. */
static int resolve_edge_state(const dstm_dgraph_t *g, size_t u, size_t v,
                              int *out_rank, intptr_t *out_off)
{
        if (u >= g->total_vertices || v >= g->total_vertices)
                return -DSTM_EINVAL;

        int    rank      = dgraph_find_rank(g, u);
        size_t local_idx = u - g->rank_vertex_offsets[rank];

        dgraph_vid_t rs, re;
        int rc = fetch_row_range(g, rank, local_idx, &rs, &re);
        if (rc != DSTM_SUCCESS)
                return rc;

        dgraph_vid_t slot;
        rc = find_edge_slot(g, rank, rs, re, v, &slot);
        if (rc != DSTM_SUCCESS)
                return rc;

        size_t stride = g->rank_edge_slot_stride[rank];

        *out_rank = rank;
        *out_off  = g->rank_edge_slot_off[rank] +
                    (intptr_t)((size_t)slot * stride) +
                    (intptr_t)sizeof(dgraph_vid_t);
        return DSTM_SUCCESS;
}

/* ---------------------------------------------------------------------------
 * Edge state operations
 * ------------------------------------------------------------------------- */

int dstm_dgraph_edge_get(dstm_dgraph_t *g, size_t u, size_t v, void *dst,
                         dstm_time_t time, dstm_time_t delta)
{
        if (g == NULL || dst == NULL)
                return -DSTM_EINVAL;

        int      rank;
        intptr_t off;
        int rc = resolve_edge_state(g, u, v, &rank, &off);
        if (rc != DSTM_SUCCESS)
                return rc;

        if (g->edge_state_size == 0)
                return DSTM_SUCCESS;

        int        region = dgraph_region_of(g, rank);
        dstm_ptr_t src    = dstm_ptr_make(region, off);

        return dstm_get(rank, region, time, delta, dst, src,
                        g->edge_state_size);
}

int dstm_dgraph_edge_put(dstm_dgraph_t *g, size_t u, size_t v, void *src,
                         dstm_time_t time, dstm_time_t delta)
{
        if (g == NULL || src == NULL)
                return -DSTM_EINVAL;

        int      rank;
        intptr_t off;
        int rc = resolve_edge_state(g, u, v, &rank, &off);
        if (rc != DSTM_SUCCESS)
                return rc;

        if (g->edge_state_size == 0)
                return DSTM_SUCCESS;

        int        region = dgraph_region_of(g, rank);
        dstm_ptr_t dst    = dstm_ptr_make(region, off);

        return dstm_put(rank, region, time, delta, dst, src,
                        g->edge_state_size);
}

/* ---------------------------------------------------------------------------
 * Neighborhood queries
 * ------------------------------------------------------------------------- */

int dstm_dgraph_degree_get(dstm_dgraph_t *g, size_t u, size_t *out_degree)
{
        if (g == NULL || out_degree == NULL)
                return -DSTM_EINVAL;
        if (u >= g->total_vertices)
                return -DSTM_EINVAL;

        int    rank      = dgraph_find_rank(g, u);
        size_t local_idx = u - g->rank_vertex_offsets[rank];

        dgraph_vid_t rs, re;
        int rc = fetch_row_range(g, rank, local_idx, &rs, &re);
        if (rc != DSTM_SUCCESS)
                return rc;

        *out_degree = (size_t)(re - rs);
        return DSTM_SUCCESS;
}

int dstm_dgraph_neighbors_get(dstm_dgraph_t *g, size_t u,
                              size_t *out_neighbors, size_t *io_count)
{
        if (g == NULL || io_count == NULL)
                return -DSTM_EINVAL;
        if (u >= g->total_vertices)
                return -DSTM_EINVAL;

        int    rank      = dgraph_find_rank(g, u);
        size_t local_idx = u - g->rank_vertex_offsets[rank];

        dgraph_vid_t rs, re;
        int rc = fetch_row_range(g, rank, local_idx, &rs, &re);
        if (rc != DSTM_SUCCESS)
                return rc;

        size_t deg = (size_t)(re - rs);

        if (deg == 0) {
                *io_count = 0;
                return DSTM_SUCCESS;
        }
        if (out_neighbors == NULL || *io_count < deg) {
                *io_count = deg;
                return -DSTM_ENOMEM;
        }

        const dgraph_vid_t *dsts = g->rank_edge_dsts[rank];
        for (size_t i = 0; i < deg; i++)
                out_neighbors[i] = (size_t)dsts[(size_t)rs + i];

        *io_count = deg;
        return DSTM_SUCCESS;
}

/* ---------------------------------------------------------------------------
 * Distance matrix derivation
 *
 * Produces a rank x rank dstm_distance_t. Cell (i, j) is:
 *   0                on the diagonal,
 *   min edge weight  over every edge crossing the i<->j rank boundary in
 *                    either direction (edges are undirected for
 *                    synchronization: if i can write to j, j's progress
 *                    constrains i), and
 *   -1               when no edge connects the two ranks.
 *
 * "min" is the correct reduction for parallel / opposing edges: the
 * smallest weight is the most constrained, and the most constrained edge
 * governs how tightly the two ranks must stay synchronized.
 *
 * Two entry points:
 *
 *   dstm_dgraph_distance_from_topology  -- operates on the raw CSR
 *      arrays, no dstm_dgraph_t required. Needed when the matrix is an
 *      input to dstm_start for the session that will host the graph:
 *      dstm_dgraph_create allocates from the session's local region,
 *      so the graph cannot exist until dstm_start has run, and
 *      dstm_start needs the matrix. Only dstm_init is required here --
 *      dstm_global_context.comm is live from init onward, including
 *      between sessions. No graph state exists that early, so there is
 *      no weight callback: every edge weighs default_delta.
 *
 *   dstm_dgraph_derive_distance{,_weighted} -- same reduction over an
 *      already-built graph, for inspection or for a subsequent
 *      session. Because state exists, per-edge weights may be computed
 *      from live vertex/edge state via a caller-supplied callback.
 * ------------------------------------------------------------------------- */

/* Sentinel stored in a per-rank weight row / gathered matrix meaning "no
 * edge to this rank". Real weights are required to be > 0, so 0 is free. */
#define DGRAPH_NO_EDGE_W 0

/* Accumulate one edge's weight into the per-destination-rank row, keeping
 * the minimum. `w` must be > 0. */
static inline void row_accumulate(int *row, int dr, int w)
{
        int cur = row[dr];
        if (cur == DGRAPH_NO_EDGE_W || w < cur)
                row[dr] = w;
}

/* Shared tail: given this rank's per-destination-rank min-weight row
 * (DGRAPH_NO_EDGE_W where this rank has no edge into that rank), produce
 * the symmetric distance matrix. Consumes nothing, frees nothing the
 * caller owns. */
static int build_distance_from_row(const int *local_row, int world_size,
                                   MPI_Comm comm, dstm_distance_t **ret)
{
        int *full = calloc((size_t)world_size * (size_t)world_size,
                           sizeof(int));
        if (full == NULL)
                return -DSTM_ENOMEM;

        if (MPI_Allgather(local_row, world_size, MPI_INT, full, world_size,
                          MPI_INT, comm) != MPI_SUCCESS) {
                free(full);
                return -DSTM_FAILURE;
        }

        int *matrix = malloc((size_t)world_size * (size_t)world_size *
                             sizeof(int));
        if (matrix == NULL) {
                free(full);
                return -DSTM_ENOMEM;
        }

        /* Symmetrize + encode. full[i*ws + j] is rank i's min weight into
         * rank j (DGRAPH_NO_EDGE_W if none). The undirected cell is the
         * min over the two directions, skipping the no-edge sentinel so a
         * missing reverse edge does not poison a real forward one. */
        for (int i = 0; i < world_size; i++) {
                for (int j = 0; j < world_size; j++) {
                        int v;
                        if (i == j) {
                                v = 0;
                        } else {
                                int a = full[(size_t)i * world_size + j];
                                int b = full[(size_t)j * world_size + i];
                                int has_a = (a != DGRAPH_NO_EDGE_W);
                                int has_b = (b != DGRAPH_NO_EDGE_W);

                                if (has_a && has_b)
                                        v = (a < b) ? a : b;
                                else if (has_a)
                                        v = a;
                                else if (has_b)
                                        v = b;
                                else
                                        v = -1;
                        }
                        matrix[(size_t)i * world_size + j] = v;
                }
        }
        free(full);

        int rc = dstm_distance_create_from_matrix(ret, (size_t)world_size,
                                                  matrix);
        free(matrix);
        return rc;
}

int dstm_dgraph_distance_from_topology(dstm_distance_t **ret,
                                       size_t local_vertices,
                                       const size_t *local_out_degrees,
                                       const size_t *local_edge_dsts,
                                       int default_delta)
{
        if (ret == NULL || default_delta <= 0)
                return -DSTM_EINVAL;
        if (local_vertices > 0 && local_out_degrees == NULL)
                return -DSTM_EINVAL;
        *ret = NULL;

        int      world_size = dstm_global_context.world_size;
        MPI_Comm comm       = dstm_global_context.comm;

        /* Rebuild the same contiguous rank-order partitioning that
         * dstm_dgraph_create will use, so dst ids map to the same
         * owners. */
        uint64_t  mine    = (uint64_t)local_vertices;
        uint64_t *counts  = calloc((size_t)world_size, sizeof(uint64_t));
        size_t   *offsets = calloc((size_t)world_size + 1, sizeof(size_t));
        int      *row     = calloc((size_t)world_size, sizeof(int));

        if (counts == NULL || offsets == NULL || row == NULL) {
                free(counts);
                free(offsets);
                free(row);
                return -DSTM_ENOMEM;
        }

        if (MPI_Allgather(&mine, 1, MPI_UINT64_T, counts, 1, MPI_UINT64_T,
                          comm) != MPI_SUCCESS) {
                free(counts);
                free(offsets);
                free(row);
                return -DSTM_FAILURE;
        }

        size_t acc = 0;
        for (int r = 0; r < world_size; r++) {
                offsets[r] = acc;
                acc += (size_t)counts[r];
        }
        offsets[world_size] = acc;

        /* No live state at topology time: every edge weighs default_delta.
         * row[] holds the constant for any rank we touch, sentinel
         * elsewhere. */
        {
                const size_t *cursor = local_edge_dsts;
                for (size_t i = 0; i < local_vertices; i++) {
                        size_t deg = local_out_degrees[i];
                        for (size_t k = 0; k < deg; k++) {
                                if (cursor[k] >= acc) {
                                        free(counts);
                                        free(offsets);
                                        free(row);
                                        return -DSTM_EINVAL;
                                }
                                int dr = find_rank_in_offsets(offsets,
                                                              world_size,
                                                              cursor[k]);
                                row_accumulate(row, dr, default_delta);
                        }
                        cursor += deg;
                }
        }

        int rc = build_distance_from_row(row, world_size, comm, ret);
        free(counts);
        free(offsets);
        free(row);
        return rc;
}

/* Pointer to self-owned local vertex `local_idx`'s state inside the
 * region's live buffer, or NULL if there is no vertex state. Valid only
 * for the duration of the derivation walk; the callback must not retain
 * it. derive_distance only walks self's own vertices and edges, so no
 * remote get is ever needed. */
static inline const void *local_vertex_state_ptr(const dstm_dgraph_t *g,
                                                 size_t local_idx)
{
        if (g->vertex_state_size == 0)
                return NULL;
        return (const char *)g->region->buffer + g->vertex_state_off +
               local_idx * g->vertex_state_size;
}

/* Pointer to self-owned edge slot `slot`'s user_state inside the
 * region's live buffer, or NULL if there is no edge state. `slot` is a
 * global edge-slot index into self's edge slab. */
static inline const void *local_edge_state_ptr(const dstm_dgraph_t *g,
                                               dgraph_vid_t slot)
{
        if (g->edge_state_size == 0)
                return NULL;
        return (const char *)g->region->buffer + g->edge_slot_off +
               (size_t)slot * g->edge_slot_stride + sizeof(dgraph_vid_t);
}

int dstm_dgraph_derive_distance_weighted(dstm_dgraph_t *g,
                                         dstm_distance_t **ret,
                                         int default_delta,
                                         dstm_dgraph_edge_weight_fn weight_fn,
                                         void *user)
{
        if (g == NULL || ret == NULL || default_delta <= 0)
                return -DSTM_EINVAL;
        *ret = NULL;

        int      world_size = g->world_size;
        int      self       = g->local_rank;
        MPI_Comm comm       = dstm_global_context.comm;

        /* State is handed to the callback as pointers into the region's
         * live buffer, so that buffer has to exist. Checked once here
         * rather than at every access. */
        if (weight_fn != NULL &&
            (g->vertex_state_size > 0 || g->edge_state_size > 0) &&
            (g->region == NULL || g->region->buffer == NULL))
                return -DSTM_ENOTLOCAL;

        int *local_row = calloc((size_t)world_size, sizeof(int));
        if (local_row == NULL)
                return -DSTM_ENOMEM;

        int err = DSTM_SUCCESS;

        const dgraph_vid_t *local_dsts = g->rank_edge_dsts[self];
        if (local_dsts != NULL) {
                for (size_t i = 0; i < g->local_vertices; i++) {
                        dgraph_vid_t rs = g->row_ptrs[i];
                        dgraph_vid_t re = g->row_ptrs[i + 1];
                        const void  *src_vs = (weight_fn != NULL)
                                ? local_vertex_state_ptr(g, i) : NULL;

                        for (dgraph_vid_t s = rs; s < re; s++) {
                                size_t dst_vid = (size_t)local_dsts[s];
                                int    dr = dgraph_find_rank(g, dst_vid);
                                int    w  = default_delta;

                                if (weight_fn != NULL) {
                                        /* dst vertex state is only
                                         * readable locally; a remote dst
                                         * gets NULL rather than a
                                         * blocking remote get. */
                                        const void *dst_vs = (dr == self)
                                                ? local_vertex_state_ptr(g,
                                                          dst_vid -
                                                          g->local_vertex_offset)
                                                : NULL;

                                        w = weight_fn(
                                                g->local_vertex_offset + i,
                                                dst_vid, src_vs, dst_vs,
                                                local_edge_state_ptr(g, s),
                                                user);
                                        if (w <= 0) {
                                                err = -DSTM_EINVAL;
                                                goto done;
                                        }
                                }

                                row_accumulate(local_row, dr, w);
                        }
                }
        }

done:
        /* build_distance_from_row is collective, so it must be reached
         * on every rank or not at all. Agree on the local outcome first,
         * the same way create does. */
        err = dstm_msg_consensus(comm, err);
        if (err == DSTM_SUCCESS)
                err = build_distance_from_row(local_row, world_size, comm,
                                              ret);

        free(local_row);
        return err;
}

int dstm_dgraph_derive_distance(dstm_dgraph_t *g, dstm_distance_t **ret,
                                int default_delta)
{
        /* Constant-weight special case: NULL callback reproduces the
         * original boolean-reachability -> default_delta / -1 matrix. */
        return dstm_dgraph_derive_distance_weighted(g, ret, default_delta,
                                                    NULL, NULL);
}
