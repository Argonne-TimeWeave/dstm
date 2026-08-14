/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef GRAPH_TRAFFIC_CHAN_H
#define GRAPH_TRAFFIC_CHAN_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/*******************************************************************************
 * Channel buffer
 *
 * One buffer travels along one directed edge, per tick. It is the edge
 * state of a dstm_dgraph, so its size is fixed for the graph's lifetime
 * and identical on every rank.
 *
 *  `[chan_header | cong_entry state[|E|] | car_rec cars[<=cap]]`
 *
 * The offsets are carried in the header rather than recomputed by the
 * reader. They are redundant with the compile-time layout today, but
 * they cost 8 bytes and turn a whole class of layout/versioning
 * mismatches into a clean rejected message.
 *
 * The congestion half is fixed size, so cars_off is a constant; only the
 * car half varies, and only its *used* prefix. The unused tail is
 * transmitted (the region slot has to be a fixed size) but zeroed.
 ******************************************************************************/

#define CHAN_MAGIC 0x43484e31u          /* "CHN1" */

/* Per-edge congestion, highest-stamp-wins. Arrival order carries no weight.
 *
 * Exactly one rank -- owner(src(edge)) -- ever writes a given entry.
 * Every other rank only relays it. Two entries carrying the same stamp
 * are therefore bit-identical, so the merge below cannot encounter a
 * genuine conflict, is commutative/associative/idempotent, and gives
 * bitwise-identical results regardless of message arrival order. */
struct cong_entry {
        float   value;          /* smoothed load on the edge            */
        int32_t stamp;          /* tick the OWNER wrote it, -1 == never */
};

/* Wire form of an agent in transit.
 *
 * dst_vertex is redundant with chan_header.dst_vertex -- every car in a
 * buffer lands at the buffer's own destination, because the buffer IS
 * the edge. It is kept because inject_car reads it per-record without
 * touching the header, and because dropping it takes car_rec to 20 B and
 * breaks the 24 B wire assert for 4 bytes x cap per message. */
struct car_rec {
        int32_t id;
        int32_t origin;
        int32_t dest;
        int32_t dst_vertex;
        int32_t depart_tick;
        int32_t hops;
};

struct chan_header {
        uint32_t magic;
        int32_t  tick;          /* ARRIVAL tick of this buffer          */
        uint32_t n_state;
        uint32_t n_cars;
        uint32_t cap;
        uint32_t state_off;     /* bytes from buffer start to state[0]  */
        uint32_t cars_off;      /* bytes from buffer start to cars[0]   */
        /* Diagnostic only: chan_validate does not check these and the
         * receiver does not compare them against the slot it read from.
         * They exist to make a hex dump interpretable. */
        uint32_t src_vertex;
        uint32_t dst_vertex;
        uint32_t reserved;
};

/* Compile-time guards: the wire layout must not drift. */
typedef char chan_assert_hdr[(sizeof(struct chan_header) == 40) ? 1 : -1];
typedef char chan_assert_ent[(sizeof(struct cong_entry) == 8) ? 1 : -1];
typedef char chan_assert_car[(sizeof(struct car_rec) == 24) ? 1 : -1];

struct chan_layout {
        uint32_t n_state;
        uint32_t cap;
        uint32_t state_off;
        uint32_t cars_off;
        size_t   bytes;         /* == dgraph edge_state_size            */
};

void chan_layout_init(struct chan_layout *L, size_t n_edges, unsigned int cap);

static inline struct chan_header *chan_hdr(void *buf)
{
        return (struct chan_header *)buf;
}

static inline const struct chan_header *chan_hdr_c(const void *buf)
{
        return (const struct chan_header *)buf;
}

static inline struct cong_entry *chan_state(const struct chan_layout *L,
                                            void *buf)
{
        return (struct cong_entry *)((char *)buf + L->state_off);
}

static inline const struct cong_entry *chan_state_c(const struct chan_layout *L,
                                                    const void *buf)
{
        return (const struct cong_entry *)((const char *)buf + L->state_off);
}

static inline struct car_rec *chan_cars(const struct chan_layout *L, void *buf)
{
        return (struct car_rec *)((char *)buf + L->cars_off);
}

static inline const struct car_rec *chan_cars_c(const struct chan_layout *L,
                                                const void *buf)
{
        return (const struct car_rec *)((const char *)buf + L->cars_off);
}

/* Zero the whole buffer and stamp a valid header. arrive_tick is when
 * the receiver is expected to consume it, i.e. send_tick + delta. */
void chan_reset(const struct chan_layout *L, void *buf, int32_t arrive_tick,
                size_t src_vertex, size_t dst_vertex);

/* Append a car. Returns 0 on success, -1 if the per-edge cap is full. */
int chan_push_car(const struct chan_layout *L, void *buf,
                  const struct car_rec *car);

/* Structural validation of a received buffer. Returns 0 if it is a
 * well-formed channel buffer for this layout, -1 otherwise (including
 * the never-written case, where the seeded slot is all zeroes). */
int chan_validate(const struct chan_layout *L, const void *buf);

/*******************************************************************************
 * Congestion replica
 ******************************************************************************/

struct cong_replica {
        struct cong_entry *e;
        size_t             n;
};

int  cong_replica_init(struct cong_replica *R, size_t n_edges);
void cong_replica_free(struct cong_replica *R);

/* Highest-stamp-wins join (a join-semilattice: commutative, associative,
 * idempotent). Returns the number of entries that advanced -- note the
 * COUNT is not idempotent even though the state is, so merge_updates is
 * a diagnostic, not an invariant. Safe to apply the same buffer twice or
 * to drop one entirely; dropping only adds staleness. */
size_t cong_merge(struct cong_replica *R, const struct cong_entry *incoming);

/* Age-weighted congestion PREMIUM. The router knows how old every number
 * it reads is, so decay the measured value toward `baseline` as
 * confidence dies. This is what turns "the data is delayed" from a defect
 * into something the algorithm reasons about explicitly. */
float cong_effective(const struct cong_entry *e, int32_t now, float tau,
                     float baseline);

#endif /* GRAPH_TRAFFIC_CHAN_H */
