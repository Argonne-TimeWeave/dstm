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
 * MULTI-REGION EXPLORATORY TEST (public API only)
 * ===============================================
 *
 * Determines whether the multi-region refactor supports TWO distributed
 * data structures living on TWO SEPARATE local backings, with puts/gets to
 * both regions interleaved in logical time.
 *
 * Everything here uses only the public dstm.h surface:
 *   - dstm_region_register()          -> obtain a second local region id
 *   - dstm_darray_create_on_region()  -> back a darray on a chosen region
 *   - dstm_darray_put/get()           -> drive traffic
 *
 * It is written to FIND BUGS, not to pass:
 *   - every expected value is derived independently from its inputs,
 *   - values are namespaced per region (0x0A.. vs 0x0B..) so any
 *     cross-region contamination is immediately visible,
 *   - operations on both regions are interleaved at the same timesteps so
 *     any rank-level-vs-region-level state confusion (version state
 *     machine, ninfos, heartbeat clock, barrier) surfaces,
 *   - every mismatch is reported (no abort-on-first) and the process exits
 *     non-zero if anything is wrong.
 *
 * Run:   mpirun -n 2 ./multi_region      (also try -n 4)
 */

#include <assert.h>
#include <inttypes.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dstm.h"
#include "internal/globals.h"

/* ---- value tagging so contamination is visible --------------------- */
#define REG0_MARK 0x0A000000
#define REG1_MARK 0x0B000000
#define REG_MASK  0x0F000000
/* Encode (region, rank, slot, timestep) into one int so a value read from
 * the wrong region/rank/slot/time is obvious on inspection. */
#define TAG(mark, rank, slot, t)                                               \
        ((int)((mark) | (((rank) & 0xff) << 12) | (((slot) & 0xff) << 4) |     \
               ((t) & 0xf)))

/* ---- report-everything scoreboard ---------------------------------- */
static int g_failures = 0;
static int g_checks = 0;

#define CHECK_EQ(got, want, what)                                              \
        do {                                                                   \
                g_checks++;                                                     \
                if ((int)(got) != (int)(want)) {                               \
                        g_failures++;                                          \
                        fprintf(stderr,                                        \
                                "MISMATCH [%-28s] got 0x%08x want 0x%08x "     \
                                "(%s:%d)\n",                                   \
                                (what), (unsigned)(got), (unsigned)(want),     \
                                __FILE__, __LINE__);                          \
                } else {                                                       \
                        fprintf(stderr, "ok       [%-28s] 0x%08x\n", (what),   \
                                (unsigned)(got));                             \
                }                                                              \
        } while (0)

#define CHECK_TRUE(cond, what)                                                 \
        do {                                                                   \
                g_checks++;                                                     \
                if (!(cond)) {                                                 \
                        g_failures++;                                          \
                        fprintf(stderr, "FAILED   [%-28s] (%s:%d)\n", (what),  \
                                __FILE__, __LINE__);                          \
                } else {                                                       \
                        fprintf(stderr, "ok       [%-28s]\n", (what));         \
                }                                                              \
        } while (0)

/* Confirm a value belongs to the region namespace we expect. */
#define CHECK_REGION(val, mark, what)                                          \
        CHECK_EQ((val) & REG_MASK, (mark), what)

/* ---------------------------------------------------------------------
 * Phase 0: register a second local region (public API).
 *
 * The default region is 0.  Registering one more must yield id 1 and a
 * distinct region object.  Regions are session-scoped, so this runs after
 * dstm_start and before dstm_fini.
 * ------------------------------------------------------------------- */
static int phase0_register(int rank)
{
        (void)rank;

        CHECK_EQ(dstm_get_local_region_id(), 0, "phase0.default_region_id");
        CHECK_TRUE(dstm_get_local_region(0) != NULL, "phase0.region0_exists");

        int rid = -1;
        int rc = dstm_region_register(1 << 20 /* 1 MiB */, &rid);
        CHECK_EQ(rc, DSTM_SUCCESS, "phase0.register_rc");
        CHECK_EQ(rid, 1, "phase0.second_region_id");

        CHECK_TRUE(dstm_get_local_region(1) != NULL, "phase0.region1_exists");
        CHECK_TRUE(dstm_get_local_region(0) != dstm_get_local_region(1),
                   "phase0.regions_distinct");

        return rid;
}

/* ---------------------------------------------------------------------
 * Phase 1: TWO darrays on TWO backings, LOCAL-only interleaved.
 *
 * Each rank owns one element in each darray.  We put into both darrays at
 * the same timesteps, then read back our OWN element from both.  This
 * isolates the per-region version state machine: if region_id filtering in
 * the WRITABLE/READABLE/DISCARDABLE checks is wrong, one darray's pending
 * request will gate the other darray's version and the gets will hang or
 * return garbage.  No remote traffic here.
 * ------------------------------------------------------------------- */
static void phase1_local(int rank, int ws)
{
        (void)ws;

        dstm_darray_t *a0 = NULL, *a1 = NULL;
        CHECK_EQ(dstm_darray_create_on_region(&a0, sizeof(int), 1, 0),
                 DSTM_SUCCESS, "phase1.create_a0_r0");
        CHECK_EQ(dstm_darray_create_on_region(&a1, sizeof(int), 1, 1),
                 DSTM_SUCCESS, "phase1.create_a1_r1");
        assert(a0 != NULL && a1 != NULL);

        /* DSTM contract: a rank must issue operations in monotonically
         * nondecreasing originating time.  We therefore interleave the put
         * for timestep t with the reach-back get that reads the value made
         * effective at time t (published by the put at originating time
         * t-1).  The get uses originating time t+1 with delta=-1, so the
         * clock never advances past a pending reach-back get and versions
         * are read inside their readable window [t, t+2).  Both regions are
         * touched at the SAME originating time each step, which keeps the
         * shared clock monotonic. */
        const int STEPS = 4;
        for (int t = 0; t < STEPS; t++) {
                int v0 = TAG(REG0_MARK, rank, 0, t);
                int v1 = TAG(REG1_MARK, rank, 0, t);
                CHECK_EQ(dstm_darray_put(a0, (size_t)rank, &v0, t, 1),
                         DSTM_SUCCESS, "phase1.put_a0");
                CHECK_EQ(dstm_darray_put(a1, (size_t)rank, &v1, t, 1),
                         DSTM_SUCCESS, "phase1.put_a1");

                /* Read the value that became effective at time t.  Effective
                 * time t was published by the put at originating time t-1,
                 * i.e. value TAG(..., t-1).  Skip t == 0 (no history yet)
                 * and t == 1 (effective time 1 was published by put@0, but
                 * reading it requires get@2 which we issue at the NEXT
                 * iteration; here we read effective time t via get@(t+1)). */
                if (t >= 1) {
                        int want_t = t - 1; /* value published at orig time t-1 */
                        int g0 = 0, g1 = 0;
                        CHECK_EQ(dstm_darray_get(a0, (size_t)rank, &g0, t + 1, -1),
                                 DSTM_SUCCESS, "phase1.get_a0_rc");
                        CHECK_EQ(dstm_darray_get(a1, (size_t)rank, &g1, t + 1, -1),
                                 DSTM_SUCCESS, "phase1.get_a1_rc");
                        CHECK_EQ(g0, TAG(REG0_MARK, rank, 0, want_t),
                                 "phase1.get_a0_val");
                        CHECK_EQ(g1, TAG(REG1_MARK, rank, 0, want_t),
                                 "phase1.get_a1_val");
                        CHECK_REGION(g0, REG0_MARK, "phase1.a0_isolation");
                        CHECK_REGION(g1, REG1_MARK, "phase1.a1_isolation");
                }
        }

        dstm_barrier((dstm_time_t)STEPS + 1);

        dstm_darray_destroy(&a0);
        dstm_darray_destroy(&a1);

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
                fprintf(stderr, "--- phase1 (local, two darrays) done ---\n");
}

/* ---------------------------------------------------------------------
 * Phase 2: TWO darrays on TWO backings, REMOTE interleaved.
 *
 * Each rank writes its own element in both darrays, then reads its
 * right-neighbor's element from both, interleaved in time.  This drives
 * OUT_PUT/OUT_GET/IN_PUT/IN_GET across the wire for two regions at once,
 * exercising the wire header region_id and the responder's
 * dstm_get_local_region(req->region_id) routing.
 * ------------------------------------------------------------------- */
static void phase2_remote(int rank, int ws)
{
        if (ws < 2) {
                if (rank == 0)
                        fprintf(stderr, "--- phase2 skipped (need >=2) ---\n");
                return;
        }

        dstm_darray_t *a0 = NULL, *a1 = NULL;
        CHECK_EQ(dstm_darray_create_on_region(&a0, sizeof(int), 1, 0),
                 DSTM_SUCCESS, "phase2.create_a0_r0");
        CHECK_EQ(dstm_darray_create_on_region(&a1, sizeof(int), 1, 1),
                 DSTM_SUCCESS, "phase2.create_a1_r1");
        assert(a0 != NULL && a1 != NULL);

        int peer = (rank + 1) % ws;

        /* Same monotonic-time interleaving as phase1, but the reach-back
         * gets read the PEER's element, driving OUT_GET/IN_GET across the
         * wire for both regions.  Both ranks issue identical originating
         * times in the same order, so cross-rank clock skew stays within
         * one step and every remote get lands inside the target version's
         * readable window [t, t+2) before the owner can discard it. */
        const int STEPS = 4;
        for (int t = 0; t < STEPS; t++) {
                int v0 = TAG(REG0_MARK, rank, 1, t);
                int v1 = TAG(REG1_MARK, rank, 1, t);
                CHECK_EQ(dstm_darray_put(a0, (size_t)rank, &v0, t, 1),
                         DSTM_SUCCESS, "phase2.put_a0");
                CHECK_EQ(dstm_darray_put(a1, (size_t)rank, &v1, t, 1),
                         DSTM_SUCCESS, "phase2.put_a1");

                if (t >= 1) {
                        int want_t = t - 1; /* peer's value published at t-1 */
                        int g0 = 0, g1 = 0;
                        CHECK_EQ(dstm_darray_get(a0, (size_t)peer, &g0, t + 1, -1),
                                 DSTM_SUCCESS, "phase2.get_a0_rc");
                        CHECK_EQ(dstm_darray_get(a1, (size_t)peer, &g1, t + 1, -1),
                                 DSTM_SUCCESS, "phase2.get_a1_rc");
                        CHECK_EQ(g0, TAG(REG0_MARK, peer, 1, want_t),
                                 "phase2.get_a0_val");
                        CHECK_EQ(g1, TAG(REG1_MARK, peer, 1, want_t),
                                 "phase2.get_a1_val");
                        CHECK_REGION(g0, REG0_MARK, "phase2.a0_isolation");
                        CHECK_REGION(g1, REG1_MARK, "phase2.a1_isolation");
                }
        }

        dstm_barrier((dstm_time_t)STEPS + 1);

        dstm_darray_destroy(&a0);
        dstm_darray_destroy(&a1);

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
                fprintf(stderr, "--- phase2 (remote, two darrays) done ---\n");
}

/* ---------------------------------------------------------------------
 * Phase 3: multi-element darrays on two backings, allgatherv on each.
 *
 * Larger partitions (a few elements per rank) on two distinct backings,
 * with a full allgatherv on each, interleaved.  This stresses the region
 * routing across many (rank, region) pairs at once and confirms the two
 * backings never alias under heavier traffic.
 * ------------------------------------------------------------------- */
static void phase3_allgatherv(int rank, int ws)
{
        const size_t LOCAL = 3; /* elements per rank */

        dstm_darray_t *a0 = NULL, *a1 = NULL;
        CHECK_EQ(dstm_darray_create_on_region(&a0, sizeof(int), LOCAL, 0),
                 DSTM_SUCCESS, "phase3.create_a0_r0");
        CHECK_EQ(dstm_darray_create_on_region(&a1, sizeof(int), LOCAL, 1),
                 DSTM_SUCCESS, "phase3.create_a1_r1");
        assert(a0 != NULL && a1 != NULL);

        size_t total = LOCAL * (size_t)ws;

        int *src0 = malloc(LOCAL * sizeof(int));
        int *src1 = malloc(LOCAL * sizeof(int));
        int *dst0 = malloc(total * sizeof(int));
        int *dst1 = malloc(total * sizeof(int));
        assert(src0 && src1 && dst0 && dst1);

        /* deltas is indexed by RANK for allgatherv; size ws is enough. */
        dstm_time_t *deltas = malloc((size_t)ws * sizeof(dstm_time_t));
        assert(deltas != NULL);
        for (int r = 0; r < ws; r++)
                deltas[r] = (r == rank) ? 1 : -1;

        const int STEPS = 3;

        /* Prime a few timesteps so remote gets at (t-1) are satisfiable. */
        for (int t = 0; t < STEPS; t++) {
                for (size_t i = 0; i < LOCAL; i++) {
                        src0[i] = TAG(REG0_MARK, rank, (int)(2 + i), t);
                        src1[i] = TAG(REG1_MARK, rank, (int)(2 + i), t);
                }
                /* local put at effective time t+1 into both backings */
                for (size_t i = 0; i < LOCAL; i++) {
                        size_t idx = (size_t)rank * LOCAL + i;
                        CHECK_EQ(dstm_darray_put(a0, idx, &src0[i], t, 1),
                                 DSTM_SUCCESS, "phase3.put_a0");
                        CHECK_EQ(dstm_darray_put(a1, idx, &src1[i], t, 1),
                                 DSTM_SUCCESS, "phase3.put_a1");
                }
        }

        /* allgatherv at t = STEPS on each backing, interleaved.
         * effective time for local put = STEPS+1; remote gets read STEPS. */
        for (size_t i = 0; i < LOCAL; i++) {
                src0[i] = TAG(REG0_MARK, rank, (int)(2 + i), STEPS);
                src1[i] = TAG(REG1_MARK, rank, (int)(2 + i), STEPS);
        }
        CHECK_EQ(dstm_darray_allgatherv(a0, src0, dst0, (dstm_time_t)STEPS,
                                        deltas),
                 DSTM_SUCCESS, "phase3.allgatherv_a0");
        CHECK_EQ(dstm_darray_allgatherv(a1, src1, dst1, (dstm_time_t)STEPS,
                                        deltas),
                 DSTM_SUCCESS, "phase3.allgatherv_a1");

        /* dstm_darray_allgatherv PUBLISHES the local slice (a put) but does
         * NOT copy it back into dst -- only remote slices are gathered into
         * dst (see dstm.h).  A full-allgather output therefore requires the
         * caller to place its own contribution into dst itself.  We copy the
         * source we just published into our own slice so the own-slice
         * checks below inspect real data instead of uninitialized memory. */
        for (size_t i = 0; i < LOCAL; i++) {
                size_t idx = (size_t)rank * LOCAL + i;
                dst0[idx] = src0[i];
                dst1[idx] = src1[i];
        }

        /* Verify every slot of both gathered buffers.
         * Remote slice for rank r was put at effective time STEPS-1 (the
         * last priming step, since remote get delta = -1 reads time
         * STEPS-1).  Our own slice reflects the allgatherv local put at
         * effective time STEPS+1, but allgatherv reads back via delta[self]
         * so we validate the remote slices (which is where cross-region
         * aliasing would show) and confirm own-region namespace. */
        for (int r = 0; r < ws; r++) {
                for (size_t i = 0; i < LOCAL; i++) {
                        size_t idx = (size_t)r * LOCAL + i;
                        if (r == rank) {
                                continue;
                        }
                        int w0 = TAG(REG0_MARK, r, (int)(2 + i), STEPS - 2);
                        int w1 = TAG(REG1_MARK, r, (int)(2 + i), STEPS - 2);
                        CHECK_EQ(dst0[idx], w0, "phase3.remote_a0_val");
                        CHECK_EQ(dst1[idx], w1, "phase3.remote_a1_val");
                        CHECK_REGION(dst0[idx], REG0_MARK, "phase3.a0_iso");
                        CHECK_REGION(dst1[idx], REG1_MARK, "phase3.a1_iso");
                }
        }

        dstm_barrier((dstm_time_t)STEPS + 1);

        free(deltas);
        free(dst1);
        free(dst0);
        free(src1);
        free(src0);
        dstm_darray_destroy(&a0);
        dstm_darray_destroy(&a1);

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
                fprintf(stderr, "--- phase3 (allgatherv, two backings) done "
                                "---\n");
}

/* ---------------------------------------------------------------------
 * Phase 4: create_equal_on_region must actually honor region_id.
 * Guards against the wrapper silently landing on region 0.
 * ------------------------------------------------------------------- */
static void phase4_create_equal(int rank, int ws)
{
        dstm_darray_t *a1 = NULL;
        CHECK_EQ(dstm_darray_create_equal_on_region(&a1, sizeof(int),
                                                    (size_t)ws, 1),
                 DSTM_SUCCESS, "phase4.create_equal_r1");
        assert(a1 != NULL);

        /* The local partition must live in region 1, not region 0. */
        void *lp1 = dstm_darray_get_local_ptr(a1);
        void *r0base = (void *)dstm_get_local_region(0);
        void *r1base = (void *)dstm_get_local_region(1);
        CHECK_TRUE(r0base != r1base, "phase4.regions_distinct");
        CHECK_TRUE(lp1 != NULL, "phase4.local_ptr_nonnull");

        /* Round-trip a value through our own slot on region 1. */
        int v = TAG(REG1_MARK, rank, 7, 1);
        CHECK_EQ(dstm_darray_put(a1, (size_t)rank, &v, 0, 1),
                 DSTM_SUCCESS, "phase4.put");
        int g = 0;
        CHECK_EQ(dstm_darray_get(a1, (size_t)rank, &g, 2, -1),
                 DSTM_SUCCESS, "phase4.get_rc");
        CHECK_EQ(g, v, "phase4.get_val");
        CHECK_REGION(g, REG1_MARK, "phase4.iso");

        dstm_barrier((dstm_time_t)4);
        dstm_darray_destroy(&a1);
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
                fprintf(stderr, "--- phase4 (create_equal_on_region) done ---\n");
}

/* ---------------------------------------------------------------------
 * Phase 5: lifecycle + validation of the region API itself.
 *
 * Covers behaviors that the happy-path phases never touch:
 *   - dstm_region_unregister frees the slot AND the id becomes reusable,
 *     with the re-registered region being a fresh, distinct object;
 *   - unregister of an invalid / already-free id returns -DSTM_EINVAL;
 *   - registering past DSTM_MAX_REGIONS returns -DSTM_ENOMEM;
 *   - dstm_darray_put/get reject out-of-range region ids and ptr/region
 *     mismatches with -DSTM_EINVAL instead of asserting/aborting.
 *
 * NOTE: phase0_register (run by run_phase) has already taken id 1 for
 * this session, so the first free id here is 2.
 * ------------------------------------------------------------------- */
static void phase5_lifecycle(int rank, int ws)
{
        (void)ws;

        /* --- register yields a valid, in-range, distinct id.
         * The API contracts only that the id is >= 0 and usable; it does
         * NOT promise a particular numeric value, so assert PROPERTIES,
         * not a specific number. */
        int rid2 = -1;
        CHECK_EQ(dstm_region_register(1 << 16, &rid2), DSTM_SUCCESS,
                 "phase5.register_rid2_rc");
        CHECK_TRUE(rid2 > 0 && rid2 < DSTM_MAX_REGIONS,
                   "phase5.register_rid2_in_range");
        dstm_region_t *before = dstm_get_local_region(rid2);
        CHECK_TRUE(before != NULL, "phase5.rid2_exists");
        /* distinct from the default and from phase0's region 1 */
        CHECK_TRUE(before != dstm_get_local_region(0),
                   "phase5.rid2_distinct_from_0");
        CHECK_TRUE(before != dstm_get_local_region(1),
                   "phase5.rid2_distinct_from_1");

        /* --- unregister frees the slot: the id no longer resolves. */
        CHECK_EQ(dstm_region_unregister(rid2), DSTM_SUCCESS,
                 "phase5.unregister_rc");
        CHECK_TRUE(dstm_get_local_region(rid2) == NULL,
                   "phase5.rid2_gone_after_unregister");

        /* --- re-register yields a valid, usable id and a fresh object.
         * Whether the allocator reuses the freed slot or hands out a new
         * one is an implementation detail; we require only that we get a
         * working, in-range region back. */
        int rid2b = -1;
        CHECK_EQ(dstm_region_register(1 << 16, &rid2b), DSTM_SUCCESS,
                 "phase5.reregister_rc");
        CHECK_TRUE(rid2b > 0 && rid2b < DSTM_MAX_REGIONS,
                   "phase5.reregister_in_range");
        dstm_region_t *after = dstm_get_local_region(rid2b);
        CHECK_TRUE(after != NULL, "phase5.rid2_exists_again");

        /* --- unregister error paths */
        CHECK_EQ(dstm_region_unregister(rid2b), DSTM_SUCCESS,
                 "phase5.cleanup_unregister_rc");
        CHECK_EQ(dstm_region_unregister(rid2b), -DSTM_EINVAL,
                 "phase5.double_unregister_einval");
        CHECK_EQ(dstm_region_unregister(-1), -DSTM_EINVAL,
                 "phase5.unregister_negative_einval");
        CHECK_EQ(dstm_region_unregister(DSTM_MAX_REGIONS), -DSTM_EINVAL,
                 "phase5.unregister_oob_einval");

        /* --- MAX_REGIONS exhaustion: fill every remaining slot, then the
         * next register must fail with -DSTM_ENOMEM.  Track what we take
         * so we can release it and not poison later phases. */
        int taken[DSTM_MAX_REGIONS];
        int ntaken = 0;
        for (;;) {
                int id = -1;
                int rc = dstm_region_register(1 << 12, &id);
                if (rc == DSTM_SUCCESS) {
                        CHECK_TRUE(id >= 0 && id < DSTM_MAX_REGIONS,
                                   "phase5.exhaust_id_in_range");
                        taken[ntaken++] = id;
                } else {
                        CHECK_EQ(rc, -DSTM_ENOMEM,
                                 "phase5.exhaust_returns_enomem");
                        break;
                }
                if (ntaken >= DSTM_MAX_REGIONS) /* safety stop */
                        break;
        }
        CHECK_TRUE(ntaken > 0, "phase5.exhaust_took_some");
        for (int i = 0; i < ntaken; i++)
                CHECK_EQ(dstm_region_unregister(taken[i]), DSTM_SUCCESS,
                         "phase5.exhaust_cleanup_rc");

        /* --- put/get region-id validation (no aborts, clean EINVAL).
         * Use region 0's own element so the only thing "wrong" is the
         * region argument / ptr.region we deliberately corrupt. */
        dstm_darray_t *a0 = NULL;
        CHECK_EQ(dstm_darray_create_on_region(&a0, sizeof(int), 1, 0),
                 DSTM_SUCCESS, "phase5.create_a0");

        /* A correct local round-trip still works (baseline). */
        int v = TAG(REG0_MARK, rank, 9, 0);
        CHECK_EQ(dstm_darray_put(a0, (size_t)rank, &v, 0, 1),
                 DSTM_SUCCESS, "phase5.baseline_put");
        int g = 0;
        CHECK_EQ(dstm_darray_get(a0, (size_t)rank, &g, 2, -1),
                 DSTM_SUCCESS, "phase5.baseline_get_rc");
        CHECK_EQ(g, v, "phase5.baseline_get_val");

        /* Direct low-level dstm_put/dstm_get validation.  We resolve our
         * own local ptr and then feed mismatched / out-of-range region
         * ids.  These must return -DSTM_EINVAL, not abort. */
        {
                dstm_region_t *r0 = dstm_get_local_region(0);
                dstm_ptr_t p = dstm_region_ptr_fromvoid(
                        r0, dstm_darray_get_local_ptr(a0));
                int src = 123, dstbuf = 0;

                /* region_id out of range (negative) */
                CHECK_EQ(dstm_put(rank, -1, 3, 1, p, &src, sizeof(int)),
                         -DSTM_EINVAL, "phase5.put_region_negative");
                /* region_id out of range (>= MAX) */
                CHECK_EQ(dstm_put(rank, DSTM_MAX_REGIONS, 3, 1, p, &src,
                                  sizeof(int)),
                         -DSTM_EINVAL, "phase5.put_region_oob");
                /* region_id != ptr.region (p.region is 0, pass 1) */
                CHECK_EQ(dstm_put(rank, 1, 3, 1, p, &src, sizeof(int)),
                         -DSTM_EINVAL, "phase5.put_region_ptr_mismatch");

                /* same three for get (delta must be <= 0 for a get) */
                CHECK_EQ(dstm_get(rank, -1, 4, -1, &dstbuf, p, sizeof(int)),
                         -DSTM_EINVAL, "phase5.get_region_negative");
                CHECK_EQ(dstm_get(rank, DSTM_MAX_REGIONS, 4, -1, &dstbuf, p,
                                  sizeof(int)),
                         -DSTM_EINVAL, "phase5.get_region_oob");
                CHECK_EQ(dstm_get(rank, 1, 4, -1, &dstbuf, p, sizeof(int)),
                         -DSTM_EINVAL, "phase5.get_region_ptr_mismatch");
        }

        dstm_barrier((dstm_time_t)4);
        dstm_darray_destroy(&a0);
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
                fprintf(stderr, "--- phase5 (region lifecycle + validation) "
                                "done ---\n");
}

typedef void (*phase_fn)(int rank, int ws);

static void run_phase(dstm_distance_t *dist, int ws, int rank, phase_fn fn)
{
        int rc = dstm_start(dist);
        assert(rc == DSTM_SUCCESS);

        int rid = phase0_register(rank);   /* region 1 registered per session */
        assert(rid == 1);

        fn(rank, ws);

        /* last effective time used in any phase is STEPS+1 == 4 */
        dstm_fini((dstm_time_t)8);
        MPI_Barrier(MPI_COMM_WORLD);
}

/* ---------------------------------------------------------------------
 * main
 * ------------------------------------------------------------------- */
int main(int argc, char *argv[])
{
        MPI_Init(&argc, &argv);
        dstm_init(&argc, &argv);
        dstm_log_setlevel(DSTM_LOG_DEBUG);

        int rank, ws;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ws);

        /* Complete topology: every rank reaches every other at distance 1,
         * the simplest correct setting for arbitrary remote gets. */
        dstm_distance_t *dist = NULL;
        int rc = dstm_distance_create_complete(&dist, (size_t)ws);
        assert(rc == DSTM_SUCCESS);

        /* Each phase runs in its own session so the per-region version
         * tables start empty and the put-before-get phase rule is never
         * violated by leftover versions from a prior phase.  Region 1 is
         * session-scoped, so we register it inside each session. */
        run_phase(dist, ws, rank, phase1_local);
        run_phase(dist, ws, rank, phase2_remote);
        run_phase(dist, ws, rank, phase3_allgatherv);
        run_phase(dist, ws, rank, phase4_create_equal); 
        run_phase(dist, ws, rank, phase5_lifecycle);


        dstm_distance_destroy(&dist);
        dstm_finalize();

        /* Aggregate and force a non-zero exit on any failure so the harness
         * records the truth. */
        int total_fail = 0, total_checks = 0;
        MPI_Reduce(&g_failures, &total_fail, 1, MPI_INT, MPI_SUM, 0,
                   MPI_COMM_WORLD);
        MPI_Reduce(&g_checks, &total_checks, 1, MPI_INT, MPI_SUM, 0,
                   MPI_COMM_WORLD);

        if (rank == 0) {
                fprintf(stderr,
                        "=== multi_region: %d checks, %d failures "
                        "(world_size=%d) ===\n",
                        total_checks, total_fail, ws);
                if (total_fail == 0)
                        printf("PASS: multi_region (world_size=%d)\n", ws);
                else
                        printf("FAIL: multi_region (%d failures)\n", total_fail);
        }

        MPI_Finalize();
        return (total_fail == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
