# TODO

Items explicitly identified as future work in `nick_dstm.v`, ordered by dependency:

## 1. Foundations

- [x] Keep push and pull as separate, non-mixed valid program classes in `valid_program`: the whole program satisfies `push_program` or the whole program satisfies `pull_program`.
- [x] State and prove progress results separately for push-only and pull-only programs rather than assuming one proof covers both.
  - [x] Prove the shared abstract head-store prerequisite and specialize it separately: every reachable push head store is a remote enabled step, while every reachable pull head store is a local enabled step.
  - [x] Require every load's exact version to have a program store producer, integrate this into `valid_program`, prove the resulting pending-load availability invariant through reachable abstract states, and prove conditional head-load progress once all possible writes have executed.
  - [x] Establish the dependency argument ensuring required producers can execute; the existing push-only and pull-only timestamp rules suffice without strengthening `valid_program`.
    - [x] Prove the reusable ranking prerequisite already induced by push-only and pull-only timestamp rules: every matching producer store has a strictly smaller originating time than its load, reachable node pools preserve nondecreasing instruction times, and any still-pending producer lies behind a head with a strictly smaller originating time than the blocked load.
    - [x] Use finiteness to choose a minimum-time head and derive an enabled head store or visible load; strict descent rules out a blocked minimum load, so `valid_program` needs no additional dependency condition.
- [x] Formalize and use the assumption that each node issues instructions at monotonically increasing logical times (`monotonic_program`, `monotonic_program_status_reachable`).
- [ ] Complete the memory/status lifecycle work begun in `nick_dstm.v`:
  - [x] Represent logical `Discardable` tombstones in the abstract semantics, guarded by the absence of pending reads, and prove retired versions preserve their value and cannot become writable again through reachable states (`retired_version_reachable`).
  - [x] Distinguish the runtime's directed minimum/readability distance from its directed maximum/discardability distance in `comm_graph`, constrain both to the same reachable channels with minimum no greater than maximum, and integrate the minimum bound through operation validity, status readability, the ring graph, and reachable heartbeat-frontier proofs.
  - [x] Add a sound `READABLE -> DISCARDABLE` rule to the heartbeat/status semantics using a separate discard-distance bound and explicit end-of-reads evidence. Remote loads carry both the minimum look-back and maximum retention bound; `discard_safe` requires local closure and permits strict maximum-distance heartbeat evidence only for remote readers; the valid-program reachable invariant proves this excludes pending reads.
  - [x] Relate logical tombstones to runtime eviction and prove that removing a discardable version preserves observable behavior.
- Refine the model with message travel time and relate it to read/write limits (`nick_dstm.v:1583`).

## 2. Heartbeats and readability

Depends on **Foundations**.

- Model end-of-writes signaling and barriers so the runtime can determine when a version is readable (`nick_dstm.v:25`, `nick_dstm.v:32`).
- Model end-of-reads signaling and the runtime's barrier evidence before using the maximum distance for concrete discardability; heartbeat time alone does not prove no same-time local operation remains.
- [x] Generalize push/pull from one timestep to each directed edge's readability lag `d`, integrate the generalized forms into `valid_program`, and prove valid programs carry operation validity through reachable status states and the readability visibility theorem.
- [x] Prove that the directed readability bound plus monotonic heartbeat frontiers makes the status readability premise imply abstract visibility (`status_readability_visible`); this requires strict local progress past the version timestamp.

## 3. Semantic correspondence

Depends on **Heartbeats and readability**.

- [x] Prove the safety direction: every reachable status execution of a valid monotonic program erases to an abstract execution, with heartbeat-only steps treated as stuttering (`valid_status_reachable_abstract`).
- Prove the promptness/completeness direction:
  - The evidence-conditional single-transition result is formalized: every abstract load or store step lifts directly, and abstract readability or discardability lifts when its respective concrete evidence is explicitly available (`abstract_step_lifts_with_evidence`).
  - [ ] Prove that sufficient heartbeat or barrier evidence eventually becomes available; this remains dependent on sound end-of-writes signaling, delivery/application ordering, and validated barrier closure.

## 4. Progress and correctness

Abstract proofs depend on **Foundations**; concrete status proofs also depend on **Semantic correspondence**.

- Prove separately that every valid push-only program and every valid pull-only program has a deadlock-free terminating schedule (`nick_dstm.v:24`, `nick_dstm.v:336`, `nick_dstm.v:1586`, `nick_dstm.v:1595`). Do not rely on mixed push/pull programs in the current scope.
- Prove deadlock freedom for the concrete status semantics (`nick_dstm.v:334`).
- Adapt the ring termination construction to the concrete status semantics: schedule heartbeats, prove the required versions become readable, retire them as discardable when safe, and account for runtime-style eviction.
- [x] Reconcile same-time local writes with readability: nondecreasing issue times allow another local write at the announced timestamp, so the local readability check is now strict and the visibility proof covers reachable legal node pools.
- Reconcile runtime local reads with future discardability; because local loads may target any past version, no heartbeat value alone excludes a later-issued local read, so the runtime needs enforceable end-of-reads/barrier evidence.
- Prove concrete ring termination only after the status semantics includes sufficient end-of-writes/end-of-reads or barrier evidence; the current abstract ring schedule does not establish those runtime facts.
- Decide whether to prove that every reachable ring state has a terminating continuation (`ring_can_always_terminate`, `nick_dstm.v:1511`).
- Prove ring result correctness using a preserved state invariant (`ring_correct`, `nick_dstm.v:1517`).

## 5. Expressiveness

Depends on **Progress and correctness** if translated programs must inherit termination or deadlock-freedom results.

- Define conservative synchronization, translate its programs to DSTM, and prove semantic preservation (`nick_dstm.v:1587`).
