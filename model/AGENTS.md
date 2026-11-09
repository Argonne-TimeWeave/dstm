# Model Context

- `nick_dstm.v` is a Rocq model.
- The abstract semantics marks a version readable after all possible writes finish; the status semantics approximates this using monotonic logical-time heartbeats and channel lag.
- A valid program has nondecreasing per-node originating times and is characterized as push (remote write in the future, local read now) or pull (local write now, remote read in the past). Deadlock freedom means every reachable state is finished or can step.
- "Monotonic" means nondecreasing, not strictly increasing; multiple operations may share an originating time, as in `ring_program`.
- Channel lags may vary. Generalizing push/pull beyond one timestep depends on formalizing heartbeat frontiers; see `TODO.md` for the dependency-ordered plan.
- Runtime behavior is documented in `../src/` and `../include/`, especially heartbeat and version-state logic in `../src/ops.c`.
- Examples of the model in use can be found at: https://github.com/Argonne-TimeWeave/dstm/tree/summer26
- Plain English description of the programming/memory model can be found at context.tex
 - context.tex is the first work that describes this programming model so it is slightly out of date but still representative and accurate.

## Assumption Register

Treat assumptions as part of the model/runtime contract, not as reasons to delete contrary evidence. Each assumption must state its scope, evidence or provenance, model consequence, and whether the runtime enforces it. When an assumption resolves a concern, move or mark only the affected claim as **Resolved under assumption**, retain the original evidence and classification, and record the residual documentation or enforcement gap.

- **Single-threaded client calls:** Each rank invokes the public DSTM API from one client thread; while that thread is blocked in `dstm_barrier` or `dstm_fini`, it cannot issue another operation. **Provenance:** confirmed project design assumption; the public header does not currently state it. **Model consequence:** barrier entry provides client-side issue quiescence, so the model need not account for concurrent post-entry API calls. **Enforcement:** assumed application discipline, not runtime-checked. This does not establish remote delivery/application completion or validate the barrier generation/frontier.
- **One logical timeline and one current time per rank:** All ranks use one shared simulation timeline, and a rank has no internally virtualized logical processes with distinct current times (`context.tex:318-326`). **Model consequence:** one node program and heartbeat frontier per rank are adequate abstractions. **Enforcement:** design scope, not runtime-checked.
- **Static ranks, ownership, and communication scope:** Ranks are static and uniquely identified, state ownership is statically partitioned (`context.tex:318-326`), and the session-global distance object covers every rank pair used by every live structure (`../include/dstm.h:388-405`). **Model consequence:** `owner` and `comm_graph` remain fixed during execution. **Enforcement:** rank/ownership stability is design scope; graph coverage is a documented caller precondition.
- **Nondecreasing originating times:** Each rank issues operations in nondecreasing logical time (`context.tex:387-393`); equal times are allowed. **Model consequence:** `monotonic_program` and reachable heartbeat frontiers may be assumed only for valid programs. **Enforcement:** heartbeat updates ignore non-increases, but the runtime does not reject a decreasing operation time.
- **Operation/graph compatibility:** Remote operation deltas respect the installed directed minimum readability and maximum discardability bounds. The installed matrix must cover all traffic (`../include/dstm.h:388-405`), and combined matrices retain the minimum and maximum bounds (`../include/dstm.h:295-315`). **Model consequence:** `operation_allowed` is a program proof obligation used by readability and discardability safety. **Enforcement:** matrix coverage and delta compatibility are documented/application obligations; basic operations currently check only delta sign and remote zero.
- **Regions registered before traffic:** Every region is registered before its rank or any neighbor issues operations on it (`../include/dstm.h:1227-1235`). **Model consequence:** relevant neighbor-clock entries are assumed to exist rather than silently starting late at zero. **Enforcement:** documented caller precondition, not fully runtime-checked.
- **Timestamp arithmetic remains in range:** Runtime signed timestamp addition/subtraction is assumed not to overflow or underflow. **Model consequence:** Rocq natural-number arithmetic represents logical timestamp calculations only under this bounded-runtime premise. **Enforcement:** not runtime-checked or presently documented as a numeric bound.

## Working Rules

- Do not edit the generated `Makefile`; keep `Makefile.conf` portable to the active Rocq installation.
- Do not edit any file outside of this directory; reading `../src/` and `../include/` to validate the model is allowed and expected.
- Keep the module-scoped stdpp compatibility alias used by `lookup_insert_eq`.
- After Rocq changes, run `make` and `make validate` in this directory.
- Before making any changes, understand the scope and logic of the current state.
- Be skeptical of prior proofs and definitions, always check back to DSTM source (and context.tex).
- Avoid re-proving/defining things many times. When given a chance to modularize or generalize existing code, take it.
- Treat `context.tex` as read-only design context. Reconcile model changes with it, but do not clarify or modernize the document itself.
- Mark a TODO complete only after its assumption is integrated into the relevant top-level definition and proved through the reachable-state invariant, not merely defined locally.
- After every model change, explain what changed and why, with `file:line` pointers to the relevant model definitions/proofs, `context.tex`, and runtime source or headers.
- Record every weak, ambiguous, inconsistent, or insufficiently formal runtime behavior discovered during modeling in `CONCERNS.md`. Include precise `file:line` evidence, the model property it affects, and whether it is an implementation bug, documentation mismatch, or unresolved design question. Keep resolved entries with their resolution rather than deleting them.
- Generalize and combine duplicated, overlapping, or unnecessarily narrow `CONCERNS.md` entries when possible, while preserving every distinct technical claim, classification, model impact, precise runtime/model evidence, and resolution.
- Before adding or closing a concern, check the Assumption Register. Do not use an assumption outside its stated scope, and do not treat an unenforced assumption as runtime evidence. Add newly confirmed assumptions here with provenance and update affected concerns with their exact resolution status.

# Response Preference

- Short, concise, and to the point responses. 
- If unconfident, terminate and ask a clarifying question.
- Always give a short description of changes made in the chat box.
- For model changes, relate the description explicitly to both the prose semantics in `context.tex` and the implementation in `../src/` or `../include/`.
- At the end of every change, explain the complete result in simple terms. Cover the big picture, each new or modified definition and theorem, how the proof works case by case, what the result guarantees, what it does not guarantee, and the validation performed. Include `file:line` references, define technical terms such as safety, refinement, and stuttering in plain language, and clearly distinguish proved model properties from runtime assumptions or unresolved concerns.
