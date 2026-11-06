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

# Response Preference

- Short, concise, and to the point responses. 
- If unconfident, terminate and ask a clarifying question.
- Always give a short description of changes made in the chat box.
- For model changes, relate the description explicitly to both the prose semantics in `context.tex` and the implementation in `../src/` or `../include/`.
