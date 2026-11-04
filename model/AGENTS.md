# Model Context

- `nick_dstm.v` is a Rocq model.
- The abstract semantics marks a version readable after all possible writes finish; the status semantics approximates this using monotonic logical-time heartbeats and channel lag.
- A valid program is currently characterized as push (remote write in the future, local read now) or pull (local write now, remote read in the past). Deadlock freedom means every reachable state is finished or can step.
- Channel lags may vary. Generalizing push/pull beyond one timestep depends on formalizing heartbeat frontiers; see `TODO.md` for the dependency-ordered plan.
- Runtime behavior is documented in `../src/` and `../include/`, especially heartbeat and version-state logic in `../src/ops.c`.
- Examples of the model in use can be found at: https://github.com/Argonne-TimeWeave/dstm/tree/summer26
- Plain English description of the programming/memory model can be found at context.tex
 - context.tex is the first work that describes this programming model so it is slightly out of date but still representative and accurate.

## Working Rules

- Do not edit the generated `Makefile`; keep `Makefile.conf` portable to the active Rocq installation.
- Do not edit context.tex
- Do not edit any file outside of this dir
- Keep the module-scoped stdpp compatibility alias used by `lookup_insert_eq`.
- After Rocq changes, run `make` and `make validate` in this directory.
- Before making any changes, understand the scope and logic of the current state.
- Be skeptical of prior proofs and definitions, always check back to DSTM source (and context.tex).

# Response Preference

- Short, concise, and to the point responses. 
- If unconfident, terminate and ask a clarifying question.
- Always give a short description of changes made in the chat box.
