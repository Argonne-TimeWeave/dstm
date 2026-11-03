# Model Context

- `nick_dstm.v` is a Rocq model.
- The abstract semantics marks a version readable after all possible writes finish; the status semantics approximates this using monotonic logical-time heartbeats and channel lag.
- A valid program is currently characterized as push (remote write in the future, local read now) or pull (local write now, remote read in the past). Deadlock freedom means every reachable state is finished or can step.
- Channel lags may vary. Generalizing push/pull beyond one timestep depends on formalizing heartbeat frontiers; see `TODO.md` for the dependency-ordered plan.
- Runtime behavior is documented in `../src/` and `../include/`, especially heartbeat and version-state logic in `../src/ops.c`.
- Examples of the model in use can be found at: https://github.com/Argonne-TimeWeave/dstm/tree/summer26

## Working Rules

- Do not edit the generated `Makefile`; keep `Makefile.conf` portable to the active Rocq installation.
- Keep the module-scoped stdpp compatibility alias used by `lookup_insert_eq`.
- After Rocq changes, run `make` and `make validate` in this directory.

# Response Preference

- Short, concise, and to the point responses. 
- If unconfident, terminate and ask a clarifying question.
