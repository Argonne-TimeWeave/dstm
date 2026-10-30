# TODO

Items explicitly identified as future work in `nick_dstm.v`, ordered by dependency:

## 1. Foundations

- Formalize and use the assumption that each node issues instructions at monotonically increasing logical times (`nick_dstm.v:21`, `nick_dstm.v:1570`).
- Revisit the memory/status model, including whether to model `Discardable` versions (`nick_dstm.v:123`).
- Refine the model with message travel time and relate it to read/write limits (`nick_dstm.v:1583`).

## 2. Heartbeats and readability

Depends on **Foundations**.

- Define heartbeats as monotonic logical-time frontiers and prove reported times bound future instructions (`nick_dstm.v:1637`).
- Model end-of-writes signaling and barriers so the runtime can determine when a version is readable (`nick_dstm.v:25`, `nick_dstm.v:32`).
- Generalize push/pull from one timestep to each edge's channel lag `d` (`nick_dstm.v:1638`).
- Prove heartbeat time plus `d` makes status readability imply abstract visibility (`nick_dstm.v:1638`).

## 3. Semantic correspondence

Depends on **Heartbeats and readability**.

- Prove that the status semantics implements the abstract semantics for monotonic programs; distinguish safety from prompt/eventual readability (`nick_dstm.v:1566`, `nick_dstm.v:1573`, `nick_dstm.v:1591`).

## 4. Progress and correctness

Abstract proofs depend on **Foundations**; concrete status proofs also depend on **Semantic correspondence**.

- Prove that every valid push or pull program has a deadlock-free terminating schedule (`nick_dstm.v:24`, `nick_dstm.v:336`, `nick_dstm.v:1586`, `nick_dstm.v:1595`).
- Prove deadlock freedom for the concrete status semantics (`nick_dstm.v:334`).
- Decide whether to prove that every reachable ring state has a terminating continuation (`ring_can_always_terminate`, `nick_dstm.v:1511`).
- Prove ring result correctness using a preserved state invariant (`ring_correct`, `nick_dstm.v:1517`).

## 5. Expressiveness

Depends on **Progress and correctness** if translated programs must inherit termination or deadlock-freedom results.

- Define conservative synchronization, translate its programs to DSTM, and prove semantic preservation (`nick_dstm.v:1587`).
