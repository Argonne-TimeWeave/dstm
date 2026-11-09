# Model and Runtime Concerns

## Open

### Local readability implementation and model use different strictness

- **Runtime evidence:** `../src/ops.c:114-124` says the local clock must move strictly past the version, but the implemented test only blocks while `local_heartbeat_time < v->time`, so equality permits `WRITABLE -> READABLE`. Heartbeats advance only when an operation has a strictly greater issue time at `../src/ops.c:794-799`, `../src/ops.c:866-871`, and `../src/ops.c:977-981`.
- **Prose semantics:** `context.tex:397-406` says readability requires that the owner and every neighbor have progressed past the version so they can no longer issue a write to it. `context.tex:390-393` requires originating times to be monotonic, which permits multiple operations at the same time.
- **Model impact:** A heartbeat equal to the version timestamp cannot exclude a remaining same-time local store. The sound readability premise therefore requires strict local progress, `t < ti`, in `readable_safe` at `nick_dstm.v:1779-1790`; `status_readability_visible` relies on that inequality at `nick_dstm.v:1938-2003`.
- **Classification:** Implementation/prose mismatch. Either the runtime needs a strict-past check or explicit end-of-writes/barrier evidence must rule out remaining same-time stores.

### Heartbeats do not order earlier operations before later clock evidence

- **Runtime evidence:** A time advance synchronously sends heartbeat messages before creating or sending the current PUT/GET request (`../src/ops.c:794-829`, `../src/ops.c:866-916`), but PUT payloads use `MPI_Isend` while heartbeats use `MPI_Send`, and every message receives a fresh, distinct tag (`../src/msg.c:168-170`, `../src/msg.c:215-233`, `../src/request.c:49-57`). The receiver probes with `MPI_ANY_TAG` and updates neighbor time immediately from a received operation or heartbeat header (`../src/msg.c:282-288`, `../src/ops.c:583-602`) before status updates run ahead of request application (`../src/ops.c:713-744`).
- **Model impact:** The status model atomically advances a heartbeat frontier for the current head operation and proves all pending operations have no earlier issue time. Runtime receipt of a later-time heartbeat does not, by itself, establish that earlier differently-tagged PUTs have been received and applied. Thus `status_readability_visible` is a conditional safety theorem for the modeled ordering premise, not yet a full runtime correspondence result.
- **Classification:** Unresolved design question / insufficiently formal ordering contract. Supply a common-tag FIFO/channel-order argument, sequence/acknowledgment evidence, or explicit end-of-writes/barrier evidence before treating a received heartbeat as closing all earlier writes.

### Directed distance orientation is inconsistent in heartbeat readability

- **Runtime evidence:** Heartbeat fanout and operation issuance use the directed cell `(sender, destination)` (`../src/ops.c:37-50`), while the destination's readability test looks up `(local owner, potential sender)` (`../src/ops.c:97-111`). User matrices are explicitly allowed to be asymmetric (`../include/dstm.h:265-270`).
- **Model impact:** `operation_allowed`, generalized push/pull, and `status_readability_visible` consistently use `(issuer, owner)`. They cannot justify a runtime check of `(owner, issuer)` for an asymmetric graph. Directed distances must not be silently symmetrized.
- **Classification:** Implementation/API semantic inconsistency. Define one direction convention and use it for operation validation, heartbeat fanout, readability, and discardability.

### Operation deltas are not checked against installed distance bounds

- **Runtime evidence:** PUT/APPEND reject only negative deltas and remote zero (`../src/ops.c:784-792`, `../src/ops.c:969-975`); GET rejects only positive deltas and remote zero (`../src/ops.c:856-864`). None checks the actual delta against `dstm_distance_get` or `dstm_distance_get_discard`, although readability treats the matrix as a lower write bound (`../src/ops.c:89-111`) and the public header describes minimum/maximum bounds (`../include/dstm.h:295-305`).
- **Model impact:** `operation_allowed` explicitly connects remote stores to the directed readability bound and remote loads to both the readable minimum and discard maximum. The reachable-state visibility proof and any future discardability proof apply only to programs satisfying those premises; runtime sign checks alone enforce neither bound. In particular, without the maximum upper bound on a load's look-back, a discard heartbeat frontier cannot exclude a farther-past pending read.
- **Classification:** Unenforced API contract / implementation weakness. Validate deltas or clearly expose the matrix relation as a required application proof obligation.

### Local discardability has no enforceable end-of-reads evidence

- **Runtime evidence:** Local GET permits any nonpositive delta (`../src/ops.c:856-864`), so an operation issued at a later time may still target an arbitrarily old local version. The discard check scans only already-issued requests (`../src/ops.c:135-152`) and otherwise permits retirement once `local_heartbeat_time >= v->time` (`../src/ops.c:179-180`), which cannot exclude a future local GET into the past.
- **Prose semantics:** `context.tex:403-406` requires progress past `t + dist` before discardability, while `context.tex:414-416` requires discarding to preserve semantics. For the local rank, distance zero and monotonic originating times do not constrain how far a GET may look back.
- **Model impact:** `discard_safe` requires explicit `end_of_reads` evidence for the owner and uses a strict `discard_lag` heartbeat frontier only for remote readers (`nick_dstm.v:1792-1803`). Treating local heartbeat progress as sufficient would invalidate `status_discardability_no_pending_read` (`nick_dstm.v:1892-1936`).
- **Classification:** Potential implementation correctness bug / missing API contract. Enforce a local retention window or provide barrier/end-of-reads evidence that prevents future local reads of the version before eviction.

### Barrier completion does not establish a validated closure frontier

Single-threaded client execution supplies issue quiescence, as recorded under **Resolved under assumption**. Barrier completion still requires all previously issued writes to have been received and applied and all ranks to agree on the barrier generation, phase, and closure frontier; the current runtime does not establish those remaining facts unconditionally.

- **Delivery/application runtime evidence:** Round 1 treats OUT_PUT/OUT_APPEND completion as drained (`../src/barrier.c:87-100`), but those requests finish on local `MPI_Test` send completion (`../src/ops.c:497-507`, `../src/ops.c:418-425`). Tokens travel only to the ring successor (`../src/barrier.c:73-85`) while writes can target arbitrary ranks. After the token, `barrier_all_drained` bypasses all neighbor and local-clock readability checks (`../src/barrier.c:156-174`, `../src/ops.c:89-125`). The progress loop probes only one incoming message before status promotion and request application (`../src/ops.c:709-744`).
- **Delivery/application model impact:** The runtime does not yet supply sound explicit end-of-writes evidence for an unconditional barrier readability transition. A message can remain in transit when a version is promoted, so concrete status correspondence and ring termination remain unproved.
- **Delivery/application classification:** Potential implementation correctness bug. Require destination acknowledgments, per-peer sequence closure, or a receive/application fence before bypassing heartbeat checks.
- **Protocol runtime and documentation evidence:** Receipt only increments a scalar counter and logs `last_time` (`../src/ops.c:685-693`); the DFA infers token phase from local state (`../src/barrier.c:142-217`). The public contract requires equal `last_time` values (`../include/dstm.h:442-444`), but no comparison is performed. Public documentation calls the algorithm two-pass (`../include/dstm.h:431-440`) while runtime/internal documentation describes three rounds (`../src/barrier.c:19-58`, `../include/internal/request.h:51-79`).
- **Protocol model impact:** An explicit barrier fact needs a generation, phase, and agreed closure frontier. The current token counter is insufficient formal evidence for end-of-writes or end-of-reads.
- **Protocol classification:** Insufficiently formal protocol and documentation mismatch.

### Request ordering is not a stable effective-time order

- **Runtime evidence:** The used comparator sorts by base time and then delta with subtraction returned as `int` (`../src/request.c:34-43`), whereas `dstm_time_t` is 64-bit (`../include/dstm.h:348-350`). An effective-time comparator exists at `../src/request.c:25-32` but is not used for the insertions at `../src/ops.c:644-659`, `../src/ops.c:829`, `../src/ops.c:916`, and `../src/ops.c:1009`. Equal keys have no sequence tie-breaker; APPEND order is therefore not a stable issue order.
- **Model impact:** The proof model intentionally permits same-time operations but executes each node's list order. Runtime overwrite/append results for equal keys can differ, and arithmetic truncation can violate even the documented departure order. This does not invalidate the all-entry pending-write scan, but it blocks stronger value-correspondence claims.
- **Classification:** Implementation weakness / underspecified nondeterminism. Use comparison without subtraction and specify or encode equal-key ordering if observable values depend on it.

### Rank-pair graph derivation loses heterogeneous pull bounds and direction

- **Runtime evidence:** Graph derivation reduces cross-rank edge weights to a minimum and symmetrizes opposing directions (`../src/dgraph.c:1235-1248`, `../src/dgraph.c:1272-1279`, `../src/dgraph.c:1309-1333`). Ordinary matrices reuse their single value as both readable and discard bounds; only combining separate matrices preserves a maximum (`../include/dstm.h:295-315`). Graph operations forward caller deltas without binding a specific edge weight to them (`../src/dgraph.c:986-1021`, `../src/dgraph.c:1133-1174`).
- **Model impact:** A minimum is conservative for future push writes, but heterogeneous pull reads require the maximum look-back for safe retention. The model therefore keeps directed readable/discard lags separate and does not claim generalized pull implies runtime discardability.
- **Classification:** Implementation/design limitation. Preserve directed per-rank minimum and maximum reductions from the underlying edges, or restrict/document supported pull patterns.

### Timestamp arithmetic relies on an unchecked boundedness premise

- **Runtime evidence:** Effective timestamps and frontier expressions use unchecked signed addition/subtraction (`../src/ops.c:64`, `../src/ops.c:110`, `../src/ops.c:175`, `../src/ops.c:257`, `../src/ops.c:289`).
- **Model impact:** Rocq timestamps are naturals with total arithmetic. Runtime correspondence therefore depends on the bounded-arithmetic assumption in `AGENTS.md`.
- **Classification:** Open undocumented/unvalidated premise. State safe numeric bounds in the API or check overflow and underflow in the runtime.

## Resolved under assumption

### Barrier entry is issue-quiescent

- **Runtime evidence:** `dstm_barrier` appends its request after existing local requests (`../src/barrier.c:238-256`), and round 1 waits for currently tracked outgoing operations before setting `barrier_all_drained` (`../src/barrier.c:87-100`, `../src/barrier.c:124-174`). The public contract describes draining in-flight requests but does not state the single-threaded client precondition (`../include/dstm.h:431-448`).
- **Assumption:** Public DSTM calls on each rank are made by one client thread; while that thread is blocked in `dstm_barrier`, it cannot issue another operation (`AGENTS.md`, **Single-threaded client calls**).
- **Model impact:** Barrier entry is valid client-side issue-quiescence evidence. This does not prove message delivery, request application, read completion, or agreement on a closure frontier; those distinct claims remain open above.
- **Original classification and resolution:** Former unresolved design/API-contract question; resolved for the accepted execution scope. Residual documentation mismatch: the public API should state that DSTM calls are single-threaded and that no operation may be issued concurrently with a barrier.

### Neighbor clocks are initialized before traffic

- **Runtime evidence:** Missing `(rank, region)` clock entries silently return time zero and callers ignore the error (`../src/region.c:461-476`, `../src/ops.c:104-110`, `../src/ops.c:169-175`); late region registration is explicitly unsafe, and the API requires registration before local or neighboring operations (`../include/dstm.h:1229-1235`).
- **Assumption:** Every region is registered before its rank or any neighbor issues operations on it (`AGENTS.md`, **Regions registered before traffic**).
- **Model impact:** The status proof may assume an initialized frontier entry for each relevant active node. Under the precondition, the silent-zero fallback is unreachable in modeled executions.
- **Original classification and resolution:** Documented precondition plus implementation robustness weakness; resolved for conforming programs under the documented pre-registration assumption. Ignoring lookup failures remains a defensive-programming weakness outside that scope.

## Closed
