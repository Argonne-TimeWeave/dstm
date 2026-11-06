# Model and Runtime Concerns

## Open

### Local readability check does not exclude same-time writes

- **Runtime evidence:** `../src/ops.c:104-114` keeps a version writable only while `local_heartbeat_time < v->time`, so equality permits `WRITABLE -> READABLE`. Heartbeats advance only when an operation has a strictly greater issue time at `../src/ops.c:674-679` and `../src/ops.c:746-751`.
- **Prose semantics:** `context.tex:397-406` says readability requires that the owner and every neighbor have progressed past the version so they can no longer issue a write to it. `context.tex:390-393` requires originating times to be monotonic, which permits multiple operations at the same time.
- **Model impact:** A heartbeat equal to the version timestamp cannot exclude a remaining same-time local store. The sound readability premise therefore requires strict local progress, `t < ti`, at `nick_dstm.v:1695-1706`; `status_readability_visible` relies on that inequality at `nick_dstm.v:1770-1838`.
- **Classification:** Implementation/prose mismatch. Either the runtime needs a strict-past check or explicit end-of-writes/barrier evidence must rule out remaining same-time stores.
