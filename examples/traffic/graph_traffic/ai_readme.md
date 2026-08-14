IT WORKS ON A FIXED 4-CONNECTED GRID
    NEED TO REPLACE city_build WITH A DIFFERENT LOADER that produces:
        row_ptr, dst, esrc, vtx_off (per-vertex adjacency sorted ascending)
    ALSO NEEDS: reverse edges must exist -- (v,u) whenever (u,v) does

FUTURE OPTIMIZATION only send state information that is new to that neighbor
        (local changes + anything you learned by relay that they haven't seen)

ROUTING IS NOW DIJKSTRA OVER THE REPLICA, NOT GREEDY MANHATTAN

# graph_traffic — routing under diffused, delayed congestion

Each rank holds a congestion replica of the **entire** graph. Each tick it
refreshes the entries it owns, then ships the whole replica — together with
the agents physically crossing — along every outgoing edge. Neighbours merge
what arrives. News about a distant edge therefore reaches a rank only after
as many ticks as there are hops between them, and every routing decision is
made against that stale view.

## Build and run

```sh
./autogen.sh && ./configure && make
mpirun -n 24 ./examples/traffic/graph_traffic/graph_traffic \
       --grid 20 --steps 200 --cars 0 --rate 8 --cap 3 --quiet
```

Any rank count from 1 up to the vertex count works.

| flag                | default | meaning                              |
|---------------------|---------|--------------------------------------|
| `--grid G`          | 12      | G x G 4-connected grid               |
| `--steps N`         | 60      | ticks                                |
| `--cars M`          | 64      | agents spawned per rank at t=0       |
| `--rate R`          | 0       | agents spawned per rank *per tick*   |
| `--cap C`           | 8       | max cars per edge per tick           |
| `--seed S`          | 42      | RNG seed                             |
| `--replica-mode M`  | stale   | `stale` \| `oracle` \| `blind`       |
| `--verbose`         | off     | per-tick progress from rank 0        |

Prefer `--cars 0 --rate R` for measurement. A single t=0 burst is one
decaying shockwave; congestion information is most valuable in steady state,
and a continuous spawn rate is the only way to reach one.

## Replica modes

Three ways of populating the replica, so the cost of staleness can be
*measured* rather than assumed. All three ship byte-identical channel
messages and run the identical tick sequence — only the contents of the
replica consulted by the router differ, so communication volume never
confounds the comparison.

| mode     | replica contents                    | age of consulted entry |
|----------|-------------------------------------|------------------------|
| `stale`  | diffusion by relay — the real model | `max(1, hopdist)`      |
| `oracle` | `MPI_Allgatherv` every tick         | exactly `1.00`         |
| `blind`  | congestion ignored, `cost == 1`     | n/a                    |

Expected ordering on mean trip duration:

```
blind  >=  stale  >=  oracle
```

with **both** inequalities strict. If `stale ≈ blind` the congestion signal
is not reaching the routing decision. If `stale ≈ oracle` the staleness costs
nothing and there is no phenomenon to study. Either collapse means the
*configuration* is wrong, and no parameter tuning is meaningful until it is
fixed. The usual cause is load: the gap is zero on an empty network (nothing
to route around) and zero at gridlock (nowhere to route to), and maximal in
between — aim for 15–35% blocked.

Oracle is cheap because vertices are block-partitioned: rank `r` owns exactly
the contiguous global edge range `[gslot_base, gslot_base + nedge_local)`, so
an allgatherv of owned slices reconstructs the full array. By the
single-writer property that is provably identical to a max-by-stamp
allreduce, without needing `MPI_Op_create`.

## Buffer layout

One buffer per directed edge per tick, sized once at startup and constant
for the run:

```
+-----------------------------------------------+
| chan_header                                   |  40 B
|   magic, tick, n_state, n_cars, cap,          |
|   state_off, cars_off, src_vertex, dst_vertex |
+-----------------------------------------------+
| cong_entry state[n_state]                     |  8 B each, ALWAYS |E|
|   { float value; int32_t stamp; }             |
+-----------------------------------------------+
| car_rec cars[n_cars],  n_cars <= cap          |  24 B each, bounded
+-----------------------------------------------+
```

`state_off` and `cars_off` are transmitted even though they are constants
for a given build. They cost 8 bytes and convert a whole class of layout and
versioning mismatches into a cleanly rejected message. `magic` doubles as the
"nobody has written this slot yet" test: a freshly seeded edge slot is all
zeroes, so `chan_validate` rejects it without needing a separate valid flag.

## The channel

For directed edge `u -> v`, the buffer lives in the dgraph edge slot
`(v, u)` — owned by **rank(v)**, the receiver:

```
sender    dstm_dgraph_edge_put(g, v, u, buf, t, +1)   -> receiver's version t+1
receiver  dstm_dgraph_edge_get(g, v, u, buf, t,  0)   -> purely local read
```

This is the `arr_traffic` inbox discipline — push into the receiver's future
version, drain at delta 0 — with an edge slot standing in for an inbox cell.
Each slot has exactly one writer and one reader, so no version can be
clobbered by a concurrent write.

`delta` is **1 on every channel**, including edges whose endpoints happen to
share a rank. DSTM only requires this for cross-rank edges (the distance
matrix is built from this same topology with `default_delta = 1`, so adjacent
ranks are exactly 1 apart). Applying it uniformly is a *modelling* choice: one
hop costs one tick everywhere, so diffusion timing does not depend on where
the partition boundary falls.

## The merge

```c
if (incoming.stamp > local.stamp) local = incoming;
```

Exactly one rank — `owner(src(e))` — ever writes a given entry; everyone else
only relays. Two entries carrying the same stamp are therefore bit-identical,
so **conflicts cannot occur**. The join is commutative, associative and
idempotent, which buys:

- MPI delivery order cannot change the result;
- re-reading an unchanged version is harmless (DSTM copies versions forward,
  so this happens constantly);
- dropping a round entirely only adds staleness, never corruption.

Averaging incoming views would destroy all three properties. Don't.

The **car** half is *not* idempotent, and that asymmetry is the one real trap
here. A version nobody wrote this tick still contains last tick's buffer, so
injecting agents from it would duplicate them. `chan_header.tick` carries the
intended arrival tick and gates injection; the state merge runs
unconditionally.

## Per-tick discipline

```c
spawn_cars(rate);      // continuous demand, if --rate > 0
publish_vertices(t);   // creates local version t, emits heartbeats
recv_channels(t);      // merge state, inject arriving agents
route_and_stage(t);    // Dijkstra over the replica, honour the cap
measure_local(t);      // stamp owned entries with tick t
replica_sync_oracle(); // oracle mode only
send_channels(t);      // [header][state][cars] at (t, +1)
```

`publish` is first for the same reason as in `arr_traffic`: it creates local
region version `t`, so the self-region `edge_get` at delta 0 cannot block on a
version that does not exist, and it advances the distributed horizon.

`measure` follows `recv` so a relayed copy of one of our own edges can never
overwrite the authoritative value. The stamp rule already guarantees that;
the ordering makes it obvious to a reader.

Routing precedes measurement, so even a rank's *own* edges are one tick stale
when consulted. That is why oracle reports age `1.00`, not `0.00` — and it
makes the stale/oracle contrast exact: the two differ by precisely the
diffusion delay and nothing else.

## Routing: Dijkstra over the replica

A one-hop greedy rule only ever reads the out-edges of the car's current
vertex — which are, by definition, edges this rank owns and just stamped
itself. The entire diffusion apparatus would be built, shipped, merged, and
never actually read. **Path planning is what makes remote congestion matter.**

Each tick, per distinct destination, a backward Dijkstra over the reverse CSR
produces cost-to-go for every vertex; the car takes the first hop. Trees are
cached per `(destination, tick)`, which pays off heavily under the downtown
destination bias.

```c
cost(e) = 1 + BETA * clamp(cong_effective(e) / cap, 0, CONG_MAX)
```

Free-flow is 1 per hop, so path length and congestion are on the same scale
and `BETA` reads directly as "hops of detour worth avoiding a saturated
edge". The clamp stops any single edge from dominating a path — without it,
congestion cost runs away and cars execute a congestion-fleeing random walk
that is *worse* than routing blind.

Confidence decays with age, toward free-flow:

```c
w    = expf(-age / CONG_TAU);
cost = w * value + (1 - w) * FREE_FLOW;
```

An agent trusts nearby news and discounts distant news in proportion to how
out of date it is. This turns "the information is delayed" from a defect into
something the algorithm reasons about explicitly, and is the natural hook for
prediction later: an agent reaching an edge in *k* ticks on data *d* ticks old
is extrapolating across `d + k`.

## Cap overflow is a model parameter, not a buffer limit

An agent denied a crossing because the edge is at `cap` does not vanish and is
not rerouted this tick. It stays put, and is counted as queueing load on the
edge it wanted (`BLOCK_PENALTY` per blocked agent). Saturated edges therefore
report higher congestion, which diffuses outward and steers later agents away.
Changing `--cap` changes the physics, not just the message size.

## Sizing a run

Each car needs one crossing per `DWELL_TICKS + 1` ticks, so steady-state
demand is `N / 2` per tick against a capacity of `ne * cap`. Target 50–70%
utilisation:

```
N  ~=  1.2 * ne * cap        (total cars in flight)
```

A 20x20 grid has 1520 directed edges; at `cap 3` that is 4560 crossings/tick
of capacity, so ~5500 cars in flight — roughly `--rate 8` across 24 ranks
once arrivals and departures balance. A 5x5 grid at the same car count is
17x oversubscribed: every edge saturates every tick, all routing choices
become equally bad, and all three modes collapse onto each other.

## What to look for in the output

- **`replica staleness ... max`** should track the graph diameter in hops.
  That is the propagation delay the example exists to model. Raise `--grid`
  and it grows; the mean tracks average hop distance.
- **`mean age of consulted entry`** is the delay actually experienced by
  routing, averaged over every edge Dijkstra relaxed. In `oracle` it must be
  exactly `1.00` — anything else means the allgatherv is landing wrong. In
  `stale` it should be close to the replica-staleness mean.
- **`lost`** is cars staged on the final tick that arrive at a tick which
  never runs. It should be small and roughly `live * blocked_fraction`.
  Agents are otherwise conserved: the pool slot is released the instant a
  record is staged and recreated by `inject_car` on arrival, so exactly one
  copy exists at all times.
- **`blocked %`** above ~80% means gridlock — the run measures queueing, not
  routing, and the modes will not separate.

## Cost and where it breaks

Per rank per tick: `local_edges * (40 + 8|E| + 24*cap)` bytes out. The `8|E|`
term dominates and grows with the **global** graph while `local_edges` shrinks
only as `1/P`, so total traffic grows as `O(|E|^2)`. Startup prints the exact
figure. Concretely: a 12x12 grid is 528 directed edges, ~4.5 KiB per message —
fine. A 64x64 grid is 16k edges, ~130 KiB per message and ~2 GiB/tick across
the job — not fine.

DSTM also retains one region copy per version, so peak memory is roughly
`steps * local_edges * edge_state_size`. Keep `--grid` modest or `--steps`
short. The per-destination tree cache is a second `O(V^2)` floats — 640 KB at
400 vertices, but it needs an LRU past ~2000.

The fix, when you want it, is the delta the layout was designed for: send only
entries whose stamp is newer than what you last sent that neighbour, as
`{edge_id, value, stamp}` triples. Because the merge is a join-semilattice,
this is **provably equivalent** to shipping the whole array — a bandwidth
change, not a semantic one. `n_state` is already in the header; only the
packer and the merge loop change. That drops the term to `O(|E|/P)`.

Note this is *not* "send only your own local changes": you must also forward
what you learned by relay and have not yet passed on, or distant news stops
propagating at you.

## Known caveats

- **Stale-information routing oscillates.** The classic traffic-assignment
  flip-flop: everyone sees A congested, everyone switches to B, B congests.
  The EWMA (`EWMA_ALPHA`), the age decay, the cost clamp and the deterministic
  tie-break jitter all damp it, but do not eliminate it. Hysteresis — only
  re-route if the alternative is better by a margin — is the next step.
- **Partition independence is claimed but untested.** `tie_jitter` is a pure
  hash of `(car id, tick, edge id)`, so nothing depends on the per-rank RNG
  stream, and diffusion timing is partition-independent by construction
  (`delta = 1` everywhere). The same `--seed` under `-n 2` and `-n 8` should
  therefore produce identical trajectories. That is the cheapest correctness
  test available here and it has not been run.
- **Undirected 4-connected grid only.** The channel scheme needs `(v,u)` to
  exist whenever `(u,v)` does. A directed graph would need explicit reverse
  channel edges.
