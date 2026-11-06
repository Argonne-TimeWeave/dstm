# graph_traffic — routing under diffused, delayed congestion

Each rank holds a congestion replica of the **entire** road network, loaded
from a POLARIS-style node/link CSV pair. Each tick it refreshes the entries
it owns, then ships the whole replica — together with the agents physically
crossing — along every outgoing edge. Neighbours merge what arrives. News
about a distant edge therefore reaches a rank only after as many ticks as
there are hops between them, and every routing decision is made against that
stale view.

## Build and run

```sh
./autogen.sh && ./configure && make
mpirun -n 24 ./examples/traffic/graph_traffic/graph_traffic \
       --data data/bloomington/ --steps 200 --cars 0 --rate 8 --cap 3 --quiet
```

| flag                | default | meaning                              |
|---------------------|---------|--------------------------------------|
| `--data DIR`        | -       | dir with node.csv + link.csv (required) |
| `--steps N`         | 60      | ticks                                |
| `--cars M`          | 64      | agents spawned total (M/ranks per rank) at t=0       |
| `--rate R`          | 0       | agents spawned total (R/ranks per rank) *per tick*   |
| `--cap C`           | 8       | max cars per edge per tick           |
| `--seed S`          | 42      | RNG seed                             |
| `--replica-mode M`  | stale   | `stale` \| `oracle` \| `blind`       |
| `--stale-delay D`   | 5       | merge relayed state every D ticks (stale mode) |
| `--verbose`         | off     | per-tick progress from rank 0        |
| `--quiet`           | off     | DSTM log level quiet (default: debug)|

Prefer `--cars 0 --rate R` for measurement. A single t=0 burst is one
decaying shockwave; congestion information is most valuable in steady state,
and a continuous spawn rate is the only way to reach one.

### Input data

`--data DIR` reads `<dir>/node.csv` (first column = node id) and
`<dir>/link.csv` (columns 3 and 4 = the two endpoint node ids). Every link
becomes two directed edges, a->b and b->a; lane, speed, capacity, geometry
and name columns are ignored. Self-loops and duplicate directed edges within
the file are dropped, and the counts are reported at startup.

The vertex set is the union of every node id in `node.csv` and every endpoint
in `link.csv`, sorted and de-duplicated, then assigned dense ids `0..nv-1` in
ascending original-id order. That map is a pure function of the input, so
every rank computes it identically with no communication. Any rank count from
1 up to the vertex count works.

## Replica modes

Three ways of populating the replica, so the cost of staleness can be
*measured* rather than assumed. All three ship byte-identical channel
messages and run the identical tick sequence — only the contents of the
replica consulted by the router differ, so communication volume never
confounds the comparison.

| mode     | replica contents                    | age of consulted entry |
|----------|-------------------------------------|------------------------|
| `stale`  | diffusion by relay, merged every `STALE_DELAY` ticks | `max(1, hopdist)`      |
| `oracle` | `MPI_Allgatherv` every tick         | exactly `1.00`         |
| `blind`  | congestion ignored, `cost == ff_ticks[e]`     | n/a                    |

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

Oracle is cheap because vertices are block-partitioned by dense id: rank `r`
owns a contiguous range of vertices (base + remainder split) and hence the
contiguous global edge range `[gslot_base, gslot_base + nedge_local)`, so an
allgatherv of owned slices reconstructs the full array. By the single-writer
property that is provably identical to a max-by-stamp allreduce, without
needing `MPI_Op_create`.

## Buffer layout

`[chan_header | cong_entry state[|E|] | car_rec cars[<=cap]]`, sized once
at startup and constant for the run. The authoritative diagram, the field
list and the `sizeof` asserts that pin it are in `chan.h` -- read that
rather than trusting a second copy here.

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

`delta` is **not fixed at 1**: each put uses `edge_delay(ge, t)` = the edge's
free-flow ticks (`ff_ticks[ge]`) plus a congestion term clamped so total delay
never exceeds `MAX_CONGESTION_MULTIPLIER` (8) × free-flow. Congested edges thus
physically take longer to cross, and the clamp bounds the shutdown drain.

This satisfies DSTM's lookahead contract — the boundary distance matrix uses
the *minimum* crossing free-flow per rank pair, and every put's `delta >= ff >=
distance`, so a larger delta is always legal. Under STALE, the delay is read
from the sender's (possibly stale) replica at `t`, so a car is slowed by what
its origin *believes* the edge looks like — the same staleness applied to
physical motion, by design.

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
intended arrival tick and gates injection.

In `stale` mode the state merge runs only when `t % STALE_DELAY == 0`
(default 5) — a throttle on replica freshness, and (by idempotency) another
staleness dial. `oracle` overwrites the replica via allgatherv and `blind`
never reads it, so both skip the fold; the message is still sent and received,
so communication volume stays constant across modes.

## Per-tick discipline

```c
spawn_cars(rate);      // continuous demand, if --rate > 0
publish_vertices(t);   // creates local version t, emits heartbeats
recv_channels(t);      // merge state, inject arriving agents
route_and_stage(t);    // Dijkstra over the replica, honour the cap
measure_local(t);      // stamp owned entries with tick t
replica_sync_oracle(); // oracle mode only
send_channels(t);      // [header][state][cars] at (t, edge_delay(ge,t))
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
destination bias — "downtown" is the top ~5% highest-degree vertices
(`max(1, nv/20)`), chosen deterministically from the loaded graph rather than
a fixed geographic box.

The routing cost is a linear (BPR-style) volume-delay function:

```c
cost(e) = ff_ticks[e] + (load / cap) * ROUTE_DELAY_FACTOR
```

ff_ticks[e] is the edge's per-edge free-flow travel time in ticks (derived from length/speed in the loaded network, not a uniform 1), and load is the age-decayed EWMA of per-tick crossings from cong_effective. The congestion penalty is additive and uncapped: an edge at 2× capacity costs ff + 2*ROUTE_DELAY_FACTOR. There is no hard cap on the routing belief — capacity makes an edge progressively more expensive, it does not block demand. An edge never observed has stamp < 0, so cong_effective returns 0 and the edge costs pure free-flow.

Confidence decays with age via cong_effective(e, t, CONG_TAU, 0.0f): the smoothed load is weighted by expf(-age / CONG_TAU) and decays toward 0 (no premium) as the entry ages. An agent trusts nearby news and discounts distant news in proportion to how out of date it is, turning "the information is delayed" from a defect into something the algorithm reasons about explicitly.

The physical crossing delay (edge_delay, the DSTM put delta) uses the same volume-delay term but rounded up to whole ticks and clamped to at most (MAX_CONGESTION_MULTIPLIER - 1) * ff of congestion — total ≤ 8× free-flow. The routing belief stays uncapped; only physical motion is bounded, so the shutdown drain terminates. A deterministic tie_jitter(car_id, t, edge_id) hash breaks ties independently of the per-rank RNG stream.

## Cap overflow is a model parameter, not a buffer limit

An agent denied a crossing because the edge is at `cap` does not vanish and is
not rerouted this tick. It stays put, and is counted as queueing load on the
edge it wanted (`BLOCK_PENALTY` per blocked agent). Saturated edges therefore
report higher congestion, which (a) diffuses outward and steers later agents
away via routing cost, and (b) raises the edge's own `edge_delay`, so cars
already committed to it physically take longer to cross. Changing `--cap`
changes the physics, not just the message size.

## Sizing a run

Each car needs one crossing per `DWELL_TICKS + 1` ticks, so steady-state
demand is `N / 2` per tick against a capacity of `ne * cap`. Target 50–70%
utilisation:

```
N  ~=  1.2 * ne * cap        (total cars in flight)
```

Read `ne` (directed edge count) off the startup summary. Bloomington is 3133
vertices / 9048 directed edges; at `cap 3` that is 27144 crossings/tick, so
~32000 cars in flight sits in the target band. Under-load and there is nothing
to route around; over-load and every edge saturates, all choices become equally
bad, and the modes collapse together. `--cars`/`--rate` are global totals, so
compare the total in-flight count against `ne * cap`.

## What to look for in the output

- **`replica staleness ... max`** should track the graph diameter in hops.
  That is the propagation delay the example exists to model. Its magnitude is
  set by the partition: the naive block-by-dense-id split scatters
  topologically adjacent vertices across ranks, so max staleness reflects the
  diameter of the *rank-adjacency* graph, which is far smaller than the road
  network's hop diameter. It rises monotonically with rank count (more, finer
  blocks → longer owner-to-owner chains); a locality-preserving partition
  would push it toward the true graph diameter.
- **`mean age of consulted entry`** is the delay actually experienced by
  routing, averaged over every edge Dijkstra relaxed. In `oracle` it must be
  exactly `1.00` — anything else means the allgatherv is landing wrong. In
  `stale` it should be close to the replica-staleness mean.
- **`lost`** is cars still in flight when the simulation stops. Because
  congested edges can take up to `MAX_CONGESTION_MULTIPLIER * ff_max` ticks to
  cross, the drain runs to `t_end + MAX_CONGESTION_MULTIPLIER * ff_max + 1`,
  but cars staged late or on slow/congested edges can still arrive past the
  last tick that runs. Blocked agents are not lost: they stay put and are
  still counted in `live`.
  Agents are otherwise conserved: the pool slot is released the instant a
  record is staged and recreated by `inject_car` on arrival, so exactly one
  copy exists at all times.
- **`blocked %`** above ~80% means gridlock — the run measures queueing, not
  routing, and the modes will not separate.

## Cost and where it breaks

Per rank per tick: `local_edges * (40 + 8|E| + 24*cap)` bytes out. The `8|E|`
term dominates and grows with the **global** graph while `local_edges` shrinks
only as `1/P`, so total traffic grows as `O(|E|^2)`. Read the "per-tick
out/rank" line the summary prints for your network before scaling up.

DSTM also retains one region copy per version, so peak memory is roughly
`steps * local_edges * edge_state_size`. Use a modest network or keep
`--steps` short. The per-destination tree cache is a second `O(V^2)` floats —
~39 MB at 3133 vertices (Bloomington), so it needs an LRU on larger networks.

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
- **Partition independence is a property of the model, not of a run.**
  `tie_jitter` is a pure hash of `(car id, tick, edge id)`, so routing
  tie-breaks never depend on the per-rank RNG stream, and diffusion timing is
  partition-independent by construction (intra- and cross-rank edges use the
  same `edge_delay`). But
  `spawn_cars` picks origins from the local vertex range with a per-rank
  stream, so `-n 2` and `-n 8` at the same `--seed` generate different
  demand and cannot produce identical trajectories. Making that comparison
  possible needs a rank-independent spawn (draw globally, keep what you
  own) — until then the cheap correctness test is unavailable.
- **Every link is treated as bidirectional.** The channel scheme needs `(v,u)`
  to exist whenever `(u,v)` does, so the loader emits both directions of every
  link and ignores any one-way flags in the source data. A genuinely directed
  network would need explicit reverse channel edges.
