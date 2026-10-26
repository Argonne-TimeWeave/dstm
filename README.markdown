DSTM: Distributed Shared Timestamped Memory
===========================================

DSTM is a C17 runtime and programming model for parallel discrete-event and
agent-based simulation. It provides a partitioned distributed memory that is
*versioned over simulation time*: producers and consumers reference the same
state at different points on a shared timeline, so ranks are free to drift out
of lockstep without global barriers and without rollback.

## The model in brief

* An address in DSTM is `(rank, offset, time)`. Time is logical simulation
  time (`dstm_time_t`), not wall clock.
* Every operation carries a `(time, delta)` pair; the version it touches is
  `time + delta`.
* A `put` requires `delta >= 0` (you cannot write the past), a `get` requires
  `delta <= 0` (you cannot read the future). `delta == 0` is legal only for
  the local rank; remote accesses must use a non-zero delta.
* Versions do **not** inherit from one another. A cell nobody wrote at time
  `t` reads as zero, even if it was written at `t-1`. Long-lived state has to
  be re-published every tick. This is the most common first-use mistake.
* A `dstm_distance_t` is the communication graph plus a per-edge *lookahead*:
  how many timesteps of slack exist between two ranks. It is what lets a rank
  decide a version is final without asking anyone. A lookahead of 0 is
  unsound; use 1 for "synchronise every tick".
* `put` is asynchronous. `get` blocks until the requested version is readable,
  and that block is the only synchronisation in the system.

## Requirements

* an MPI C compiler (tested with OpenMPI and MPICH)
* autoconf, automake, libtool, pkgconf
* [jansson](https://github.com/akheron/jansson) >= 2.0
* [check](https://libcheck.github.io/check/)
* [bats](https://github.com/bats-core/bats-core)
* valgrind (optional, for `make check-valgrind`)

Note that `check` and `bats` are currently hard `configure` errors even if you
do not intend to run the test suite.

## Building

```sh
./autogen.sh
./configure
make -j
make install
```

A Nix development environment with every dependency is provided:

```sh
nix-shell          # drops you in a shell with the full toolchain
./autogen.sh && ./configure && make -j
```

## Testing

```sh
make check            # serial unit tests + MPI tests under bats
make check-valgrind   # same, under valgrind
```

## Usage

DSTM does not initialise MPI for you: call `MPI_Init` first and `MPI_Finalize`
last. Use `dstm_init_comm` to run on a communicator other than
`MPI_COMM_WORLD`.

The `last_time` passed to `dstm_fini` must be greater than or equal to the
highest `time + delta` this rank ever submitted, otherwise the drain loop will
not terminate.

A typical simulation replaces the single put/get above with a loop over
timesteps, consuming neighbour state at `t` and publishing results at
`t + delta`. There is no barrier at the tick boundary.

## Beyond the basics

All of the public API is documented in `include/dstm.h`.

* **Topologies.** Besides `dstm_distance_create_ring` there are `_complete`,
  `_star`, `_line`, `_mesh`, `_torus`, `_tree`, `_hypercube` and `_dragonfly`
  constructors, plus `dstm_distance_create_from_matrix` for arbitrary graphs.
  If a program uses several data structures, build one matrix per structure
  and merge them with `dstm_distance_combine` before calling `dstm_start`:
  the session takes a single, global distance matrix.
* **Graphs.** `dstm_dgraph_t` is a distributed graph with versioned vertex and
  edge state and a static topology, built from packed CSR. It can derive its
  own rank-level distance matrix via `dstm_dgraph_derive_distance`.

## Status

This is research software under active development and the API is not yet stable.
