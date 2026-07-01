#!/usr/bin/env python3
"""
Parse DSTM airport_sim debug logs and report version lifetime stats per rank.

Recognized lines (rank and total parsed from `info:R/N:` or `normal:R/N:` prefix):

  version readable, time: T        -> version (rank, T) created
  version discardable, time: T     -> version (rank, T) marked discardable
  evicting version, time: T        -> version (rank, T) destroyed

`t=N` in normal lines is used only to seed the global timeline so we have a
tick even when no version event happens on a given rank.

Active definition (per user): a version is "active" from its `readable` event
until its `evicting` event (exclusive of evict, inclusive of readable).

Output:
  - stdout: per-rank summary table (min/max/avg active versions across time)
  - <logfile>.versions.csv: detailed per-(rank, t) table with columns
        rank, t, n_active, n_readable, n_discardable, created_this_t, evicted_this_t

Usage:
  python parse_versions.py <logfile>          # read file
  python parse_versions.py -                  # read stdin
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

# ---- line patterns ---------------------------------------------------------

# Prefix like:  airport_sim:    info:6/8:   ...   or   normal:4/8:
_PREFIX_RE = re.compile(r"\b(?:info|normal):(\d+)/(\d+):")

# Version events. We match the suffix after the prefix; time is required.
_READABLE_RE    = re.compile(r"version readable,\s*time:\s*(\d+)")
_DISCARDABLE_RE = re.compile(r"version discardable,\s*time:\s*(\d+)")
_EVICT_RE       = re.compile(r"evicting version,\s*time:\s*(\d+)")

# Simulation time hint from normal lines: "Rank 4 [t=15]:"
_TICK_RE = re.compile(r"\[t=(\d+)\]")


# ---- parsing ---------------------------------------------------------------

class VersionTracker:
    """
    Tracks per-rank version state as we walk the log in order.

    A version is identified by (rank, version_time). We process events in log
    order and, for each event, snapshot the rank's active count AT that
    sim-time tick (the most recent simulation time observed so far for that
    rank, or the event's own time if none seen yet).
    """

    def __init__(self) -> None:
        # (rank, vtime) -> state in {"readable", "discardable"}; absent => not active
        self.active: dict[tuple[int, int], str] = {}
        # Most recent sim time observed for each rank (from version events or t= ticks)
        self.cur_t: dict[int, int] = {}
        # Per (rank, t) accumulator. Values are dicts so we can update in place.
        # We store the *last seen* n_active/readable/discardable snapshot for
        # that (rank, t), and a running count of created/evicted events.
        self.snapshots: dict[tuple[int, int], dict[str, int]] = {}
        self.total_ranks: int | None = None

    # --- helpers ---

    def _touch(self, rank: int, t: int) -> dict[str, int]:
        self.cur_t[rank] = t
        key = (rank, t)
        snap = self.snapshots.get(key)
        if snap is None:
            snap = {
                "n_active": 0,
                "n_readable": 0,
                "n_discardable": 0,
                "created_this_t": 0,
                "evicted_this_t": 0,
            }
            self.snapshots[key] = snap
        return snap

    def _recount(self, rank: int) -> tuple[int, int, int]:
        n_active = n_readable = n_discardable = 0
        for (r, _vt), state in self.active.items():
            if r != rank:
                continue
            n_active += 1
            if state == "readable":
                n_readable += 1
            else:
                n_discardable += 1
        return n_active, n_readable, n_discardable

    def _snapshot_now(self, rank: int) -> None:
        t = self.cur_t.get(rank)
        if t is None:
            return
        snap = self._touch(rank, t)
        n_a, n_r, n_d = self._recount(rank)
        snap["n_active"] = n_a
        snap["n_readable"] = n_r
        snap["n_discardable"] = n_d

    # --- event handlers ---

    def on_readable(self, rank: int, vtime: int) -> None:
        key = (rank, vtime)
        was_present = key in self.active
        self.active[key] = "readable"
        snap = self._touch(rank, vtime)  # advances cur_t[rank] to vtime
        if not was_present:
            snap["created_this_t"] += 1
        self._snapshot_now(rank)

    def on_discardable(self, rank: int, vtime: int) -> None:
        key = (rank, vtime)
        if key in self.active:
            self.active[key] = "discardable"
        # Don't advance cur_t to vtime here: the discardable event is about an
        # older version, but the rank's "now" is whatever we last saw. Only
        # snapshot under the rank's current t.
        if rank in self.cur_t:
            self._snapshot_now(rank)

    def on_evict(self, rank: int, vtime: int) -> None:
        key = (rank, vtime)
        if key in self.active:
            del self.active[key]
            if rank in self.cur_t:
                snap = self._touch(rank, self.cur_t[rank])
                snap["evicted_this_t"] += 1
                self._snapshot_now(rank)

    def on_tick(self, rank: int, t: int) -> None:
        # Tick from a `[t=N]` line in a normal log entry. Advance current time
        # for this rank and re-snapshot so we record "no change" ticks too.
        prev = self.cur_t.get(rank, -1)
        if t > prev:
            self._touch(rank, t)
            self._snapshot_now(rank)


def parse_stream(lines) -> VersionTracker:
    vt = VersionTracker()
    for line in lines:
        m = _PREFIX_RE.search(line)
        if not m:
            continue
        rank = int(m.group(1))
        total = int(m.group(2))
        if vt.total_ranks is None:
            vt.total_ranks = total

        # Order matters: check version events first, then fall back to a tick.
        if (mm := _READABLE_RE.search(line)) is not None:
            vt.on_readable(rank, int(mm.group(1)))
        elif (mm := _DISCARDABLE_RE.search(line)) is not None:
            vt.on_discardable(rank, int(mm.group(1)))
        elif (mm := _EVICT_RE.search(line)) is not None:
            vt.on_evict(rank, int(mm.group(1)))
        elif (mm := _TICK_RE.search(line)) is not None:
            vt.on_tick(rank, int(mm.group(1)))
        # else: not a line we care about
    return vt


# ---- reporting -------------------------------------------------------------

def write_detailed_csv(vt: VersionTracker, out_path: Path) -> None:
    rows = []
    for (rank, t), snap in vt.snapshots.items():
        rows.append((rank, t, snap["n_active"], snap["n_readable"],
                     snap["n_discardable"], snap["created_this_t"],
                     snap["evicted_this_t"]))
    rows.sort(key=lambda r: (r[0], r[1]))

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "t", "n_active", "n_readable", "n_discardable",
                    "created_this_t", "evicted_this_t"])
        w.writerows(rows)

def print_summary(vt: VersionTracker, per_rank_csv: Path | None = None) -> None:
    # Per-rank active-version stats over observed ticks.
    by_rank: dict[int, list[int]] = defaultdict(list)
    by_rank_readable: dict[int, list[int]] = defaultdict(list)
    by_rank_discardable: dict[int, list[int]] = defaultdict(list)
    for (rank, _t), snap in vt.snapshots.items():
        by_rank[rank].append(snap["n_active"])
        by_rank_readable[rank].append(snap["n_readable"])
        by_rank_discardable[rank].append(snap["n_discardable"])

    if not by_rank:
        print("No version events found in input.", file=sys.stderr)
        return

    total = vt.total_ranks if vt.total_ranks is not None else "?"
    print(f"\nVersion activity per rank (total ranks seen: {total})")
    print(f"{'rank':>4} {'ticks':>6} {'min':>5} {'max':>5} {'avg':>7} "
          f"{'r_avg':>7} {'d_avg':>7}")

    per_rank_rows = []
    for rank in sorted(by_rank):
        vals = by_rank[rank]
        r_vals = by_rank_readable[rank]
        d_vals = by_rank_discardable[rank]
        print(f"{rank:>4} {len(vals):>6} {min(vals):>5} {max(vals):>5} "
              f"{mean(vals):>7.2f} {mean(r_vals):>7.2f} {mean(d_vals):>7.2f}")
        per_rank_rows.append((rank, len(vals), min(vals), max(vals),
                              mean(vals), mean(r_vals), mean(d_vals)))

    if per_rank_csv is not None:
        with per_rank_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "ticks", "min", "max", "avg", "r_avg", "d_avg"])
            for rank, ticks, mn, mx, avg, ra, da in per_rank_rows:
                w.writerow([rank, ticks, mn, mx,
                            f"{avg:.4f}", f"{ra:.4f}", f"{da:.4f}"])

    # Global rollup
    all_vals = [v for lst in by_rank.values() for v in lst]
    print(f"\nGlobal active-versions: min={min(all_vals)} max={max(all_vals)} "
          f"avg={mean(all_vals):.2f}  (across {len(all_vals)} rank-tick samples)")

# ---- main ------------------------------------------------------------------

def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2

    src = argv[1]
    if src == "-":
        vt = parse_stream(sys.stdin)
        out_path = Path("stdin.versions.csv")
    else:
        in_path = Path(src)
        if not in_path.is_file():
            print(f"ERROR: not a file: {in_path}", file=sys.stderr)
            return 1
        with in_path.open("r", errors="replace") as f:
            vt = parse_stream(f)
        out_path = in_path.with_suffix(in_path.suffix + ".versions.csv")

        write_detailed_csv(vt, out_path)

    per_rank_path = out_path.with_name(out_path.stem + ".per_rank.csv")
    print_summary(vt, per_rank_csv=per_rank_path)
    print(f"\nDetailed per-(rank, t) CSV written to: {out_path}")
    print(f"Per-rank summary CSV written to:    {per_rank_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
