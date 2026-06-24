#!/usr/bin/env python3
"""
Count MPI communication operations in a zombie_sim log.

Recognized op tokens (matched on the trailing portion of each log line):
    Recv
    Irecv
    Isend
    Send h        (heartbeat send)
    Send b        (barrier send)
    Send          (plain Send, i.e. not "Send h" / "Send b" / "ISend")
    alltoall
    Waitall

Lines may appear with or without the "zombie_sim:  normal:R/N:" prefix:
some logging goes through dstm_log_normal (prefixed) and some through bare
printf (no prefix). By default both are counted; use --prefixed-only to
require the prefix.

Usage:
    python parse.py <logfile> [<logfile> ...]
    python parse.py --per-rank <logfile>      # break out by rank (prefixed lines only)
    python parse.py --csv <logfile>           # CSV emission for piping into sweeps
    cat run.log | python parse.py -
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Order matters for reporting and CSV column order; counting itself is unambiguous.
OP_NAMES = ["Recv", "Irecv", "Isend", "Send h", "Send b", "Send", "alltoall", "Waitall"]

# Optional trailing number after the op name: int OR float (e.g. "0.001159",
# "1.5e-4", "42"). Lets us match both the plain "Recv" lines from your sample
# and the "_PUT: 0.000004" style timing lines without confusing them.
_TAIL = r"(?:\s*:\s*[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)?\s*$"

# All patterns are case-insensitive so "ALLTOALL" and "alltoall" both match.
_F = re.IGNORECASE

# Negative lookbehind/lookahead so "Send" doesn't match inside "Isend" etc.
# Anchored ^...$ on the strippable portion so we don't accidentally pick up
# the word "Recv" appearing inside some narrative log line.
_BOUND_L = r"(?<![A-Za-z])_?"   # allow optional leading underscore
_BOUND_R = r"(?![A-Za-z])"

OP_PATTERNS = {
    "Recv":     re.compile(r"^\s*" + _BOUND_L + r"Recv"      + _BOUND_R + _TAIL, _F),
    "Irecv":    re.compile(r"^\s*" + _BOUND_L + r"Irecv"     + _BOUND_R + _TAIL, _F),
    "Isend":    re.compile(r"^\s*" + _BOUND_L + r"Isend"     + _BOUND_R + _TAIL, _F),
    "Send h":   re.compile(r"^\s*" + _BOUND_L + r"Send\s+h"  + _BOUND_R + _TAIL, _F),
    "Send b":   re.compile(r"^\s*" + _BOUND_L + r"Send\s+b"  + _BOUND_R + _TAIL, _F),
    "Send":     re.compile(r"^\s*" + _BOUND_L + r"Send"      + _BOUND_R + _TAIL, _F),  # plain Send only
    "alltoall": re.compile(r"^\s*" + _BOUND_L + r"Alltoall"  + _BOUND_R + _TAIL, _F),
    "Waitall":  re.compile(r"^\s*" + _BOUND_L + r"Waitall"   + _BOUND_R + _TAIL, _F),
}

# Strips the "zombie_sim:    normal:21/32:   " style prefix so the op regexes
# (which anchor with ^) can match on the trailing content.
PREFIX_RE = re.compile(r"^\s*zombie_sim:\s*\S+:\s*\d+\s*/\s*\d+:\s*", _F)

# Pulls "<rank>/<size>" out of the "normal:21/32:" style prefix, if present.
RANK_RE = re.compile(r":\s*(\d+)\s*/\s*(\d+)\s*:")


def csv_header():
    cols = [op.replace(" ", "_") for op in OP_NAMES] + ["TOTAL"]
    return ",".join(cols)


def format_csv(totals):
    counts = [totals.get(op, 0) for op in OP_NAMES]
    return ",".join(str(c) for c in counts + [sum(counts)])


def strip_prefix(line):
    """Strip the zombie_sim log prefix if present. Returns (stripped, had_prefix)."""
    m = PREFIX_RE.match(line)
    if m:
        return line[m.end():], True
    return line, False


def classify(line, prefixed_only):
    """Return (op_name, had_prefix) for this line, or (None, False)."""
    stripped, had_prefix = strip_prefix(line)

    if prefixed_only and not had_prefix:
        return None, False

    # Skip narrative lines from dstm_log_normal — those start with "Rank N [t=...]"
    # and shouldn't be classified as comm ops just because they contain the word
    # "Send" or "Recv" somewhere in the middle (e.g. "SHIP human ...").
    # Since our op regexes anchor with ^, any "Rank N [...]" prefix already
    # disqualifies them naturally. But guard explicitly just in case.
    if stripped.lstrip().startswith("Rank "):
        return None, had_prefix

    for op in OP_NAMES:
        if OP_PATTERNS[op].search(stripped):
            return op, had_prefix
    return None, had_prefix


def rank_of(line):
    m = RANK_RE.search(line)
    return m.group(1) if m else None


def count_stream(fp, per_rank, prefixed_only):
    totals = Counter()
    by_rank = defaultdict(Counter)
    for raw in fp:
        line = raw.rstrip("\n")
        op, had_prefix = classify(line, prefixed_only)
        if op is None:
            continue
        totals[op] += 1
        if per_rank:
            r = rank_of(line) if had_prefix else None
            r = r or "?"
            by_rank[r][op] += 1
    return totals, by_rank


def print_totals(label, totals):
    print("== {} ==".format(label))
    width = max(len(n) for n in OP_NAMES)
    grand = 0
    for op in OP_NAMES:
        c = totals.get(op, 0)
        grand += c
        print("  {}  {}".format(op.ljust(width), c))
    print("  {}  {}".format("TOTAL".ljust(width), grand))


def print_per_rank(by_rank):
    if not by_rank:
        return
    print("  -- per rank --")
    header = ["rank"] + OP_NAMES + ["total"]
    print("  " + ",".join(header))

    def sort_key(r):
        try:
            return (0, int(r))
        except ValueError:
            return (1, r)

    for r in sorted(by_rank.keys(), key=sort_key):
        row = by_rank[r]
        counts = [row.get(op, 0) for op in OP_NAMES]
        total = sum(counts)
        print("  " + ",".join([r] + [str(c) for c in counts] + [str(total)]))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("logs", nargs="+",
                    help="Log file(s) to parse. Use '-' to read stdin.")
    ap.add_argument("--per-rank", action="store_true",
                    help="Also break counts down by rank "
                         "(parsed from 'zombie_sim:name:rank/size:' prefix; "
                         "unprefixed lines bucket into '?').")
    ap.add_argument("--prefixed-only", action="store_true",
                    help="Only count lines that carry the zombie_sim:...: "
                         "log prefix. By default, bare op lines from "
                         "printf-style logging are also counted.")
    ap.add_argument("--csv", action="store_true",
                    help="Emit CSV: header on line 1, then one data row per "
                         "input file. Columns: " + csv_header())
    args = ap.parse_args()

    if args.csv:
        print(csv_header())

    grand = Counter()
    rc = 0

    for path in args.logs:
        try:
            if path == "-":
                totals, by_rank = count_stream(
                    sys.stdin, args.per_rank, args.prefixed_only)
                label = "<stdin>"
            else:
                p = Path(path)
                with p.open("r", encoding="utf-8", errors="replace") as fp:
                    totals, by_rank = count_stream(
                        fp, args.per_rank, args.prefixed_only)
                label = str(p)
        except OSError as e:
            print("ERROR: {}: {}".format(path, e), file=sys.stderr)
            rc = 1
            continue

        if args.csv:
            print(format_csv(totals))
        else:
            print_totals(label, totals)
            if args.per_rank:
                print_per_rank(by_rank)
            print()
        grand.update(totals)

    if not args.csv and len(args.logs) > 1:
        print_totals("GRAND TOTAL", grand)

    return rc


if __name__ == "__main__":
    sys.exit(main())
