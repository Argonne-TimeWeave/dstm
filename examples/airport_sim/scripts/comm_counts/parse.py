#!/usr/bin/env python3
"""
Count MPI communication operations in an airport_sim log.

Recognized op tokens (matched on the trailing portion of each log line):
    Recv
    ISend
    Send h
    Send b
    Send          (plain Send, i.e. not "Send h" / "Send b" / "ISend")
    alltoall

Usage:
    python count_comms.py <logfile> [<logfile> ...]
    python count_comms.py --per-rank <logfile>
    cat run.log | python count_comms.py -
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Order matters for reporting; counting itself is unambiguous.
OP_NAMES = ["Recv", "Irecv", "Isend", "Send h", "Send b", "Send", "alltoall"]

# Optional trailing number: int OR float (e.g. "0.001159", "1.5e-4", "42").
_NUM  = r"(?:[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)"
# After the op token we require: optional whitespace+number, then end-of-line
# (or end of string). MULTILINE makes $ match at every newline.
_TAIL = rf"(?:[ \t]+{_NUM})?(?=\s|$|[^A-Za-z0-9])"

# Case-insensitive + multiline so bare "Recv\n" and "Send b\n" both match.
_F = re.IGNORECASE | re.MULTILINE

OP_PATTERNS = {
    # Plain Recv — must not be followed by a letter (so "Recvd" wouldn't match)
    "Recv":     re.compile(r"(?<![A-Za-z])Recv(?![A-Za-z])"            + _TAIL, _F),
    "Irecv":    re.compile(r"(?<![A-Za-z])Irecv(?![A-Za-z])"           + _TAIL, _F),
    "Isend":    re.compile(r"(?<![A-Za-z])Isend(?![A-Za-z])"           + _TAIL, _F),
    "Send h":   re.compile(r"(?<![A-Za-z])Send\s+h(?![A-Za-z])"        + _TAIL, _F),
    "Send b":   re.compile(r"(?<![A-Za-z])Send\s+b(?![A-Za-z])"        + _TAIL, _F),
    # Plain Send only: must NOT be followed by whitespace+h/b (those are their own ops),
    # and must not be part of a longer word like "Sender".
    "Send":     re.compile(r"(?<![A-Za-z])Send(?![A-Za-z])(?!\s+[hb](?![A-Za-z]))" + _TAIL, _F),
    "alltoall": re.compile(r"(?<![A-Za-z])alltoall(?![A-Za-z])"        + _TAIL, _F),
}

# Pulls "<rank>/<size>" out of the "normal:94/256:" style prefix, if present.
RANK_RE = re.compile(r":\s*(\d+)\s*/\s*(\d+)\s*:")

def csv_header():
    cols = [op.replace(" ", "_") for op in OP_NAMES] + ["TOTAL"]
    return ",".join(cols)


def format_csv(totals):
    counts = [totals.get(op, 0) for op in OP_NAMES]
    return ",".join(str(c) for c in counts + [sum(counts)])

def classify(line):
    hits = []
    for op in ["Send h", "Send b", "Irecv", "Isend", "alltoall", "Recv", "Send"]:
        for m in OP_PATTERNS[op].finditer(line):
            hits.append((m.start(), op))
    hits.sort()
    return [op for _, op in hits]


def rank_of(line):
    m = RANK_RE.search(line)
    return m.group(1) if m else None


def count_stream(fp, per_rank):
    totals = Counter()
    by_rank = defaultdict(Counter)
    for raw in fp:
        line = raw.rstrip("\n")
        ops = classify(line)
        if not ops:
            continue
        r = rank_of(line) if per_rank else None
        for op in ops:
            totals[op] += 1
            if per_rank:
                by_rank[r or "?"][op] += 1
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
                    help="Also break counts down by rank (parsed from 'name:rank/size:' prefix).")
    ap.add_argument("--csv", action="store_true",
                    help="Emit CSV: header on line 1, then one data row per input file. "
                         "Columns: " + csv_header())
    args = ap.parse_args()

    if args.csv:
        print(csv_header())

    grand = Counter()
    rc = 0

    for path in args.logs:
        try:
            if path == "-":
                totals, by_rank = count_stream(sys.stdin, args.per_rank)
                label = "<stdin>"
            else:
                p = Path(path)
                with p.open("r", encoding="utf-8", errors="replace") as fp:
                    totals, by_rank = count_stream(fp, args.per_rank)
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
