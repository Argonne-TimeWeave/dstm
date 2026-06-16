#!/usr/bin/env python3
"""
Parse MPI_Wtime() timing samples from an airport_sim log and report what
percentage of total application runtime was spent in each comm op.

Recognized op tokens (each followed by a float number of seconds at end of line):
    PUT       <seconds>
    GET       <seconds>
    ALLTOALL  <seconds>
    WAITALL   <seconds>

Total runtime is parsed from the simulator's summary line:
    Elapsed time:       0.0464 seconds

Per-op "comm time per rank" is the sum of that op's samples across all ranks,
divided by the number of ranks observed in the log (from the "name:rank/size:"
prefix). Percent is that per-rank comm time divided by total runtime.

Usage:
    python parse_times.py <logfile> [<logfile> ...]
    python parse_times.py --csv <logfile> [<logfile> ...]
    cat run.log | python parse_times.py -
"""

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

OP_NAMES = ["PUT", "GET", "ALLTOALL", "WAITALL"]

FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"

OP_PATTERNS = {
    op: re.compile(
        r"(?<![A-Za-z])" + op + r"\s+(" + FLOAT + r")\s*$",
        re.IGNORECASE,
    )
    for op in OP_NAMES
}

# "normal:3/16:" -> rank=3, size=16
RANK_RE = re.compile(r":\s*(\d+)\s*/\s*(\d+)\s*:")

# "Elapsed time:       0.0464 seconds"
ELAPSED_RE = re.compile(
    r"Elapsed\s+time\s*:\s*(" + FLOAT + r")\s*seconds?",
    re.IGNORECASE,
)


def _fmt(x):
    if x is None:
        return ""
    return "{:.9g}".format(x)


def scan_stream(fp):
    sums = defaultdict(float)   # op -> sum of seconds across all ranks
    ranks_seen = set()
    size_from_prefix = None
    elapsed = None

    for raw in fp:
        line = raw.rstrip("\n")

        m_e = ELAPSED_RE.search(line)
        if m_e:
            try:
                elapsed = float(m_e.group(1))
            except ValueError:
                pass
            continue

        if "airport_sim" not in line:
            continue

        m_r = RANK_RE.search(line)
        if m_r:
            ranks_seen.add(int(m_r.group(1)))
            try:
                size_from_prefix = int(m_r.group(2))
            except ValueError:
                pass

        for op in OP_NAMES:
            m = OP_PATTERNS[op].search(line)
            if m:
                try:
                    sums[op] += float(m.group(1))
                except ValueError:
                    pass
                break

    # Prefer the size advertised in the prefix; fall back to count of ranks seen.
    nranks = size_from_prefix if size_from_prefix else (len(ranks_seen) or 1)
    return sums, nranks, elapsed


def per_rank_and_pct(sums, nranks, elapsed):
    """Return dict op -> (per_rank_seconds, percent_of_runtime_or_None)."""
    out = {}
    for op in OP_NAMES:
        per_rank = sums.get(op, 0.0) / nranks
        pct = (per_rank / elapsed * 100.0) if elapsed and elapsed > 0 else None
        out[op] = (per_rank, pct)
    return out


# ---------- CSV ----------

def csv_header():
    cols = ["nranks", "elapsed_s"]
    for op in OP_NAMES:
        cols.append("{}_s_per_rank".format(op))
        cols.append("{}_pct".format(op))
    cols.append("COMM_s_per_rank")
    cols.append("COMM_pct")
    return ",".join(cols)


def format_csv(sums, nranks, elapsed):
    per_op = per_rank_and_pct(sums, nranks, elapsed)
    parts = [str(nranks), _fmt(elapsed)]
    total_per_rank = 0.0
    for op in OP_NAMES:
        per_rank, pct = per_op[op]
        parts.append(_fmt(per_rank))
        parts.append(_fmt(pct))
        total_per_rank += per_rank
    parts.append(_fmt(total_per_rank))
    total_pct = (total_per_rank / elapsed * 100.0) if elapsed and elapsed > 0 else None
    parts.append(_fmt(total_pct))
    return ",".join(parts)


# ---------- Pretty ----------

def print_report(label, sums, nranks, elapsed):
    print("== {} ==".format(label))
    print("  nranks   : {}".format(nranks))
    print("  elapsed  : {} s".format(_fmt(elapsed) if elapsed else "(not found)"))
    print("  {:<10} {:>14} {:>10}".format("op", "s/rank", "% of run"))
    per_op = per_rank_and_pct(sums, nranks, elapsed)
    total_per_rank = 0.0
    for op in OP_NAMES:
        per_rank, pct = per_op[op]
        total_per_rank += per_rank
        pct_str = "{:.2f}%".format(pct) if pct is not None else "  n/a"
        print("  {:<10} {:>14} {:>10}".format(op, _fmt(per_rank), pct_str))
    total_pct = (total_per_rank / elapsed * 100.0) if elapsed and elapsed > 0 else None
    total_pct_str = "{:.2f}%".format(total_pct) if total_pct is not None else "  n/a"
    print("  {:<10} {:>14} {:>10}".format("COMM", _fmt(total_per_rank), total_pct_str))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("logs", nargs="+",
                    help="Log file(s) to parse. Use '-' to read stdin.")
    ap.add_argument("--csv", action="store_true",
                    help="Emit CSV: header on line 1, then one data row per "
                         "input file. Columns: " + csv_header())
    args = ap.parse_args()

    if args.csv:
        print(csv_header())

    rc = 0
    for path in args.logs:
        try:
            if path == "-":
                sums, nranks, elapsed = scan_stream(sys.stdin)
                label = "<stdin>"
            else:
                p = Path(path)
                with p.open("r", encoding="utf-8", errors="replace") as fp:
                    sums, nranks, elapsed = scan_stream(fp)
                label = str(p)
        except OSError as e:
            print("ERROR: {}: {}".format(path, e), file=sys.stderr)
            rc = 1
            continue

        if args.csv:
            print(format_csv(sums, nranks, elapsed))
        else:
            print_report(label, sums, nranks, elapsed)
            print()

    return rc


if __name__ == "__main__":
    sys.exit(main())
