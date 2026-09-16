#!/usr/bin/env python3
"""
Parse a single experiment log and append one CSV row.

Counts substring occurrences of DSTM communication tokens, plus the
GLOBAL METRICS block emitted by mini.c.

Tokens are matched as whole-word-ish substrings; "Send b" and "Send h"
are matched literally so they don't collide with each other.
"""

import argparse
import csv
import os
import re
import sys


# Order here defines what we count. Use regex so we can be precise.
PATTERNS = {
    "recv":   re.compile(r"\bRecv\b"),
    "isend":  re.compile(r"\bIsend\b"),
    "send_b": re.compile(r"\bSend b\b"),
    "send_h": re.compile(r"\bSend h\b"),
}

# Pull the global totals printed by mini.c at the end of the run.
GLOBAL_PATTERNS = {
    "sent":      re.compile(r"sent\s*=\s*(\d+)"),
    "delivered": re.compile(r"delivered\s*=\s*(\d+)"),
    "forwarded": re.compile(r"forwarded\s*=\s*(\d+)"),
    "in_flight": re.compile(r"in-flight\s*=\s*(-?\d+)"),
}


def parse(log_path: str) -> dict:
    counts = {k: 0 for k in PATTERNS}
    globals_ = {k: "" for k in GLOBAL_PATTERNS}

    in_global_block = False
    remaining = set(GLOBAL_PATTERNS)  # keys we still need to fill

    with open(log_path, "r", errors="replace") as f:
        for line in f:
            for key, pat in PATTERNS.items():
                counts[key] += len(pat.findall(line))

            if "GLOBAL METRICS" in line:
                in_global_block = True
                continue

            if in_global_block and remaining:
                # Only accept lines that look like the indented summary,
                # never the per-rank "Rank N METRICS:" lines.
                if "Rank" in line and "METRICS" in line:
                    continue
                for key in list(remaining):
                    m = GLOBAL_PATTERNS[key].search(line)
                    if m:
                        globals_[key] = m.group(1)
                        remaining.discard(key)
                if not remaining:
                    in_global_block = False  # done; ignore later noise

    return {**counts, **globals_}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist",   type=int, required=True)
    ap.add_argument("--nranks", type=int, required=True)
    ap.add_argument("--log",    required=True, help="path to run log")
    ap.add_argument("--csv",    required=True, help="csv to append to")
    args = ap.parse_args()

    if not os.path.exists(args.log):
        print(f"log not found: {args.log}", file=sys.stderr)
        return 1

    data = parse(args.log)

    row = [
        args.dist,
        args.nranks,
        data["recv"],
        data["isend"],
        data["send_b"],
        data["send_h"],
        data["sent"],
        data["delivered"],
        data["forwarded"],
        data["in_flight"],
    ]

    # Append; assume header was written by the bash driver.
    write_header = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
    with open(args.csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "dist", "nranks",
                "recv", "isend", "send_b", "send_h",
                "sent", "delivered", "forwarded", "in_flight",
            ])
        w.writerow(row)

    print(f"  dist={args.dist}: "
          f"recv={data['recv']} isend={data['isend']} "
          f"send_b={data['send_b']} send_h={data['send_h']} "
          f"sent={data['sent']} delivered={data['delivered']} "
          f"forwarded={data['forwarded']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
