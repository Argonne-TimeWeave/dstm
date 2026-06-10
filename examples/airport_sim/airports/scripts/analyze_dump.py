#!/usr/bin/env python3
"""
Parse an airports dump.log and report the share of elapsed time spent
in get_or_default and put operations (averaged across MPI ranks).

Usage:
    python analyze_dump.py dump.log [--ranks N]

If --ranks is not given, the script tries to read it from the
"Ranks: N" line in the log header.
"""

import argparse
import re
import sys


def parse_log(path):
    get_total = 0.0
    get_count = 0
    put_total = 0.0
    put_count = 0
    elapsed = None
    ranks = None

    line_re = re.compile(r'^\s*(get_or_default|put)\s*:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*$')
    ranks_re = re.compile(r'^\s*Ranks:\s*(\d+)')
    elapsed_re = re.compile(r'^\s*Elapsed time:\s*([0-9]*\.?[0-9]+)\s*seconds')

    with open(path, 'r') as f:
        for line in f:
            m = line_re.match(line)
            if m:
                op, val = m.group(1), float(m.group(2))
                if op == 'get_or_default':
                    get_total += val
                    get_count += 1
                else:
                    put_total += val
                    put_count += 1
                continue

            m = ranks_re.match(line)
            if m:
                ranks = int(m.group(1))
                continue

            m = elapsed_re.match(line)
            if m:
                elapsed = float(m.group(1))

    return {
        'get_total': get_total,
        'get_count': get_count,
        'put_total': put_total,
        'put_count': put_count,
        'elapsed': elapsed,
        'ranks': ranks,
    }


def main():
    ap = argparse.ArgumentParser(description="Summarize get_or_default / put timings from a dump.log")
    ap.add_argument('logfile', help='Path to dump.log')
    ap.add_argument('--ranks', type=int, default=None,
                    help='Number of MPI ranks (overrides value parsed from log)')
    args = ap.parse_args()

    stats = parse_log(args.logfile)

    ranks = args.ranks if args.ranks is not None else stats['ranks']
    if not ranks:
        print("ERROR: could not determine number of ranks; pass --ranks N", file=sys.stderr)
        sys.exit(1)
    if stats['elapsed'] is None:
        print("ERROR: could not find 'Elapsed time:' line in log", file=sys.stderr)
        sys.exit(1)

    elapsed = stats['elapsed']
    # Sum across all ranks divided by number of ranks = per-rank average wall time
    get_avg = stats['get_total'] / ranks
    put_avg = stats['put_total'] / ranks

    get_pct = 100.0 * get_avg / elapsed
    put_pct = 100.0 * put_avg / elapsed

    print(f"File:           {args.logfile}")
    print(f"Ranks:          {ranks}")
    print(f"Elapsed time:   {elapsed:.4f} s")
    print()
    print(f"get_or_default: count={stats['get_count']}, "
          f"sum={stats['get_total']:.4f} s, "
          f"per-rank avg={get_avg:.4f} s, "
          f"{get_pct:.2f}% of elapsed")
    print(f"put:            count={stats['put_count']}, "
          f"sum={stats['put_total']:.4f} s, "
          f"per-rank avg={put_avg:.4f} s, "
          f"{put_pct:.2f}% of elapsed")


if __name__ == '__main__':
    main()
