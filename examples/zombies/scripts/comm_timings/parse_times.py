#!/usr/bin/env python3
"""
Parse per-op MPI_Wtime() logs from a zombies (or airports) run and emit
a single-row CSV summarizing per-op communication time vs wall-clock.

Recognized op tokens (case-insensitive, leading underscore optional):
    PUT, GET            (DSTM)
    ISEND, IRECV,
    ALLTOALL, WAITALL   (idiomatic MPI)

For each op the parser:
  * sums every "_OP: <seconds>" occurrence across all ranks in the log
  * divides by num_ranks to get an "average per-rank seconds" figure
  * computes that figure as a percentage of the wall-clock elapsed_s

Wall-clock elapsed_s is read from the program's summary line:
    "Elapsed time:       0.0130 seconds"

num_ranks is taken from --ranks N if provided, otherwise inferred from
the maximum rank-id seen in any "[rank=N nranks=M]"-style log prefix
(e.g. ":21/32:" => 32). Falls back to 1 if nothing is found.

Usage:
    python parse_times.py <logfile>                    # human-readable
    python parse_times.py --csv <logfile>              # one CSV row (+ header)
    python parse_times.py --csv --ranks 256 <logfile>  # force num_ranks
    python parse_times.py --csv /dev/null              # print header only
"""

import argparse
import re
import sys
from pathlib import Path


# Ops we recognize. Order is fixed so the CSV header is stable.
OPS = ["PUT", "GET", "ISEND", "IRECV", "ALLTOALL", "WAITALL"]

# Matches a per-op timing line. The leading underscore is optional and the
# op name is case-insensitive. We accept any prefix in front (e.g. the
# "zombie_sim:    normal:21/32:   " stuff openmpi adds with --tag-output)
# by just searching, so a bare `_Waitall: 0.002332` matches too.
OP_RE = re.compile(
    r"_?(" + "|".join(OPS) + r")\s*:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)

# Matches "Elapsed time:    0.0130 seconds"
ELAPSED_RE = re.compile(
    r"Elapsed\s+time\s*:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?\d+)?)\s*seconds",
    re.IGNORECASE,
)

# Matches "Timesteps/second:   9852.79"
STEPS_RE = re.compile(
    r"Timesteps?/second\s*:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)

# Best-effort: log-prefix style ":21/32:" -> total ranks = 32.
# Only used as a fallback when --ranks is not given.
RANK_TOTAL_RE = re.compile(r":(\d+)\s*/\s*(\d+):")


def csv_header():
    """Build a stable header row for the CSV emission mode."""
    cols = ["num_ranks", "elapsed_s", "steps_per_s"]
    for op in OPS:
        lo = op.lower()
        cols += [lo + "_s_per_rank", lo + "_pct"]
    cols += ["comm_s_per_rank", "comm_pct"]
    return ",".join(cols)


def parse_log(path, num_ranks_override=None):
    """Scan the log and return aggregate stats."""
    op_sums = dict((op, 0.0) for op in OPS)
    elapsed = None
    steps = None
    max_total_ranks = 0

    with path.open("r", errors="replace") as f:
        for line in f:
            # Track num_ranks from the prefix when present (fallback only).
            if num_ranks_override is None:
                m = RANK_TOTAL_RE.search(line)
                if m:
                    total = int(m.group(2))
                    if total > max_total_ranks:
                        max_total_ranks = total

            # Op timing — there can be at most one op per line in practice,
            # but use findall to be safe.
            for op_name, t_str in OP_RE.findall(line):
                try:
                    op_sums[op_name.upper()] += float(t_str)
                except ValueError:
                    pass

            # Elapsed and steps/sec summary lines.
            if elapsed is None:
                m = ELAPSED_RE.search(line)
                if m:
                    try:
                        elapsed = float(m.group(1))
                    except ValueError:
                        pass
            if steps is None:
                m = STEPS_RE.search(line)
                if m:
                    try:
                        steps = float(m.group(1))
                    except ValueError:
                        pass

    if num_ranks_override is not None and num_ranks_override > 0:
        num_ranks = num_ranks_override
    elif max_total_ranks > 0:
        num_ranks = max_total_ranks
    else:
        num_ranks = 1

    # Convert sums-across-ranks to per-rank averages.
    per_rank = dict((op, op_sums[op] / num_ranks) for op in OPS)
    comm_per_rank = sum(per_rank.values())

    def pct(x):
        if elapsed and elapsed > 0:
            return 100.0 * x / elapsed
        return 0.0

    return {
        "num_ranks":     num_ranks,
        "elapsed_s":     elapsed if elapsed is not None else 0.0,
        "steps_per_s":   steps   if steps   is not None else 0.0,
        "op_sums":       op_sums,
        "per_rank":      per_rank,
        "pct":           dict((op, pct(per_rank[op])) for op in OPS),
        "comm_per_rank": comm_per_rank,
        "comm_pct":      pct(comm_per_rank),
    }


def emit_csv(stats):
    row = [
        "{}".format(stats["num_ranks"]),
        "{:.6f}".format(stats["elapsed_s"]),
        "{:.6f}".format(stats["steps_per_s"]),
    ]
    for op in OPS:
        row += [
            "{:.6f}".format(stats["per_rank"][op]),
            "{:.4f}".format(stats["pct"][op]),
        ]
    row += [
        "{:.6f}".format(stats["comm_per_rank"]),
        "{:.4f}".format(stats["comm_pct"]),
    ]
    return ",".join(row)


def emit_human(stats):
    lines = []
    nr = stats["num_ranks"]
    el = stats["elapsed_s"]
    lines.append("num_ranks     : {}".format(nr))
    lines.append("elapsed_s     : {:.6f}".format(el))
    lines.append("steps_per_s   : {:.2f}".format(stats["steps_per_s"]))
    lines.append("")
    lines.append("{:<10} {:>12} {:>12} {:>12}".format("op", "sum_s", "s/rank", "% of total"))
    lines.append("-" * 50)
    for op in OPS:
        lines.append("{:<10} {:>12.6f} {:>12.6f} {:>11.2f}%".format(
            op,
            stats["op_sums"][op],
            stats["per_rank"][op],
            stats["pct"][op],
        ))
    lines.append("-" * 50)
    lines.append("{:<10} {:>12.6f} {:>12.6f} {:>11.2f}%".format(
        "TOTAL",
        sum(stats["op_sums"].values()),
        stats["comm_per_rank"],
        stats["comm_pct"],
    ))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("logfile", type=Path,
                    help="path to log file (use /dev/null with --csv to "
                         "just print the header)")
    ap.add_argument("--csv", action="store_true",
                    help="emit one CSV header line + one CSV data row")
    ap.add_argument("--ranks", type=int, default=None,
                    help="number of MPI ranks for this run (used to "
                         "divide per-op sums into per-rank averages). "
                         "If omitted, the parser tries to infer from log "
                         "prefixes; otherwise falls back to 1.")
    args = ap.parse_args()

    # Header-only mode: useful for the runner to learn the column layout
    # without having to keep it in sync by hand.
    if args.csv and str(args.logfile) in ("/dev/null", "-"):
        print(csv_header())
        return

    if not args.logfile.is_file():
        print("ERROR: not a file: {}".format(args.logfile), file=sys.stderr)
        sys.exit(1)

    stats = parse_log(args.logfile, num_ranks_override=args.ranks)

    if args.csv:
        print(csv_header())
        print(emit_csv(stats))
    else:
        print(emit_human(stats))


if __name__ == "__main__":
    main()
