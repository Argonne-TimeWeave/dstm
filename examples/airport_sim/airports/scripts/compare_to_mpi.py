#!/usr/bin/env python3
"""
Clustered stacked bar chart comparing two weak-scaling CSVs:
  * DSTM CSV: columns include get_avg_s, put_avg_s, elapsed_s
  * MPI all-to-all CSV: columns include all_avg_s, elapsed_s

For each rank count, two side-by-side stacked bars are drawn:
  DSTM     : [get_or_default | put | other]
  MPI a2a  : [alltoall       | other       ]

Rank counts that don't appear in BOTH CSVs are skipped (e.g. 128
which is intentionally absent from the MPI run).

Usage:
    python3 plot_compare.py <dstm.csv> <mpi.csv> [-o out.png] [--show]
"""

import argparse
import csv
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def _avg(rows, key):
    vals = []
    for r in rows:
        v = r.get(key, "")
        if v == "" or v is None:
            continue
        try:
            vals.append(float(v))
        except ValueError:
            pass
    return float(np.mean(vals)) if vals else 0.0


def load_csv(path, required):
    """Return dict: ranks -> averaged metrics dict."""
    by_rank = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"ERROR: {path} missing columns: {sorted(missing)}")
        for row in reader:
            try:
                r = int(row["ranks"])
            except (KeyError, ValueError):
                continue
            by_rank[r].append(row)

    out = {}
    for r, rows in by_rank.items():
        out[r] = {k: _avg(rows, k) for k in required if k != "ranks"}
    return out


def main():
    ap = argparse.ArgumentParser(description="Clustered stacked bar chart: DSTM vs MPI all-to-all.")
    ap.add_argument("dstm_csv", help="DSTM CSV (get_avg_s, put_avg_s, elapsed_s)")
    ap.add_argument("mpi_csv",  help="MPI all-to-all CSV (all_avg_s, elapsed_s)")
    ap.add_argument("-o", "--output", default=None,
                    help="Output image path (default: compare_<dstm-stem>_vs_<mpi-stem>.png)")
    ap.add_argument("--show", action="store_true", help="Open interactive window")
    ap.add_argument("--title", default=None, help="Override plot title")
    ap.add_argument("--dstm-label", default="DSTM", help="Legend label for first dataset")
    ap.add_argument("--mpi-label",  default="MPI all-to-all", help="Legend label for second dataset")
    args = ap.parse_args()

    dstm = load_csv(args.dstm_csv, {"ranks", "elapsed_s", "get_avg_s", "put_avg_s"})
    mpi  = load_csv(args.mpi_csv,  {"ranks", "elapsed_s", "all_avg_s"})

    common = sorted(set(dstm) & set(mpi))
    if not common:
        sys.exit("ERROR: no rank counts appear in both CSVs")

    dstm_only = sorted(set(dstm) - set(mpi))
    mpi_only  = sorted(set(mpi)  - set(dstm))
    if dstm_only:
        print(f"NOTE: skipping ranks only in DSTM CSV: {dstm_only}")
    if mpi_only:
        print(f"NOTE: skipping ranks only in MPI CSV:  {mpi_only}")

    # DSTM stack components
    d_elapsed = np.array([dstm[r]["elapsed_s"]                    for r in common])
    d_get     = np.array([dstm[r]["get_avg_s"]                    for r in common])
    d_put     = np.array([dstm[r]["put_avg_s"]                    for r in common])
    d_other   = np.maximum(d_elapsed - d_get - d_put, 0.0)

    # MPI stack components
    m_elapsed = np.array([mpi[r]["elapsed_s"]                     for r in common])
    m_all     = np.array([mpi[r]["all_avg_s"]                     for r in common])
    m_other   = np.maximum(m_elapsed - m_all, 0.0)

    x = np.arange(len(common), dtype=float)
    width = 0.38
    off = width / 2 + 0.01  # tiny gap between the two bars in a cluster

    fig, ax = plt.subplots(figsize=(max(8, 0.95 * len(common) + 3), 6))

    # Colors: keep the DSTM "other" and MPI "other" visually distinct but related.
    c_get   = "#4C72B0"
    c_put   = "#DD8452"
    c_all   = "#55A868"
    c_other_d = "#BFBFBF"
    c_other_m = "#8C8C8C"

    # DSTM bars (left of each cluster)
    xd = x - off
    ax.bar(xd, d_get,   width, label=f"{args.dstm_label}: get_or_default", color=c_get)
    ax.bar(xd, d_put,   width, bottom=d_get,
           label=f"{args.dstm_label}: put", color=c_put)
    ax.bar(xd, d_other, width, bottom=d_get + d_put,
           label=f"{args.dstm_label}: other", color=c_other_d)

    # MPI bars (right of each cluster)
    xm = x + off
    ax.bar(xm, m_all,   width, label=f"{args.mpi_label}: alltoall", color=c_all)
    ax.bar(xm, m_other, width, bottom=m_all,
           label=f"{args.mpi_label}: other", color=c_other_m)

    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Per-rank average time (s)")
    title = args.title or f"Time breakdown vs. ranks: {args.dstm_label} vs {args.mpi_label}"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in common])
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=8, ncol=1)

    # Annotate each bar with its elapsed total.
    ymax = max(d_elapsed.max(), m_elapsed.max())
    pad = ymax * 0.01
    for xi, total in zip(xd, d_elapsed):
        ax.text(xi, total + pad, f"{total:.2f}s",
                ha="center", va="bottom", fontsize=7, color="#333")
    for xi, total in zip(xm, m_elapsed):
        ax.text(xi, total + pad, f"{total:.2f}s",
                ha="center", va="bottom", fontsize=7, color="#333")
    ax.set_ylim(0, ymax * 1.14)

    fig.tight_layout()

    out = args.output
    if out is None:
        d_stem = args.dstm_csv.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].rsplit(".", 1)[0]
        m_stem = args.mpi_csv.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].rsplit(".", 1)[0]
        out = f"compare_{d_stem}_vs_{m_stem}.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
