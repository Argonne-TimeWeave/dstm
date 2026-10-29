#!/usr/bin/env python3
"""
plot_throughput.py -- single-panel throughput figure for the PLANES sweep.

Reads the merged runs.csv produced by plane_sweep.sh (one row per
(impl, planes, rep), both the timing pass and the MPI-activity pass) and
produces ONE figure: throughput (hops/s) vs planes, one line per backend.

Because plane_sweep.sh fixes ranks and sweeps the plane count, the x-axis is
PLANES (CSV col "planes"), not ranks. Both queue implementations are drawn:

    array (darray), graph (dgraph), queue (dqueue, original),
    queue-batch (dqueue, batched), ROSS-opt, ROSS-cons.

Throughput median is computed here directly from runs.csv:
    * median tsps over rows with status == ok, bucketed by planes.

USAGE
    plot_throughput.py RUNS_CSV [options]

    plot_throughput.py results_planes_sweep/<stamp>/runs.csv --out throughput.pdf

OPTIONS
    --mode NAME            mode value to filter on (default: auto-detect;
                           plane_sweep.sh writes mode == "planes")
    --variant NAME         which variant to plot (default: base)
    --out PATH             output figure path (default: throughput.pdf)
    --title STR            overall figure suptitle (default: none)
    --logx / --no-logx     log2 x (planes) axis  (default: on)
    --logy                 log-scale the y axis
    --dpi N                raster DPI for png output (default: 300)

FONTS (conference-safe)
    Output embeds TrueType (fonttype 42) for both PDF and PS backends, so the
    figure contains NO Type 3 fonts (IEEE/ACM PDF-express compliant). Font
    sizes are enlarged for print legibility.
"""

import argparse
import csv
import sys
from collections import defaultdict
from statistics import median

import matplotlib
matplotlib.use("Agg")                 # headless / no DISPLAY on compute nodes

# ---- Conference-safe fonts: embed TrueType (42), never Type 3. ------------
matplotlib.rcParams["pdf.fonttype"] = 42      # TrueType in PDF (not Type 3)
matplotlib.rcParams["ps.fonttype"] = 42       # TrueType in PS/EPS (not Type 3)
matplotlib.rcParams["text.usetex"] = False    # keep mathtext, avoid Type 3

# ---- Larger fonts for a conference paper. --------------------------------
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size":        16,     # base
    "axes.titlesize":   19,
    "axes.labelsize":   18,
    "xtick.labelsize":  15,
    "ytick.labelsize":  15,
    "legend.fontsize":  16,
    "figure.titlesize": 21,
    "lines.linewidth":  2.4,
    "lines.markersize":  8,
    "axes.grid":        True,
    "grid.linestyle":   ":",
    "grid.alpha":       0.5,
})

import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Backend identity: how each impl appears in runs.csv, how it is drawn.
# runs.csv column 1 (impl) uses:
#   darray, graph, queue, queue_batch, ross_opt, ross_cons.
# --------------------------------------------------------------------------
IMPL_ORDER = ["darray", "graph", "queue", "queue_batch", "ross_opt", "ross_cons"]
IMPL_LABEL = {
    "darray":      "array",
    "graph":       "graph",
    "queue":       "queue",
    "queue_batch": "queue-batch",
    "ross_opt":    "ROSS-opt",
    "ross_cons":   "ROSS-cons",
}
IMPL_STYLE = {
    "darray":      dict(color="#1f77b4", marker="o", linestyle="-"),
    "graph":       dict(color="#2ca02c", marker="s", linestyle="--"),
    "queue":       dict(color="#d62728", marker="^", linestyle="-."),
    "queue_batch": dict(color="#8c564b", marker="P", linestyle=":"),
    "ross_opt":    dict(color="#9467bd", marker="D", linestyle="-"),
    "ross_cons":   dict(color="#ff7f0e", marker="v", linestyle="--"),
}

# --------------------------------------------------------------------------
# runs.csv column names we need (match plane_sweep.sh header exactly).
# --------------------------------------------------------------------------
COL_IMPL    = "impl"
COL_VARIANT = "variant"
COL_MODE    = "mode"
COL_PLANES  = "planes"
COL_STATUS  = "status"          # timing-pass status
COL_TSPS    = "tsps"            # throughput: hops/s


def _to_float(tok):
    try:
        return float(tok)
    except (TypeError, ValueError):
        return None


def load_rows(path):
    try:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
    except OSError as exc:
        sys.exit(f"error: cannot read {path}: {exc}")
    if not rows:
        sys.exit(f"error: {path} has no data rows")
    # Sanity: required columns present.
    need = [COL_IMPL, COL_VARIANT, COL_MODE, COL_PLANES, COL_STATUS, COL_TSPS]
    missing = [c for c in need if c not in rows[0]]
    if missing:
        sys.exit(f"error: {path} is missing expected columns: {missing}\n"
                 f"       (is this the runs.csv from plane_sweep.sh?)")
    return rows


def detect_modes(rows):
    seen = []
    for r in rows:
        m = r[COL_MODE]
        if m and m not in seen:
            seen.append(m)
    return seen


def _median_by_planes(rows, mode, variant, value_col, status_col,
                      status_ok="ok"):
    """
    Return {impl: {planes: median_value}} for the given mode/variant, taking
    the median of value_col over rows whose status_col == status_ok.
    Non-numeric or sentinel (-1 / nan) values are ignored.
    """
    buckets = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r[COL_MODE] != mode:
            continue
        if r[COL_VARIANT] != variant:
            continue
        if r.get(status_col) != status_ok:
            continue
        impl = r[COL_IMPL]
        if impl not in IMPL_ORDER:
            continue
        planes = _to_float(r[COL_PLANES])
        val = _to_float(r[value_col])
        if planes is None or val is None:
            continue
        if val < 0:                      # -1 sentinel for skipped/failed
            continue
        buckets[impl][int(planes)].append(val)

    out = defaultdict(dict)
    for impl, byplanes in buckets.items():
        for planes, vals in byplanes.items():
            if vals:
                out[impl][planes] = median(vals)
    return out


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def _plot_panel(ax, data, xlabel, ylabel, logx, logy):
    """data: {impl: {planes: value}}. One line per impl, sorted by planes."""
    any_points = False
    for impl in IMPL_ORDER:
        series = data.get(impl)
        if not series:
            continue
        xs = sorted(series.keys())
        ys = [series[x] for x in xs]
        if not xs:
            continue
        ax.plot(xs, ys, label=IMPL_LABEL[impl], **IMPL_STYLE[impl])
        any_points = True

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    # Use the actual plane values as ticks so a log2 axis reads cleanly.
    all_planes = sorted({p for s in data.values() for p in s})
    if all_planes:
        ax.set_xticks(all_planes)
        ax.set_xticklabels([str(p) for p in all_planes])
        ax.minorticks_off()
    return any_points


def main():
    ap = argparse.ArgumentParser(
        description="Single-panel throughput vs planes figure from the "
                    "plane-sweep runs.csv (handles both queue backends).")
    ap.add_argument("runs_csv",
                    help="runs.csv from plane_sweep.sh")
    ap.add_argument("--mode", default=None,
                    help="mode value to filter on (default: auto-detect; "
                         "plane_sweep.sh writes mode == 'planes')")
    ap.add_argument("--variant", default="base",
                    help="variant to plot (default: base)")
    ap.add_argument("--out", default="throughput.pdf",
                    help="output figure path (default: throughput.pdf)")
    ap.add_argument("--title", default=None, help="figure suptitle")
    ap.add_argument("--logx", dest="logx", action="store_true", default=True)
    ap.add_argument("--no-logx", dest="logx", action="store_false")
    ap.add_argument("--logy", action="store_true", default=False)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    rows = load_rows(args.runs_csv)

    # Pick the mode.
    mode = args.mode
    if mode is None:
        modes = detect_modes(rows)
        if "planes" in modes:
            mode = "planes"
        elif modes:
            mode = modes[0]
        else:
            sys.exit("error: no mode values found in runs.csv.")
        print(f"[plot] auto-selected mode: {mode}  (found: {modes})")

    throughput = _median_by_planes(rows, mode, args.variant,
                                   COL_TSPS, COL_STATUS)

    # Warn loudly about anything missing rather than drawing a silent blank.
    present = [IMPL_LABEL[i] for i in IMPL_ORDER if throughput.get(i)]
    if not present:
        print(f"[plot] WARNING: no throughput data for mode '{mode}', "
              f"variant '{args.variant}'. Figure will be empty.",
              file=sys.stderr)
    else:
        print(f"[plot] throughput: {', '.join(present)}")

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.0))
    _plot_panel(ax, throughput, "planes per airport", "throughput (hops/s)",
                args.logx, args.logy)
    ax.set_title("Throughput vs Planes")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   ncol=min(len(labels), 3), frameon=False,
                   bbox_to_anchor=(0.5, 1.06))

    if args.title:
        fig.suptitle(args.title, y=1.14)

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"[plot] wrote {args.out}  (mode={mode}, variant={args.variant})")


if __name__ == "__main__":
    main()
