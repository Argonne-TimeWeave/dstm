#!/usr/bin/env python3
"""
plot.py -- three-panel backend comparison figure (array/graph/queue/ROSS).

Reads the MERGED runs.csv produced by run_all.sh (one row per config, both the
timing pass and the MPI-activity pass) and produces one figure with three
side-by-side panels sharing ranks on the x-axis:

    (1) throughput        hops/s                (tsps,  CSV col 16)
    (2) MPI calls         calls/rank/tick mean  (CSV col 27)
    (3) bytes sent        bytes_sent/rank/tick  (CSV col 48)

Each panel draws one line per backend:
    array (darray), graph (dgraph), queue (dqueue),
    ROSS-opt (ross_opt), ROSS-cons (ross_cons).

Medians are computed here directly from runs.csv:
    * throughput  : median tsps over rows with status == ok
    * activity    : median of the per-tick columns over rows with
                    act_status == ok

USAGE
    plot.py RUNS_CSV [options]

    plot.py results/<stamp>/runs.csv --mode weak --out compare.pdf

OPTIONS
    --mode {weak|strong}   which scaling mode to plot (default: auto; prefers
                           'strong' if present, else 'weak', else first found)
    --variant NAME         which variant to plot (default: base)
    --out PATH             output figure path (default: three_panel.pdf)
    --title STR            overall figure suptitle (default: none)
    --logx / --no-logx     log2 x (ranks) axis  (default: on)
    --logy                 log-scale all y axes
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
# Must be set BEFORE pyplot import side effects matter; set on rcParams.
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
# runs.csv column 1 (impl) uses: darray, graph, queue, ross_opt, ross_cons.
# --------------------------------------------------------------------------
IMPL_ORDER = ["darray", "graph", "queue", "ross_opt", "ross_cons"]
IMPL_LABEL = {
    "darray":    "array",
    "graph":     "graph",
    "queue":     "queue",
    "ross_opt":  "ROSS-opt",
    "ross_cons": "ROSS-cons",
}
IMPL_STYLE = {
    "darray":    dict(color="#1f77b4", marker="o", linestyle="-"),
    "graph":     dict(color="#2ca02c", marker="s", linestyle="--"),
    "queue":     dict(color="#d62728", marker="^", linestyle="-."),
    "ross_opt":  dict(color="#9467bd", marker="D", linestyle="-"),
    "ross_cons": dict(color="#ff7f0e", marker="v", linestyle="--"),
}

# --------------------------------------------------------------------------
# runs.csv column names we need (match run_all.sh header exactly).
# --------------------------------------------------------------------------
COL_IMPL      = "impl"
COL_VARIANT   = "variant"
COL_MODE      = "mode"
COL_RANKS     = "ranks"
COL_STATUS    = "status"          # timing-pass status
COL_TSPS      = "tsps"            # throughput: hops/s
COL_ACTSTATUS = "act_status"      # activity-pass status
COL_CALLS     = "calls_pertick_mean"
COL_BYTES     = "bytes_sent_pertick_mean"


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
    need = [COL_IMPL, COL_VARIANT, COL_MODE, COL_RANKS, COL_STATUS,
            COL_TSPS, COL_ACTSTATUS, COL_CALLS, COL_BYTES]
    missing = [c for c in need if c not in rows[0]]
    if missing:
        sys.exit(f"error: {path} is missing expected columns: {missing}\n"
                 f"       (is this the merged runs.csv from run_all.sh?)")
    return rows


def detect_modes(rows):
    seen = []
    for r in rows:
        m = r[COL_MODE]
        if m and m not in seen:
            seen.append(m)
    return seen


def _median_by_rank(rows, mode, variant, value_col, status_col, status_ok="ok"):
    """
    Return {impl: {ranks: median_value}} for the given mode/variant, taking the
    median of value_col over rows whose status_col == status_ok. Non-numeric or
    sentinel (-1 / nan) values are ignored.
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
        ranks = _to_float(r[COL_RANKS])
        val = _to_float(r[value_col])
        if ranks is None or val is None:
            continue
        if val < 0:                      # -1 sentinel for skipped/failed
            continue
        buckets[impl][int(ranks)].append(val)

    out = defaultdict(dict)
    for impl, byrank in buckets.items():
        for ranks, vals in byrank.items():
            if vals:
                out[impl][ranks] = median(vals)
    return out


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def _plot_panel(ax, data, ylabel, logx, logy):
    """data: {impl: {ranks: value}}. One line per impl, sorted by ranks."""
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

    ax.set_xlabel("ranks")
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    # Use the actual rank values as ticks so a log2 axis reads cleanly.
    all_ranks = sorted({r for s in data.values() for r in s})
    if all_ranks:
        ax.set_xticks(all_ranks)
        ax.set_xticklabels([str(r) for r in all_ranks])
        ax.minorticks_off()
    return any_points


def main():
    ap = argparse.ArgumentParser(
        description="Three-panel array/graph/queue/ROSS comparison figure "
                    "from the merged runs.csv.")
    ap.add_argument("runs_csv",
                    help="merged runs.csv from run_all.sh")
    ap.add_argument("--mode", choices=["weak", "strong"], default=None,
                    help="scaling mode to plot (default: auto-detect)")
    ap.add_argument("--variant", default="base",
                    help="variant to plot (default: base)")
    ap.add_argument("--out", default="three_panel.pdf",
                    help="output figure path (default: three_panel.pdf)")
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
        if "strong" in modes:
            mode = "strong"
        elif "weak" in modes:
            mode = "weak"
        elif modes:
            mode = modes[0]
        else:
            sys.exit("error: no mode values found in runs.csv.")
        print(f"[plot] auto-selected mode: {mode}  (found: {modes})")

    throughput = _median_by_rank(rows, mode, args.variant,
                                 COL_TSPS, COL_STATUS)
    calls      = _median_by_rank(rows, mode, args.variant,
                                 COL_CALLS, COL_ACTSTATUS)
    bytes_sent = _median_by_rank(rows, mode, args.variant,
                                 COL_BYTES, COL_ACTSTATUS)

    # Warn loudly about anything missing rather than drawing a silent blank.
    for name, d in (("throughput", throughput),
                    ("calls/t", calls),
                    ("bytes_sent/t", bytes_sent)):
        present = [IMPL_LABEL[i] for i in IMPL_ORDER if d.get(i)]
        if not present:
            print(f"[plot] WARNING: no {name} data for mode '{mode}', "
                  f"variant '{args.variant}'. Panel will be empty.",
                  file=sys.stderr)
        else:
            print(f"[plot] {name}: {', '.join(present)}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    _plot_panel(axes[0], throughput, "throughput (hops/s)",
                args.logx, args.logy)
    _plot_panel(axes[1], calls, "MPI calls / rank / tick",
                args.logx, args.logy)
    _plot_panel(axes[2], bytes_sent, "bytes sent / rank / tick",
                args.logx, args.logy)

    axes[0].set_title("(a) Throughput")
    axes[1].set_title("(b) MPI Call Volume")
    axes[2].set_title("(c) Bytes Sent")

    # Single shared legend across the top, from whichever panel has lines.
    handles, labels = None, None
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   ncol=len(labels), frameon=False,
                   bbox_to_anchor=(0.5, 1.04))

    if args.title:
        fig.suptitle(args.title, y=1.10)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"[plot] wrote {args.out}  (mode={mode}, variant={args.variant})")


if __name__ == "__main__":
    main()
