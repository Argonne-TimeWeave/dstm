#!/usr/bin/env bash
#
# strong_scale.sh — strong scaling sweep for DSTM vs RepastHPC.
#
# Fixed GLOBAL agent count N; per-rank counts shrink as ranks grow.
# Produces ONE results directory with a single summary.csv and
# mpi_summary.csv, matching the schema written by compare.sh.
#
# Defaults are chosen so that the FINAL point in SWEEP_CONFIG
# (256:32:8) matches the weak-scaling comparison invocation:
#
#   ./compare.sh --hostfile "$COBALT_NODEFILE" \
#       --sweep-config "1:1:1 2:1:2 4:1:4 8:1:8 16:2:8 32:4:8 64:8:8 128:16:8 256:32:8" \
#       --nh 512 --time 256 --density 1.0 --repeats 4
#
# At 256 ranks: N/ranks = 132352/256 = 517 = nh(512) + nz(5). Exact.
# All other rank counts pack the same 132352 total agents across fewer ranks.
#
# usage:
#   ./strong_scale.sh                       # defaults: N=132352, TIME=256
#   ./strong_scale.sh 132352 256
#   N=16000 TIME=200 SWEEP_CONFIG="4:4:1 8:8:1 16:16:1" ./strong_scale.sh
#
# Env knobs (all optional):
#   N              total agents across all ranks         (default 132352)
#   TIME           ticks to simulate                     (default 256)
#   NZ             zombies per rank                      (default 5)
#   SEED           RNG seed                              (default 42)
#   WORK           DSTM synthetic-work knob              (default 0)
#   REPEATS        repeats per (ranks) point             (default 4)
#   DENSITY        agents/cell target                    (default 1.0)
#   SWEEP_CONFIG   "ranks:ppn:nodes ..." points to sweep
#   ONLY           dstm | repast | both                  (default both)
#   HOSTFILE       hostfile (default: $COBALT_NODEFILE)
#
# Layout expected (same as compare.sh):
#   simple_ABMS_z/
#     zombies
#     scripts/
#       strong_scale.sh          <- this file
#       compare.sh
#       zombie_model
#       config.props
#       model.props
#       results/
#

#set -uo pipefail

export LD_LIBRARY_PATH=~/sfw/Boost/Boost_1.83/lib:${LD_LIBRARY_PATH:-}

# --------------------------------------------------------------------------
# Paths.
# --------------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(   cd "${SCRIPT_DIR}/.." && pwd )"

DSTM_EXE="${DSTM_EXE:-${ROOT_DIR}/zombies}"
REPAST_EXE="${REPAST_EXE:-${SCRIPT_DIR}/zombie_model}"
REPAST_CONFIG_TMPL="${REPAST_CONFIG_TMPL:-${SCRIPT_DIR}/config.props}"
REPAST_MODEL_TMPL="${REPAST_MODEL_TMPL:-${SCRIPT_DIR}/model.props}"

MPIRUN="${MPIRUN:-mpirun}"

# --------------------------------------------------------------------------
# DSTM header we rewrite per-iteration to size the strip arrays.
# --------------------------------------------------------------------------
DSTM_HEADER="${DSTM_HEADER:-${ROOT_DIR}/zombie.h}"
DSTM_BUILD_DIR="${DSTM_BUILD_DIR:-${ROOT_DIR}}"

# --------------------------------------------------------------------------
# Config (positional args + env overrides).
#
# Chosen so that at ranks=256 (the last SWEEP_CONFIG point below):
#   per_rank_total = N / ranks = 132352 / 256 = 517
#   nh = 517 - NZ(5) = 512   <-- matches --nh 512 in compare.sh invocation
# --------------------------------------------------------------------------
N="${1:-${N:-132352}}"
TIME="${2:-${TIME:-256}}"

NZ="${NZ:-5}"
SEED="${SEED:-42}"
WORK="${WORK:-0}"
REPEATS="${REPEATS:-4}"
DENSITY="${DENSITY:-1.0}"
ONLY="${ONLY:-both}"

SWEEP_CONFIG="${SWEEP_CONFIG:-4:1:4 8:1:8 16:2:8 32:4:8 64:8:8 128:16:8 256:32:8}"
HOSTFILE="${HOSTFILE:-${COBALT_NODEFILE:-}}"

case "$ONLY" in
    dstm)   RUN_DSTM=1; RUN_REPAST=0 ;;
    repast) RUN_DSTM=0; RUN_REPAST=1 ;;
    both)   RUN_DSTM=1; RUN_REPAST=1 ;;
    *) echo "bad ONLY value: $ONLY (want dstm|repast|both)" >&2; exit 1 ;;
esac

if (( RUN_DSTM )); then
    if [[ ! -w "$DSTM_HEADER" ]]; then
        echo "!! DSTM header not writable: $DSTM_HEADER" >&2
        exit 1
    fi
fi

# Snapshot the current MAX_CELL_SIDE line so we can restore it on exit.
ORIG_MCS_LINE=""
if (( RUN_DSTM )); then
    ORIG_MCS_LINE="$(grep -E '^[[:space:]]*#define[[:space:]]+MAX_CELL_SIDE[[:space:]]+' "$DSTM_HEADER" | head -n1)"
    if [[ -z "$ORIG_MCS_LINE" ]]; then
        echo "!! could not find '#define MAX_CELL_SIDE ...' in $DSTM_HEADER" >&2
        exit 1
    fi
fi

restore_header() {
    if [[ -n "$ORIG_MCS_LINE" ]]; then
        sed -i -E "s|^[[:space:]]*#define[[:space:]]+MAX_CELL_SIDE[[:space:]]+[0-9]+.*|${ORIG_MCS_LINE}|" "$DSTM_HEADER"
        echo "[cleanup] restored MAX_CELL_SIDE line in $DSTM_HEADER"
    fi
}
trap restore_header EXIT

if [[ -z "$HOSTFILE" ]]; then
    echo "!! HOSTFILE not set and \$COBALT_NODEFILE empty" >&2
    exit 1
fi
if [[ ! -r "$HOSTFILE" ]]; then
    echo "!! hostfile not readable: $HOSTFILE" >&2
    exit 1
fi

# Derive --prefix like compare.sh does.
MPI_PREFIX_OVERRIDE="$(dirname "$(dirname "$(command -v "$MPIRUN")")")"
MAP_BY_TEMPLATE="ppr:PPN:node"
MPI_ENV_FORWARD="LD_LIBRARY_PATH PATH MPI_WRAP_OUTDIR"

# --------------------------------------------------------------------------
# Sanity: executables.
# --------------------------------------------------------------------------
if (( RUN_DSTM )) && [[ ! -x "$DSTM_EXE" ]]; then
    echo "!! DSTM executable not found or not executable: $DSTM_EXE" >&2
    exit 1
fi
if (( RUN_REPAST )); then
    if [[ ! -x "$REPAST_EXE" ]]; then
        echo "!! Repast executable not found or not executable: $REPAST_EXE" >&2
        exit 1
    fi
    for f in "$REPAST_CONFIG_TMPL" "$REPAST_MODEL_TMPL"; do
        [[ -f "$f" ]] || { echo "!! missing: $f" >&2; exit 1; }
    done
fi

# --------------------------------------------------------------------------
# Geometry helpers (copied from compare.sh so this script is standalone).
# --------------------------------------------------------------------------
factor_procs() {
    local n=$1
    local best_x=1 x
    for (( x=1; x*x<=n; x++ )); do
        if (( n % x == 0 )); then best_x=$x; fi
    done
    local y=$(( n / best_x ))
    echo "$best_x $y"
}

compute_geometry() {
    local nprocs=$1
    local agents_per_rank=$2
    local density=$3
    local total_agents=$(( nprocs * agents_per_rank ))

    local ppx ppy
    read -r ppx ppy < <(factor_procs "$nprocs")

    local total_cells
    total_cells=$(awk -v a="$total_agents" -v d="$density" 'BEGIN{
        c = a / d; if (c < 1) c = 1; printf "%d", (c==int(c) ? c : int(c)+1)
    }')

    local nl_per_rank=$(( total_cells / nprocs ))
    (( nl_per_rank < 1 )) && nl_per_rank=1
    local nl=1
    while (( nl < nl_per_rank )); do nl=$(( nl * 2 )); done

    local repast_total=$(( nl * nprocs ))
    local side
    side=$(awk -v n="$repast_total" 'BEGIN{
        s = sqrt(n); si = int(s); if (si*si < n) si++; printf "%d", si
    }')

    local min_patch=5
    local min_side_x=$(( ppx * min_patch ))
    local min_side_y=$(( ppy * min_patch ))
    (( side < min_side_x )) && side=$min_side_x
    (( side < min_side_y )) && side=$min_side_y

    local a=$ppx b=$ppy tmp
    while (( b != 0 )); do tmp=$(( a % b )); a=$b; b=$tmp; done
    local gcd=$a
    local lcm=$(( ppx * ppy / gcd ))
    local rem=$(( side % lcm ))
    if (( rem != 0 )); then side=$(( side + lcm - rem )); fi

    local half=$(( side / 2 ))
    local min=$(( -half ))
    local max=$(( min + side - 1 ))

    echo "$nl $min $max"
}

# --------------------------------------------------------------------------
# Compute the smallest cell_side such that cell_side^2 >= nl.
# Mirrors zombie_init's ceil-sqrt logic exactly.
# --------------------------------------------------------------------------
cell_side_for_nl() {
    local nl=$1
    local s=1
    while (( s * s < nl )); do s=$(( s + 1 )); done
    echo "$s"
}

# --------------------------------------------------------------------------
# Rewrite MAX_CELL_SIDE in the DSTM header, then rebuild.
# --------------------------------------------------------------------------
set_max_cell_side_and_build() {
    local new_val=$1
    local header=$DSTM_HEADER

    local cur_val
    cur_val="$(grep -E '^[[:space:]]*#define[[:space:]]+MAX_CELL_SIDE[[:space:]]+' "$header" \
               | head -n1 | awk '{print $3}')"

    if [[ "$cur_val" == "$new_val" && -x "$DSTM_EXE" ]]; then
        echo "[build] MAX_CELL_SIDE already $new_val and executable present; skipping rebuild"
        return 0
    fi

    echo "[build] setting MAX_CELL_SIDE=$new_val in $header (was $cur_val)"
    sed -i -E "s|^([[:space:]]*#define[[:space:]]+MAX_CELL_SIDE[[:space:]]+)[0-9]+.*|\1${new_val} /* auto-set by strong_scale.sh */|" \
        "$header"

    echo "[build] make -C $DSTM_BUILD_DIR"
    if ! ( cd "$DSTM_BUILD_DIR" && make ) >"${OUT_ROOT}/build_mcs${new_val}.log" 2>&1; then
        echo "!! build failed (MAX_CELL_SIDE=$new_val); see ${OUT_ROOT}/build_mcs${new_val}.log" >&2
        exit 1
    fi
}

# --------------------------------------------------------------------------
# mpirun invocation prefix for (ranks, ppn).
# --------------------------------------------------------------------------
mpirun_cmd() {
    local ranks=$1 ppn=$2
    local -a cmd=("$MPIRUN" -n "$ranks"
                  --hostfile "$HOSTFILE"
                  --prefix   "$MPI_PREFIX_OVERRIDE")

    local map_by="${MAP_BY_TEMPLATE//PPN/$ppn}"
    cmd+=(--map-by "$map_by")

    local v
    for v in $MPI_ENV_FORWARD; do
        cmd+=(-x "$v")
    done

    printf '%s\n' "${cmd[@]}"
}

# --------------------------------------------------------------------------
# Output setup — SINGLE results dir, SINGLE summary + mpi_summary.
# --------------------------------------------------------------------------
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="${SCRIPT_DIR}/results/strong_${STAMP}"
mkdir -p "$OUT_ROOT"

SUMMARY_CSV="${OUT_ROOT}/summary.csv"
echo "framework,ranks,ppn,nodes,nh,nz,time,density,nl,seed,repeat,elapsed_s,tps,status" > "$SUMMARY_CSV"

MPI_SUMMARY_CSV="${OUT_ROOT}/mpi_summary.csv"
echo "framework,ranks,ppn,nodes,nh,nz,time,density,nl,seed,repeat,elapsed_s,\
n_ranks_seen,\
isend,irecv,send,recv,wait,waitall,test,\
barrier,reduce,allreduce,bcast,gather,allgather,alltoall,\
bytes_sent,bytes_recvd,\
total_calls,calls_per_rank_per_tick,bytes_per_rank_per_tick" > "$MPI_SUMMARY_CSV"

echo "strong-scaling sweep"
echo "  N (total agents) = $N"
echo "  TIME             = $TIME"
echo "  REPEATS          = $REPEATS"
echo "  DENSITY          = $DENSITY"
echo "  NZ (per rank)    = $NZ"
echo "  SWEEP_CONFIG     = $SWEEP_CONFIG"
echo "  hostfile         = $HOSTFILE"
echo "  results          -> $OUT_ROOT"
echo

# --------------------------------------------------------------------------
# Elapsed-time extractors.
# --------------------------------------------------------------------------
extract_dstm_elapsed() {
    grep -E '^Elapsed time:' "$1" | awk '{print $3}' | tail -n1
}
extract_repast_elapsed() {
    local logdir=$1
    local csv="${logdir}/Zombies.csv"
    if [[ -f "$csv" ]]; then
        awk -F, 'NR==1{ for (i=1;i<=NF;i++) if ($i=="run.time") c=i }
                 NR>1 && c { last=$c }
                 END { if (last!="") print last }' "$csv"
    fi
}

# --------------------------------------------------------------------------
# MPI wrap aggregation (same as compare.sh).
# --------------------------------------------------------------------------
aggregate_mpi_counts() {
    local rundir=$1 ranks=$2 timesteps=$3

    if ! ls "$rundir"/mpi_counts_rank*.txt >/dev/null 2>&1; then
        printf "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0000,0.00\n"
        return
    fi

    awk -v timesteps="$timesteps" '
        BEGIN {
            n_seen = 0
            isend = irecv = send = recv = 0
            wait = waitall = test = 0
            barrier = reduce = allreduce = 0
            bcast = gather = allgather = alltoall = 0
            bytes_s = bytes_r = 0
        }
        FNR == 1 { n_seen++ }
        /^isend/    { isend += $2; irecv += $4; send += $6; recv += $8 }
        /^wait/     { wait  += $2; waitall += $4; test += $6 }
        /^barrier/  {
            barrier   += $2; reduce   += $4; allreduce += $6
            bcast     += $8; gather   += $10; allgather += $12; alltoall += $14
        }
        /^bytes_sent/ { bytes_s += $2; bytes_r += $4 }
        END {
            total_calls = isend + irecv + send + recv + wait + waitall + test \
                        + barrier + reduce + allreduce \
                        + bcast + gather + allgather + alltoall
            if (n_seen > 0 && timesteps > 0) {
                cpt = total_calls / n_seen / timesteps
                bpt = bytes_s     / n_seen / timesteps
            } else { cpt = 0; bpt = 0 }
            printf "%d,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.4f,%.2f\n",
                n_seen,
                isend, irecv, send, recv, wait, waitall, test,
                barrier, reduce, allreduce, bcast, gather, allgather, alltoall,
                bytes_s, bytes_r,
                total_calls, cpt, bpt
        }
    ' "$rundir"/mpi_counts_rank*.txt
}

emit_mpi_row() {
    local framework=$1 ranks=$2 ppn=$3 nodes=$4 nh=$5 nz=$6 time=$7
    local density=$8 nl=$9 seed=${10} rep=${11} elapsed=${12} rundir=${13}
    local frag
    frag=$(aggregate_mpi_counts "$rundir" "$ranks" "$time")
    echo "${framework},${ranks},${ppn},${nodes},${nh},${nz},${time},${density},${nl},${seed},${rep},${elapsed},${frag}" \
        >> "$MPI_SUMMARY_CSV"
}

# --------------------------------------------------------------------------
# Runners (mirror compare.sh, single-dir output).
# --------------------------------------------------------------------------
run_dstm() {
    local ranks=$1 ppn=$2 nodes=$3 nh=$4 nz=$5 time=$6 nl=$7 density=$8 seed=$9 rep=${10}
    local rundir="${OUT_ROOT}/dstm/np${ranks}_ppn${ppn}_nh${nh}_nz${nz}_t${time}_nl${nl}_d${density}_s${seed}/r${rep}"
    mkdir -p "$rundir"
    local log="${rundir}/stdout.log"

    local -a launcher
    mapfile -t launcher < <(mpirun_cmd "$ranks" "$ppn")

    local -a cmd=("${launcher[@]}"
        "$DSTM_EXE"
        --seed "$seed" --time "$time"
        --nl   "$nl"   --nh   "$nh"   --nz "$nz"
        --work "$WORK" --quiet
    )

    echo "[dstm  ] ${cmd[*]}"
    local rc=0
    ( cd "$rundir" && MPI_WRAP_OUTDIR="$rundir" "${cmd[@]}" ) > "$log" 2>&1 || rc=$?

    local status="ok" elapsed tps
    if (( rc != 0 )); then status="exit_${rc}"; fi
    elapsed=$(extract_dstm_elapsed "$log")
    if [[ -z "$elapsed" ]]; then
        [[ "$status" == "ok" ]] && status="parse_fail"
        elapsed=NA; tps=NA
    else
        tps=$(awk -v t="$time" -v e="$elapsed" 'BEGIN{ if (e>0) printf "%.2f", t/e; else print "NA" }')
    fi

    echo "dstm,$ranks,$ppn,$nodes,$nh,$nz,$time,$density,$nl,$seed,$rep,$elapsed,$tps,$status" >> "$SUMMARY_CSV"
    printf "         elapsed=%ss  tps=%s  status=%s\n" "$elapsed" "$tps" "$status"

    emit_mpi_row "dstm" "$ranks" "$ppn" "$nodes" "$nh" "$nz" "$time" \
                 "$density" "$nl" "$seed" "$rep" "$elapsed" "$rundir"
}

render_repast_props() {
    local outfile=$1 nprocs=$2 nh=$3 nz=$4 time=$5 seed=$6 rmin=$7 rmax=$8
    local ppx ppy
    read -r ppx ppy < <(factor_procs "$nprocs")

    cp "$REPAST_MODEL_TMPL" "$outfile"

    _set() {
        local key=$1 val=$2 file=$3
        if grep -Eq "^[[:space:]]*${key}[[:space:]]*=" "$file"; then
            local esc; esc=$(printf '%s' "$val" | sed -e 's/[\/&]/\\&/g')
            sed -i -E "s|^([[:space:]]*${key}[[:space:]]*=).*|\1 ${esc}|" "$file"
        else
            echo "${key} = ${val}" >> "$file"
        fi
    }

    _set "stop.at"      "$time" "$outfile"
    _set "human.count"  "$nh"   "$outfile"
    _set "zombie.count" "$nz"   "$outfile"
    _set "random.seed"  "$seed" "$outfile"
    _set "proc.per.x"   "$ppx"  "$outfile"
    _set "proc.per.y"   "$ppy"  "$outfile"
    _set "min.x"        "$rmin" "$outfile"
    _set "max.x"        "$rmax" "$outfile"
    _set "min.y"        "$rmin" "$outfile"
    _set "max.y"        "$rmax" "$outfile"
}

run_repast() {
    local ranks=$1 ppn=$2 nodes=$3 nh=$4 nz=$5 time=$6 density=$7 seed=$8 rep=$9 rmin=${10} rmax=${11}
    local rundir="${OUT_ROOT}/repast/np${ranks}_ppn${ppn}_nh${nh}_nz${nz}_t${time}_d${density}_s${seed}/r${rep}"
    mkdir -p "$rundir"
    mkdir -p "${rundir}/logs"

    local model_props="${rundir}/model.props"
    render_repast_props "$model_props" "$ranks" "$nh" "$nz" "$time" "$seed" "$rmin" "$rmax"

    local log="${rundir}/stdout.log"

    local -a launcher
    mapfile -t launcher < <(mpirun_cmd "$ranks" "$ppn")

    local -a cmd=("${launcher[@]}"
        "$REPAST_EXE" "$REPAST_CONFIG_TMPL" "$model_props"
    )

    echo "[repast] ${cmd[*]}"
    local rc=0
    ( cd "$rundir" && MPI_WRAP_OUTDIR="$rundir" "${cmd[@]}" ) > "$log" 2>&1 || rc=$?

    local status="ok" elapsed tps
    if (( rc != 0 )); then status="exit_${rc}"; fi
    elapsed=$(extract_repast_elapsed "$rundir")
    if [[ -z "$elapsed" ]]; then
        [[ "$status" == "ok" ]] && status="parse_fail"
        elapsed=NA; tps=NA
    else
        tps=$(awk -v t="$time" -v e="$elapsed" 'BEGIN{ if (e>0) printf "%.2f", t/e; else print "NA" }')
    fi

    echo "repast,$ranks,$ppn,$nodes,$nh,$nz,$time,$density,-,$seed,$rep,$elapsed,$tps,$status" >> "$SUMMARY_CSV"
    printf "         elapsed=%ss  tps=%s  status=%s\n" "$elapsed" "$tps" "$status"

    emit_mpi_row "repast" "$ranks" "$ppn" "$nodes" "$nh" "$nz" "$time" \
                 "$density" "-" "$seed" "$rep" "$elapsed" "$rundir"
}

# --------------------------------------------------------------------------
# The strong-scaling sweep itself.
#
# For each (ranks, ppn, nodes) point:
#   per_rank_total = max(1, N / ranks)
#   nh             = max(0, per_rank_total - NZ)
# Global agent count is (approximately) fixed at N across the sweep.
# --------------------------------------------------------------------------
for entry in $SWEEP_CONFIG; do
    if [[ ! "$entry" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
        echo "!! bad SWEEP_CONFIG entry: '$entry' (want ranks:ppn:nodes)" >&2
        exit 1
    fi
    ranks=${BASH_REMATCH[1]}
    ppn=${BASH_REMATCH[2]}
    nodes=${BASH_REMATCH[3]}

    per_rank_total=$(( N / ranks ))
    (( per_rank_total < 1 )) && per_rank_total=1
    nh=$(( per_rank_total - NZ ))
    (( nh < 0 )) && nh=0
    nz=$NZ

    apr=$(( nh + nz ))
    read -r nl_auto rmin rmax < <(compute_geometry "$ranks" "$apr" "$DENSITY")

    if (( RUN_DSTM )); then
        needed_cs=$(cell_side_for_nl "$nl_auto")
        set_max_cell_side_and_build "$needed_cs"
    fi

    for (( rep=1; rep<=REPEATS; rep++ )); do
        echo "===== ranks=$ranks ppn=$ppn nodes=$nodes N=$N nh=$nh nz=$nz time=$TIME density=$DENSITY nl=$nl_auto repast=[${rmin},${rmax}]^2 rep=$rep ====="

        if (( RUN_DSTM   )); then
            run_dstm   "$ranks" "$ppn" "$nodes" "$nh" "$nz" "$TIME" "$nl_auto" "$DENSITY" "$SEED" "$rep"
        fi
        if (( RUN_REPAST )); then
            run_repast "$ranks" "$ppn" "$nodes" "$nh" "$nz" "$TIME" "$DENSITY" "$SEED" "$rep" "$rmin" "$rmax"
        fi
        echo
    done
done

# --------------------------------------------------------------------------
# Pretty summary (same as compare.sh).
# --------------------------------------------------------------------------
echo "===== summary ($SUMMARY_CSV) ====="
if command -v column >/dev/null 2>&1; then
    column -s, -t < "$SUMMARY_CSV"
else
    cat "$SUMMARY_CSV"
fi

echo
echo "===== MPI activity summary ($MPI_SUMMARY_CSV) ====="
if command -v column >/dev/null 2>&1; then
    awk -F, 'BEGIN{OFS=","} {
        print $1,$2,$3,$4,$5,$6,$7,$12, $14,$15,$16,$17,$18,$21,$23,$24,$29,$30,$31
    }' "$MPI_SUMMARY_CSV" | column -s, -t
else
    cat "$MPI_SUMMARY_CSV"
fi
