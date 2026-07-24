#!/usr/bin/env bash
#
# strong_scale_timing.sh — strong-scaling sweep for timing_zombie_model + DSTM.
#
# Fixed GLOBAL agent count N; per-rank counts shrink as ranks grow.
# Per sweep point runs BOTH:
#   1) timing_zombie_model  -> emits per-rank sync floats to stdout
#   2) ../zombies (DSTM)    -> emits _GET/_PUT timings to stdout
#
# Emits ONE row per binary per repeat into summary.csv with a unified
# schema. Unused columns for a given framework are NA.
#
# See summary.csv header for column meanings; same schema as weak_scale_timing.sh.
#

set -uo pipefail

export LD_LIBRARY_PATH=~/sfw/Boost/Boost_1.83/lib:${LD_LIBRARY_PATH:-}

# --------------------------------------------------------------------------
# Paths.
# --------------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(   cd "${SCRIPT_DIR}/.." && pwd )"

TIMING_EXE="${TIMING_EXE:-${SCRIPT_DIR}/timing_zombie_model}"
DSTM_EXE="${DSTM_EXE:-${ROOT_DIR}/zombies}"
REPAST_CONFIG_TMPL="${REPAST_CONFIG_TMPL:-${SCRIPT_DIR}/config.props}"
REPAST_MODEL_TMPL="${REPAST_MODEL_TMPL:-${SCRIPT_DIR}/model.props}"

DSTM_HEADER="${DSTM_HEADER:-${ROOT_DIR}/zombie.h}"
DSTM_BUILD_DIR="${DSTM_BUILD_DIR:-${ROOT_DIR}}"

MPIRUN="${MPIRUN:-mpirun}"

# --------------------------------------------------------------------------
# Config (positional args + env overrides), matching strong_scale.sh.
# --------------------------------------------------------------------------
N="${1:-${N:-132352}}"
TIME="${2:-${TIME:-256}}"

NZ="${NZ:-5}"
SEED="${SEED:-42}"
WORK="${WORK:-0}"
REPEATS="${REPEATS:-4}"
DENSITY="${DENSITY:-1.0}"
ONLY="${ONLY:-both}"    # timing | dstm | both

SWEEP_CONFIG="${SWEEP_CONFIG:-4:1:4 8:1:8 16:2:8 32:4:8 64:8:8 128:16:8 256:32:8}"
HOSTFILE="${HOSTFILE:-${COBALT_NODEFILE:-}}"

case "$ONLY" in
    timing) RUN_TIMING=1; RUN_DSTM=0 ;;
    dstm)   RUN_TIMING=0; RUN_DSTM=1 ;;
    both)   RUN_TIMING=1; RUN_DSTM=1 ;;
    *) echo "bad ONLY value: $ONLY (want timing|dstm|both)" >&2; exit 1 ;;
esac

if [[ -z "$HOSTFILE" ]]; then
    echo "!! HOSTFILE not set and \$COBALT_NODEFILE empty" >&2
    exit 1
fi
if [[ ! -r "$HOSTFILE" ]]; then
    echo "!! hostfile not readable: $HOSTFILE" >&2
    exit 1
fi

# --------------------------------------------------------------------------
# Sanity.
# --------------------------------------------------------------------------
if (( RUN_TIMING )) && [[ ! -x "$TIMING_EXE" ]]; then
    echo "!! timing_zombie_model not found or not executable: $TIMING_EXE" >&2
    exit 1
fi
if (( RUN_DSTM )) && [[ ! -x "$DSTM_EXE" ]]; then
    echo "!! DSTM executable not found or not executable: $DSTM_EXE" >&2
    exit 1
fi
if (( RUN_TIMING )); then
    for f in "$REPAST_CONFIG_TMPL" "$REPAST_MODEL_TMPL"; do
        [[ -f "$f" ]] || { echo "!! missing: $f" >&2; exit 1; }
    done
fi

ORIG_MCS_LINE=""
if (( RUN_DSTM )); then
    if [[ ! -w "$DSTM_HEADER" ]]; then
        echo "!! DSTM header not writable: $DSTM_HEADER" >&2
        exit 1
    fi
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

MPI_PREFIX_OVERRIDE="$(dirname "$(dirname "$(command -v "$MPIRUN")")")"
MAP_BY_TEMPLATE="ppr:PPN:node"
MPI_ENV_FORWARD="LD_LIBRARY_PATH PATH"

# --------------------------------------------------------------------------
# Geometry helpers.
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

cell_side_for_nl() {
    local nl=$1
    local s=1
    while (( s * s < nl )); do s=$(( s + 1 )); done
    echo "$s"
}

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
    sed -i -E "s|^([[:space:]]*#define[[:space:]]+MAX_CELL_SIDE[[:space:]]+)[0-9]+.*|\1${new_val} /* auto-set by strong_scale_timing.sh */|" \
        "$header"

    echo "[build] make -C $DSTM_BUILD_DIR"
    if ! ( cd "$DSTM_BUILD_DIR" && make ) >"${OUT_ROOT}/build_mcs${new_val}.log" 2>&1; then
        echo "!! build failed (MAX_CELL_SIDE=$new_val); see ${OUT_ROOT}/build_mcs${new_val}.log" >&2
        exit 1
    fi
}

# --------------------------------------------------------------------------
# mpirun invocation prefix.
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
# Output setup.
# --------------------------------------------------------------------------
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="${SCRIPT_DIR}/results/strong_timing_${STAMP}"
mkdir -p "$OUT_ROOT"

SUMMARY_CSV="${OUT_ROOT}/summary.csv"
echo "framework,ranks,ppn,nodes,nh,nz,time,density,nl,seed,repeat,elapsed_s,tps,\
sync_total_s,sync_per_rank_s,sync_pct,n_sync_samples,\
get_total_s,get_per_rank_s,put_total_s,put_per_rank_s,\
comm_total_s,comm_per_rank_s,comm_pct,n_get,n_put,\
status" > "$SUMMARY_CSV"

echo "strong-scaling sweep (timing_zombie_model + DSTM)"
echo "  N (total agents) = $N"
echo "  TIME             = $TIME"
echo "  REPEATS          = $REPEATS"
echo "  DENSITY          = $DENSITY"
echo "  NZ (per rank)    = $NZ"
echo "  SWEEP_CONFIG     = $SWEEP_CONFIG"
echo "  ONLY             = $ONLY"
echo "  hostfile         = $HOSTFILE"
echo "  results          -> $OUT_ROOT"
echo

# --------------------------------------------------------------------------
# Extractors.
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

aggregate_sync_times() {
    local log=$1
    awk '
        /^[[:space:]]*[+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?[[:space:]]*$/ {
            s += $1 + 0; n += 1
        }
        END { printf "%.6f %d\n", s+0, n+0 }
    ' "$log"
}

aggregate_comm_times() {
    local log=$1
    awk '
        /^[[:space:]]*_GET:[[:space:]]+[+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?[[:space:]]*$/ {
            gs += $2 + 0; gn += 1
        }
        /^[[:space:]]*_PUT:[[:space:]]+[+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?[[:space:]]*$/ {
            ps += $2 + 0; pn += 1
        }
        END { printf "%.6f %d %.6f %d\n", gs+0, gn+0, ps+0, pn+0 }
    ' "$log"
}

# --------------------------------------------------------------------------
# Repast props renderer.
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# Runner: timing_zombie_model.
# --------------------------------------------------------------------------
run_timing() {
    local ranks=$1 ppn=$2 nodes=$3 nh=$4 nz=$5 time=$6 density=$7 nl=$8 \
          seed=$9 rep=${10} rmin=${11} rmax=${12}
    local rundir="${OUT_ROOT}/timing/np${ranks}_ppn${ppn}_nh${nh}_nz${nz}_t${time}_d${density}_s${seed}/r${rep}"
    mkdir -p "${rundir}/logs"

    local model_props="${rundir}/model.props"
    render_repast_props "$model_props" "$ranks" "$nh" "$nz" "$time" "$seed" "$rmin" "$rmax"

    local log="${rundir}/stdout.log"

    local -a launcher
    mapfile -t launcher < <(mpirun_cmd "$ranks" "$ppn")

    local -a cmd=("${launcher[@]}"
        "$TIMING_EXE" "$REPAST_CONFIG_TMPL" "$model_props"
    )

    echo "[timing] ${cmd[*]}"
    local rc=0
    ( cd "$rundir" && "${cmd[@]}" ) > "$log" 2>&1 || rc=$?

    local status="ok" elapsed tps
    if (( rc != 0 )); then status="exit_${rc}"; fi
    elapsed=$(extract_repast_elapsed "$rundir")
    if [[ -z "$elapsed" ]]; then
        [[ "$status" == "ok" ]] && status="parse_fail"
        elapsed=NA; tps=NA
    else
        tps=$(awk -v t="$time" -v e="$elapsed" 'BEGIN{ if (e>0) printf "%.2f", t/e; else print "NA" }')
    fi

    local sync_total sync_n sync_per_rank sync_pct
    read -r sync_total sync_n < <(aggregate_sync_times "$log")

    if [[ "$elapsed" == "NA" || "$sync_n" == "0" ]]; then
        sync_per_rank=NA; sync_pct=NA
    else
        sync_per_rank=$(awk -v s="$sync_total" -v r="$ranks" 'BEGIN{ printf "%.6f", s/r }')
        sync_pct=$(awk -v s="$sync_total" -v r="$ranks" -v e="$elapsed" \
                   'BEGIN{ if (e>0) printf "%.2f", 100.0*(s/r)/e; else print "NA" }')
    fi

    echo "timing,$ranks,$ppn,$nodes,$nh,$nz,$time,$density,$nl,$seed,$rep,$elapsed,$tps,\
$sync_total,$sync_per_rank,$sync_pct,$sync_n,\
NA,NA,NA,NA,NA,NA,NA,NA,NA,\
$status" >> "$SUMMARY_CSV"
    printf "         elapsed=%ss  tps=%s  sync_total=%ss  sync/rank=%ss  sync%%=%s  n=%s  status=%s\n" \
        "$elapsed" "$tps" "$sync_total" "$sync_per_rank" "$sync_pct" "$sync_n" "$status"
}

# --------------------------------------------------------------------------
# Runner: DSTM ../zombies.
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
    ( cd "$rundir" && "${cmd[@]}" ) > "$log" 2>&1 || rc=$?

    local status="ok" elapsed tps
    if (( rc != 0 )); then status="exit_${rc}"; fi
    elapsed=$(extract_dstm_elapsed "$log")
    if [[ -z "$elapsed" ]]; then
        [[ "$status" == "ok" ]] && status="parse_fail"
        elapsed=NA; tps=NA
    else
        tps=$(awk -v t="$time" -v e="$elapsed" 'BEGIN{ if (e>0) printf "%.2f", t/e; else print "NA" }')
    fi

    local get_total get_n put_total put_n
    read -r get_total get_n put_total put_n < <(aggregate_comm_times "$log")

    local comm_total get_per_rank put_per_rank comm_per_rank comm_pct
    comm_total=$(awk -v g="$get_total" -v p="$put_total" 'BEGIN{ printf "%.6f", g+p }')

    if [[ "$elapsed" == "NA" || ( "$get_n" == "0" && "$put_n" == "0" ) ]]; then
        get_per_rank=NA; put_per_rank=NA; comm_per_rank=NA; comm_pct=NA
    else
        get_per_rank=$(awk  -v s="$get_total"  -v r="$ranks" 'BEGIN{ printf "%.6f", s/r }')
        put_per_rank=$(awk  -v s="$put_total"  -v r="$ranks" 'BEGIN{ printf "%.6f", s/r }')
        comm_per_rank=$(awk -v s="$comm_total" -v r="$ranks" 'BEGIN{ printf "%.6f", s/r }')
        comm_pct=$(awk -v s="$comm_total" -v r="$ranks" -v e="$elapsed" \
                   'BEGIN{ if (e>0) printf "%.2f", 100.0*(s/r)/e; else print "NA" }')
    fi

    echo "dstm,$ranks,$ppn,$nodes,$nh,$nz,$time,$density,$nl,$seed,$rep,$elapsed,$tps,\
NA,NA,NA,NA,\
$get_total,$get_per_rank,$put_total,$put_per_rank,\
$comm_total,$comm_per_rank,$comm_pct,$get_n,$put_n,\
$status" >> "$SUMMARY_CSV"
    printf "         elapsed=%ss  tps=%s  get=%ss(n=%s)  put=%ss(n=%s)  comm/rank=%ss  comm%%=%s  status=%s\n" \
        "$elapsed" "$tps" "$get_total" "$get_n" "$put_total" "$put_n" \
        "$comm_per_rank" "$comm_pct" "$status"
}

# --------------------------------------------------------------------------
# Strong-scaling sweep.
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

        if (( RUN_TIMING )); then
            run_timing "$ranks" "$ppn" "$nodes" "$nh" "$nz" "$TIME" "$DENSITY" \
                       "$nl_auto" "$SEED" "$rep" "$rmin" "$rmax"
        fi
        if (( RUN_DSTM )); then
            run_dstm   "$ranks" "$ppn" "$nodes" "$nh" "$nz" "$TIME" \
                       "$nl_auto" "$DENSITY" "$SEED" "$rep"
        fi
        echo
    done
done

# --------------------------------------------------------------------------
# Pretty summary.
# --------------------------------------------------------------------------
echo "===== summary ($SUMMARY_CSV) ====="
if command -v column >/dev/null 2>&1; then
    column -s, -t < "$SUMMARY_CSV"
else
    cat "$SUMMARY_CSV"
fi
