#!/usr/bin/env bash
#
# weak_scale_timing.sh — weak-scaling sweep for timing_zombie_model + DSTM.
#
# Per sweep point runs BOTH:
#   1) timing_zombie_model  -> emits per-rank sync floats to stdout
#   2) ../zombies (DSTM)    -> emits _GET/_PUT timings to stdout
#
# Emits ONE row per binary per repeat into summary.csv with a unified
# schema. Unused columns for a given framework are NA.
#
# Sync columns (filled for framework=timing):
#   sync_total_s      sum of all bare-float lines in the log
#   sync_per_rank_s   sync_total_s / ranks
#   sync_pct          100 * sync_per_rank_s / elapsed_s
#   n_sync_samples    number of bare-float lines parsed
#
# Comm columns (filled for framework=dstm):
#   get_total_s       sum of _GET: <sec>
#   get_per_rank_s    get_total_s / ranks
#   put_total_s       sum of _PUT: <sec>
#   put_per_rank_s    put_total_s / ranks
#   comm_total_s      get_total_s + put_total_s
#   comm_per_rank_s   comm_total_s / ranks
#   comm_pct          100 * comm_per_rank_s / elapsed_s
#   n_get, n_put      sample counts
#
# Weak-scaling setup (matches compare.sh weak invocation):
#   ./weak_scale_timing.sh --hostfile "$COBALT_NODEFILE" \
#       --sweep-config "1:1:1 2:1:2 4:1:4 8:1:8 16:2:8 32:4:8 64:8:8 128:16:8 256:32:8" \
#       --nh 512 --nz 5 --time 256 --density 1.0 --repeats 4
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

# DSTM rebuild machinery.
DSTM_HEADER="${DSTM_HEADER:-${ROOT_DIR}/zombie.h}"
DSTM_BUILD_DIR="${DSTM_BUILD_DIR:-${ROOT_DIR}}"

MPIRUN="${MPIRUN:-mpirun}"

# --------------------------------------------------------------------------
# Defaults (match compare.sh weak-scaling invocation).
# --------------------------------------------------------------------------
NH_DEFAULT=512
NZ_DEFAULT=5
TIME_DEFAULT=256
SEED_DEFAULT=42
WORK_DEFAULT=0
REPEATS_DEFAULT=4
DENSITY_DEFAULT="1.0"

NH="$NH_DEFAULT"
NZ="$NZ_DEFAULT"
TIME="$TIME_DEFAULT"
SEED="$SEED_DEFAULT"
WORK="$WORK_DEFAULT"
REPEATS="$REPEATS_DEFAULT"
DENSITY="$DENSITY_DEFAULT"

SWEEP_CONFIG="4:1:4 8:1:8 16:2:8 32:4:8 64:8:8 128:16:8 256:32:8"
HOSTFILE=""

ONLY="both"   # timing | dstm | both

# --------------------------------------------------------------------------
# Argument parsing.
# --------------------------------------------------------------------------
usage() {
    cat <<EOF
usage: $0 [options]

  --nh N              humans per rank            (default $NH_DEFAULT)
  --nz N              zombies per rank           (default $NZ_DEFAULT)
  --time N            ticks to simulate          (default $TIME_DEFAULT)
  --density D         agents/cell target         (default $DENSITY_DEFAULT)
  --seed N            RNG seed                   (default $SEED_DEFAULT)
  --work N            DSTM synthetic-work knob   (default $WORK_DEFAULT)
  --repeats N         repeats per point          (default $REPEATS_DEFAULT)
  --sweep-config STR  "ranks:ppn:nodes ..." points (default: weak-scaling ladder)
  --hostfile FILE     hostfile (default: \$COBALT_NODEFILE)
  --only WHICH        timing | dstm | both       (default: both)
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nh)           NH="$2"; shift 2 ;;
        --nz)           NZ="$2"; shift 2 ;;
        --time)         TIME="$2"; shift 2 ;;
        --density)      DENSITY="$2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        --work)         WORK="$2"; shift 2 ;;
        --repeats)      REPEATS="$2"; shift 2 ;;
        --sweep-config) SWEEP_CONFIG="$2"; shift 2 ;;
        --hostfile)     HOSTFILE="$2"; shift 2 ;;
        --only)         ONLY="$2"; shift 2 ;;
        -h|--help)      usage; exit 0 ;;
        *) echo "unknown arg: $1"; usage; exit 1 ;;
    esac
done

case "$ONLY" in
    timing) RUN_TIMING=1; RUN_DSTM=0 ;;
    dstm)   RUN_TIMING=0; RUN_DSTM=1 ;;
    both)   RUN_TIMING=1; RUN_DSTM=1 ;;
    *) echo "bad --only value: $ONLY (want timing|dstm|both)" >&2; exit 1 ;;
esac

HOSTFILE="${HOSTFILE:-${COBALT_NODEFILE:-}}"
if [[ -z "$HOSTFILE" ]]; then
    echo "!! HOSTFILE not set and \$COBALT_NODEFILE empty" >&2
    exit 1
fi
if [[ ! -r "$HOSTFILE" ]]; then
    echo "!! hostfile not readable: $HOSTFILE" >&2
    exit 1
fi

# --------------------------------------------------------------------------
# Sanity: executables + templates.
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

# DSTM header snapshot for restore-on-exit.
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
    sed -i -E "s|^([[:space:]]*#define[[:space:]]+MAX_CELL_SIDE[[:space:]]+)[0-9]+.*|\1${new_val} /* auto-set by weak_scale_timing.sh */|" \
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
OUT_ROOT="${SCRIPT_DIR}/results/weak_timing_${STAMP}"
mkdir -p "$OUT_ROOT"

SUMMARY_CSV="${OUT_ROOT}/summary.csv"
echo "framework,ranks,ppn,nodes,nh,nz,time,density,nl,seed,repeat,elapsed_s,tps,\
sync_total_s,sync_per_rank_s,sync_pct,n_sync_samples,\
get_total_s,get_per_rank_s,put_total_s,put_per_rank_s,\
comm_total_s,comm_per_rank_s,comm_pct,n_get,n_put,\
status" > "$SUMMARY_CSV"

echo "weak-scaling sweep (timing_zombie_model + DSTM)"
echo "  NH (per rank)    = $NH"
echo "  NZ (per rank)    = $NZ"
echo "  TIME             = $TIME"
echo "  REPEATS          = $REPEATS"
echo "  DENSITY          = $DENSITY"
echo "  SWEEP_CONFIG     = $SWEEP_CONFIG"
echo "  ONLY             = $ONLY"
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

# Sum bare-float lines in the log ("<float>" alone). Prints "<sum> <count>".
aggregate_sync_times() {
    local log=$1
    awk '
        /^[[:space:]]*[+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?[[:space:]]*$/ {
            s += $1 + 0; n += 1
        }
        END { printf "%.6f %d\n", s+0, n+0 }
    ' "$log"
}

# Sum _GET: <sec> and _PUT: <sec> lines. Prints "<get_sum> <get_n> <put_sum> <put_n>".
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
# Runner: timing_zombie_model (fills sync_* columns).
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

    # sync row: comm cols are NA.
    echo "timing,$ranks,$ppn,$nodes,$nh,$nz,$time,$density,$nl,$seed,$rep,$elapsed,$tps,\
$sync_total,$sync_per_rank,$sync_pct,$sync_n,\
NA,NA,NA,NA,NA,NA,NA,NA,NA,\
$status" >> "$SUMMARY_CSV"
    printf "         elapsed=%ss  tps=%s  sync_total=%ss  sync/rank=%ss  sync%%=%s  n=%s  status=%s\n" \
        "$elapsed" "$tps" "$sync_total" "$sync_per_rank" "$sync_pct" "$sync_n" "$status"
}

# --------------------------------------------------------------------------
# Runner: DSTM ../zombies (fills get_*/put_*/comm_* columns).
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

    # comm row: sync cols are NA.
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
# Weak-scaling sweep.
# --------------------------------------------------------------------------
for entry in $SWEEP_CONFIG; do
    if [[ ! "$entry" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
        echo "!! bad SWEEP_CONFIG entry: '$entry' (want ranks:ppn:nodes)" >&2
        exit 1
    fi
    ranks=${BASH_REMATCH[1]}
    ppn=${BASH_REMATCH[2]}
    nodes=${BASH_REMATCH[3]}

    nh=$NH; nz=$NZ
    apr=$(( nh + nz ))
    read -r nl_auto rmin rmax < <(compute_geometry "$ranks" "$apr" "$DENSITY")

    if (( RUN_DSTM )); then
        needed_cs=$(cell_side_for_nl "$nl_auto")
        set_max_cell_side_and_build "$needed_cs"
    fi

    for (( rep=1; rep<=REPEATS; rep++ )); do
        echo "===== ranks=$ranks ppn=$ppn nodes=$nodes nh=$nh nz=$nz time=$TIME density=$DENSITY nl=$nl_auto repast=[${rmin},${rmax}]^2 rep=$rep ====="

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
