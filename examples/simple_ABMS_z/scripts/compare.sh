#!/usr/bin/env bash
#
# compare.sh — run the DSTM and RepastHPC zombie sims with matched
# configuration and report wall-clock + tick throughput side-by-side.
#
# Layout expected:
#   simple_ABMS_z/
#     zombies                     <- DSTM executable
#     scripts/
#       compare.sh                <- this file
#       zombie_model              <- Repast executable
#       config.props              <- Repast logger config (template)
#       model.props               <- Repast model config   (template)
#       results/                  <- created; per-run logs + summary CSV
#
# Single-node usage (unchanged):
#   ./compare.sh
#   ./compare.sh --nprocs 4 --time 200
#   ./compare.sh --sweep-nprocs "1 2 4 8"
#
# Multi-node (Cobalt) usage:
#   ./compare.sh --hostfile "$COBALT_NODEFILE" \
#                --sweep-config "1:1:1 2:1:2 4:1:4 8:1:8 16:2:8 32:4:8 64:8:8"
#
#   Each --sweep-config entry is "ranks:ppn:nodes".
#     ranks = MPI rank count (-n)
#     ppn   = processes per node (--map-by ppr:PPN:node)
#     nodes = nodes actually used (recorded in the CSV; informational)
#

set -uo pipefail   # deliberately NOT -e; we want to keep going on per-run failure

export LD_LIBRARY_PATH=~/sfw/Boost/Boost_1.83/lib:$LD_LIBRARY_PATH

# --------------------------------------------------------------------------
# Paths — resolve relative to this script's directory.
# --------------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

DSTM_EXE="${DSTM_EXE:-${ROOT_DIR}/zombies}"
REPAST_EXE="${REPAST_EXE:-${SCRIPT_DIR}/zombie_model}"
REPAST_CONFIG_TMPL="${REPAST_CONFIG_TMPL:-${SCRIPT_DIR}/config.props}"
REPAST_MODEL_TMPL="${REPAST_MODEL_TMPL:-${SCRIPT_DIR}/model.props}"

MPIRUN="${MPIRUN:-mpirun}"

# --------------------------------------------------------------------------
# Defaults.
# --------------------------------------------------------------------------
NPROCS_DEFAULT=4
NH_DEFAULT=500
NZ_DEFAULT=5
TIME_DEFAULT=100
NL_DEFAULT=4
SEED_DEFAULT=42
WORK_DEFAULT=0
REPEATS_DEFAULT=1
DENSITY_DEFAULT="1.0"

SWEEP_NPROCS=""
SWEEP_NH=""
SWEEP_NZ=""
SWEEP_TIME=""
SWEEP_NL=""
SWEEP_DENSITY=""
SWEEP_CONFIG=""     # "ranks:ppn:nodes ranks:ppn:nodes ..."   (multi-node mode)

NPROCS="$NPROCS_DEFAULT"
NH="$NH_DEFAULT"
NZ="$NZ_DEFAULT"
TIME="$TIME_DEFAULT"
NL="$NL_DEFAULT"
SEED="$SEED_DEFAULT"
WORK="$WORK_DEFAULT"
REPEATS="$REPEATS_DEFAULT"
DENSITY="$DENSITY_DEFAULT"

# Multi-node knobs.
HOSTFILE=""                             # e.g. "$COBALT_NODEFILE"
MPI_PREFIX_OVERRIDE=""                  # override --prefix arg to mpirun
MPI_ENV_FORWARD="LD_LIBRARY_PATH PATH"  # space-separated env vars to -x
MAP_BY_TEMPLATE=""                      # override the whole --map-by expr

RUN_DSTM=1
RUN_REPAST=1

# --------------------------------------------------------------------------
# Argument parsing.
# --------------------------------------------------------------------------
usage() {
    cat <<EOF
usage: $0 [options]

Single-point overrides:
  --nprocs N          MPI ranks (default $NPROCS_DEFAULT)
  --nh N              humans per rank (default $NH_DEFAULT)
  --nz N              zombies per rank (default $NZ_DEFAULT)
  --time N            ticks to simulate (default $TIME_DEFAULT)
  --nl N              DSTM local cells per rank (default $NL_DEFAULT)
                      (ignored when --density is set; auto-computed instead)
  --density D         agents/cell target; auto-sizes DSTM nl & Repast world
  --seed N            RNG seed (default $SEED_DEFAULT)
  --work N            DSTM synthetic-work knob (default $WORK_DEFAULT)
  --repeats N         run each config point N times (default $REPEATS_DEFAULT)

Sweeps (cross-product):
  --sweep-nprocs "1 2 4 8"
  --sweep-nh     "100 500 1000"
  --sweep-nz     "5 20"
  --sweep-time   "50 100 200"
  --sweep-nl     "4 8 16"
  --sweep-density "0.1 1.0 10.0"

Multi-node launch (Cobalt/PBS-style):
  --hostfile FILE     hostfile passed to mpirun --hostfile
                      (usually "\$COBALT_NODEFILE")
  --sweep-config "ranks:ppn:nodes ranks:ppn:nodes ..."
                      alternative to --sweep-nprocs; each entry specifies
                      rank count, processes-per-node, and node count.
                      Overrides --sweep-nprocs / --nprocs when present.
  --mpi-prefix PATH   passed as mpirun --prefix (defaults to auto from PATH)
  --mpi-env  "VAR1 VAR2 ..."   env vars to forward via mpirun -x (default: "$MPI_ENV_FORWARD")
  --map-by  EXPR      override the --map-by argument (default: "ppr:PPN:node")

Selection:
  --only {dstm,repast,both}   (default: both)

Other:
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nprocs)         NPROCS="$2"; shift 2 ;;
        --nh)             NH="$2"; shift 2 ;;
        --nz)             NZ="$2"; shift 2 ;;
        --time)           TIME="$2"; shift 2 ;;
        --nl)             NL="$2"; shift 2 ;;
        --density)        DENSITY="$2"; shift 2 ;;
        --seed)           SEED="$2"; shift 2 ;;
        --work)           WORK="$2"; shift 2 ;;
        --repeats)        REPEATS="$2"; shift 2 ;;

        --sweep-nprocs)   SWEEP_NPROCS="$2"; shift 2 ;;
        --sweep-nh)       SWEEP_NH="$2"; shift 2 ;;
        --sweep-nz)       SWEEP_NZ="$2"; shift 2 ;;
        --sweep-time)     SWEEP_TIME="$2"; shift 2 ;;
        --sweep-nl)       SWEEP_NL="$2"; shift 2 ;;
        --sweep-density)  SWEEP_DENSITY="$2"; shift 2 ;;
        --sweep-config)   SWEEP_CONFIG="$2"; shift 2 ;;

        --hostfile)       HOSTFILE="$2"; shift 2 ;;
        --mpi-prefix)     MPI_PREFIX_OVERRIDE="$2"; shift 2 ;;
        --mpi-env)        MPI_ENV_FORWARD="$2"; shift 2 ;;
        --map-by)         MAP_BY_TEMPLATE="$2"; shift 2 ;;

        --only)
            case "$2" in
                dstm)   RUN_DSTM=1; RUN_REPAST=0 ;;
                repast) RUN_DSTM=0; RUN_REPAST=1 ;;
                both)   RUN_DSTM=1; RUN_REPAST=1 ;;
                *) echo "bad --only value: $2"; exit 1 ;;
            esac
            shift 2 ;;

        -h|--help)  usage; exit 0 ;;
        *) echo "unknown arg: $1"; usage; exit 1 ;;
    esac
done

# Populate defaults for sweep lists.
[[ -z "$SWEEP_NPROCS"  ]] && SWEEP_NPROCS="$NPROCS"
[[ -z "$SWEEP_NH"      ]] && SWEEP_NH="$NH"
[[ -z "$SWEEP_NZ"      ]] && SWEEP_NZ="$NZ"
[[ -z "$SWEEP_TIME"    ]] && SWEEP_TIME="$TIME"
[[ -z "$SWEEP_NL"      ]] && SWEEP_NL="$NL"
[[ -z "$SWEEP_DENSITY" ]] && SWEEP_DENSITY="$DENSITY"

# --------------------------------------------------------------------------
# Multi-node mode: figure out --prefix and validate hostfile.
# --------------------------------------------------------------------------
MULTINODE=0
if [[ -n "$HOSTFILE" || -n "$SWEEP_CONFIG" ]]; then
    MULTINODE=1
    if [[ -z "$HOSTFILE" ]]; then
        echo "!! --sweep-config requires --hostfile" >&2
        exit 1
    fi
    if [[ ! -r "$HOSTFILE" ]]; then
        echo "!! hostfile not readable: $HOSTFILE" >&2
        exit 1
    fi

    if [[ -z "$MPI_PREFIX_OVERRIDE" ]]; then
        # Derive from `which mpirun`, mirroring your weak_scaling.sh.
        MPI_PREFIX_OVERRIDE="$(dirname "$(dirname "$(command -v "$MPIRUN")")")"
    fi
    [[ -z "$MAP_BY_TEMPLATE" ]] && MAP_BY_TEMPLATE="ppr:PPN:node"
fi

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
# factor_procs / compute_geometry (unchanged from prior version, with the
# proc-grid-aware side-length snap fix).
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
# Build the mpirun invocation prefix for a given (ranks, ppn).
# Prints an array-safe, whitespace-separated command line.
# --------------------------------------------------------------------------
mpirun_cmd() {
    local ranks=$1 ppn=$2
    local -a cmd=("$MPIRUN" -n "$ranks")

    if (( MULTINODE )); then
        cmd+=(--hostfile "$HOSTFILE")
        cmd+=(--prefix "$MPI_PREFIX_OVERRIDE")

        # Substitute PPN into the map-by template.
        local map_by="${MAP_BY_TEMPLATE//PPN/$ppn}"
        cmd+=(--map-by "$map_by")

        # Forward env vars.
        local v
        for v in $MPI_ENV_FORWARD; do
            cmd+=(-x "$v")
        done
    fi

    printf '%s\n' "${cmd[@]}"
}

# --------------------------------------------------------------------------
# Output setup.
# --------------------------------------------------------------------------
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="${SCRIPT_DIR}/results/${STAMP}"
mkdir -p "$OUT_ROOT"

SUMMARY_CSV="${OUT_ROOT}/summary.csv"
echo "framework,ranks,ppn,nodes,nh,nz,time,density,nl,seed,repeat,elapsed_s,tps,status" > "$SUMMARY_CSV"

echo "results -> $OUT_ROOT"
if (( MULTINODE )); then
    echo "multi-node mode: hostfile=$HOSTFILE prefix=$MPI_PREFIX_OVERRIDE"
    echo "                 map-by=$MAP_BY_TEMPLATE  env-forward='$MPI_ENV_FORWARD'"
fi
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
# Runners.
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

    echo "dstm,$ranks,$ppn,$nodes,$nh,$nz,$time,$density,$nl,$seed,$rep,$elapsed,$tps,$status" >> "$SUMMARY_CSV"
    printf "         elapsed=%ss  tps=%s  status=%s\n" "$elapsed" "$tps" "$status"
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

    echo "repast,$ranks,$ppn,$nodes,$nh,$nz,$time,$density,-,$seed,$rep,$elapsed,$tps,$status" >> "$SUMMARY_CSV"
    printf "         elapsed=%ss  tps=%s  status=%s\n" "$elapsed" "$tps" "$status"
}

# --------------------------------------------------------------------------
# Build the list of (ranks, ppn, nodes) points to sweep.
# --------------------------------------------------------------------------
declare -a POINTS
if [[ -n "$SWEEP_CONFIG" ]]; then
    for entry in $SWEEP_CONFIG; do
        # entry format: ranks:ppn:nodes
        if [[ "$entry" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
            POINTS+=("${BASH_REMATCH[1]} ${BASH_REMATCH[2]} ${BASH_REMATCH[3]}")
        else
            echo "!! bad --sweep-config entry: '$entry' (want ranks:ppn:nodes)" >&2
            exit 1
        fi
    done
else
    # Single-node fallback: derive from --sweep-nprocs, ppn = ranks, nodes = 1.
    for ranks in $SWEEP_NPROCS; do
        POINTS+=("$ranks $ranks 1")
    done
fi

# --------------------------------------------------------------------------
# Cross-product sweep.
# --------------------------------------------------------------------------
for point in "${POINTS[@]}"; do
  read -r ranks ppn nodes <<< "$point"
  for density in $SWEEP_DENSITY; do
    for nh in $SWEEP_NH; do
      for nz in $SWEEP_NZ; do
        for time in $SWEEP_TIME; do
          for (( rep=1; rep<=REPEATS; rep++ )); do

            apr=$(( nh + nz ))
            read -r nl_auto rmin rmax < <(compute_geometry "$ranks" "$apr" "$density")

            echo "===== ranks=$ranks ppn=$ppn nodes=$nodes nh=$nh nz=$nz time=$time density=$density nl=$nl_auto repast=[${rmin},${rmax}]^2 rep=$rep ====="

            if (( RUN_DSTM ));   then run_dstm   "$ranks" "$ppn" "$nodes" "$nh" "$nz" "$time" "$nl_auto" "$density" "$SEED" "$rep"; fi
            if (( RUN_REPAST )); then run_repast "$ranks" "$ppn" "$nodes" "$nh" "$nz" "$time" "$density" "$SEED" "$rep" "$rmin" "$rmax"; fi
            echo

          done
        done
      done
    done
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
