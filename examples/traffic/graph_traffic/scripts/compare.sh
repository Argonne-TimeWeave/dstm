#!/bin/bash
set -u
N=${N:-24}; GRID=${GRID:-20}; STEPS=${STEPS:-200}
CARS=${CARS:-0}; RATE=${RATE:-8}; CAP=${CAP:-3}; SEED=${SEED:-42}

echo "grid=$GRID ranks=$N steps=$STEPS rate=$RATE cap=$CAP seed=$SEED"
for m in blind stale oracle; do
  o=$(mpirun -n $N ./traffic --grid $GRID --steps $STEPS --cars $CARS \
      --rate $RATE --cap $CAP --seed $SEED --replica-mode $m --quiet)
  printf "%-7s dur=%-7s hops=%-6s blk=%-6s arr=%-7s age=%s\n" "$m" \
    "$(sed -n 's/.*mean duration=\([0-9.]*\).*/\1/p' <<<"$o")" \
    "$(sed -n 's/.*mean hops=\([0-9.]*\).*/\1/p'     <<<"$o")" \
    "$(sed -n 's/.*blocked=[0-9]* (\([0-9.]*\)%.*/\1/p' <<<"$o")" \
    "$(sed -n 's/^spawned=[0-9]* arrived=\([0-9]*\).*/\1/p' <<<"$o")" \
    "$(sed -n 's/.*consulted entry=\([0-9.]*\).*/\1/p' <<<"$o")"
done
