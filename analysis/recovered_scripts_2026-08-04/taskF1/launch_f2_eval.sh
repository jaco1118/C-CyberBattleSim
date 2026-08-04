#!/bin/bash
# F2 STEP 2 eval: run one eval condition across all 5 replicates (seed-topology pairs), 200 eps.
# Usage: bash launch_f2_eval.sh <eval_cond> [change_interval]
set -u
COND="$1"; CI="${2:-}"
BASE=/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1
RUNS=$BASE/f2_runs; OUT=$BASE/f2eval_out; LOGS=$BASE/f2eval_logs
mkdir -p "$OUT" "$LOGS"
DRIVER=$BASE/taskF2_eval.py; N_EP=200
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
declare -A TOPO=( [42]=5 [100]=100 [123]=18 [200]=2 [300]=67 )
echo "$(date +%H:%M:%S) F2eval launching cond=$COND ci=${CI:-natural} (5 replicates)" >> "$LOGS/_progress.log"
for SEED in 42 100 123 200 300; do
  TID=${TOPO[$SEED]}
  RUN="$RUNS/trpo_250k_F2_static_band80-100_seed${SEED}"
  TAG="seed${SEED}_${COND}"
  ( python "$DRIVER" "$RUN" "$SEED" "$TID" "$COND" "$N_EP" "$OUT" $CI > "$LOGS/$TAG.out" 2>&1; echo "$(date +%H:%M:%S) DONE $TAG (exit $?)" >> "$LOGS/_progress.log" ) &
done
wait
echo "$(date +%H:%M:%S) F2eval cond=$COND COMPLETE" >> "$LOGS/_progress.log"
