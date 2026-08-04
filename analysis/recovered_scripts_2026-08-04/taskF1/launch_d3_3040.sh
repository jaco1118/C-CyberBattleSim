#!/bin/bash
# Task D3 STEP 2.1: band 30-40 property-substitution eval, 5 static seeds on topo44, 250k ckpts,
# N_EP=200, ci=20 (default). Low concurrency (2) so F4 training keeps priority.
set -u
B=/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1
OUT=$B/d3_eval_out
LOGS=$B/d3_eval_logs
mkdir -p "$OUT" "$LOGS"
N_EP=200
MAXJOBS=4
SEEDS="${SEEDS:-42 100 123 200 300}"
run() {
  local SEED=$1
  local RUN_FOLDER="$B/runs/trpo_250k_F1_static_seed${SEED}"
  OMP_NUM_THREADS=4 python "$B/taskF1_eval.py" "$RUN_FOLDER" static "$SEED" property_substitution "$N_EP" "$OUT" \
     > "$LOGS/3040_sub_seed${SEED}.out" 2>&1
  echo "$(date +%H:%M:%S) DONE 3040_sub_seed${SEED} (exit $?)" >> "$LOGS/_progress.log"
}
echo "$(date +%H:%M:%S) launching D3 substitution cells (30-40) seeds=[$SEEDS], N_EP=$N_EP, concurrency $MAXJOBS" >> "$LOGS/_progress.log"
for SEED in $SEEDS; do
  run "$SEED" &
  while [ "$(jobs -r | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
done
wait
echo "$(date +%H:%M:%S) ALL D3 3040 substitution cells done" >> "$LOGS/_progress.log"
