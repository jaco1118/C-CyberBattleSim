#!/bin/bash
# Task F Pass 1, STEP 2+3 evaluation grid, STOCHASTIC eval, 200 episodes/cell, concurrency 4.
# Condition A: 2 agents x 5 seeds x {static,membership,property} = 30 cells
# Condition B: static agent x 5 seeds x pn{0.01,0.10,0.25,0.50} = 20 cells
set -u
BASE=/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1
RUNS=$BASE/runs
OUT=$BASE/eval_out
LOGS=$BASE/eval_logs
mkdir -p "$OUT" "$LOGS"
DRIVER=$BASE/taskF1_eval.py
N_EP=200
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
MAXJOBS=4

declare -a CELLS
# Condition A
for SEED in 42 100 123 200 300; do
  for AG in static adapted; do
    for EC in static membership property; do
      CELLS+=("$AG $SEED $EC -")
    done
  done
done
# Condition B (static agent only)
for SEED in 42 100 123 200 300; do
  for PN in 0.01 0.10 0.25 0.50; do
    CELLS+=("static $SEED defender $PN")
  done
done

echo "$(date +%H:%M:%S) launching ${#CELLS[@]} eval cells (N_EP=$N_EP, stochastic, concurrency $MAXJOBS)" >> "$LOGS/_progress.log"
running=0
for cell in "${CELLS[@]}"; do
  set -- $cell
  AG=$1; SEED=$2; EC=$3; PN=$4
  RUN_FOLDER="$RUNS/trpo_250k_F1_${AG}_seed${SEED}"
  if [ "$EC" = "defender" ]; then
    TAG="${AG}_seed${SEED}_${EC}_p${PN}"; ARGS="$RUN_FOLDER $AG $SEED $EC $N_EP $OUT $PN"
  else
    TAG="${AG}_seed${SEED}_${EC}"; ARGS="$RUN_FOLDER $AG $SEED $EC $N_EP $OUT"
  fi
  ( python "$DRIVER" $ARGS > "$LOGS/$TAG.out" 2>&1; echo "$(date +%H:%M:%S) DONE $TAG (exit $?)" >> "$LOGS/_progress.log" ) &
  running=$((running+1))
  if [ "$running" -ge "$MAXJOBS" ]; then wait -n; running=$((running-1)); fi
done
wait
echo "$(date +%H:%M:%S) ALL EVAL CELLS COMPLETE" >> "$LOGS/_progress.log"
