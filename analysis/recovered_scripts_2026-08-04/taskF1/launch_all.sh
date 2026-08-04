#!/bin/bash
# Task F Pass 1, STEP 1: launch all 10 training runs (2 conditions x 5 seeds), concurrency 4,
# each capped at 4 torch threads (16 threads max, under 32 cores; single large GPU shared).
set -u
BASE=/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1
RUNS=$BASE/runs
LOGS=$BASE/train_logs
mkdir -p "$RUNS" "$LOGS"
PY=python
DRIVER=$BASE/taskF1_train.py
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

MAXJOBS=4
declare -a SPECS
for SEED in 42 100 123 200 300; do
  for COND in static adapted; do
    SPECS+=("$SEED $COND")
  done
done

running=0
for spec in "${SPECS[@]}"; do
  set -- $spec
  SEED=$1; COND=$2
  NAME="trpo_250k_F1_${COND}_seed${SEED}"
  RUN_FOLDER="$RUNS/$NAME"
  echo "$(date +%H:%M:%S) launching $NAME"
  ( $PY "$DRIVER" "$SEED" "$COND" "$RUN_FOLDER" > "$LOGS/$NAME.out" 2>&1; echo "$(date +%H:%M:%S) DONE $NAME (exit $?)" >> "$LOGS/_progress.log" ) &
  running=$((running+1))
  if [ "$running" -ge "$MAXJOBS" ]; then
    wait -n
    running=$((running-1))
  fi
done
wait
echo "$(date +%H:%M:%S) ALL 10 RUNS COMPLETE" >> "$LOGS/_progress.log"
echo "ALL DONE"
