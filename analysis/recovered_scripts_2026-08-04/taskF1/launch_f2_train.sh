#!/bin/bash
# JOB 2 STEP 1: 5 static-only TRPO runs, band 80-100, one topology per seed. Concurrency 5,
# 3 threads each (leaves cores for JOB 1 running concurrently).
set -u
BASE=/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1
RUNS=$BASE/f2_runs; LOGS=$BASE/f2_train_logs
mkdir -p "$RUNS" "$LOGS"
DRIVER=$BASE/taskF2_train.py
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
# seed -> topology pairing (0.3)
declare -A TOPO=( [42]=5 [100]=100 [123]=18 [200]=2 [300]=67 )
echo "$(date +%H:%M:%S) JOB2 STEP1 launching 5 static runs" >> "$LOGS/_progress.log"
for SEED in 42 100 123 200 300; do
  TID=${TOPO[$SEED]}
  NAME="trpo_250k_F2_static_band80-100_seed${SEED}"
  RUN="$RUNS/$NAME"
  ( python "$DRIVER" "$SEED" "$TID" "$RUN" > "$LOGS/$NAME.out" 2>&1; echo "$(date +%H:%M:%S) DONE $NAME topo=$TID (exit $?)" >> "$LOGS/_progress.log" ) &
done
wait
echo "$(date +%H:%M:%S) JOB2 STEP1 ALL COMPLETE" >> "$LOGS/_progress.log"
