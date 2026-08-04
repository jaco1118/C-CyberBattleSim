#!/bin/bash
set -u
export USER=slchan LOGNAME=slchan
B=/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1
OUT=$B/conv_results_750k.csv; LOGS=$B/conv_750k_logs; mkdir -p "$LOGS"; rm -f "$OUT"
N=60; MAXJOBS=5; STEPS="650000 700000 750000"
declare -A T80=( [42]=5 [100]=100 [123]=18 [200]=2 [300]=67 )
running=0
for SEED in 42 100 123 200 300; do
  RF="$B/f4_runs/f4_static_80-100_seed${SEED}"; TOPO="scalability_80_100/${T80[$SEED]}"
  for st in $STEPS; do
    OMP_NUM_THREADS=4 python "$B/conv_eval.py" "$RF" "$TOPO" "$st" "$N" "$OUT" "80-100" "$SEED" > "$LOGS/seed${SEED}_${st}.out" 2>&1 &
    running=$((running+1)); [ "$running" -ge "$MAXJOBS" ] && { wait -n; running=$((running-1)); }
  done
done
wait
echo "$(date +%H:%M:%S) 750k re-check evals done" >> "$LOGS/_progress.log"
