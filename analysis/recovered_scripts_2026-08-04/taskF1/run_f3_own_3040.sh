#!/bin/bash
set -u
B=/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1
OUT=$B/f3_own_out; LOGS=$B/f3_own_logs; mkdir -p "$OUT" "$LOGS"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for SEED in 42 100 123 200 300; do
  ( python "$B/taskF3_ownership_eval.py" "$B/runs/trpo_250k_F1_static_seed${SEED}" "$SEED" "scalability_30_40/44" membership 200 "$OUT" > "$LOGS/3040_seed${SEED}.out" 2>&1; echo "$(date +%H:%M:%S) DONE 3040_seed${SEED} (exit $?)" >> "$LOGS/_progress.log" ) &
done
wait
echo "$(date +%H:%M:%S) F3 30-40 ownership eval COMPLETE" >> "$LOGS/_progress.log"
