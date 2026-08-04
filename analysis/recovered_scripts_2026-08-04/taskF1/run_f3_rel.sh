#!/bin/bash
set -u
B=/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1
OUT=$B/f3_rel_out; LOGS=$B/f3_rel_logs; mkdir -p "$OUT" "$LOGS"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
declare -A T80=( [42]=5 [100]=100 [123]=18 [200]=2 [300]=67 )
for SEED in 42 100 123 200 300; do
  TID=${T80[$SEED]}
  RUN="$B/f2_runs/trpo_250k_F2_static_band80-100_seed${SEED}"
  # drift+leaveown+score at ci=8 (membership_matched applies the CI override)
  ( python "$B/taskF3_ownership_eval.py" "$RUN" "$SEED" "scalability_80_100/${TID}" membership_matched 200 "$OUT" 8 > "$LOGS/drift_seed${SEED}.out" 2>&1; echo "$(date +%H:%M:%S) DONE drift_seed${SEED} (exit $?)" >> "$LOGS/_progress.log" ) &
  # mechanical split at ci=8
  ( python "$B/taskF3_mech_eval.py" "$RUN" "$SEED" "scalability_80_100/${TID}" 200 "$OUT" "80-100rel" 8 > "$LOGS/mech_seed${SEED}.out" 2>&1; echo "$(date +%H:%M:%S) DONE mech_seed${SEED} (exit $?)" >> "$LOGS/_progress.log" ) &
done
wait
echo "$(date +%H:%M:%S) F3 relative-churn sweep COMPLETE" >> "$LOGS/_progress.log"
