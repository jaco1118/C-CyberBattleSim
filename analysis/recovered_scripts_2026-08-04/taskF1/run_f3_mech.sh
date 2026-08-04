#!/bin/bash
set -u
B=/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1
OUT=$B/f3_mech_out; LOGS=$B/f3_mech_logs; mkdir -p "$OUT" "$LOGS"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
declare -A T80=( [42]=5 [100]=100 [123]=18 [200]=2 [300]=67 )
for SEED in 42 100 123 200 300; do
  ( python "$B/taskF3_mech_eval.py" "$B/runs/trpo_250k_F1_static_seed${SEED}" "$SEED" "scalability_30_40/44" 100 "$OUT" "30-40" > "$LOGS/3040_seed${SEED}.out" 2>&1 ) &
  ( python "$B/taskF3_mech_eval.py" "$B/f2_runs/trpo_250k_F2_static_band80-100_seed${SEED}" "$SEED" "scalability_80_100/${T80[$SEED]}" 100 "$OUT" "80-100" > "$LOGS/80100_seed${SEED}.out" 2>&1 ) &
done
wait
echo "$(date +%H:%M:%S) F3 mech eval COMPLETE" >> "$LOGS/_progress.log"
