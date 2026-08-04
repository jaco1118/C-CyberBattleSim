#!/bin/bash
set -u
B=/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1
D=$B/taskZ_eval.py; OUT=$B/z_repro_out; LOGS=$B/z_repro_logs; mkdir -p "$OUT" "$LOGS"
export USER=slchan OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
NEP=200; MAXJOBS=5
declare -A T80=( [42]=5 [100]=100 [123]=18 [200]=2 [300]=67 )
launch(){ ( "$@" ; ) & running=$((running+1)); [ "$running" -ge "$MAXJOBS" ] && { wait -n; running=$((running-1)); }; }
running=0
echo "$(date +%H:%M:%S) Z repro eval (arm1 static, NEW vs OLD, same harness) start" >> "$LOGS/_progress.log"
for S in 42 100 123 200 300; do
  # NEW arm1 (fresh Z retrain)
  launch bash -c "python '$D' '$B/z_runs/z_arm1_30-40_seed${S}' scalability_30_40/44 1 $S static $NEP '$OUT/new_30-40' > '$LOGS/new_30-40_seed${S}.out' 2>&1"
  launch bash -c "python '$D' '$B/z_runs/z_arm1_80-100_seed${S}' scalability_80_100/${T80[$S]} 1 $S static $NEP '$OUT/new_80-100' > '$LOGS/new_80-100_seed${S}.out' 2>&1"
  # OLD reported checkpoints (F1/F2), same eval harness
  launch bash -c "python '$D' '$B/runs/trpo_250k_F1_static_seed${S}' scalability_30_40/44 1 $S static $NEP '$OUT/old_30-40' > '$LOGS/old_30-40_seed${S}.out' 2>&1"
  launch bash -c "python '$D' '$B/f2_runs/trpo_250k_F2_static_band80-100_seed${S}' scalability_80_100/${T80[$S]} 1 $S static $NEP '$OUT/old_80-100' > '$LOGS/old_80-100_seed${S}.out' 2>&1"
done
wait
echo "$(date +%H:%M:%S) Z repro eval ALL COMPLETE" >> "$LOGS/_progress.log"
