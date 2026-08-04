#!/bin/bash
# F4 second resume (pre-registered ceiling): the five 80-100 runs 500k -> 750k, since they failed the
# convergence criterion at 500k. Resumes from each run's checkpoint_500000 + its vecnorm into the SAME
# run folder (continues num_timesteps 500k->750k). Two compounded resume discontinuities (250k->500k,
# 500k->750k) -- disclosed per evidence_taskF4.md 0.5. 30-40 and 10-15 are converged, not touched.
set -u
B=/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1
RUNS=$B/f4_runs; LOGS=$B/f4_750k_logs; mkdir -p "$LOGS"
D=$B/taskF4_train.py; TARGET=750000; MAXJOBS=5
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
declare -A T80=( [42]=5 [100]=100 [123]=18 [200]=2 [300]=67 )
running=0
echo "$(date +%H:%M:%S) F4 80-100 SECOND RESUME 500k->750k launching 5 runs" >> "$LOGS/_progress.log"
for SEED in 42 100 123 200 300; do
  RF="$RUNS/f4_static_80-100_seed${SEED}"
  CK="$RF/checkpoints/1/checkpoint_500000_steps.zip"
  VN="$RF/checkpoints/1/checkpoint_vecnormalize_500000_steps.pkl"
  bash -c "python '$D' $SEED scalability_80_100/${T80[$SEED]} '$RF' $TARGET '$CK' '$VN' > '$LOGS/80-100_seed${SEED}.out' 2>&1; echo \"\$(date +%H:%M:%S) DONE 80-100_seed${SEED} (exit \$?)\" >> '$LOGS/_progress.log'" &
  running=$((running+1))
  [ "$running" -ge "$MAXJOBS" ] && { wait -n; running=$((running-1)); }
done
wait
echo "$(date +%H:%M:%S) F4 80-100 750k ALL COMPLETE" >> "$LOGS/_progress.log"
