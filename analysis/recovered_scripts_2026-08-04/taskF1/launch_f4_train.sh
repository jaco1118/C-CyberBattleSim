#!/bin/bash
set -u
B=/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1
RUNS=$B/f4_runs; LOGS=$B/f4_train_logs; mkdir -p "$RUNS" "$LOGS"
D=$B/taskF4_train.py; TARGET=500000; MAXJOBS=5
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
declare -A T80=( [42]=5 [100]=100 [123]=18 [200]=2 [300]=67 )
declare -A T10=( [42]=61 [100]=54 [123]=62 [200]=29 [300]=4 )
launch(){ ( "$@" ; ) & running=$((running+1)); [ "$running" -ge "$MAXJOBS" ] && { wait -n; running=$((running-1)); }; }
running=0
echo "$(date +%H:%M:%S) F4 STEP1 launching 15 runs -> 500k" >> "$LOGS/_progress.log"
for SEED in 42 100 123 200 300; do
  # 30-40 resume (topo44, F1 static ckpt)
  CK30=$B/runs/trpo_250k_F1_static_seed${SEED}/checkpoints/1/checkpoint_250000_steps.zip
  VN30=$B/runs/trpo_250k_F1_static_seed${SEED}/checkpoints/1/checkpoint_vecnormalize_250000_steps.pkl
  launch bash -c "python '$D' $SEED scalability_30_40/44 '$RUNS/f4_static_30-40_seed${SEED}' $TARGET '$CK30' '$VN30' > '$LOGS/30-40_seed${SEED}.out' 2>&1; echo \"\$(date +%H:%M:%S) DONE 30-40_seed${SEED} (exit \$?)\" >> '$LOGS/_progress.log'"
  # 80-100 resume (per-seed topo, F2 static ckpt)
  CK80=$B/f2_runs/trpo_250k_F2_static_band80-100_seed${SEED}/checkpoints/1/checkpoint_250000_steps.zip
  VN80=$B/f2_runs/trpo_250k_F2_static_band80-100_seed${SEED}/checkpoints/1/checkpoint_vecnormalize_250000_steps.pkl
  launch bash -c "python '$D' $SEED scalability_80_100/${T80[$SEED]} '$RUNS/f4_static_80-100_seed${SEED}' $TARGET '$CK80' '$VN80' > '$LOGS/80-100_seed${SEED}.out' 2>&1; echo \"\$(date +%H:%M:%S) DONE 80-100_seed${SEED} (exit \$?)\" >> '$LOGS/_progress.log'"
  # 10-15 fresh (per-seed distinct topo)
  launch bash -c "python '$D' $SEED scalability_10_15/${T10[$SEED]} '$RUNS/f4_static_10-15_seed${SEED}' $TARGET > '$LOGS/10-15_seed${SEED}.out' 2>&1; echo \"\$(date +%H:%M:%S) DONE 10-15_seed${SEED} (exit \$?)\" >> '$LOGS/_progress.log'"
done
wait
echo "$(date +%H:%M:%S) F4 STEP1 ALL COMPLETE" >> "$LOGS/_progress.log"
