#!/bin/bash
set -u
B=/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1
RUNS=$B/z_runs; LOGS=$B/z_train_logs; mkdir -p "$RUNS" "$LOGS"
D=$B/taskZ_train.py; TARGET=250000; MAXJOBS=5
export USER=slchan OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
declare -A T80=( [42]=5 [100]=100 [123]=18 [200]=2 [300]=67 )
SEEDS="42 100 123 200 300"
launch(){ ( "$@" ; ) & running=$((running+1)); [ "$running" -ge "$MAXJOBS" ] && { wait -n; running=$((running-1)); }; }
running=0
echo "$(date +%H:%M:%S) Z STEP1 launching 30 runs (3 arms x 2 bands x 5 seeds) -> ${TARGET}, MAXJOBS=$MAXJOBS" >> "$LOGS/_progress.log"
# ARM ORDER 1,2,3 so arm1 (the reproduction check) finishes earliest
for A in 1 2 3; do
  for SEED in $SEEDS; do
    # 30-40, single topo 44
    RF=$RUNS/z_arm${A}_30-40_seed${SEED}
    launch bash -c "python '$D' $SEED scalability_30_40/44 $A '$RF' $TARGET > '$LOGS/arm${A}_30-40_seed${SEED}.out' 2>&1; echo \"\$(date +%H:%M:%S) DONE arm${A}_30-40_seed${SEED} (exit \$?)\" >> '$LOGS/_progress.log'"
    # 80-100, per-seed topo
    RF=$RUNS/z_arm${A}_80-100_seed${SEED}
    launch bash -c "python '$D' $SEED scalability_80_100/${T80[$SEED]} $A '$RF' $TARGET > '$LOGS/arm${A}_80-100_seed${SEED}.out' 2>&1; echo \"\$(date +%H:%M:%S) DONE arm${A}_80-100_seed${SEED} (exit \$?)\" >> '$LOGS/_progress.log'"
  done
done
wait
echo "$(date +%H:%M:%S) Z STEP1 ALL COMPLETE" >> "$LOGS/_progress.log"
