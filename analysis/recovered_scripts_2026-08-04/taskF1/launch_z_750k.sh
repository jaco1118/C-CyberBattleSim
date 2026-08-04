#!/bin/bash
set -u
B=/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1
RUNS=$B/z750_runs; LOGS=$B/z750_logs; mkdir -p "$RUNS" "$LOGS"
D=$B/taskZ_train.py; TARGET=750000; MAXJOBS=5
export USER=slchan OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0
declare -A T80=( [42]=5 [100]=100 [123]=18 [200]=2 [300]=67 )
launch(){ ( "$@" ; ) & running=$((running+1)); [ "$running" -ge "$MAXJOBS" ] && { wait -n; running=$((running-1)); }; }
running=0
date +%s > "$LOGS/_start_epoch.txt"
echo "$(date +%H:%M:%S) Z 750k robustness (3 arms x 5 seeds @80-100, resume 250k->750k)" >> "$LOGS/_progress.log"
for A in 1 2 3; do
  for S in 42 100 123 200 300; do
    TID=${T80[$S]}
    CK=$B/z_runs/z_arm${A}_80-100_seed${S}/checkpoints/1/checkpoint_250000_steps.zip
    VN=$B/z_runs/z_arm${A}_80-100_seed${S}/checkpoints/1/checkpoint_vecnormalize_250000_steps.pkl
    RF=$RUNS/z750_arm${A}_80-100_seed${S}
    launch bash -c "python '$D' $S scalability_80_100/${TID} $A '$RF' $TARGET '$CK' '$VN' > '$LOGS/arm${A}_seed${S}.out' 2>&1; echo \"\$(date +%H:%M:%S) DONE arm${A}_seed${S} (exit \$?)\" >> '$LOGS/_progress.log'"
  done
done
wait
echo "$(date +%H:%M:%S) Z 750k ALL COMPLETE" >> "$LOGS/_progress.log"
