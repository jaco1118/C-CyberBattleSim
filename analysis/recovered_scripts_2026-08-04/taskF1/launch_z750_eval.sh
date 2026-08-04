#!/bin/bash
set -u
B=/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1
D=$B/taskZ_eval.py; OUT=$B/z750_eval; LOGS=$B/z750_eval_logs; mkdir -p "$OUT/static" "$OUT/rel" "$OUT/abs" "$LOGS"
export USER=slchan OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0 CKPT_STEP=750000
NEP=200; MAXJOBS=5
declare -A T80=( [42]=5 [100]=100 [123]=18 [200]=2 [300]=67 )
launch(){ ( "$@" ; ) & running=$((running+1)); [ "$running" -ge "$MAXJOBS" ] && { wait -n; running=$((running-1)); }; }
running=0
echo "$(date +%H:%M:%S) Z750 eval start (80-100, static+rel+abs, CKPT_STEP=750000)" >> "$LOGS/_progress.log"
for A in 1 2 3; do for S in 42 100 123 200 300; do
  R=$B/z750_runs/z750_arm${A}_80-100_seed${S}; TID=${T80[$S]}
  launch bash -c "CKPT_STEP=750000 python '$D' '$R' scalability_80_100/${TID} $A $S static     $NEP '$OUT/static' > '$LOGS/static_arm${A}_s${S}.out' 2>&1"
  launch bash -c "CKPT_STEP=750000 python '$D' '$R' scalability_80_100/${TID} $A $S membership $NEP '$OUT/rel' 8 > '$LOGS/rel_arm${A}_s${S}.out' 2>&1"
  launch bash -c "CKPT_STEP=750000 python '$D' '$R' scalability_80_100/${TID} $A $S membership $NEP '$OUT/abs' 20 > '$LOGS/abs_arm${A}_s${S}.out' 2>&1"
done; done
wait
echo "$(date +%H:%M:%S) Z750 eval COMPLETE" >> "$LOGS/_progress.log"
