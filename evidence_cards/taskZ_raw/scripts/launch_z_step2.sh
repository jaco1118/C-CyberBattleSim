#!/bin/bash
set -u
B=/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1
D=$B/taskZ_eval.py; OUT=$B/z_step2_out; LOGS=$B/z_step2_logs
mkdir -p "$OUT/static" "$OUT/chg_3040" "$OUT/chg_80100_abs" "$OUT/chg_80100_rel" "$LOGS"
export USER=slchan OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
NEP=200; MAXJOBS=5
declare -A T80=( [42]=5 [100]=100 [123]=18 [200]=2 [300]=67 )
SEEDS="42 100 123 200 300"
launch(){ ( "$@" ; ) & running=$((running+1)); [ "$running" -ge "$MAXJOBS" ] && { wait -n; running=$((running-1)); }; }
running=0
echo "$(date +%H:%M:%S) Z STEP2 eval start (fixed-rel PRIMARY, fixed-abs SECONDARY; +static null)" >> "$LOGS/_progress.log"
for A in 1 2 3; do
  for S in $SEEDS; do
    R30=$B/z_runs/z_arm${A}_30-40_seed${S}; R80=$B/z_runs/z_arm${A}_80-100_seed${S}; TID=${T80[$S]}
    # static (null baseline, both bands)
    launch bash -c "python '$D' '$R30' scalability_30_40/44 $A $S static  $NEP '$OUT/static' > '$LOGS/static_arm${A}_30-40_s${S}.out' 2>&1"
    launch bash -c "python '$D' '$R80' scalability_80_100/${TID} $A $S static $NEP '$OUT/static' > '$LOGS/static_arm${A}_80-100_s${S}.out' 2>&1"
    # 30-40 change CI=20 (shared by fixed-abs AND fixed-rel)
    launch bash -c "python '$D' '$R30' scalability_30_40/44 $A $S membership $NEP '$OUT/chg_3040' 20 > '$LOGS/chg3040_arm${A}_s${S}.out' 2>&1"
    # 80-100 fixed-abs CI=20
    launch bash -c "python '$D' '$R80' scalability_80_100/${TID} $A $S membership $NEP '$OUT/chg_80100_abs' 20 > '$LOGS/abs_arm${A}_s${S}.out' 2>&1"
    # 80-100 fixed-rel CI=8
    launch bash -c "python '$D' '$R80' scalability_80_100/${TID} $A $S membership $NEP '$OUT/chg_80100_rel' 8 > '$LOGS/rel_arm${A}_s${S}.out' 2>&1"
  done
done
wait
echo "$(date +%H:%M:%S) Z STEP2 eval ALL COMPLETE" >> "$LOGS/_progress.log"
