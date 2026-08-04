#!/bin/bash
set -u
cd /cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/
export USER=slchan OMP_NUM_THREADS=2
B=/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1
STEPS=800
for BAND in 30-40 80-100; do
  OUT=$B/L_reg_$BAND; rm -rf "$OUT"; mkdir -p "$OUT"
  python "$B/run_L_regression.py" new_off 42 $STEPS $BAND "$OUT" > "$OUT/new_off.log" 2>&1
  python "$B/run_L_regression.py" new_on  42 $STEPS $BAND "$OUT" > "$OUT/new_on.log" 2>&1
done
echo "NEW runs done $(date +%H:%M:%S)"
git stash push -m "taskL-step2" cyberbattle/_env/cyberbattle_env_compressed.py > "$B/L_reg_stash.log" 2>&1
STASHED=$?
for BAND in 30-40 80-100; do
  OUT=$B/L_reg_$BAND
  python "$B/run_L_regression.py" old 42 $STEPS $BAND "$OUT" > "$OUT/old.log" 2>&1
done
git stash pop >> "$B/L_reg_stash.log" 2>&1
echo "OLD runs done + stash popped $(date +%H:%M:%S); post-pop diff:"
git diff --stat cyberbattle/_env/cyberbattle_env_compressed.py | tail -1
