#!/bin/bash
# JOB 1: Condition B on gate 30-40 multi-topology checkpoints. 5 seeds x 5 pn = 25 cells,
# 200 episodes each, stochastic. Concurrency 4, 3 threads each.
set -u
BASE=/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1
OUT=$BASE/condB_multitopo_out; LOGS=$BASE/condB_multitopo_logs
mkdir -p "$OUT" "$LOGS"
GLOGS=/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents/logs
DRIVER=$BASE/condB_multitopo_eval.py
N_EP=200; MAXJOBS=4
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
echo "$(date +%H:%M:%S) JOB1 launching 25 cells (N_EP=$N_EP)" >> "$LOGS/_progress.log"
running=0
for SEED in 42 100 123 200 300; do
  RUN=$(ls -d $GLOGS/trpo_250k_tuned_compressed_band30-40_seed${SEED}_*/TRPO_x_control_SecureBERT 2>/dev/null | head -1)
  for PN in 0.0 0.01 0.10 0.25 0.50; do
    TAG="seed${SEED}_pn${PN}"
    ( python "$DRIVER" "$RUN" "$SEED" "$PN" "$N_EP" "$OUT" > "$LOGS/$TAG.out" 2>&1; echo "$(date +%H:%M:%S) DONE $TAG (exit $?)" >> "$LOGS/_progress.log" ) &
    running=$((running+1)); [ "$running" -ge "$MAXJOBS" ] && { wait -n; running=$((running-1)); }
  done
done
wait
echo "$(date +%H:%M:%S) JOB1 ALL COMPLETE" >> "$LOGS/_progress.log"
