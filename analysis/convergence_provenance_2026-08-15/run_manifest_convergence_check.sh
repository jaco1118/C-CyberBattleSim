#!/bin/bash
# Task CONVERGENCE-PROVENANCE, Question 1: apply the EXACT method Table IV.2 uses (the F4 band
# rule, train/Root owned nodes, 50k windows, |Delta%|<5%, >=4/5 seeds -- confirmed from
# cyberbattle/agents/compute_convergence_check.py's own docstring and Task Y's commit message,
# "F4-check each cell after every stage... same F4 rule throughout") to the 15 manifest
# checkpoints of 2026-07-26 -- the ones that actually back this project's reported attenuation/
# pooling/SNR results, which have never had this check run on them (Task THREE-LOOKUPS).
#
# This is a pure read of already-logged tfevents scalars via the ALREADY-COMMITTED, UNMODIFIED
# compute_convergence_check.py (commit 4173c53, confirmed an ancestor of current HEAD) -- no
# training, no resumption, no new evaluation episodes. --stop 250000 matches the "250k" budget
# label these runs were trained under (tfevents' own max logged step is ~253952-254976, a normal
# few-thousand-step rollout-batch overshoot past the round target, per the script's own docstring
# note about this exact behaviour).
set -e
REPO=/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim
AGENTS=$REPO/cyberbattle/agents
PY=/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/bin/python
OUT=$REPO/analysis/convergence_provenance_2026-08-15

for BAND in 10-15 30-40 80-100; do
  echo "==================== band $BAND ===================="
  ARGS=()
  for SEED in 42 100 123 200 300; do
    DIR=$(ls -d "$AGENTS"/logs/trpo_250k_tuned_compressed_band${BAND}_seed${SEED}_2026-07-26_*/)
    ARGS+=(--run "seed${SEED}=${DIR}TRPO_x_control_SecureBERT/TRPO_1")
  done
  "$PY" "$AGENTS/compute_convergence_check.py" "${ARGS[@]}" --stop 250000
  echo ""
done
