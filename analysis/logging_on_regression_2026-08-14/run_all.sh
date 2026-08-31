#!/bin/bash
# Task LOGGING-ON-REGRESSION: driver. Extracts the pre-instrumentation commit (7cdfb2b) via
# git archive into a scratch tree, runs run_side.py for Side A (7cdfb2b) and Side B (current
# HEAD, drift_logging=True) across seeds, then runs compare_sides.py. No training; evaluation
# rollouts only, on the band 10-15 seed 42 manifest checkpoint (see run_side.py docstring).
set -e
REPO=/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim
OUT=$REPO/analysis/logging_on_regression_2026-08-14
SCRATCH=$OUT/scratch_side_a_7cdfb2b
PY=/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/bin/python
N_STEPS=${N_STEPS:-2000}
SEEDS=${SEEDS:-"42 100 123"}

echo "[1/4] extracting pre-instrumentation commit 7cdfb2b into $SCRATCH"
rm -rf "$SCRATCH"
mkdir -p "$SCRATCH"
git -C "$REPO" archive 7cdfb2b | tar -x -C "$SCRATCH"
echo "  done: $(find "$SCRATCH/cyberbattle/_env" -maxdepth 1 -name '*.py' | wc -l) files under _env"

echo "[2/4] Side A runs (pre-instrumentation, $SCRATCH)"
for SEED in $SEEDS; do
  echo "  seed=$SEED"
  "$PY" "$OUT/run_side.py" "$SCRATCH" A "$SEED" "$N_STEPS" "$OUT/side_A_seed${SEED}.pkl"
done

echo "[3/4] Side B runs (current HEAD, $REPO, drift_logging=True)"
for SEED in $SEEDS; do
  echo "  seed=$SEED"
  "$PY" "$OUT/run_side.py" "$REPO" B "$SEED" "$N_STEPS" "$OUT/side_B_seed${SEED}.pkl"
done

echo "[4/4] comparing"
"$PY" "$OUT/compare_sides.py" "$OUT" $SEEDS
