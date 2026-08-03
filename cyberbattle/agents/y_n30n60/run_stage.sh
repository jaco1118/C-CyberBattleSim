#!/bin/bash
# Y-N30-N60 staged convergence orchestrator (one 250k stage per invocation).
#
# Pre-registered checkpoint stopping rule (evidence: dissertation_log_v2.md Y-N30-N60):
#   train in 250k increments; after each increment run the F4 band rule
#   (compute_convergence_check.py, threshold 5%, window 50k) on the cell's 5 seeds;
#   stop a cell at the FIRST increment whose band CONVERGES; hard cap 1.25M (stage 5).
# Each stage k resumes stage k-1 via --finetune_model (constant LR, VecNormalize rebuilt
# fresh = documented transient), exactly as the N=90 probe was extended.
#
# The F4 check for stage k uses that stage's own run folder: SB3 restarts the tensorboard
# step counter at 0 on resume, so stop=250000 (local) anchors the final-50k vs preceding-50k
# windows at absolute step k*250k -- the "converged by k*250k?" question.
#
# Usage:
#   run_stage.sh <stage_k> "<N>:<seed>:<finetune_rel_or_NONE>" ["<N>:<seed>:..." ...]
#     <N>              cell node count (30 or 60)
#     <seed>           42|100|123|200|300
#     <finetune_rel>   path to stage k-1 checkpoint .zip RELATIVE TO logs/, or NONE for stage 1
set -u
STAGE="$1"; shift
SPECS=("$@")
AG=/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents
PY=/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/bin/python
export LD_LIBRARY_PATH=/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/lib:${LD_LIBRARY_PATH:-}
# THREAD CAP (critical): the env step is single-threaded Python; the policy nets are tiny. Left
# uncapped, each of the ~10 concurrent PyTorch procs spawns ~ncore intra-op threads -> ~320 threads
# on 32 cores, load ~116, fps collapses 245 -> 4 (aggregate WORSE than one proc). Cap every math
# backend to 1 thread so the procs coexist without scheduler thrashing (must be set before python
# imports torch/numpy). Measured effect: restores per-proc fps to near the single-proc rate.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
OUT="$AG/y_n30n60/stage${STAGE}"
mkdir -p "$OUT"
cd "$AG"

declare -A CELLS
pids=()
for spec in "${SPECS[@]}"; do
  IFS=':' read -r N S FT <<< "$spec"
  CELLS[$N]=1
  name="yN${N}_s${S}_stg${STAGE}"
  topo="graphs_yN${N}_s${S}_2026-08-03_17-15-34"
  args=(--train_config y_n30n60/y_base.yaml --name "$name"
        --load_envs "$topo" --load_seeds "$AG/y_n30n60/seeds/seeds_$S")
  [ "$FT" != "NONE" ] && args+=(--finetune_model "$FT")
  echo "[launch] $name  finetune=$FT"
  nohup nice -n 5 "$PY" train_agent.py "${args[@]}" > "$OUT/${name}.log" 2>&1 &
  pids+=($!)
done

echo "[wait] ${#pids[@]} training runs (stage $STAGE): ${pids[*]}"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "[wait] done; failed=$fail"

# F4 band check per cell touched this stage
for N in "${!CELLS[@]}"; do
  cargs=()
  for S in 42 100 123 200 300; do
    rf=$(ls -dt "$AG"/logs/yN${N}_s${S}_stg${STAGE}_* 2>/dev/null | head -1)
    [ -n "$rf" ] && cargs+=(--run "seed${S}=$rf")
  done
  vf="$OUT/verdict_N${N}.txt"
  {
    echo "=== F4 BAND CHECK  cell N=$N  stage=$STAGE  (absolute $((STAGE*250))k) ==="
    "$PY" compute_convergence_check.py "${cargs[@]}" 2>&1 | grep -vE "Warning|warn"
  } | tee "$vf"
done
echo "STAGE ${STAGE} DONE (failed=$fail)"
