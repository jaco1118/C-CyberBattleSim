#!/bin/bash
# Y2-pilot-N30 (Task Y3) staged convergence orchestrator (one 250k stage per invocation).
# Adapted from y2_n50/run_stage.sh (itself from y_n30n60/run_stage.sh) -- same F4 checkpoint
# stopping rule, same thread-cap fix, same --finetune_model resume mechanism. Differs only in:
# single cell (N=30 fixed, degree ~12.35 -- matches N=50's actual achieved ~12.50), topology
# folder looked up by glob (each seed's own generation timestamp), paths under y2_n30/.
#
# Usage: run_stage.sh <stage_k> "<seed>:<finetune_rel_or_NONE>" [...]
set -u
STAGE="$1"; shift
SPECS=("$@")
AG=/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents
PY=/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/bin/python
export LD_LIBRARY_PATH=/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
OUT="$AG/y2_n30/stage${STAGE}"
mkdir -p "$OUT"
cd "$AG"

pids=()
for spec in "${SPECS[@]}"; do
  IFS=':' read -r S FT <<< "$spec"
  name="yN30_s${S}_stg${STAGE}"
  topo=$(basename "$(ls -dt "$AG"/../data/env_samples/graphs_yN30_s${S}_2026-08-05* 2>/dev/null | head -1)")
  args=(--train_config y2_n30/y2_base.yaml --name "$name"
        --load_envs "$topo" --load_seeds "$AG/y2_n30/seeds/seeds_$S")
  [ "$FT" != "NONE" ] && args+=(--finetune_model "$FT")
  echo "[launch] $name  topo=$topo  finetune=$FT"
  nohup nice -n 5 "$PY" train_agent.py "${args[@]}" > "$OUT/${name}.log" 2>&1 &
  pids+=($!)
done

echo "[wait] ${#pids[@]} training runs (stage $STAGE): ${pids[*]}"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "[wait] done; failed=$fail"

cargs=()
for S in 42 100 123 200 300; do
  rf=$(ls -dt "$AG"/logs/yN30_s${S}_stg${STAGE}_* 2>/dev/null | head -1)
  [ -n "$rf" ] && cargs+=(--run "seed${S}=$rf")
done
mkdir -p "$AG/y2_n30/verdicts"
vf="$AG/y2_n30/verdicts/stage${STAGE}_N30.txt"
{
  echo "=== F4 BAND CHECK  cell N=30 (Y3 pilot, degree ~12.35)  stage=$STAGE  (absolute $((STAGE*250))k) ==="
  "$PY" compute_convergence_check.py "${cargs[@]}" 2>&1 | grep -vE "Warning|warn"
} | tee "$vf"
echo "STAGE ${STAGE} DONE (failed=$fail)"
