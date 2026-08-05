#!/bin/bash
# Task Y-EXTEND: N=90 per-seed extension orchestrator (one 250k stage per invocation).
# Only seeds 100 and 300 are ever passed here (seeds 42/123/200 are already within band per
# STEP 0.3 and must not be touched, per the task's own explicit restriction). New per-seed hard
# cap for this task: 1,750,000 steps -- do not exceed regardless of outcome.
#
# Unlike y_n30n60/y_n2_pilot's clean staged naming, N=90's original topology folders use the
# ad hoc "graphs_yprobe_n90_s<seed>_2026-08-03_08-38-04" prefix (reconstructed from each run's own
# stored train_config.yaml, not from a shared convention) -- hardcoded per seed below rather than
# glob-derived, since there is exactly one topology folder per seed and no ambiguity risk.
#
# Usage: run_stage.sh <stage_k> "<seed>:<finetune_rel_to_logs/>" [...]
set -u
STAGE="$1"; shift
SPECS=("$@")
AG=/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents
PY=/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/bin/python
export LD_LIBRARY_PATH=/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
OUT="$AG/y_n90_extend/stage${STAGE}"
mkdir -p "$OUT"
cd "$AG"

declare -A TOPO=( [100]="graphs_yprobe_n90_s100_2026-08-03_08-38-04" [300]="graphs_yprobe_n90_s300_2026-08-03_08-38-04" )

pids=()
for spec in "${SPECS[@]}"; do
  IFS=':' read -r S FT <<< "$spec"
  name="yN90ext_s${S}_stg${STAGE}"
  topo="${TOPO[$S]}"
  args=(--train_config y_n90_extend/y_base.yaml --name "$name"
        --load_envs "$topo" --load_seeds "$AG/y_n90_extend/seeds/seeds_$S"
        --finetune_model "$FT")
  echo "[launch] $name  topo=$topo  finetune=$FT"
  nohup nice -n 5 "$PY" train_agent.py "${args[@]}" > "$OUT/${name}.log" 2>&1 &
  pids+=($!)
done

echo "[wait] ${#pids[@]} training runs (stage $STAGE): ${pids[*]}"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "[wait] done; failed=$fail"
echo "STAGE ${STAGE} DONE (failed=$fail)"
