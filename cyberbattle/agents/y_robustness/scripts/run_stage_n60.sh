#!/bin/bash
# Task Y-ROBUSTNESS STEP 1.3: run static + membership_matched eval for all 10 seed-cells
# (N=30 x 5 seeds, N=90 x 5 seeds), 200 episodes each, concurrently, thread-capped.
set -u
AG=/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents
PY=/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/bin/python
export LD_LIBRARY_PATH=/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
N_EP=200
mkdir -p "$AG/y_robustness/out/n60" "$AG/y_robustness/logs_run"
cd "$AG"

declare -A RUN CS TOPO
RUN[n60_42]=logs/yN60_s42_stg7_2026-08-05_23-47-20/TRPO_x_control_SecureBERT;   CS[n60_42]=250000;  TOPO[n60_42]="./graphs_yN60_s42_2026-08-03_17-15-34/1"
RUN[n60_100]=logs/yN60_s100_stg7_2026-08-05_23-47-20/TRPO_x_control_SecureBERT; CS[n60_100]=250000; TOPO[n60_100]="./graphs_yN60_s100_2026-08-03_17-15-34/1"
RUN[n60_123]=logs/yN60_s123_stg7_2026-08-05_23-47-20/TRPO_x_control_SecureBERT; CS[n60_123]=250000; TOPO[n60_123]="./graphs_yN60_s123_2026-08-03_17-15-34/1"
RUN[n60_200]=logs/yN60_s200_stg7_2026-08-05_23-47-20/TRPO_x_control_SecureBERT; CS[n60_200]=250000; TOPO[n60_200]="./graphs_yN60_s200_2026-08-03_17-15-34/1"
RUN[n60_300]=logs/yN60_s300_stg7_2026-08-05_23-47-20/TRPO_x_control_SecureBERT; CS[n60_300]=250000; TOPO[n60_300]="./graphs_yN60_s300_2026-08-03_17-15-34/1"

pids=()
for cell in n60_42 n60_100 n60_123 n60_200 n60_300; do
  seed="${cell#n60_}"
  CKPT_STEP="${CS[$cell]}" nohup nice -n 5 "$PY" y_robustness/scripts/taskF2_eval.py \
    "${RUN[$cell]}" "$seed" "${TOPO[$cell]}" static "$N_EP" y_robustness/out/n60 \
    > "y_robustness/logs_run/${cell}_static.log" 2>&1 &
  pids+=($!)
  CKPT_STEP="${CS[$cell]}" nohup nice -n 5 "$PY" y_robustness/scripts/taskF2_eval.py \
    "${RUN[$cell]}" "$seed" "${TOPO[$cell]}" membership_matched "$N_EP" y_robustness/out/n60 "${CI_N60:?set CI_N60 to the STEP 1 calibrated change_interval before running}" \
    > "y_robustness/logs_run/${cell}_membership_matched.log" 2>&1 &
  pids+=($!)
done

echo "[wait] ${#pids[@]} eval runs: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do
  wait "$p"; status=$?
  echo "[wait] pid $p exit=$status"
  [ "$status" -ne 0 ] && fail=$((fail+1))
done
echo "[wait] done; failed=$fail"
echo "RUN COMPLETE (failed=$fail)"
