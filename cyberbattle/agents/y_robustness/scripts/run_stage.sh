#!/bin/bash
# Task Y-ROBUSTNESS STEP 1.3: run static + membership_matched eval for all 10 seed-cells
# (N=30 x 5 seeds, N=90 x 5 seeds), 200 episodes each, concurrently, thread-capped.
set -u
AG=/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents
PY=/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/bin/python
export LD_LIBRARY_PATH=/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
N_EP=200
mkdir -p "$AG/y_robustness/out/n30" "$AG/y_robustness/out/n90" "$AG/y_robustness/logs_run"
cd "$AG"

declare -A RUN CS TOPO
RUN[n30_42]=logs/yN30_s42_stg1_2026-08-03_21-28-47/TRPO_x_control_SecureBERT;   CS[n30_42]=250000;  TOPO[n30_42]="./graphs_yN30_s42_2026-08-03_17-15-34/1"
RUN[n30_100]=logs/yN30_s100_stg1_2026-08-03_21-28-47/TRPO_x_control_SecureBERT; CS[n30_100]=250000; TOPO[n30_100]="./graphs_yN30_s100_2026-08-03_17-15-34/1"
RUN[n30_123]=logs/yN30_s123_stg1_2026-08-03_21-28-47/TRPO_x_control_SecureBERT; CS[n30_123]=250000; TOPO[n30_123]="./graphs_yN30_s123_2026-08-03_17-15-34/1"
RUN[n30_200]=logs/yN30_s200_stg1_2026-08-03_21-28-47/TRPO_x_control_SecureBERT; CS[n30_200]=250000; TOPO[n30_200]="./graphs_yN30_s200_2026-08-03_17-15-34/1"
RUN[n30_300]=logs/yN30_s300_stg1_2026-08-03_21-28-47/TRPO_x_control_SecureBERT; CS[n30_300]=250000; TOPO[n30_300]="./graphs_yN30_s300_2026-08-03_17-15-34/1"
RUN[n90_42]=logs/yprobe_n90_s42_ext125M_2026-08-03_13-42-07/TRPO_x_control_SecureBERT;  CS[n90_42]=500000;  TOPO[n90_42]="./graphs_yprobe_n90_2026-08-03_02-10-10/1"
RUN[n90_100]=logs/yN90ext_s100_stg1_2026-08-05_22-55-50/TRPO_x_control_SecureBERT;       CS[n90_100]=250000; TOPO[n90_100]="./graphs_yprobe_n90_s100_2026-08-03_08-38-04/1"
RUN[n90_123]=logs/yprobe_n90_s123_static500k_2026-08-03_08-43-27/TRPO_x_control_SecureBERT; CS[n90_123]=500000; TOPO[n90_123]="./graphs_yprobe_n90_s123_2026-08-03_08-38-04/1"
RUN[n90_200]=logs/yprobe_n90_s200_static500k_2026-08-03_08-43-27/TRPO_x_control_SecureBERT; CS[n90_200]=500000; TOPO[n90_200]="./graphs_yprobe_n90_s200_2026-08-03_08-38-04/1"
RUN[n90_300]=logs/yN90ext_s300_stg1_2026-08-05_22-55-50/TRPO_x_control_SecureBERT;       CS[n90_300]=250000; TOPO[n90_300]="./graphs_yprobe_n90_s300_2026-08-03_08-38-04/1"

pids=()
for cell in n30_42 n30_100 n30_123 n30_200 n30_300; do
  seed="${cell#n30_}"
  CKPT_STEP="${CS[$cell]}" nohup nice -n 5 "$PY" y_robustness/scripts/taskF2_eval.py \
    "${RUN[$cell]}" "$seed" "${TOPO[$cell]}" static "$N_EP" y_robustness/out/n30 \
    > "y_robustness/logs_run/${cell}_static.log" 2>&1 &
  pids+=($!)
  CKPT_STEP="${CS[$cell]}" nohup nice -n 5 "$PY" y_robustness/scripts/taskF2_eval.py \
    "${RUN[$cell]}" "$seed" "${TOPO[$cell]}" membership_matched "$N_EP" y_robustness/out/n30 20 \
    > "y_robustness/logs_run/${cell}_membership_matched.log" 2>&1 &
  pids+=($!)
done
for cell in n90_42 n90_100 n90_123 n90_200 n90_300; do
  seed="${cell#n90_}"
  CKPT_STEP="${CS[$cell]}" nohup nice -n 5 "$PY" y_robustness/scripts/taskF2_eval.py \
    "${RUN[$cell]}" "$seed" "${TOPO[$cell]}" static "$N_EP" y_robustness/out/n90 \
    > "y_robustness/logs_run/${cell}_static.log" 2>&1 &
  pids+=($!)
  CKPT_STEP="${CS[$cell]}" nohup nice -n 5 "$PY" y_robustness/scripts/taskF2_eval.py \
    "${RUN[$cell]}" "$seed" "${TOPO[$cell]}" membership_matched "$N_EP" y_robustness/out/n90 4 \
    > "y_robustness/logs_run/${cell}_membership_matched.log" 2>&1 &
  pids+=($!)
done

echo "[wait] ${#pids[@]} eval runs: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "[wait] done; failed=$fail"
echo "RUN COMPLETE (failed=$fail)"
