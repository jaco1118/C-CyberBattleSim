#!/bin/bash
# Task Y-EARLYCKPT STEP 1: robustness eval at each cell's earliest durably-converged
# reward checkpoint (N=60 stage 3 = cumulative 750k; N=90 static500k stage = 500k),
# same harness/episodes/change_interval as the currently-reported figures. New output
# paths only -- does not touch the existing n60/n90 committed evaluation output.
set -u
AG=/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents
PY=/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/bin/python
export LD_LIBRARY_PATH=/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
N_EP=200
mkdir -p "$AG/y_robustness/out/n60_stage3" "$AG/y_robustness/out/n90_static500k" "$AG/y_robustness/logs_run"
cd "$AG"

declare -A RUN CS TOPO CI
RUN[n60_42]=logs/yN60_s42_stg3_2026-08-03_23-00-46/TRPO_x_control_SecureBERT;   CS[n60_42]=250000;  TOPO[n60_42]="./graphs_yN60_s42_2026-08-03_17-15-34/1"
RUN[n60_100]=logs/yN60_s100_stg3_2026-08-03_23-00-46/TRPO_x_control_SecureBERT; CS[n60_100]=250000; TOPO[n60_100]="./graphs_yN60_s100_2026-08-03_17-15-34/1"
RUN[n60_123]=logs/yN60_s123_stg3_2026-08-03_23-00-46/TRPO_x_control_SecureBERT; CS[n60_123]=250000; TOPO[n60_123]="./graphs_yN60_s123_2026-08-03_17-15-34/1"
RUN[n60_200]=logs/yN60_s200_stg3_2026-08-03_23-00-46/TRPO_x_control_SecureBERT; CS[n60_200]=250000; TOPO[n60_200]="./graphs_yN60_s200_2026-08-03_17-15-34/1"
RUN[n60_300]=logs/yN60_s300_stg3_2026-08-03_23-00-46/TRPO_x_control_SecureBERT; CS[n60_300]=250000; TOPO[n60_300]="./graphs_yN60_s300_2026-08-03_17-15-34/1"
RUN[n90_42]=logs/yprobe_n90_static500k_2026-08-03_02-20-13/TRPO_x_control_SecureBERT;      CS[n90_42]=500000;  TOPO[n90_42]="./graphs_yprobe_n90_2026-08-03_02-10-10/1"
RUN[n90_100]=logs/yprobe_n90_s100_static500k_2026-08-03_08-43-27/TRPO_x_control_SecureBERT; CS[n90_100]=500000; TOPO[n90_100]="./graphs_yprobe_n90_s100_2026-08-03_08-38-04/1"
RUN[n90_123]=logs/yprobe_n90_s123_static500k_2026-08-03_08-43-27/TRPO_x_control_SecureBERT; CS[n90_123]=500000; TOPO[n90_123]="./graphs_yprobe_n90_s123_2026-08-03_08-38-04/1"
RUN[n90_200]=logs/yprobe_n90_s200_static500k_2026-08-03_08-43-27/TRPO_x_control_SecureBERT; CS[n90_200]=500000; TOPO[n90_200]="./graphs_yprobe_n90_s200_2026-08-03_08-38-04/1"
RUN[n90_300]=logs/yprobe_n90_s300_static500k_2026-08-03_08-43-27/TRPO_x_control_SecureBERT; CS[n90_300]=500000; TOPO[n90_300]="./graphs_yprobe_n90_s300_2026-08-03_08-38-04/1"

pids=()
for cell in n60_42 n60_100 n60_123 n60_200 n60_300; do
  seed="${cell#n60_}"
  CKPT_STEP="${CS[$cell]}" nohup nice -n 5 "$PY" y_robustness/scripts/taskF2_eval.py \
    "${RUN[$cell]}" "$seed" "${TOPO[$cell]}" static "$N_EP" y_robustness/out/n60_stage3 \
    > "y_robustness/logs_run/${cell}_stage3_static.log" 2>&1 &
  pids+=($!)
  CKPT_STEP="${CS[$cell]}" nohup nice -n 5 "$PY" y_robustness/scripts/taskF2_eval.py \
    "${RUN[$cell]}" "$seed" "${TOPO[$cell]}" membership_matched "$N_EP" y_robustness/out/n60_stage3 7 \
    > "y_robustness/logs_run/${cell}_stage3_membership_matched.log" 2>&1 &
  pids+=($!)
done
for cell in n90_42 n90_100 n90_123 n90_200 n90_300; do
  seed="${cell#n90_}"
  CKPT_STEP="${CS[$cell]}" nohup nice -n 5 "$PY" y_robustness/scripts/taskF2_eval.py \
    "${RUN[$cell]}" "$seed" "${TOPO[$cell]}" static "$N_EP" y_robustness/out/n90_static500k \
    > "y_robustness/logs_run/${cell}_static500k_static.log" 2>&1 &
  pids+=($!)
  CKPT_STEP="${CS[$cell]}" nohup nice -n 5 "$PY" y_robustness/scripts/taskF2_eval.py \
    "${RUN[$cell]}" "$seed" "${TOPO[$cell]}" membership_matched "$N_EP" y_robustness/out/n90_static500k 4 \
    > "y_robustness/logs_run/${cell}_static500k_membership_matched.log" 2>&1 &
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
