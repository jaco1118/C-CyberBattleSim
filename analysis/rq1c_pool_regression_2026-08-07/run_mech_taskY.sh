#!/bin/bash
# Task RQ1C-POOL STEP 1.1: run taskF3_mech_eval.py (change arm, was_root/root_owned_departures)
# against Task Y's 14 already-trained seeds, reusing the exact checkpoints, topologies, and
# calibrated churn conditions already established in TASK Y-ROBUSTNESS / TASK Y-NEIGHBOUR.
# No new training. No modification to taskF3_mech_eval.py -- pure copy-and-point, per STEP 0B.1.
set -u
AG=/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents
PY=/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/bin/python
export LD_LIBRARY_PATH=/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
N_EP=200
OUT="$AG/y_robustness/out/mech_taskY"
mkdir -p "$OUT"
cd "$AG"

declare -A RUN CS TOPO CI BANDLABEL
# N=30 high-degree (attenuation-pooling-scale cell, ci=20 default)
for s in 42 100 123 200 300; do
  RUN[hi_$s]="logs/yN30_s${s}_stg1_2026-08-03_21-28-47/TRPO_x_control_SecureBERT"
  CS[hi_$s]=250000
  CI[hi_$s]=20
  BANDLABEL[hi_$s]="taskY_n30hi"
done
TOPO[hi_42]="./graphs_yN30_s42_2026-08-03_17-15-34/1"
TOPO[hi_100]="./graphs_yN30_s100_2026-08-03_17-15-34/1"
TOPO[hi_123]="./graphs_yN30_s123_2026-08-03_17-15-34/1"
TOPO[hi_200]="./graphs_yN30_s200_2026-08-03_17-15-34/1"
TOPO[hi_300]="./graphs_yN30_s300_2026-08-03_17-15-34/1"

# N=90 (taskY-probe-n90 cell, ci=4 calibrated)
RUN[n90_42]="logs/yprobe_n90_s42_ext125M_2026-08-03_13-42-07/TRPO_x_control_SecureBERT";  CS[n90_42]=500000
RUN[n90_100]="logs/yN90ext_s100_stg1_2026-08-05_22-55-50/TRPO_x_control_SecureBERT";       CS[n90_100]=250000
RUN[n90_123]="logs/yprobe_n90_s123_static500k_2026-08-03_08-43-27/TRPO_x_control_SecureBERT"; CS[n90_123]=500000
RUN[n90_200]="logs/yprobe_n90_s200_static500k_2026-08-03_08-43-27/TRPO_x_control_SecureBERT"; CS[n90_200]=500000
RUN[n90_300]="logs/yN90ext_s300_stg1_2026-08-05_22-55-50/TRPO_x_control_SecureBERT";       CS[n90_300]=250000
TOPO[n90_42]="./graphs_yprobe_n90_2026-08-03_02-10-10/1"
TOPO[n90_100]="./graphs_yprobe_n90_s100_2026-08-03_08-38-04/1"
TOPO[n90_123]="./graphs_yprobe_n90_s123_2026-08-03_08-38-04/1"
TOPO[n90_200]="./graphs_yprobe_n90_s200_2026-08-03_08-38-04/1"
TOPO[n90_300]="./graphs_yprobe_n90_s300_2026-08-03_08-38-04/1"
for s in 42 100 123 200 300; do CI[n90_$s]=4; BANDLABEL[n90_$s]="taskY_n90"; done

# N=30 low-degree (taskY2-pilot-n30 cell, ci=20 default; seed42 excluded, never converged)
for s in 100 123 200 300; do
  RUN[lo_$s]="logs/yN30_s${s}_stg1_2026-08-05_21-24-52/TRPO_x_control_SecureBERT"
  CS[lo_$s]=250000
  CI[lo_$s]=20
  BANDLABEL[lo_$s]="taskY_n30lo"
done
TOPO[lo_100]="./graphs_yN30_s100_2026-08-05_21-16-40/1"
TOPO[lo_123]="./graphs_yN30_s123_2026-08-05_21-17-08/1"
TOPO[lo_200]="./graphs_yN30_s200_2026-08-05_21-17-46/1"
TOPO[lo_300]="./graphs_yN30_s300_2026-08-05_21-18-13/1"

pids=()
for cell in "${!RUN[@]}"; do
  seed="${cell#*_}"
  CKPT_STEP="${CS[$cell]}" nohup nice -n 5 "$PY" y_robustness/scripts/taskF3_mech_eval.py \
    "${RUN[$cell]}" "$seed" "${TOPO[$cell]}" "$N_EP" "$OUT" "${BANDLABEL[$cell]}" "${CI[$cell]}" \
    > "$OUT/${BANDLABEL[$cell]}_s${seed}.log" 2>&1 &
  pids+=($!)
  echo "[launch] $cell  run=${RUN[$cell]}  ckpt=${CS[$cell]}  topo=${TOPO[$cell]}  ci=${CI[$cell]}  pid=$!"
done

echo "[wait] ${#pids[@]} eval runs: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "[wait] done; failed=$fail"
echo "RUN COMPLETE (failed=$fail)"
