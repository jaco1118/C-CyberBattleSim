#!/bin/bash
# F4 threshold RECALIBRATION on the eval signal, BEFORE the 80-100 500k verdict.
# Reference data (identical to the training-curve criterion's references, evidence_taskF4.md):
#   30-40  original 250k runs  = KNOWN CONVERGED  -> eval-signal noise floor
#   80-100 original 250k runs  = KNOWN NOT-CONVERGED at 250k (Task R: still +12% on the training curve)
# Windows 150k/200k/250k -> preceding [150,200] vs final [200,250]. Same conv_eval.py, N=60 as verdict.
set -u
B=/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1
OUT=$B/conv_calib.csv
LOGS=$B/conv_calib_logs
mkdir -p "$LOGS"; rm -f "$OUT"
N=${N:-60}; MAXJOBS=${MAXJOBS:-4}
STEPS="150000 200000 250000"
declare -a RUNS=(
  "30-40|42|scalability_30_40/44|runs/trpo_250k_F1_static_seed42"
  "30-40|100|scalability_30_40/44|runs/trpo_250k_F1_static_seed100"
  "30-40|123|scalability_30_40/44|runs/trpo_250k_F1_static_seed123"
  "30-40|200|scalability_30_40/44|runs/trpo_250k_F1_static_seed200"
  "30-40|300|scalability_30_40/44|runs/trpo_250k_F1_static_seed300"
  "80-100|42|scalability_80_100/5|f2_runs/trpo_250k_F2_static_band80-100_seed42"
  "80-100|100|scalability_80_100/100|f2_runs/trpo_250k_F2_static_band80-100_seed100"
  "80-100|123|scalability_80_100/18|f2_runs/trpo_250k_F2_static_band80-100_seed123"
  "80-100|200|scalability_80_100/2|f2_runs/trpo_250k_F2_static_band80-100_seed200"
  "80-100|300|scalability_80_100/67|f2_runs/trpo_250k_F2_static_band80-100_seed300"
)
run_one() {
  local band=$1 seed=$2 topo=$3 rel=$4 step=$5
  OMP_NUM_THREADS=4 python "$B/conv_eval.py" "$B/$rel" "$topo" "$step" "$N" "$OUT" "$band" "$seed" \
     > "$LOGS/${band}_seed${seed}_${step}.out" 2>&1
  echo "$(date +%H:%M:%S) DONE ${band}_seed${seed}_${step} (exit $?)" >> "$LOGS/_progress.log"
}
echo "$(date +%H:%M:%S) launching CALIBRATION eval: 10 runs x 3 ckpts, N=$N, concurrency $MAXJOBS" >> "$LOGS/_progress.log"
for entry in "${RUNS[@]}"; do
  IFS='|' read -r band seed topo rel <<< "$entry"
  for step in $STEPS; do
    run_one "$band" "$seed" "$topo" "$rel" "$step" &
    while [ "$(jobs -r | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
  done
done
wait
echo "$(date +%H:%M:%S) ALL calibration evals done" >> "$LOGS/_progress.log"
