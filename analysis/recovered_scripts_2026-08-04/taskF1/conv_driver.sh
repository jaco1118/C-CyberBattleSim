#!/bin/bash
# F4 convergence eval: each of 15 runs at 3 window-boundary checkpoints (400k/450k/500k), N=60 static
# episodes, mean root_owned. Window-vs-window Δ% criterion applied afterward by conv_analyze.py.
set -u
B=/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1
OUT=$B/conv_results.csv
LOGS=$B/conv_logs
mkdir -p "$LOGS"
rm -f "$OUT"
N=${N:-60}
MAXJOBS=${MAXJOBS:-5}
STEPS="400000 450000 500000"

# band|seed|topo_subpath  (topologies as built by launch_f4_train.sh)
declare -a RUNS=(
  "30-40|42|scalability_30_40/44"   "30-40|100|scalability_30_40/44"  "30-40|123|scalability_30_40/44"
  "30-40|200|scalability_30_40/44"  "30-40|300|scalability_30_40/44"
  "80-100|42|scalability_80_100/5"  "80-100|100|scalability_80_100/100" "80-100|123|scalability_80_100/18"
  "80-100|200|scalability_80_100/2" "80-100|300|scalability_80_100/67"
  "10-15|42|scalability_10_15/61"   "10-15|100|scalability_10_15/54"  "10-15|123|scalability_10_15/62"
  "10-15|200|scalability_10_15/29"  "10-15|300|scalability_10_15/4"
)
run_one() {
  local band=$1 seed=$2 topo=$3 step=$4
  local rf="$B/f4_runs/f4_static_${band}_seed${seed}"
  OMP_NUM_THREADS=4 python "$B/conv_eval.py" "$rf" "$topo" "$step" "$N" "$OUT" "$band" "$seed" \
     > "$LOGS/${band}_seed${seed}_${step}.out" 2>&1
  echo "$(date +%H:%M:%S) DONE ${band}_seed${seed}_${step} (exit $?)" >> "$LOGS/_progress.log"
}
echo "$(date +%H:%M:%S) launching convergence eval: 15 runs x 3 ckpts, N=$N, concurrency $MAXJOBS" >> "$LOGS/_progress.log"
for entry in "${RUNS[@]}"; do
  IFS='|' read -r band seed topo <<< "$entry"
  for step in $STEPS; do
    run_one "$band" "$seed" "$topo" "$step" &
    while [ "$(jobs -r | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
  done
done
wait
echo "$(date +%H:%M:%S) ALL convergence evals done" >> "$LOGS/_progress.log"
