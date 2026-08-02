#!/bin/bash
# Resume RQ2C-1 collection for the seeds not finished on 2026-08-02 (deadline cut-off).
# 10-15 is COMPLETE (5 seeds) -- do NOT re-run it. 30-40 and 80-100 have only a PARTIAL seed-42
# in append-mode files, so we DELETE their partial rq2c logs and re-run those two bands in full.
set -e
cd "$(dirname "$0")/.."          # -> cyberbattle/agents
OUT="$PWD/rq2c_replay"
echo "[resume] deleting partial 30-40 / 80-100 rq2c logs (append-mode; would otherwise double-count)"
rm -f "$OUT"/rq2c/rq2c_30-40_*.jsonl "$OUT"/rq2c/rq2c_80-100_*.jsonl
for b in 30-40 80-100; do
  echo "[resume] launching band $b (all 5 seeds)"
  env PYTHONHASHSEED=0 YEG_DRIFT_DIR="$OUT" RQ2C=1 RQ2C_DIR="$OUT/rq2c" \
    nohup python compute_attenuation_analysis.py --manifest "$OUT/rq2c_manifest_$b.yaml" --collect \
    > "$OUT/run_$b.log" 2>&1 &
done
echo "[resume] both bands launched (nohup, survive logout). When done:"
echo "  python compute_rq2c_action_divergence.py --input-dir $OUT/rq2c --out $OUT/rq2c_action_divergence_table.md --n-boot 10000"
wait
