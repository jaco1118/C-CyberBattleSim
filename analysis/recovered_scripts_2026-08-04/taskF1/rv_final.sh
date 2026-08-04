#!/bin/bash
export USER=slchan
B=/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1
# wait for the reverted-logger regression to finish
for i in $(seq 1 40); do
  if [ -f $B/L_regd_30-40/drift_old.csv ] && [ -f $B/L_regd_80-100/drift_old.csv ] && ! pgrep -f drive_L_regression_det >/dev/null; then break; fi
  sleep 30
done
cd /cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/
echo "post-regression git: $(git diff --stat cyberbattle/_env/cyberbattle_env_compressed.py|tail -1) stash:$(git stash list|wc -l)"
python3 - <<'PY'
import pandas as pd, numpy as np
def cmp(f1,f2):
    a=pd.read_csv(f1).drop(columns=["run_id"]); b=pd.read_csv(f2).drop(columns=["run_id"])
    x,y=a.values,b.values; return int((~((x==y)|(pd.isna(x)&pd.isna(y)))).sum()) if a.shape==b.shape else "SHAPE"
B="/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1"
print("=== reverted STEP-1 logger byte-identical regression ===")
for BAND in ["30-40","80-100"]:
    d=f"{B}/L_regd_{BAND}"
    print(f"  {BAND}: old-vs-new_off={cmp(f'{d}/drift_old.csv',f'{d}/drift_new_off.csv')} | old-vs-new_on={cmp(f'{d}/drift_old.csv',f'{d}/drift_new_on.csv')} cells")
PY
# measure reverted disk via the demo
export OMP_NUM_THREADS=2 PYTHONHASHSEED=0
cd $B; python demo_L.py >/dev/null 2>&1
python3 -c "
import os; d='$B/L_demo/eventgraph'; j=os.path.getsize(d+'/event_graph.jsonl'); f=os.path.getsize(d+'/event_obs.f32'); n=sum(1 for _ in open(d+'/event_graph.jsonl'))
print(f'reverted logger: {n} change-steps, per-step {(j+f)/n:.0f} B -> ~50k steps = {(j+f)/n*50000/1e6:.0f} MB (back to STEP-1 ~440 MB)')
print('  fields:', list(__import__('json').loads(open(d+'/event_graph.jsonl').readline()).keys()))
"
echo "RV_FINAL DONE"
