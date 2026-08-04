#!/bin/bash
export USER=slchan OMP_NUM_THREADS=1 PYTHONHASHSEED=0
B=/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1
python replay_verify.py 30-40 replay "$B/rv_30-40_replay" "$B/rv_30-40_p1/actions.npy" >/dev/null 2>&1
python replay_verify.py 80-100 policy "$B/rv_80-100_p1" >/dev/null 2>&1
python replay_verify.py 80-100 policy "$B/rv_80-100_p2" >/dev/null 2>&1
python replay_verify.py 80-100 replay "$B/rv_80-100_replay" "$B/rv_80-100_p1/actions.npy" >/dev/null 2>&1
echo "rv_rest done $(date +%H:%M:%S)"
