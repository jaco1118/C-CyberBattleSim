"""Calibrate change_interval to ~11.23 leave/episode under the TRAINED F2 policy.
Runs the F2 eval harness in a lightweight mode: membership condition, small episode count, at
candidate change_interval values, reporting leave/ep. Uses seed42/topology5 as the representative.
Usage: python taskF2_calibrate_rate.py <candidate_ci_1> <candidate_ci_2> ..."""
import sys, subprocess, os, re, tempfile
BASE="/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1"
RUN=f"{BASE}/f2_runs/trpo_250k_F2_static_band80-100_seed42"
tmpout=tempfile.mkdtemp()
for ci in sys.argv[1:]:
    r=subprocess.run(["python",f"{BASE}/taskF2_eval.py",RUN,"42","5","membership_matched","30",tmpout,ci],
                     capture_output=True,text=True)
    line=[l for l in (r.stdout+r.stderr).splitlines() if "leave/ep=" in l]
    m=re.search(r"leave/ep=([\d.]+)", line[0]) if line else None
    print(f"change_interval={ci}: leave/ep={m.group(1) if m else 'NA'} (30 eps, seed42/topo5, trained policy)")
