#!/bin/bash
export USER=slchan
PILOT=/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1/y_pilot_configs
for i in $(seq 1 40); do grep -q "ITER2 COMPLETE" "$PILOT/progress2.log" 2>/dev/null && break; sleep 30; done
cd /cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/
echo "=== iter-2 rejection rates (G-level restarts for 5 accepted) ==="
for N in 30 60 90; do echo "  ${N}: $(grep -c 'does not satisfy' "$PILOT/gen2_${N}.log") restarts / 5 accepted"; done
python3 - <<'PY' 2>&1 | grep -vE "UserWarning|data_dict"
import pickle, glob, numpy as np, os
REPO="/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim"
def newest(pat):
    ds=sorted(glob.glob(f"{REPO}/cyberbattle/data/env_samples/{pat}"))
    return ds[-1] if ds else None
def load(d): return [pickle.load(open(p,"rb")) for p in sorted(glob.glob(f"{d}/*/network_SecureBERT.pkl"))]
def vpn(nets): return np.mean([len(nd["data"].vulnerabilities) for net in nets for _,nd in net.network.nodes(data=True)])
print("\n=== iter-2 FINAL-p: degree (target 22) + vulns/node ===")
for N in [30,60,90]:
    d=newest(f"graphs_ypilot2_{N}_*"); nets=load(d)
    ko=[np.mean([x for _,x in net.knows_graph.out_degree()]) for net in nets]
    ac=[net.access_connectivity for net in nets]
    print(f"  {N}-node ({len(nets)}g): KNOWS out-deg={np.mean(ko):.1f} (spread {min(ko):.1f}-{max(ko):.1f}) | access_conn={np.mean(ac):.3f} | vulns/node={vpn(nets):.1f}")
# existing anchor comparison
print("\n=== existing bands vulns/node (for the confound comparison at FINAL p) ===")
for band,sub in [("30-40","scalability_30_40"),("80-100","scalability_80_100")]:
    print(f"  existing {band}: vulns/node={vpn(load2 := [pickle.load(open(p,'rb')) for p in sorted(glob.glob(f'{REPO}/cyberbattle/data/env_samples/{sub}/*/network_SecureBERT.pkl'))[:6]]):.1f}")
PY
echo "Y_MEASURE2 DONE"
