import pandas as pd, numpy as np
BASE="/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents/attenuation_drift_logs"
cols=["change_type","event_phase","touched_node_visible","attributed","node_origin_is_join","change_drift_full","change_fired"]
print("Q1.3 + Q1.4 — join events in the RQ1 gate drift logs (change_drift_full exact-zero split)\n")
for band in ["10-15","30-40","80-100"]:
    d=pd.read_csv(f"{BASE}/drift_{band}.csv", usecols=cols)
    ct=d.change_type.value_counts().to_dict()
    j=d[d.change_type=="membership_join"]
    print(f"### band {band}: total rows={len(d)} ; change_type counts={ct}")
    if len(j)==0:
        print("   no membership_join rows\n"); continue
    # split by whether the joined/touched node is visible (discovered within episode)
    vis=j[j.touched_node_visible==True]; inv=j[j.touched_node_visible==False]
    def z(x): return (x.change_drift_full==0.0).sum()
    print(f"   membership_join rows={len(j)} | visible(discovered)={len(vis)}  not-visible(undiscovered)={len(inv)}")
    if len(inv): print(f"     UNDISCOVERED joins: change_drift_full==0 exactly: {z(inv)}/{len(inv)} = {z(inv)/len(inv):.4f}  (max|drift|={inv.change_drift_full.abs().max():.3e})")
    if len(vis): print(f"     DISCOVERED   joins: change_drift_full==0 exactly: {z(vis)}/{len(vis)} = {z(vis)/len(vis):.4f}  (median drift={vis.change_drift_full.median():.4f})")
    # also split by attributed flag as a cross-check
    at=j[j.attributed==True]; nat=j[j.attributed==False]
    print(f"   [xcheck by attributed] attributed={len(at)} (zero-drift {z(at)}/{len(at) if len(at) else 0}) ; not-attributed={len(nat)} (zero-drift {z(nat)}/{len(nat) if len(nat) else 0})")
    print()
