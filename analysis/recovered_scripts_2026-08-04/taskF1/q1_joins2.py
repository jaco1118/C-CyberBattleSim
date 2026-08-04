import pandas as pd, numpy as np
BASE="/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents/attenuation_drift_logs"
cols=["run_id","seed","episode","step","change_type","event_phase","touched_node_visible",
      "attributed","node_origin_is_join","change_drift_full","delta_h_v_norm","n_touched_nodes"]
print("Q1.4 resolution: join's OWN contribution (delta_h_v_norm) vs the global shared change_drift_full\n")
for band in ["10-15","30-40","80-100"]:
    d=pd.read_csv(f"{BASE}/drift_{band}.csv", usecols=cols)
    j=d[d.change_type=="membership_join"]
    inv=j[j.touched_node_visible==False]   # undiscovered joins
    # join's OWN effect on the vector:
    dv0=(inv.delta_h_v_norm==0.0).sum()
    print(f"### {band}: undiscovered-join rows={len(inv)}")
    print(f"   delta_h_v_norm (the JOIN's own contribution) == 0 exactly: {dv0}/{len(inv)} = {dv0/len(inv):.4f}  max={inv.delta_h_v_norm.abs().max():.2e}")
    # of the undiscovered joins whose GLOBAL change_drift_full != 0, does a leave/property share that step?
    nz=inv[inv.change_drift_full!=0.0]
    # for each such (seed,episode,step), was there a co-firing leave/property row?
    key=["run_id","seed","episode","step"]
    steps_with_leaveprop = set(map(tuple, d[d.change_type.isin(["membership_leave","property"])][key].drop_duplicates().values.tolist()))
    nzkeys = list(map(tuple, nz[key].values.tolist()))
    co = sum(1 for k in nzkeys if k in steps_with_leaveprop)
    print(f"   undiscovered-joins with NONZERO global change_drift_full: {len(nz)}")
    print(f"     of those, step also had a co-firing leave/property event: {co}/{len(nz)} = {co/max(len(nz),1):.4f}")
    print(f"     their own delta_h_v_norm==0: {(nz.delta_h_v_norm==0.0).sum()}/{len(nz)}\n")
