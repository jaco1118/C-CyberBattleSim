import pandas as pd, numpy as np
BASE="/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents/attenuation_drift_logs"
cols=["seed","episode","step","change_type","event_phase","touched_node_visible",
      "change_drift_full","delta_h_v_norm"]
for band in ["10-15","30-40","80-100"]:
    d=pd.read_csv(f"{BASE}/drift_{band}.csv", usecols=cols)
    key=["seed","episode","step"]
    # per step: did ANY event move the vector (delta_h_v_norm>0)?
    mover = d[d.delta_h_v_norm.fillna(0)>0].groupby(key).size()
    mover_steps = set(map(tuple, [list(x) for x in mover.index]))
    j=d[d.change_type=="membership_join"]
    inv=j[j.touched_node_visible==False]
    nz=inv[inv.change_drift_full.fillna(0)!=0.0]   # nonzero global drift
    z =inv[inv.change_drift_full.fillna(-1)==0.0]   # exactly zero global drift
    nzkeys=list(map(tuple,nz[key].values.tolist()))
    co=sum(1 for k in nzkeys if k in mover_steps)
    # NaN breakdown for delta_h_v of undiscovered joins
    ndel_nan = inv.delta_h_v_norm.isna().sum(); ndel_zero=(inv.delta_h_v_norm==0).sum()
    print(f"### {band}: undiscovered joins={len(inv)}  [own delta: zero={ndel_zero}, NaN={ndel_nan}, nonzero={(inv.delta_h_v_norm.fillna(0)>0).sum()}]")
    print(f"   global change_drift_full: exactly-zero={len(z)}  nonzero={len(nz)}")
    print(f"   of nonzero-drift undiscovered joins, step had ANOTHER vector-moving event: {co}/{len(nz)} = {co/max(len(nz),1):.4f}")
    # the residual: nonzero drift but NO co-firing mover -> investigate
    resid=[k for k in nzkeys if k not in mover_steps]
    print(f"   residual (nonzero drift, no co-firing mover in step): {len(resid)}")
    if resid:
        rk=set(resid); sample=nz[nz.apply(lambda r:(r.seed,r.episode,r.step) in rk,axis=1)].head(1)
        for _,r in sample.iterrows():
            same=d[(d.seed==r.seed)&(d.episode==r.episode)&(d.step==r.step)]
            print(f"     e.g. seed{r.seed} ep{r.episode} step{r.step}: rows this step ->")
            print(same[["change_type","touched_node_visible","change_drift_full","delta_h_v_norm"]].to_string(index=False))
    print()
