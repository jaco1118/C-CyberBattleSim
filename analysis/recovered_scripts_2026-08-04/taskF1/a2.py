"""Task A2: why do gate and F-series disagree on max/min response rate at 80-100? Analysis only.
Same filter both pipelines: membership_leave & relevant & touched_node_visible & event_phase in
{immediate,attributed}; response = change_drift_slice > 0."""
import pandas as pd, numpy as np, os
GB = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/attenuation_gate_archive/2026-07-26_trpo_5seed_gate/attenuation_drift_logs"
FB = "/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1"
SEEDS = [42, 100, 123, 200, 300]

def gate(band):
    return pd.read_csv(f"{GB}/drift_{band}.csv")
def fseries(band):
    folder = "eval_out" if band == "30-40" else "f2eval_out"
    return pd.concat([pd.read_csv(f"{FB}/{folder}/drift_static_seed{s}_evalmembership.csv").assign(seed=s)
                      for s in SEEDS if os.path.exists(f"{FB}/{folder}/drift_static_seed{s}_evalmembership.csv")], ignore_index=True)
def filt(d):
    return d[(d.change_type == "membership_leave") & (d.relevant == True) & (d.touched_node_visible == True) &
             (d.event_phase.isin(["immediate", "attributed"]))].copy()

PIPES = {"gate": gate, "F-series": fseries}
print("="*82); print("TASK A2 — gate vs F-series max/min response rate @80-100"); print("="*82)

# STEP 1: reproduce the four figures + counts, same filter
print("\n### STEP 1.1/1.3 response rates (same filter both pipelines) [ARTIFACT] ###")
data = {}
for pipe, fn in PIPES.items():
    for band in ["30-40", "80-100"]:
        e = filt(fn(band)); data[(pipe, band)] = e
        rmax = e.groupby("seed").apply(lambda g: (g.change_drift_max > 0).mean(), include_groups=False)
        rmin = e.groupby("seed").apply(lambda g: (g.change_drift_min > 0).mean(), include_groups=False)
        print(f"  {pipe:9s} {band:7s}: max={rmax.mean():.3f}(sd{rmax.std(ddof=1):.3f}) min={rmin.mean():.3f}(sd{rmin.std(ddof=1):.3f})  "
              f"n_events={len(e)} seeds={e.seed.nunique()} scenarios={e.scenario_id.nunique()}")

# STEP 2.1: n_discovered distribution at leave events
print("\n### STEP 2.1 n_discovered at leave events, per pipeline per band [FINDING] ###")
for band in ["30-40", "80-100"]:
    for pipe in PIPES:
        e = data[(pipe, band)]; nd = e.n_discovered
        print(f"  {pipe:9s} {band:7s}: n_discovered mean={nd.mean():.1f} median={nd.median():.0f} p10={nd.quantile(.1):.0f} p90={nd.quantile(.9):.0f} min={nd.min():.0f} max={nd.max():.0f}")

# STEP 2.2: response rate vs n_discovered, binned, both pipelines overlaid
print("\n### STEP 2.2 response rate vs n_discovered (pooled across bands), max & min slice [FINDING] ###")
allpipe = {p: pd.concat([data[(p, b)] for b in ["30-40", "80-100"]], ignore_index=True) for p in PIPES}
bins = [0, 5, 10, 15, 20, 30, 40, 55, 70, 85, 200]
print("  n_disc bin |    max: gate / F-series (n_g,n_f)    |    min: gate / F-series")
for i in range(len(bins) - 1):
    lo, hi = bins[i], bins[i+1]; row = f"  [{lo:3d},{hi:3d}) | "
    vals = {}
    for sl in ["max", "min"]:
        for p in PIPES:
            e = allpipe[p]; sub = e[(e.n_discovered >= lo) & (e.n_discovered < hi)]
            vals[(sl, p)] = ((sub[f"change_drift_{sl}"] > 0).mean(), len(sub))
    ng, nf = vals[("max","gate")][1], vals[("max","F-series")][1]
    if ng + nf == 0: continue
    row += f"{vals[('max','gate')][0]:.2f}/{vals[('max','F-series')][0]:.2f} ({ng},{nf})".ljust(34)
    row += f"| {vals[('min','gate')][0]:.2f}/{vals[('min','F-series')][0]:.2f}"
    print(row)

# STEP 2.3: same curve? vertical gap at matched n_discovered (overlap region)
print("\n### STEP 2.3 same curve? gap at matched n_discovered [FINDING] ###")
for sl in ["max", "min"]:
    gaps = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        g = allpipe["gate"]; f = allpipe["F-series"]
        gs = g[(g.n_discovered >= lo) & (g.n_discovered < hi)]; fs = f[(f.n_discovered >= lo) & (f.n_discovered < hi)]
        if len(gs) >= 50 and len(fs) >= 50:
            gaps.append((f"[{lo},{hi})", (gs[f'change_drift_{sl}']>0).mean() - (fs[f'change_drift_{sl}']>0).mean(), len(gs), len(fs)))
    print(f"  {sl} slice: gate-minus-Fseries response-rate gap where BOTH have >=50 events:")
    for lab, gap, ng, nf in gaps:
        print(f"    {lab:10s}: gap={gap:+.3f} (n_gate={ng}, n_f={nf})")
    if gaps:
        mg = np.mean([g[1] for g in gaps]); print(f"    mean gap over overlapping bins = {mg:+.3f}")
