"""Task X STEP A: perception axis for node-PROPERTY change (all 4 slices), existing F-series data.
Property = patch (vuln removal). Provenance: F1 static (30-40/topo44) + F2 static (80-100) @250k.
Response rate at tau=0 per slice = fraction with change_drift_slice > 0, over the standard filtered
event set (relevant & touched_node_visible & event_phase in {immediate,attributed}), per seed."""
import pandas as pd, numpy as np, glob, os
B = "/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1"
SEEDS = [42, 100, 123, 200, 300]
BANDS = {"30-40": "eval_out", "80-100": "f2eval_out"}

def load(folder, cond):
    fs = [f"{B}/{folder}/drift_static_seed{s}_eval{cond}.csv" for s in SEEDS]
    return pd.concat([pd.read_csv(f).assign(_seed=s) for f, s in zip(fs, SEEDS) if os.path.exists(f)], ignore_index=True)

def filt(d, ct):
    return d[(d.change_type == ct) & (d.relevant == True) & (d.touched_node_visible == True) &
             (d.event_phase.isin(["immediate", "attributed"]))]

def rr_table(d, ct):
    """per-slice response rate at tau=0: mean across seeds (sd across seeds)."""
    e = filt(d, ct); out = {}
    for sl in ["full", "mean", "max", "min"]:
        col = f"change_drift_{sl}"
        per = e.groupby("_seed").apply(lambda g: (g[col] > 0).mean(), include_groups=False)
        out[sl] = (per.mean(), per.std(ddof=1))
    return out, len(e)

print("="*88)
print("TASK X STEP A — perception axis for node-PROPERTY change (patch)  [F1/F2 static @250k]")
print("="*88)

# A.1 / A.2 : property + membership response rates, all 4 slices, same basis (F-series)
print("\n### A.1 + A.2 response rate at tau=0, all 4 slices; property vs membership_leave, per band [FINDING] ###")
print("  (mean across 5 seeds (sd); membership computed on the SAME F-series agents for like-for-like)")
hdr = f"  {'band':7s} {'change':16s} | " + " | ".join(f"{s:^13s}" for s in ["full","mean","max","min"])
print(hdr)
for band, folder in BANDS.items():
    prop = load(folder, "property"); mem = load(folder, "membership")
    for ct, dd in [("property(patch)", (prop, "property")), ("membership_leave", (mem, "membership_leave"))]:
        tbl, n = rr_table(dd[0], dd[1])
        row = f"  {band:7s} {ct:16s} | " + " | ".join(f"{tbl[s][0]:.3f}(sd{tbl[s][1]:.3f})" for s in ["full","mean","max","min"])
        print(row + f"   n={n}")

# A.3 : exact-zero full-vector change drift among property events
print("\n### A.3 property events with EXACTLY zero full-vector change drift, per band [FINDING] ###")
zero_detail = {}
for band, folder in BANDS.items():
    e = filt(load(folder, "property"), "property")
    n = len(e); z_exact = (e.change_drift_full == 0.0).sum(); z_eps = (e.change_drift_full.abs() <= 1e-12).sum()
    zero_detail[band] = e[e.change_drift_full.abs() <= 1e-12]
    print(f"  {band:7s}: n_property_events={n}  exact_zero(==0.0)={z_exact} ({z_exact/n:.4f})  |Δ|<=1e-12={z_eps} ({z_eps/n:.4f})")

# A.4 : characterise the zeros (seed distribution; node/operation not loggable)
print("\n### A.4 characterisation of the exact zeros [FINDING] ###")
for band in BANDS:
    zd = zero_detail[band]
    if len(zd) == 0:
        print(f"  {band}: no exact zeros -> nothing to characterise."); continue
    by_seed = zd.groupby("_seed").size().to_dict()
    print(f"  {band}: {len(zd)} zeros; by seed = {by_seed}")
    print(f"    (operation: ALL property events are 'patch' (vuln removal); node identity is NOT logged,")
    print(f"     so per-node / node-kind characterisation is not possible from existing data.)")

# A.5 : exact-zero full-vector drift among membership_leave (expected 0)
print("\n### A.5 exact-zero full-vector change drift among MEMBERSHIP_LEAVE (expected 0) [FINDING] ###")
for band, folder in BANDS.items():
    e = filt(load(folder, "membership"), "membership_leave")
    z = (e.change_drift_full == 0.0).sum()
    print(f"  {band}: n_leave_events={len(e)}  exact_zero_full={z} ({z/max(len(e),1):.4f})")
