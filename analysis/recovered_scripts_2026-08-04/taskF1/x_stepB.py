"""Task X STEP B: relevance axis + graded perception (B.3a). F-series data, F1/F2 static @250k.
Channel count = # of {mean,max,min} slices that responded at tau=0 (full excluded; full always moves)."""
import pandas as pd, numpy as np, os
B = "/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1"
SEEDS = [42, 100, 123, 200, 300]
BANDS = {"30-40": "eval_out", "80-100": "f2eval_out"}

def load(folder, cond):
    return pd.concat([pd.read_csv(f"{B}/{folder}/drift_static_seed{s}_eval{cond}.csv").assign(_seed=s)
                      for s in SEEDS if os.path.exists(f"{B}/{folder}/drift_static_seed{s}_eval{cond}.csv")], ignore_index=True)
def filt(d, ct):
    return d[(d.change_type == ct) & (d.relevant == True) & (d.touched_node_visible == True) &
             (d.event_phase.isin(["immediate", "attributed"]))].copy()

print("="*84); print("TASK X STEP B — relevance axis + graded perception  [F1/F2 static @250k]"); print("="*84)

# B.3a: channel-count distribution (0/1/2/3), per band + change type
print("\n### B.3a GRADED perception: # of {mean,max,min} slices responding at tau=0, dist 0-3 [FINDING] ###")
print("  (per-event count; mean(sd) of the per-seed proportion across 5 seeds)")
for band, folder in BANDS.items():
    for ct in ["property", "membership_leave"]:
        cond = "property" if ct == "property" else "membership"
        e = filt(load(folder, cond), ct)
        if len(e) == 0: continue
        e["cc"] = (e.change_drift_mean > 0).astype(int) + (e.change_drift_max > 0).astype(int) + (e.change_drift_min > 0).astype(int)
        # per-seed proportion of each count level, then mean/sd across seeds
        dist = {}
        for k in [0, 1, 2, 3]:
            per = e.groupby("_seed").apply(lambda g: (g.cc == k).mean(), include_groups=False)
            dist[k] = (per.mean(), per.std(ddof=1))
        row = f"  {band:7s} {ct:16s}: " + "  ".join(f"cc={k}:{dist[k][0]:.3f}(sd{dist[k][1]:.3f})" for k in [0,1,2,3])
        print(row + f"   n={len(e)}")

# B.3: relevance rate -- raw flag (constant) + the one varying component (was_owned, membership only)
print("\n### B.3 relevance rate [FINDING] ###")
print("  raw flag `relevant`: CONSTANT True for property and membership_leave at both bands (B.2), so it")
print("  cannot carry the axis. Only varying component available = `was_owned` (leaveown CSV, membership):")
for band, folder in BANDS.items():
    per = []
    for s in SEEDS:
        f = f"{B}/{folder}/leaveown_static_seed{s}_evalmembership.csv"
        if os.path.exists(f):
            lo = pd.read_csv(f); per.append(lo.was_owned.mean())
    if per:
        per = np.array(per)
        print(f"  {band:7s} membership_leave was_owned rate = {per.mean():.3f} (sd {per.std(ddof=1):.3f}) across {len(per)} seeds")
print("  property: NO varying relevance component logged (no was_owned; property is not a departure).")

# B.4: cross-tab relevance x perception. Binary relevance is constant -> report the (degenerate) binary
#      table and the discriminating channel-count table, both in the single 'relevant' column.
print("\n### B.4 cross-tab: relevance (constant) x perception [FINDING] ###")
print("  Table 1 (binary relevance x binary full-vector perception): both saturated ->")
for band, folder in BANDS.items():
    for ct in ["property", "membership_leave"]:
        cond = "property" if ct == "property" else "membership"
        e = filt(load(folder, cond), ct)
        if len(e) == 0: continue
        rel = (e.relevant == True).mean(); perc = (e.change_drift_full > 0).mean()
        print(f"    {band:7s} {ct:16s}: relevant={rel:.3f}, full-responds={perc:.3f} -> ALL {len(e)} events in [relevant x perceived]")
print("  Table 2 (binary relevance x channel count): relevance=1 (one column), channel-count rows = the")
print("  B.3a distribution above -> the discrimination is entirely on the perception (channel) axis.")
print("  The was_owned split that would make relevance non-degenerate CANNOT be joined to perception:")
print("  drift CSV has no node_id, leaveown CSV has no step; only (seed,episode) shared, and batch leave")
print("  events break rank-alignment -> the joint was_owned x channel-count table is NOT computable here.")
