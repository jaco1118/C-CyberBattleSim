"""STEP 2.1/2.3: byte-identical comparison. old vs new_off and old vs new_on, both bands. Compares the
drift CSV cell-by-cell (NaN==NaN treated equal) over EVERY pre-existing column, and the trajectory npz
(returned obs, reward, done) for exact equality."""
import pandas as pd, numpy as np, os
B="/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1"
ID_COLS=['run_id']  # identifier tag set per-run; not behavioural. seed/scenario_id are held equal.
def diff_cells(a, b):
    a=a.drop(columns=[c for c in ID_COLS if c in a.columns]); b=b.drop(columns=[c for c in ID_COLS if c in b.columns])
    assert list(a.columns)==list(b.columns), f"column mismatch: {set(a.columns)^set(b.columns)}"
    assert a.shape==b.shape, f"shape {a.shape} vs {b.shape}"
    eq = (a.values==b.values) | (pd.isna(a.values) & pd.isna(b.values))
    return int((~eq).sum())
allpass=True
for BAND in ["30-40","80-100"]:
    OUT=f"{B}/L_reg_{BAND}"
    old=pd.read_csv(f"{OUT}/drift_old.csv")
    print(f"\n=== BAND {BAND}: old drift rows={len(old)}, cols={len(old.columns)} ===")
    for tag in ["new_off","new_on"]:
        new=pd.read_csv(f"{OUT}/drift_{tag}.csv")
        # new drift CSV schema must EQUAL old (side data is separate); assert same columns
        same_cols = [c for c in new.columns if c not in ID_COLS]==[c for c in old.columns if c not in ID_COLS]
        dc = diff_cells(old, new) if same_cols and old.shape==new.shape else -1
        # trajectory
        to=np.load(f"{OUT}/traj_old.npz"); tn=np.load(f"{OUT}/traj_{tag}.npz")
        r_eq=np.array_equal(to["reward"],tn["reward"]); d_eq=np.array_equal(to["done"],tn["done"])
        o_eq=np.array_equal(to["obs"],tn["obs"]); o_max=float(np.max(np.abs(to["obs"]-tn["obs"]))) if to["obs"].shape==tn["obs"].shape else -1
        ok = (same_cols and dc==0 and r_eq and d_eq and o_eq)
        allpass = allpass and ok
        print(f"  old vs {tag}: same_drift_cols={same_cols} drift_differing_cells={dc} | "
              f"traj reward_eq={r_eq} done_eq={d_eq} obs_eq={o_eq} (max|Δobs|={o_max:.2e}) -> {'PASS' if ok else 'FAIL'}")
print(f"\nOVERALL: {'PASS — byte-identical (drift_logging=True path unchanged by Task L)' if allpass else 'FAIL'}")
