"""Task GRAPH-DEPTH STEP 2.1/2.3: byte-identical comparison. old vs new_off (2a, must pass) and
old vs new_on (2b, the untested path this task exists to check). Compares the drift CSV
cell-by-cell (NaN==NaN treated equal) over EVERY pre-existing column, and the trajectory npz
(returned obs, reward, done) for exact equality. Modelled directly on the already-proven
compare_L_regression.py (commit c05a16a, attenuation-pooling-scale).
"""
import sys
import pandas as pd
import numpy as np
import os

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
BANDS = ["30-40", "80-100"]
ID_COLS = ["run_id"]  # identifier tag set per-run; not behavioural. seed/scenario_id are held equal.


def diff_cells(a, b):
    a = a.drop(columns=[c for c in ID_COLS if c in a.columns])
    b = b.drop(columns=[c for c in ID_COLS if c in b.columns])
    assert list(a.columns) == list(b.columns), f"column mismatch: {set(a.columns) ^ set(b.columns)}"
    assert a.shape == b.shape, f"shape {a.shape} vs {b.shape}"
    eq = (a.values == b.values) | (pd.isna(a.values) & pd.isna(b.values))
    return int((~eq).sum())


def main():
    allpass = True
    results = []
    for band in BANDS:
        band_out = os.path.join(OUT, band)
        old = pd.read_csv(os.path.join(band_out, "drift_old.csv"))
        print(f"\n=== BAND {band}: old drift rows={len(old)}, cols={len(old.columns)} ===")
        for tag in ["new_off", "new_on"]:
            new = pd.read_csv(os.path.join(band_out, f"drift_{tag}.csv"))
            same_cols = [c for c in new.columns if c not in ID_COLS] == [c for c in old.columns if c not in ID_COLS]
            dc = diff_cells(old, new) if same_cols and old.shape == new.shape else -1
            to = np.load(os.path.join(band_out, "traj_old.npz"))
            tn = np.load(os.path.join(band_out, f"traj_{tag}.npz"))
            r_eq = np.array_equal(to["reward"], tn["reward"])
            d_eq = np.array_equal(to["done"], tn["done"])
            o_eq = np.array_equal(to["obs"], tn["obs"])
            o_max = float(np.max(np.abs(to["obs"] - tn["obs"]))) if to["obs"].shape == tn["obs"].shape else -1
            ok = (same_cols and dc == 0 and r_eq and d_eq and o_eq)
            allpass = allpass and ok
            print(f"  old vs {tag}: same_drift_cols={same_cols} drift_differing_cells={dc} | "
                  f"traj reward_eq={r_eq} done_eq={d_eq} obs_eq={o_eq} (max|delta_obs|={o_max:.2e}) -> {'PASS' if ok else 'FAIL'}")
            results.append({"band": band, "comparison": f"old_vs_{tag}", "same_drift_cols": same_cols,
                             "drift_differing_cells": dc, "reward_eq": r_eq, "done_eq": d_eq,
                             "obs_eq": o_eq, "max_abs_delta_obs": o_max, "pass": ok})
    print(f"\nOVERALL: {'PASS' if allpass else 'FAIL'}")
    pd.DataFrame(results).to_csv(os.path.join(OUT, "regression_results.csv"), index=False)
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
