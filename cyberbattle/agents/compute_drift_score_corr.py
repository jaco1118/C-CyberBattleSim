"""O-8 Part 1: within-episode drift vs per-step action-success (binary proxy) correlation.

Point-biserial correlation between per-step observation drift (agent_drift_full = the obs change
caused by the agent's OWN action, h1->h2) and agent_action_succeeded (binary: reward>0 that step).

IMPORTANT (labelling): this is "drift vs per-step action-success (binary proxy)". It is NOT the same
claim as "drift vs magnitude of score change" -- that would need a continuous per-step reward, which is
NOT logged anywhere in the drift CSVs (only the binary agent_action_succeeded is). A continuous version
would require a new run and is flagged as a separate future item, not done here.

Reads the drift CSVs already logged (no new training). Reports per band: point-biserial r, n steps, p.

Usage: python compute_drift_score_corr.py [--drift-dir <dir with drift_<band>.csv>]
"""
import argparse, glob, os, re
import numpy as np, pandas as pd
from scipy import stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift-dir", default="cyberbattle/agents/attenuation_step3_logs")
    ap.add_argument("--drift-col", default="agent_drift_full")
    args = ap.parse_args()
    print(f"drift source: {args.drift_dir}  | drift col: {args.drift_col}  | score proxy: agent_action_succeeded (binary reward>0)")
    print(f"{'band':>8} {'n_steps':>10} {'point_biserial_r':>18} {'p_value':>12}  {'mean_drift|succ=1':>18} {'mean_drift|succ=0':>18}")
    for f in sorted(glob.glob(os.path.join(args.drift_dir, "drift_*.csv"))):
        band = re.search(r"drift_(.+)\.csv", os.path.basename(f)).group(1)
        df = pd.read_csv(f, low_memory=False)
        # per-step rows with both signals present; coerce action-success to 0/1
        d = df[[args.drift_col, "agent_action_succeeded"]].copy()
        d[args.drift_col] = pd.to_numeric(d[args.drift_col], errors="coerce")
        d["s"] = d["agent_action_succeeded"].map({True: 1, False: 0, "True": 1, "False": 0})
        d = d.dropna(subset=[args.drift_col, "s"])
        drift = d[args.drift_col].to_numpy(float); succ = d["s"].to_numpy(int)
        if len(np.unique(succ)) < 2:
            print(f"{band:>8} {len(d):>10}  (action-success has one class only -> correlation undefined)"); continue
        r, p = stats.pointbiserialr(succ, drift)
        m1 = drift[succ == 1].mean(); m0 = drift[succ == 0].mean()
        print(f"{band:>8} {len(d):>10} {r:>18.4f} {p:>12.2e}  {m1:>18.4f} {m0:>18.4f}")


if __name__ == "__main__":
    main()
