"""Task RQ1A-RECOVER STEP 1: recovered RQ1a regression -- conquest + log(network size)
predicting relative fragility, across all 117 (band, seed, topology) cells.

Background: the original script that produced RQ1a's headline regression (R^2=0.376,
coef conquest=+0.309, coef log(size)=+0.054) was never committed to the repo and could not be
recovered (see evidence_cards/ for the investigation trail: STEP 0 found the raw logs behind it,
0B pinned down "network size" and confirmed the model is a genuine two-predictor regression
(not three, despite band-related numbers mentioned nearby in the surviving patch prose), 0C/0D
confirmed which relative-fragility formula the original analysis actually used -- formula B
(behavioural residual, not gross loss) -- by reproducing the patch's own cited per-band reference
figures to within 0.0002-0.0004 on the full 117-cell dataset).

This script recomputes the three quantities from the raw event logs and fits the regression fresh,
rather than reproducing (or being able to reproduce) the original code, which does not survive.

Raw data: cyberbattle/agents/cx_step2_static/ (undisturbed arm), cx_step2_registration/ (disturbed
arm), eventgraph_<band>/s<seed>_<topology>/event_episode.jsonl.

Definitions used (all confirmed against source in the STEP 0/0B/0C/0D investigation, not assumed):
  conquest        = root_owned_static / ownable_count           (both read from the static arm)
  network_size    = num_nodes_start (== initial_alive)          (fixed per cell, static arm)
  relative_fragility (formula B) = behavioural_residual / root_owned_static, where
      root_owned_static  = mean(final_root_owned_count) over the cell's STATIC-arm episodes
      cost               = root_owned_static - mean(final_root_owned_count over CHANGE-arm episodes)
      mechanical         = mean(root_owned_departures over CHANGE-arm episodes)
      behavioural_residual = cost - mechanical
  (cost/mechanical convention: arm-mean first, then combine -- matches
  analysis/recovered_scripts_2026-08-04/taskF1/taskF3_count_recompute.py on attenuation-pooling-scale,
  confirmed identical in STEP 0D.1.)

Usage: run from cyberbattle/agents/ (where the raw cx_step2_* data lives):
  python ../../analysis/rq1a_regression_recovered_2026-08-07/compute_rq1a_regression.py [--out-csv PATH]
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

STATIC_BASE = "cx_step2_static/eventgraph_{band}/s{seed}_{topo}/event_episode.jsonl"
CHANGE_BASE = "cx_step2_registration/eventgraph_{band}/s{seed}_{topo}/event_episode.jsonl"
SEEDS = [42, 100, 123, 200, 300]
BANDS = ["10-15", "30-40", "80-100"]
NBOOT = 10000
BOOT_SEED = 11


def topos_for(band, seed):
    pattern = f"cx_step2_registration/eventgraph_{band}/s{seed}_*"
    return sorted(d.split("_")[-1] for d in glob.glob(pattern))


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def compute_cell(band, seed, topo):
    static_path = STATIC_BASE.format(band=band, seed=seed, topo=topo)
    change_path = CHANGE_BASE.format(band=band, seed=seed, topo=topo)
    if not os.path.exists(change_path):
        return None  # the 3 known-missing change-arm cells (band 10-15: s100/t29, s100/t45, s123/t29)
    static_rows = load_rows(static_path)
    change_rows = load_rows(change_path)

    # network size + ownable_count: confirmed fixed per cell (static arm) -- verify, don't assume
    nns_vals = {r["num_nodes_start"] for r in static_rows}
    oc_vals = {r["ownable_count"] for r in static_rows}
    nns_fixed = len(nns_vals) == 1
    oc_fixed = len(oc_vals) == 1
    network_size = next(iter(nns_vals)) if nns_fixed else None
    ownable_count = next(iter(oc_vals)) if oc_fixed else None

    root_owned_static = sum(r["final_root_owned_count"] for r in static_rows) / len(static_rows)
    mean_change = sum(r["final_root_owned_count"] for r in change_rows) / len(change_rows)
    cost = root_owned_static - mean_change
    mechanical = sum(r["root_owned_departures"] for r in change_rows) / len(change_rows)
    behavioural_residual = cost - mechanical
    relative_fragility = behavioural_residual / root_owned_static

    conquest = root_owned_static / ownable_count if oc_fixed else None

    return {
        "band": band, "seed": seed, "topo": topo,
        "n_static_ep": len(static_rows), "n_change_ep": len(change_rows),
        "network_size": network_size, "network_size_fixed": nns_fixed,
        "ownable_count": ownable_count, "ownable_count_fixed": oc_fixed,
        "root_owned_static": root_owned_static, "cost": cost, "mechanical": mechanical,
        "behavioural_residual": behavioural_residual,
        "conquest": conquest, "relative_fragility": relative_fragility,
    }


def build_dataset():
    rows = []
    skipped = []
    for band in BANDS:
        for seed in SEEDS:
            for topo in topos_for(band, seed):
                c = compute_cell(band, seed, topo)
                if c is None:
                    skipped.append((band, seed, topo))
                else:
                    rows.append(c)
    return pd.DataFrame(rows), skipped


def fit_ols(df, log_base):
    if log_base == "ln":
        log_size = np.log(df["network_size"])
    elif log_base == "log10":
        log_size = np.log10(df["network_size"])
    else:
        raise ValueError(log_base)
    X = sm.add_constant(np.column_stack([df["conquest"].to_numpy(), log_size.to_numpy()]))
    y = df["relative_fragility"].to_numpy()
    model = sm.OLS(y, X).fit()
    return model


def bootstrap_ci(df, log_base, n_boot=NBOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    n = len(df)
    conquest_coefs, size_coefs, intercepts, r2s = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = df.iloc[idx]
        m = fit_ols(sample, log_base)
        intercepts.append(m.params[0])
        conquest_coefs.append(m.params[1])
        size_coefs.append(m.params[2])
        r2s.append(m.rsquared)
    def ci(a):
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {
        "intercept_ci": ci(intercepts), "conquest_ci": ci(conquest_coefs),
        "log_size_ci": ci(size_coefs), "r2_ci": ci(r2s),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", default="../../analysis/rq1a_regression_recovered_2026-08-07/rq1a_cells.csv")
    args = ap.parse_args()

    df, skipped = build_dataset()
    print(f"Cells computed: {len(df)}  (expected 117)")
    print(f"Skipped (no change-arm data): {skipped}")
    assert len(df) == 117, f"expected 117 cells, got {len(df)}"

    bad_size = df[~df["network_size_fixed"]]
    bad_oc = df[~df["ownable_count_fixed"]]
    print(f"Cells with non-fixed network_size within static arm: {len(bad_size)}")
    print(f"Cells with non-fixed ownable_count within static arm: {len(bad_oc)}")

    print("\n" + "=" * 70)
    print("STEP 1B.3 -- summary statistics across all 117 cells")
    print("=" * 70)
    for col in ["conquest", "network_size", "relative_fragility"]:
        s = df[col]
        print(f"  {col:20s}: mean={s.mean():.4f} sd={s.std(ddof=1):.4f} min={s.min():.4f} max={s.max():.4f}")

    print("\nPer-band relative_fragility means (cross-check vs STEP 0D):")
    print(df.groupby("band")["relative_fragility"].mean())

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nPer-cell data written to {args.out_csv}")

    for log_base in ["ln", "log10"]:
        print("\n" + "=" * 70)
        print(f"STEP 1C -- OLS fit, log base = {log_base}")
        print("=" * 70)
        model = fit_ols(df, log_base)
        print(model.summary())
        print(f"\nR^2 (plain) = {model.rsquared:.4f}")
        print(f"n observations = {int(model.nobs)}")
        print(f"intercept = {model.params[0]:+.4f} (SE {model.bse[0]:.4f})")
        print(f"coef conquest = {model.params[1]:+.4f} (SE {model.bse[1]:.4f})")
        print(f"coef log_size = {model.params[2]:+.4f} (SE {model.bse[2]:.4f})")

        boot = bootstrap_ci(df, log_base)
        print(f"\nBootstrap 95% CI (10,000 resamples of the 117 cells, NEW -- not in the original thesis):")
        print(f"  intercept: {boot['intercept_ci']}")
        print(f"  conquest : {boot['conquest_ci']}")
        print(f"  log_size : {boot['log_size_ci']}")
        print(f"  R^2      : {boot['r2_ci']}")


if __name__ == "__main__":
    main()
