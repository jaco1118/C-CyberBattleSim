"""Task RQ1B-MECH-SPLIT-ROBUST: robustness check on the mechanical-share-vs-size finding from the
immediately preceding task (compute_mechanical_share_scale.py, commit dc14831/e5cd7ea).

That task found mechanical_share ~ log(network_size): log-size coef -0.1677, bootstrap 95% CI
[-0.1941,-0.1431], R^2=0.648, on 2,931/4,410 episodes -- excluding episodes with gross_root_loss==0,
gross_root_loss<0, OR mechanical_share>1 (the last exclusion added mid-task after a near-zero-
denominator blowup was caught). That exclusion rate is UNEVEN across bands (STEP 0 0.1, computed
fresh here from the per-episode data, exact not rounded): 189/588=32.14% at 10-15, 239/1569=15.23%
at 30-40, 110/1312=8.38% at 80-100. This task tests whether that unevenness is DRIVING the finding's
direction, by re-running the identical regression under an alternative, INCLUSIVE treatment: instead
of excluding share>1 episodes, cap mechanical_share at 1.0 and keep them in.

This is a SEPARATE, ADDITIVE computation. It does not modify or overwrite the original excluded-
version result files (mechanical_share_per_cell.csv / mechanical_share_regression_results.csv,
commit e5cd7ea). The capped-version numbers themselves are computed fresh from
cx_step2_registration/ and rq1a_cells.csv, independently of the prior script's output -- the only
place the original results CSV is touched at all is a read-only load for the side-by-side print at
the end (item 4), never written to.

DEFINITIONS -- only the share>1 branch differs from the original script:
  gross_root_loss_i == 0            -> undefined, EXCLUDE (unchanged from the original task)
  gross_root_loss_i < 0             -> undefined, EXCLUDE (unchanged from the original task)
  gross_root_loss_i > 0, ratio > 1  -> mechanical_share_i = 1.0 (top-coded), INCLUDE (the only change)
  gross_root_loss_i > 0, ratio <= 1 -> mechanical_share_i = ratio, INCLUDE (unchanged)

SAFETY: reads only already-existing files (cx_step2_registration/, the existing rq1a_cells.csv). No
training, no environment reset, no checkpoint or encoder touched, no step()/encode()/reward path
modified. Nothing beyond reading existing files and computing statistics is run.

Usage: run from cyberbattle/agents/ (where the raw data lives):
  python ../../analysis/rq1b_mech_split_scale_2026-08-08/compute_mechanical_share_capped_robustness.py
"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

RQ1A_CSV = "../../analysis/rq1a_regression_recovered_2026-08-07/rq1a_cells.csv"
CHANGE_BASE = "cx_step2_registration/eventgraph_{band}/s{seed}_{topo}/event_episode.jsonl"
ORIGINAL_RESULTS_CSV = "../../analysis/rq1b_mech_split_scale_2026-08-08/mechanical_share_regression_results.csv"
BANDS = ["10-15", "30-40", "80-100"]
NBOOT = 10000
BOOT_SEED = 11
OUT_DIR = "../../analysis/rq1b_mech_split_scale_2026-08-08"


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def bootstrap_ols(logn, y, n_boot=NBOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    slopes, r2s = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        Xb = sm.add_constant(logn[idx])
        m = sm.OLS(y[idx], Xb).fit()
        slopes.append(m.params[1])
        r2s.append(m.rsquared)
    return (float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))), \
           (float(np.percentile(r2s, 2.5)), float(np.percentile(r2s, 97.5)))


def main():
    rq1a = pd.read_csv(RQ1A_CSV)
    cell_rows = []

    # 0.1: exact per-band totals, computed fresh here (not re-derived from rounded chat figures)
    band_totals = {b: {"total": 0, "zero": 0, "neg": 0, "over1": 0, "at_or_under1": 0} for b in BANDS}

    for _, cell in rq1a.iterrows():
        band, seed, topo = cell["band"], int(cell["seed"]), int(cell["topo"])
        root_owned_static = cell["root_owned_static"]
        network_size = cell["network_size"]
        rows = load_rows(CHANGE_BASE.format(band=band, seed=seed, topo=topo))

        capped_shares = []
        n_zero = n_neg = n_over1 = n_capped_in = 0
        for r in rows:
            band_totals[band]["total"] += 1
            gross = root_owned_static - r["final_root_owned_count"]
            if gross == 0:
                n_zero += 1
                band_totals[band]["zero"] += 1
            elif gross < 0:
                n_neg += 1
                band_totals[band]["neg"] += 1
            else:
                mech = r["root_owned_departures"]
                ratio = mech / gross
                if ratio > 1:
                    n_over1 += 1
                    n_capped_in += 1
                    capped_shares.append(1.0)
                    band_totals[band]["over1"] += 1
                else:
                    capped_shares.append(ratio)
                    band_totals[band]["at_or_under1"] += 1

        cell_rows.append({
            "band": band, "seed": seed, "topo": topo, "network_size": network_size,
            "mechanical_share_mean_capped": float(np.mean(capped_shares)) if capped_shares else None,
            "n_episodes_total": len(rows),
            "n_episodes_included_capped": len(capped_shares),
            "n_episodes_excluded_zero": n_zero, "n_episodes_excluded_negative": n_neg,
            "n_episodes_capped_to_1": n_over1,
            "share_over1_rate_of_gross_positive": (n_over1 / (n_over1 + (len(capped_shares) - n_over1))
                                                    if (n_over1 + (len(capped_shares) - n_over1)) > 0 else None),
        })

    df = pd.DataFrame(cell_rows)
    df.to_csv(os.path.join(OUT_DIR, "mechanical_share_capped_per_cell.csv"), index=False)

    print("=" * 78)
    print("0.1 (recomputed exactly, fresh from per-episode data): share>1 rate by band")
    print("=" * 78)
    for b in BANDS:
        t = band_totals[b]
        gross_pos = t["over1"] + t["at_or_under1"]
        rate = 100 * t["over1"] / gross_pos
        print(f"  band {b}: total={t['total']}  gross==0={t['zero']}  gross<0={t['neg']}  "
              f"gross>0={gross_pos}  share>1={t['over1']} ({rate:.2f}%)  share<=1={t['at_or_under1']}")

    print()
    print("=" * 78)
    print("CAPPED-VERSION episode/cell counts")
    print("=" * 78)
    tot_by_band = df.groupby("band")["n_episodes_total"].sum()
    inc_by_band = df.groupby("band")["n_episodes_included_capped"].sum()
    capped_by_band = df.groupby("band")["n_episodes_capped_to_1"].sum()
    for b in BANDS:
        print(f"  band {b}: total={tot_by_band[b]}  included(capped)={inc_by_band[b]}  "
              f"of which capped-to-1={capped_by_band[b]}")
    print(f"  TOTAL: included(capped)={inc_by_band.sum()} vs original excluded-version included="
          f"2931 (prior task) -- capped version adds {inc_by_band.sum()-2931} episodes")

    n_missing = df["mechanical_share_mean_capped"].isna().sum()
    print(f"\nCells with a usable capped mechanical_share_mean: {len(df)-n_missing} / {len(df)}")

    reg_df = df.dropna(subset=["mechanical_share_mean_capped"]).copy()
    print()
    print("=" * 78)
    print(f"REGRESSION [CAPPED VERSION]: mechanical_share_mean_capped ~ log(network_size), n={len(reg_df)}")
    print("=" * 78)
    logn = np.log(reg_df["network_size"].to_numpy())
    y = reg_df["mechanical_share_mean_capped"].to_numpy()
    X = sm.add_constant(logn)
    model = sm.OLS(y, X).fit()
    const, slope = model.params
    print(f"  const={const:+.4f}  log_size_coef={slope:+.4f}  (SE {model.bse[1]:.4f}, "
          f"p={model.pvalues[1]:.4g})  R^2={model.rsquared:.4f}  n={int(model.nobs)}")
    ci_slope, ci_r2 = bootstrap_ols(logn, y)
    print(f"  Bootstrap 95% CI (10,000 resamples over {len(reg_df)} cells):")
    print(f"    log_size_coef: [{ci_slope[0]:+.4f}, {ci_slope[1]:+.4f}]")
    print(f"    R^2          : [{ci_r2[0]:.4f}, {ci_r2[1]:.4f}]")
    band_means = reg_df.groupby("band")["mechanical_share_mean_capped"].agg(["mean", "std", "count"])
    print("  band means [capped]:")
    print(band_means.to_string())

    print()
    print("=" * 78)
    print("SIDE-BY-SIDE: original excluded-version vs. capped-version regression")
    print("=" * 78)
    orig = pd.read_csv(ORIGINAL_RESULTS_CSV)
    orig_primary = orig[orig["version"] == "clean_primary"].iloc[0]
    print(f"  ORIGINAL (excluded, n={int(orig_primary['n_cells'])}): "
          f"log_size_coef={orig_primary['log_size_coef']:+.4f}  "
          f"CI=[{orig_primary['log_size_coef_ci_lo']:+.4f}, {orig_primary['log_size_coef_ci_hi']:+.4f}]  "
          f"p={orig_primary['p_value']:.3e}  R^2={orig_primary['r_squared']:.4f}")
    print(f"  CAPPED   (n={len(reg_df)}): log_size_coef={slope:+.4f}  "
          f"CI=[{ci_slope[0]:+.4f}, {ci_slope[1]:+.4f}]  p={model.pvalues[1]:.3e}  R^2={model.rsquared:.4f}")
    same_sign = (slope < 0) == (orig_primary["log_size_coef"] < 0)
    print(f"  SAME SIGN: {same_sign}")
    if same_sign:
        ratio_mag = abs(slope) / abs(orig_primary["log_size_coef"])
        print(f"  magnitude ratio (capped/original): {ratio_mag:.3f} "
              f"({'similar' if 0.5 < ratio_mag < 2.0 else 'notably different'} magnitude)")

    print()
    print("=" * 78)
    print("Item 5: does the ORIGINAL version's per-cell share>1 rate correlate with log(network_size)?")
    print("(descriptive only, not causal)")
    print("=" * 78)
    rate_df = df.dropna(subset=["share_over1_rate_of_gross_positive"]).copy()
    logn_r = np.log(rate_df["network_size"].to_numpy())
    rate_v = rate_df["share_over1_rate_of_gross_positive"].to_numpy()
    r = float(np.corrcoef(logn_r, rate_v)[0, 1])
    rng = np.random.default_rng(BOOT_SEED + 2)
    n = len(rate_df)
    rs = []
    for _ in range(NBOOT):
        idx = rng.integers(0, n, n)
        rs.append(np.corrcoef(logn_r[idx], rate_v[idx])[0, 1])
    ci_r = (float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5)))
    print(f"  r(share>1 rate, log(network_size)) = {r:+.4f}  bootstrap 95% CI [{ci_r[0]:+.4f}, {ci_r[1]:+.4f}]  "
          f"n={n} cells")

    results = {
        "n_cells": len(reg_df), "const": const, "log_size_coef": slope, "log_size_coef_se": model.bse[1],
        "log_size_coef_ci_lo": ci_slope[0], "log_size_coef_ci_hi": ci_slope[1],
        "p_value": model.pvalues[1], "r_squared": model.rsquared,
        "r_squared_ci_lo": ci_r2[0], "r_squared_ci_hi": ci_r2[1],
        "same_sign_as_original": same_sign,
        "share_over1_rate_vs_logsize_r": r,
        "share_over1_rate_vs_logsize_ci_lo": ci_r[0],
        "share_over1_rate_vs_logsize_ci_hi": ci_r[1],
    }
    for b in BANDS:
        results[f"band_{b}_mean"] = band_means.loc[b, "mean"]
        results[f"band_{b}_std"] = band_means.loc[b, "std"]
        results[f"band_{b}_n_cells"] = band_means.loc[b, "count"]
    pd.DataFrame([results]).to_csv(os.path.join(OUT_DIR, "mechanical_share_capped_regression_results.csv"), index=False)
    print(f"\nWrote: {OUT_DIR}/mechanical_share_capped_per_cell.csv ({len(df)} rows), "
          f"mechanical_share_capped_regression_results.csv")
    print("\nOriginal excluded-version files (mechanical_share_per_cell.csv, "
          "mechanical_share_regression_results.csv) were NOT read for computation (only for the "
          "side-by-side print above) and were NOT modified.")


if __name__ == "__main__":
    main()
