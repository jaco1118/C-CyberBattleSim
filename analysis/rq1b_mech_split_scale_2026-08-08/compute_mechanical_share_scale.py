"""Task RQ1B-MECH-SPLIT-SCALE STEP 1: does the mechanical share of root loss shift with network
size, on the real 117-cell CX main-study population.

Background: an older, three-figure "69%/72%/76% mechanical share" number exists
(evidence_cards/evidence_taskF3.md:237-239), but it is NOT a size-ordered three-band series -- it
is two node bands (30-40, 80-100) under three CHURN CONDITIONS (30-40 fixed-absolute; 80-100
fixed-absolute; 80-100 fixed-relative), with no 10-15 figure and 80-100 counted twice. This was
already caught and documented by this project itself (evidence_cards/claims_audit.md, CA-1;
evidence_cards/evidence_taskZ.md, 0.1) -- "the premise is mislabelled; the numbers are sound."
Verified again in this task (STEP 1C-MECHANISM STEP 0's own investigation trail): the raw data
behind those figures lives in a separate F2/F3 characterisation-eval collection
(job-scratch taskF1/{eval_out,f2eval_out,f3_mech_out,f3_rel_out}/, computed via
taskF3_mech_analyze.py), NOT the CX main-study population (cx_step2_registration/) FORMULA B and
the RQ1a regression use -- confirmed still present on disk, contrary to an earlier session's belief
it was lost, but a genuinely different dataset regardless. This task does not use that data or
attempt to reconcile with those three figures; it computes the same question properly, from
scratch, on the correct, already-verified 117-cell population.

DEFINITIONS (verified against FORMULA B's own source, RQ1C-MECHANISM STEP 1's 0.2/0.3 -- exact
algebraic match, ~1e-15 floating point noise, already checked once for the same construction):
  gross_root_loss_i      = root_owned_static_cell - final_root_owned_count_i
  mechanical_root_loss_i = root_owned_departures_i
  mechanical_share_i     = mechanical_root_loss_i / gross_root_loss_i
  (behavioural_residual_i = gross_root_loss_i - mechanical_root_loss_i, matching FORMULA B exactly;
  mean_i(gross_root_loss_i - mechanical_root_loss_i) over a cell's change-arm episodes equals that
  cell's already-committed behavioural_residual value, by linearity of the mean.)
root_owned_static_cell is read directly from the existing, committed 117-cell CSV
(analysis/rq1a_regression_recovered_2026-08-07/rq1a_cells.csv, commit 418abc3);
final_root_owned_count_i / root_owned_departures_i are read per-episode from
cx_step2_registration/eventgraph_<band>/s<seed>_<topo>/event_episode.jsonl (same raw data FORMULA B
and RQ1a's regression already use -- no new evaluation run).

DEGENERATE CASES (STEP 0 numerics convention, verified counts reported below): gross_root_loss_i ==
0 or < 0 -> mechanical_share_i is UNDEFINED for that episode, excluded, not floored. Checked STEP 0
0.3: 0/9/2 episodes (10-15/30-40/80-100) have gross_root_loss==0; 52/192/686 have gross_root_loss<0
(the disturbed episode ended up owning MORE root nodes than the cell's own static-arm mean --
expected, especially at 80-100 where static-arm conquest itself is lower and episode-to-episode
root-owned-count variance is larger, not a data error). All 117 cells retain at least 5 usable
episodes after exclusion (checked STEP 0 0.5, min 5/median 16-36/max 83 across bands) -- per-cell
means are stable, no cell dropped entirely.

AGGREGATION: one mechanical_share value per cell = mean of per-episode mechanical_share_i over that
cell's usable (gross_root_loss_i > 0) change-arm episodes, matching the granularity (n=117) and the
arm-mean convention RQ1a's own regression uses.

SCOPE: per the task, this fits ONLY mechanical_share ~ log(network_size) -- no conquest, no other
covariate, no attempt to connect the result to RQ1a's own coefficients.

SAFETY: reads only already-existing files (cx_step2_registration/, the existing rq1a_cells.csv). No
training, no environment reset, no checkpoint or encoder touched, no step()/encode()/reward path
modified. Nothing beyond reading existing files and computing statistics is run.

Usage: run from cyberbattle/agents/ (where the raw data lives):
  python ../../analysis/rq1b_mech_split_scale_2026-08-08/compute_mechanical_share_scale.py
"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

RQ1A_CSV = "../../analysis/rq1a_regression_recovered_2026-08-07/rq1a_cells.csv"
CHANGE_BASE = "cx_step2_registration/eventgraph_{band}/s{seed}_{topo}/event_episode.jsonl"
BANDS = ["10-15", "30-40", "80-100"]
NBOOT = 10000
BOOT_SEED = 11
OUT_DIR = "../../analysis/rq1b_mech_split_scale_2026-08-08"


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    rq1a = pd.read_csv(RQ1A_CSV)
    cell_rows = []
    n_zero_total = {b: 0 for b in BANDS}
    n_neg_total = {b: 0 for b in BANDS}
    n_usable_total = {b: 0 for b in BANDS}

    for _, cell in rq1a.iterrows():
        band, seed, topo = cell["band"], int(cell["seed"]), int(cell["topo"])
        root_owned_static = cell["root_owned_static"]
        network_size = cell["network_size"]
        rows = load_rows(CHANGE_BASE.format(band=band, seed=seed, topo=topo))

        shares = []
        n_zero = n_neg = 0
        for r in rows:
            gross = root_owned_static - r["final_root_owned_count"]
            if gross == 0:
                n_zero += 1
            elif gross < 0:
                n_neg += 1
            else:
                mech = r["root_owned_departures"]
                shares.append(mech / gross)
        n_zero_total[band] += n_zero
        n_neg_total[band] += n_neg
        n_usable_total[band] += len(shares)

        cell_rows.append({
            "band": band, "seed": seed, "topo": topo, "network_size": network_size,
            "mechanical_share_mean": float(np.mean(shares)) if shares else None,
            "n_episodes_total": len(rows), "n_episodes_included": len(shares),
            "n_episodes_excluded_zero": n_zero, "n_episodes_excluded_negative": n_neg,
        })

    df = pd.DataFrame(cell_rows)
    df.to_csv(os.path.join(OUT_DIR, "mechanical_share_per_cell.csv"), index=False)

    print("=" * 78)
    print("EPISODE COUNTS (per band)")
    print("=" * 78)
    for b in BANDS:
        total = int(rq1a[rq1a["band"] == b].shape[0])  # placeholder, replaced below
    tot_by_band = df.groupby("band")["n_episodes_total"].sum()
    for b in BANDS:
        print(f"  band {b}: total={tot_by_band[b]}  included={n_usable_total[b]}  "
              f"excluded_zero={n_zero_total[b]}  excluded_negative={n_neg_total[b]}")
    print(f"  TOTAL across all bands: {tot_by_band.sum()} (expect 4410); "
          f"included={sum(n_usable_total.values())}")

    n_missing_cell = df["mechanical_share_mean"].isna().sum()
    print(f"\nCells with a usable mechanical_share_mean: {len(df) - n_missing_cell} / {len(df)}")
    if n_missing_cell:
        print(f"  CELLS WITH ZERO USABLE EPISODES (excluded from regression):")
        print(df[df["mechanical_share_mean"].isna()][["band", "seed", "topo"]].to_string(index=False))

    reg_df = df.dropna(subset=["mechanical_share_mean"]).copy()

    print()
    print("=" * 78)
    print(f"REGRESSION: mechanical_share_mean ~ log(network_size), n={len(reg_df)}")
    print("=" * 78)
    X = sm.add_constant(np.log(reg_df["network_size"].to_numpy()))
    y = reg_df["mechanical_share_mean"].to_numpy()
    model = sm.OLS(y, X).fit()
    print(model.summary())
    const, slope = model.params
    print(f"\n  const={const:+.4f}  log_size_coef={slope:+.4f}  (SE {model.bse[1]:.4f}, "
          f"p={model.pvalues[1]:.4g})  R^2={model.rsquared:.4f}  n={int(model.nobs)}")

    rng = np.random.default_rng(BOOT_SEED)
    n = len(reg_df)
    slopes, r2s = [], []
    logn = np.log(reg_df["network_size"].to_numpy())
    yv = reg_df["mechanical_share_mean"].to_numpy()
    for _ in range(NBOOT):
        idx = rng.integers(0, n, n)
        Xb = sm.add_constant(logn[idx])
        m = sm.OLS(yv[idx], Xb).fit()
        slopes.append(m.params[1])
        r2s.append(m.rsquared)
    ci_slope = (float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5)))
    ci_r2 = (float(np.percentile(r2s, 2.5)), float(np.percentile(r2s, 97.5)))
    print(f"\n  Bootstrap 95% CI (10,000 resamples over {n} cells):")
    print(f"    log_size_coef: [{ci_slope[0]:+.4f}, {ci_slope[1]:+.4f}]")
    print(f"    R^2          : [{ci_r2[0]:.4f}, {ci_r2[1]:.4f}]")

    print()
    print("=" * 78)
    print("CONTEXT ONLY: band-level mean mechanical_share, vs the old eyeballed 69/72/76% series")
    print("(NOT the primary finding -- the continuous regression above is)")
    print("=" * 78)
    band_means = reg_df.groupby("band")["mechanical_share_mean"].agg(["mean", "std", "count"])
    print(band_means.to_string())
    print("\n  NOTE: the old 69%/72%/76% series is NOT a 10-15/30-40/80-100 band comparison -- it is")
    print("  two bands (30-40, 80-100) under three different churn conditions (evidence_taskF3.md,")
    print("  claims_audit.md CA-1). There is no historical 10-15 figure to compare against, and the")
    print("  old series cannot be lined up one-to-one with these three band means.")

    results = {
        "n_cells": len(reg_df), "log_size_coef": slope, "log_size_coef_se": model.bse[1],
        "log_size_coef_ci_lo": ci_slope[0], "log_size_coef_ci_hi": ci_slope[1],
        "p_value": model.pvalues[1], "r_squared": model.rsquared,
        "r_squared_ci_lo": ci_r2[0], "r_squared_ci_hi": ci_r2[1],
    }
    for b in BANDS:
        results[f"band_{b}_mean"] = band_means.loc[b, "mean"]
        results[f"band_{b}_std"] = band_means.loc[b, "std"]
        results[f"band_{b}_n_cells"] = band_means.loc[b, "count"]
    pd.DataFrame([results]).to_csv(os.path.join(OUT_DIR, "mechanical_share_regression_results.csv"), index=False)
    print(f"\nWrote: {OUT_DIR}/mechanical_share_per_cell.csv ({len(df)} rows), "
          f"mechanical_share_regression_results.csv")


if __name__ == "__main__":
    main()
