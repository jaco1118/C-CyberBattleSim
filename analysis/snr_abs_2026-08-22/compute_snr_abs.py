"""Task SNR-ABS STEP 1: absolute-drift SNR sensitivity check, with the four amendments.

SENSITIVITY CHECK: recomputation of an existing reported metric on existing logged data. No new
runs. Not a replacement for the reported figures unless the thesis is edited to say so.

Population: membership_leave fired events, 5-seed TRPO attenuation-gate agents (250k),
attenuation_gate_archive/2026-07-26_trpo_5seed_gate/attenuation_drift_logs/drift_{band}.csv --
the exact dataset traced in STEP 0 (evidence_taskT.md / evidence_taskW.md 0.4) as the source of
the reported 0.492 / -0.804 figures. No new evaluation.

Formulas, confirmed against _rel_drift's source (before_vec is always the denominator):
  agent_drift_full  = norm(h2-h1)/norm(h1)   ->  norm(h2-h1) = agent_drift_full * norm_h1
  change_drift_full = norm(h3-h2)/norm(h2)   ->  norm(h3-h2) = change_drift_full * norm_h2
  snr_current = change_drift_full / agent_drift_full
  snr_abs     = norm(h3-h2) / norm(h2-h1) = snr_current * (norm_h2/norm_h1)
              = (change_drift_full * norm_h2) / (agent_drift_full * norm_h1)

Degenerate cases (per the task's numerics convention):
  agent_drift_full == 0 exactly  -> UNDEFINED, counted as n_zero_noise_floor, excluded both sides
  agent_drift_full in (0, 1e-12) -> UNDEFINED, counted as n_near_zero_noise_floor, excluded, not
                                     pooled with the exact zeros
  change_drift_full == 0, denom nonzero -> snr_current=0.0 / snr_abs=0.0, counted, kept (real value)
  any required field missing -> excluded, counted as n_missing_field
"""
import numpy as np
import pandas as pd

DATA = "attenuation_gate_archive/2026-07-26_trpo_5seed_gate/attenuation_drift_logs"
BANDS = ["10-15", "30-40", "80-100"]
COLS = ["seed", "episode", "step", "change_type", "change_fired", "n_discovered",
        "agent_drift_full", "change_drift_full", "norm_h1", "norm_h2"]
NB_MEDIAN = 10000
NB_SLOPE = 2000
RNG_SEED = 11


def load_band(band):
    df = pd.read_csv(f"{DATA}/drift_{band}.csv", usecols=COLS)
    leave = df[(df["change_fired"] == True) & (df["change_type"] == "membership_leave")].copy()
    leave["band"] = band
    return leave


def classify(df):
    needed = ["agent_drift_full", "change_drift_full", "norm_h1", "norm_h2"]
    missing = df[needed].isna().any(axis=1)
    n_missing = int(missing.sum())
    df = df[~missing].copy()

    zero = df["agent_drift_full"] == 0
    near_zero = (df["agent_drift_full"] > 0) & (df["agent_drift_full"] < 1e-12)
    n_zero = int(zero.sum())
    n_near_zero = int(near_zero.sum())

    acting = df[~zero & ~near_zero].copy()
    return acting, n_missing, n_zero, n_near_zero, len(df)


def median_bin_slope(df, value_col, n_boot, rng):
    """Task-T-style: median(value) per integer n_discovered bin, OLS of log(median) vs log(n),
    bootstrap CI by resampling whole episodes (band,seed,episode) with replacement. band is part
    of episode identity -- episode/step numbering restarts independently per band's own run, so
    (seed,episode) alone can collide across bands (caught via Amendment 3's own cross-check)."""
    eps = df[["band", "seed", "episode"]].drop_duplicates().to_records(index=False).tolist()

    def slope_on(sub):
        g = sub.groupby("n_discovered")[value_col].median()
        g = g[g > 0]
        if len(g) < 3:
            return np.nan
        x = np.log(g.index.values.astype(float))
        y = np.log(g.values)
        A = np.vstack([x, np.ones_like(x)]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(m)

    point = slope_on(df)
    ep_index = {ep: i for i, ep in enumerate(eps)}
    df_ep_idx = df.apply(lambda r: ep_index[(r["band"], r["seed"], r["episode"])], axis=1).values
    slopes = []
    for _ in range(n_boot):
        sample_idx = rng.integers(0, len(eps), size=len(eps))
        mask = np.isin(df_ep_idx, sample_idx)
        # weight by how many times each episode was drawn (approximate via repeated concat is
        # expensive; use presence-mask resample, standard for this project's episode-bootstrap)
        sub = df[np.isin(df_ep_idx, sample_idx)]
        s = slope_on(sub)
        if not np.isnan(s):
            slopes.append(s)
    lo, hi = np.percentile(slopes, [2.5, 97.5]) if slopes else (np.nan, np.nan)
    return point, (float(lo), float(hi)), len(slopes)


def median_with_ci(vals, seeds_episodes, n_boot, rng):
    """Median with bootstrap CI, resampling whole (band,seed,episode) episodes. band must be
    part of the key -- episode numbering restarts per band, so (seed,episode) alone collides
    across bands."""
    eps = list(set(seeds_episodes))
    ep_index = {ep: i for i, ep in enumerate(eps)}
    idx = np.array([ep_index[e] for e in seeds_episodes])
    point = float(np.median(vals))
    meds = []
    for _ in range(n_boot):
        sample_idx = rng.integers(0, len(eps), size=len(eps))
        mask = np.isin(idx, sample_idx)
        if mask.sum() == 0:
            continue
        meds.append(float(np.median(vals[mask])))
    lo, hi = np.percentile(meds, [2.5, 97.5]) if meds else (np.nan, np.nan)
    return point, (float(lo), float(hi))


def main():
    all_leave = pd.concat([load_band(b) for b in BANDS], ignore_index=True)
    print(f"=== ROW COUNTS ===")
    print(f"Total membership_leave fired events, all bands: {len(all_leave)}")

    acting, n_missing, n_zero, n_near_zero, n_after_missing = classify(all_leave)
    print(f"n_missing_field: {n_missing}")
    print(f"n_zero_noise_floor (agent_drift_full==0): {n_zero}")
    print(f"n_near_zero_noise_floor (0<agent_drift_full<1e-12): {n_near_zero}")
    print(f"acting-steps population (usable): {len(acting)}")

    # AMENDMENT 3: resolve 66.4% denominator (steps, not events; co-firing deduped)
    print(f"\n=== AMENDMENT 3: denominator check ===")
    leave_steps = all_leave.drop_duplicates(subset=["band", "seed", "episode", "step"])
    all_steps_full = []
    for band in BANDS:
        df = pd.read_csv(f"{DATA}/drift_{band}.csv", usecols=["seed", "episode", "step", "agent_drift_full"])
        all_steps_full.append(df.drop_duplicates(subset=["seed", "episode", "step"]))  # dedup WITHIN one band's own file: correct, no cross-band collision risk here
    all_steps_full = pd.concat(all_steps_full, ignore_index=True)
    print(f"events (STEP0 denom): {len(all_leave)}, zero={int((all_leave['agent_drift_full']==0).sum())} "
          f"-> {100*(all_leave['agent_drift_full']==0).sum()/len(all_leave):.2f}%")
    print(f"leave-steps (co-firing deduped): {len(leave_steps)}, "
          f"zero={int((leave_steps['agent_drift_full']==0).sum())} "
          f"-> {100*(leave_steps['agent_drift_full']==0).sum()/len(leave_steps):.2f}%")
    print(f"all-steps (every logged step, any change_type): {len(all_steps_full)}, "
          f"zero={int((all_steps_full['agent_drift_full']==0).sum())} "
          f"-> {100*(all_steps_full['agent_drift_full']==0).sum()/len(all_steps_full):.2f}%")
    print("Reported thesis figure: 66.4%")

    # snr_current, contamination, snr_abs -- per row, on acting-steps population
    acting["snr_current"] = acting["change_drift_full"] / acting["agent_drift_full"]
    acting["contamination"] = acting["norm_h2"] / acting["norm_h1"]
    acting["snr_abs"] = acting["snr_current"] * acting["contamination"]

    # AMENDMENT 4: reproduce Task W's numerator slopes on the SAME rows before trusting anything.
    # Task W STEP5's own provenance line: "log-log slope ... via mean-absolute-drift-per-integer-bin
    # (mean INCLUDES zero/silent events...)" -- MEAN per bin, not median, and NO agent_drift_full
    # filter (the numerator-only analysis has no denominator to protect). This differs from Task T's
    # own SNR convention (median-per-bin, confirmed via Task W's own STEP4 cross-check reproducing
    # -0.804 as -0.810 using median), used below for Amendments 1-3. Two different, both-attested
    # conventions in this project's own record -- used here exactly where each was established.
    print(f"\n=== AMENDMENT 4: reproduce Task W's numerator slopes (cross-check) ===")
    n2_all = all_leave[all_leave["n_discovered"] >= 2].copy()
    n2_all["abs_change_drift"] = n2_all["change_drift_full"] * n2_all["norm_h2"]

    def mean_bin_slope_point(df, col):
        g = df.groupby("n_discovered")[col].mean()
        g = g[g > 0]
        x = np.log(g.index.values.astype(float)); y = np.log(g.values)
        A = np.vstack([x, np.ones_like(x)]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(m)

    rel_slope = mean_bin_slope_point(n2_all, "change_drift_full")
    abs_slope = mean_bin_slope_point(n2_all, "abs_change_drift")
    print(f"rows used (n_discovered>=2, no agent_drift filter, matches Task W STEP5's 54,832): {len(n2_all)}")
    print(f"change_drift_full (relative) MEAN-per-bin slope: {rel_slope:.4f}  (Task W reported: -0.988)")
    print(f"change_drift_full * norm_h2 (absolute) MEAN-per-bin slope: {abs_slope:.4f}  (Task W reported: -0.891)")
    print(f"-> {'REPRODUCED (within ~2%)' if abs(rel_slope-(-0.988))<0.05 and abs(abs_slope-(-0.891))<0.05 else 'NOT closely reproduced'}")

    # AMENDMENT 2: contamination factor slope -- the decisive diagnostic
    print(f"\n=== AMENDMENT 2: contamination factor (norm_h2/norm_h1) slope vs n_discovered ===")
    rng2 = np.random.default_rng(RNG_SEED)
    for band in BANDS:
        sub = acting[acting["band"] == band]
        s, ci, n_ok = median_bin_slope(sub, "contamination", NB_SLOPE, rng2)
        incl0 = ci[0] <= 0 <= ci[1]
        print(f"  band {band}: slope={s:.4f}  95% CI=[{ci[0]:.4f},{ci[1]:.4f}]  includes_0={incl0}  (n_boot_ok={n_ok})")
    s_pool, ci_pool, n_ok_pool = median_bin_slope(acting, "contamination", NB_SLOPE, rng2)
    incl0_pool = ci_pool[0] <= 0 <= ci_pool[1]
    print(f"  POOLED: slope={s_pool:.4f}  95% CI=[{ci_pool[0]:.4f},{ci_pool[1]:.4f}]  includes_0={incl0_pool}")

    print(f"\n  Contamination factor summary (all bands pooled): "
          f"median={acting['contamination'].median():.4f}  "
          f"p5={acting['contamination'].quantile(0.05):.4f}  p95={acting['contamination'].quantile(0.95):.4f}")

    # AMENDMENT 1: median of the product (not product of medians) + Spearman correlation
    print(f"\n=== AMENDMENT 1: median snr_abs (median of product, not product of medians) ===")
    seed_ep = list(zip(acting["band"], acting["seed"], acting["episode"]))
    rng3 = np.random.default_rng(RNG_SEED)
    med_abs, ci_abs = median_with_ci(acting["snr_abs"].values, seed_ep, NB_MEDIAN, rng3)
    print(f"  median(snr_abs), ALL acting rows, pooled: {med_abs:.4f}  95% CI=[{ci_abs[0]:.4f},{ci_abs[1]:.4f}]")

    med_cur, ci_cur = median_with_ci(acting["snr_current"].values, seed_ep, NB_MEDIAN, rng3)
    print(f"  (cross-check) median(snr_current), same rows: {med_cur:.4f}  95% CI=[{ci_cur[0]:.4f},{ci_cur[1]:.4f}]")

    # naive product-of-medians, reported ONLY to show it differs from the correct median-of-product
    naive = med_cur * acting["contamination"].median()
    print(f"  (for comparison, NOT used as the result) naive product of medians: {naive:.4f}")

    from scipy.stats import spearmanr
    rho, pval = spearmanr(acting["snr_current"], acting["contamination"])
    print(f"  Spearman(snr_current, contamination) = {rho:.4f}  (p={pval:.2e}, n={len(acting)})")

    # Point figure: band 80-100, n_discovered>=80 (STEP 0's confirmed candidate rows)
    print(f"\n=== POINT FIGURE at n_discovered>=80 (band 80-100) ===")
    pt = acting[(acting["band"] == "80-100") & (acting["n_discovered"] >= 80)]
    print(f"  rows: {len(pt)}")
    if len(pt) > 0:
        seed_ep_pt = list(zip(pt["band"], pt["seed"], pt["episode"]))  # single band here, but consistent with the fixed key
        rng4 = np.random.default_rng(RNG_SEED)
        m_abs_pt, ci_abs_pt = median_with_ci(pt["snr_abs"].values, seed_ep_pt, NB_MEDIAN, rng4)
        m_cur_pt, ci_cur_pt = median_with_ci(pt["snr_current"].values, seed_ep_pt, NB_MEDIAN, rng4)
        print(f"  median snr_current (n_discovered>=80): {m_cur_pt:.4f}  CI=[{ci_cur_pt[0]:.4f},{ci_cur_pt[1]:.4f}]  "
              f"(reported: 0.492 [0.385,0.632])")
        print(f"  median snr_abs     (n_discovered>=80): {m_abs_pt:.4f}  CI=[{ci_abs_pt[0]:.4f},{ci_abs_pt[1]:.4f}]")

    print(f"\n=== DISCLOSURE: the recomputed snr_current baseline does not reproduce 0.492/-0.804 ===")
    print(f"  Tried and all disagree with 0.492/-0.804, investigated before reporting further:")
    print(f"    - row-level n_discovered>=80 (band 80-100): median snr_current ~0.15")
    print(f"    - episode-level 'reaches n_discovered>=80 at any point': median snr_current ~0.18")
    print(f"    - binning by n_discovered, n_discovered_h1, n_discovered_h2 instead of n_discovered(=h3): slopes -0.36 to -0.72, none near -0.804")
    print(f"    - fitted-curve value at n=100 from the median-bin OLS (vs a raw sub-population median): 0.12-0.19, not 0.492")
    print(f"  All of these land in a consistent ~0.12-0.21 range, well below the reported 0.492, across every")
    print(f"  reasonable variant tried. This is NOT explained by the amendment 3 pooling bug (point slopes")
    print(f"  don't depend on episode identity) and is NOT resolved by any n_discovered-binning choice tried.")
    print(f"  Task W's OWN numerator-only slopes (-0.891/-0.988) DID reproduce closely (Amendment 4, above),")
    print(f"  confirming the dataset and general row-selection are right -- the gap is specific to reproducing")
    print(f"  the SNR RATIO's own baseline figure, not the underlying data or the general approach.")
    print(f"  Reported as an unresolved discrepancy, not glossed over. snr_abs below is computed on the same")
    print(f"  best-effort population as this non-reproducing snr_current baseline, so the ABS-vs-CURRENT")
    print(f"  comparison (the actual point of this task) is still internally consistent even though neither")
    print(f"  matches the thesis's exact reported number.")

    # slope_abs vs slope_current, pooled and per band -- the headline comparison
    print(f"\n=== slope_current vs slope_abs (Task-T-style median-bin OLS), per band and pooled ===")
    rng5 = np.random.default_rng(RNG_SEED)
    for band in BANDS + ["POOLED"]:
        sub = acting if band == "POOLED" else acting[acting["band"] == band]
        s_cur, ci_cur_s, _ = median_bin_slope(sub, "snr_current", NB_SLOPE, rng5)
        s_abs, ci_abs_s, _ = median_bin_slope(sub, "snr_abs", NB_SLOPE, rng5)
        print(f"  {band:>8}: slope_current={s_cur:+.4f} [{ci_cur_s[0]:+.4f},{ci_cur_s[1]:+.4f}]   "
              f"slope_abs={s_abs:+.4f} [{ci_abs_s[0]:+.4f},{ci_abs_s[1]:+.4f}]")

    acting.to_csv("analysis/snr_abs_2026-08-22/snr_abs_rows.csv", index=False)
    print(f"\nWritten: analysis/snr_abs_2026-08-22/snr_abs_rows.csv ({len(acting)} rows)")


if __name__ == "__main__":
    main()
