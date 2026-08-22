"""Task SNR-TRACE: identify the exact statistic behind the thesis's published
SNR (membership_leave) figures, 0.492 [0.385, 0.632] LEVEL and -0.804 [-0.842, -0.765] SLOPE
(evidence_taskT.md:294-298, sourced to compute_attenuation_analysis.py per its own
"Files written or changed" section -- no dedicated Task T script exists).

STEP 1 finding: the production script computes SNR in TWO aggregation levels, not one:
  1. Per EVENT: snr = change_drift_full/agent_drift_full, NaN if agent_drift_full < 1e-9
     (compute_attenuation_analysis.py:572-576, ZERO_NOISE_FLOOR_THRESHOLD=1e-9).
  2. Per EPISODE: group by (seed, scenario_id, episode, change_type), agg_dict['snr']='median'
     (compute_attenuation_analysis.py:582-605) -- i.e. each episode's own MEDIAN snr across its
     own leave events is the base analysis unit, computed separately per band then concatenated
     across all three bands into one pooled `all_episode_df` (compute_attenuation_analysis.py
     line ~1364, `pd.concat(all_episode_dfs)`).
  3. LEVEL (line ~1064): near_100 = ct_df[(n_discovered>=80) & (n_discovered<=120)] -- a RANGE
     filter on the per-episode median n_discovered, pooled across bands (not restricted to band
     80-100 alone). mean_snr, lo, hi = bootstrap_series_ci(near_100['snr']) -- bootstrap_ci()
     (cyberbattle/utils/math_utils.py:48-54) reports np.mean(data), i.e. LEVEL is a MEAN of
     per-episode median-SNR values in that window, not a median-of-medians and not a per-event
     statistic at all.
  4. SLOPE (line ~1076-1078): x=ct_df['n_discovered'], y=ct_df['snr'] fed straight into
     fit_loglog_slope(x,y) (line 723) -- an UNBINNED log-log OLS on the per-episode median-SNR
     vs per-episode median n_discovered, pooled across all bands and all valid-SNR episodes.
     No integer-bin aggregation of any kind (neither Task-T's median-per-bin nor Task-W's
     mean-per-bin convention -- both are different schemes from what this code actually does).

This script verifies that finding two ways:
  (A) recomputes LEVEL/SLOPE directly from the ARCHIVED per-episode CSV written by the actual
      2026-07-26 gate run (attenuation_gate_archive/2026-07-26_trpo_5seed_gate/
      attenuation_analysis_output/attenuation_episode_aggregates.csv) using the documented
      formula, and checks it against the ARCHIVED gate_summary.txt's own printed line (the
      literal, un-reconstructed original output);
  (B) computes the four STEP-2-prescribed candidate statistics (median of per-event ratio; mean
      of per-event ratio; ratio of pooled means; ratio-per-Task-W-bin-then-median-across-bins)
      on the matching population, from the already-committed per-event snr_abs_rows.csv
      (analysis/snr_abs_2026-08-22/, 18,658-row acting-steps population, pooled across bands),
      to confirm they do NOT reproduce 0.492/-0.804 -- exactly four, no more, per the task's own
      explicit instruction not to search for a fit.

No new evaluation or training. No source file touched. No metric definition changed. No epsilon
floor substituted for any zero denominator. Reads only already-committed/archived CSVs.
"""
import numpy as np
import pandas as pd

ARCHIVE_EPISODE_CSV = (
    "attenuation_gate_archive/2026-07-26_trpo_5seed_gate/attenuation_analysis_output/"
    "attenuation_episode_aggregates.csv"
)
ARCHIVE_GATE_SUMMARY = (
    "attenuation_gate_archive/2026-07-26_trpo_5seed_gate/attenuation_analysis_output/"
    "gate_summary.txt"
)
SNR_ABS_ROWS_CSV = "analysis/snr_abs_2026-08-22/snr_abs_rows.csv"

REPORTED_LEVEL = (0.492, 0.385, 0.632)
REPORTED_SLOPE = (-0.804, -0.842, -0.765)


def fit_loglog_slope(x, y, n_bootstrap=2000):
    """Copied verbatim from compute_attenuation_analysis.py:723-742 (not reimplemented
    independently) so part (A) below is a faithful replay of the production code, not a
    reconstruction that might silently diverge from it."""
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return float('nan'), float('nan'), float('nan'), len(x)
    log_x, log_y = np.log(x), np.log(y)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    rng = np.random.default_rng(42)
    slopes = []
    n = len(x)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            s, _ = np.polyfit(log_x[idx], log_y[idx], 1)
            slopes.append(s)
        except np.linalg.LinAlgError:
            continue
    lower, upper = np.percentile(slopes, [2.5, 97.5]) if slopes else (float('nan'), float('nan'))
    return float(slope), float(lower), float(upper), n


def part_a_archived_episode_replay():
    print("=" * 100)
    print("PART A: recompute LEVEL/SLOPE from the ARCHIVED per-episode CSV of the actual "
          "2026-07-26 gate run")
    print("=" * 100)
    df = pd.read_csv(ARCHIVE_EPISODE_CSV)
    ct_df = df[df['change_type'] == 'membership_leave']
    print(f"membership_leave episode-rows (pooled across bands): {len(ct_df)}  "
          f"(archived gate_summary.txt: n episodes = 4641)")
    print(f"valid (non-NaN) snr episode-rows: {ct_df['snr'].notna().sum()}")

    near_100 = ct_df[(ct_df['n_discovered'] >= 80) & (ct_df['n_discovered'] <= 120)]
    vals = near_100['snr'].dropna().values
    mean_snr = float(np.mean(vals))
    rng = np.random.default_rng(0)  # bootstrap_ci() itself is unseeded; CI will vary slightly
    boots = np.array([np.mean(rng.choice(vals, replace=True, size=len(vals))) for _ in range(10000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\nLEVEL: mean of per-episode median-snr, n_discovered in [80,120], n={len(vals)}: "
          f"{mean_snr:.3f} [{lo:.3f}, {hi:.3f}]")
    print(f"       reported: {REPORTED_LEVEL[0]:.3f} [{REPORTED_LEVEL[1]:.3f}, {REPORTED_LEVEL[2]:.3f}]")
    level_match = abs(mean_snr - REPORTED_LEVEL[0]) < 0.001
    print(f"       point-estimate exact match (< 0.001): {level_match}")

    x = ct_df['n_discovered'].values.astype(float)
    y = ct_df['snr'].values.astype(float)
    slope, slo, shi, n_slope = fit_loglog_slope(x, y)
    print(f"\nSLOPE: unbinned log-log OLS of per-episode median-snr vs per-episode median "
          f"n_discovered, all valid rows, n={n_slope}: {slope:.3f} [{slo:.3f}, {shi:.3f}]")
    print(f"       reported: {REPORTED_SLOPE[0]:.3f} [{REPORTED_SLOPE[1]:.3f}, {REPORTED_SLOPE[2]:.3f}]")
    slope_match = abs(slope - REPORTED_SLOPE[0]) < 0.001
    ci_match = abs(slo - REPORTED_SLOPE[1]) < 0.001 and abs(shi - REPORTED_SLOPE[2]) < 0.001
    print(f"       point-estimate exact match (< 0.001): {slope_match}; "
          f"CI exact match (fit_loglog_slope's rng is seed=42, deterministic): {ci_match}")

    with open(ARCHIVE_GATE_SUMMARY) as f:
        gate_text = f.read()
    archived_level_line = [l for l in gate_text.splitlines() if l.strip().startswith("LEVEL:")][0]
    archived_slope_line = [l for l in gate_text.splitlines() if l.strip().startswith("SLOPE:")][0]
    print(f"\nArchived gate_summary.txt's own printed lines (literal original output, not "
          f"reconstructed):")
    print(f"  {archived_level_line.strip()}")
    print(f"  {archived_slope_line.strip()}")

    return dict(level_mean=mean_snr, level_ci=(lo, hi), level_match=level_match,
                slope=slope, slope_ci=(slo, shi), slope_match=slope_match, ci_match=ci_match)


def part_b_four_prescribed_candidates():
    print()
    print("=" * 100)
    print("PART B: the four STEP-2-prescribed candidate statistics (exactly four, no more), "
          "tested against 0.492")
    print("=" * 100)
    df = pd.read_csv(SNR_ABS_ROWS_CSV)
    df = df[df['change_type'] == 'membership_leave'].copy()
    near = df[(df['n_discovered'] >= 80) & (df['n_discovered'] <= 120)]
    print(f"per-event rows (pooled across bands, acting-steps population), membership_leave, "
          f"n_discovered in [80,120]: {len(near)}")

    a = float(near['snr_current'].median())
    b = float(near['snr_current'].mean())
    c = float(near['change_drift_full'].mean() / near['agent_drift_full'].mean())

    df['bin'] = (df['n_discovered'] // 10) * 10
    bin_means = df.groupby('bin')['snr_current'].mean()
    d = float(bin_means.median())
    d_bin90 = float(bin_means.get(90, float('nan')))

    print(f"\n(a) median of per-event ratio, n in [80,120]:            {a:.4f}")
    print(f"(b) mean of per-event ratio, n in [80,120]:               {b:.4f}  "
          f"(dominated by extreme low-n_discovered outliers; code comment notes observed "
          f"values up to 1.86e9 -- exactly why the real code aggregates per-episode MEDIAN "
          f"first, before ever taking a mean)")
    print(f"(c) ratio of pooled means (mean(change_drift)/mean(agent_drift)), n in [80,120]: "
          f"{c:.4f}")
    print(f"(d) Task-W mean-per-integer-n_discovered-bin, median across ALL bins: {d:.4f}  "
          f"(bin [90,100) alone: {d_bin90:.4f})")
    print(f"\nreported LEVEL: {REPORTED_LEVEL[0]:.3f}")
    for label, val in [("a", a), ("b", b), ("c", c), ("d (median-across-bins)", d),
                        ("d (bin [90,100) only)", d_bin90)]:
        print(f"  {label}: {'MATCH' if abs(val - REPORTED_LEVEL[0]) < 0.02 else 'no match'} "
              f"(|diff|={abs(val - REPORTED_LEVEL[0]):.4f})")

    return dict(a=a, b=b, c=c, d_median_across_bins=d, d_bin90=d_bin90)


def main():
    a_result = part_a_archived_episode_replay()
    b_result = part_b_four_prescribed_candidates()

    print()
    print("=" * 100)
    if a_result['level_match'] and a_result['slope_match']:
        print("0.492 AND -0.804 IDENTIFIED AS: the MEAN (bootstrap-CI'd) of per-episode "
              "MEDIAN snr values -- snr computed per membership_leave event as "
              "change_drift_full/agent_drift_full (NaN below ZERO_NOISE_FLOOR_THRESHOLD=1e-9), "
              "median-aggregated to one value per (seed, scenario_id, episode) episode within "
              "each band, then pooled (concatenated, not re-aggregated) across all three bands. "
              "LEVEL = mean of these episode-medians restricted to episodes whose own median "
              "n_discovered falls in the RANGE [80,120]. SLOPE = an UNBINNED log-log OLS "
              "(np.polyfit on log(n_discovered), log(episode-median-snr), bootstrap CI with "
              "fixed rng seed=42) over every episode with a valid (non-NaN) snr, pooled across "
              "all three bands. This is (compute_attenuation_analysis.py's own "
              "compute_episode_aggregates() + the LEVEL/SLOPE emission block near line 1063), "
              "NOT any of the four candidates named in STEP 2 -- none of which used per-episode "
              "median-first aggregation.")
    else:
        print("0.492 AND -0.804 NOT IDENTIFIABLE FROM THE SURVIVING RECORD")
    print("=" * 100)


if __name__ == "__main__":
    main()
