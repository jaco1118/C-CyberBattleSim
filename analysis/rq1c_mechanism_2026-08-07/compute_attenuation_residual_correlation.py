"""Task RQ1C-MECHANISM STEP 1: fresh, verified measurement of whether perception-level
attenuation predicts episode-level behavioural loss, on the real 117-cell CX main-study
population.

Background: an old, never-merged patch (evidence_cards/evidence_taskCX.md, section
"ATTENUATION vs behavioural residual", 2026-08-02, recovered as prose in
analysis/recovered_stash0_cardedits_2026-08-03/stash0_evidence_card_edits.patch) reports a
correlation between a perception-side "attenuation" quantity and the same behavioural residual
FORMULA B uses (r = -0.295 / -0.418 / -0.400 across bands, all p << 1e-14). The script that
produced that number does not survive anywhere (exhaustive search, STEP 0B) -- confirmed lost
by a first-party admission already committed to the repo (commit 1d6aaab: "the CX compute
pipeline...was in ephemeral job scratch...still lost"). This script is a FRESH, independently
written computation of the same relationship, built from raw data that does survive. It is not
an attempt to reproduce or resurrect the old code.

FORMULA (verified against the patch text, STEP 1 0.1):
  evidence_cards/evidence_taskCX.md, "ATTENUATION vs behavioural residual" section:
    att_strength (per event, mean) = ||delta_h_v|| - ||delta_h_G||   (>0 = pool swallowed the change)
  ||delta_h_v|| is the drift log's own `delta_h_v_norm` column (direct).
  ||delta_h_G|| is NOT any of the other raw drift columns directly -- the patch's own ITEM B.1
  explicitly warns `change_drift_full` is a *relative*, differently-scaled quantity (corr 0.61
  with delta_h_v_norm, not usable as a literal substitute) and instructs recovering ||delta_h_G||
  via the ratio: attenuation_ratio_full = ||delta_h_v|| / ||delta_h_G||, i.e.
    ||delta_h_G|| = delta_h_v_norm / attenuation_ratio_full
  so: att_strength = delta_h_v_norm - delta_h_v_norm / attenuation_ratio_full
                    = delta_h_v_norm * (1 - 1/attenuation_ratio_full)
  This is computed PER FIRED CHANGE EVENT (change_fired == True rows only -- the drift logs also
  contain one row per routine simulation step with change_fired == False, where delta_h_v_norm
  and attenuation_ratio_full are trivially 0; including those would silently redefine "per event"
  as "per step" and dilute the mean toward zero. Verified: change_type is only ever
  membership_join/membership_leave among fired events in this population -- no "property" events
  fire in this data, consistent with attenuation_step3_logs/shortfall_*.json recording property
  events as "structurally_impossible" for this run).

DEGENERATE CASE (STEP 1 numerics convention): when attenuation_ratio_full is 0 or NaN for a fired
event (observed: always co-occurs with delta_h_v_norm == 0 -- a feature-level no-op, the changed
node's own embedding did not move at all), att_strength is UNDEFINED for that event (0/0), not 0.
Such events are dropped from that episode's mean, not floored to zero.

EPISODE AGGREGATION: att_strength per episode = mean over that episode's valid (non-degenerate)
fired events. An episode is EXCLUDED from the correlation if it has zero fired events with a
defined att_strength -- counted separately as "zero events" (no fired events logged at all, drift
log absent or no change_fired==True rows) vs "degenerate ratio" (has fired events, but every one
of them has an undefined att_strength).

DRIFT LOG SOURCE: two candidate directories exist on disk (found STEP 0B):
attenuation_drift_logs/ (earlier collection) and attenuation_step3_logs/ (later "Task-L STEP3"
collection). Checked directly (STEP 1 0.4/0.5 verification): for episodes present in BOTH, the
logged events are NOT reliably step-aligned or row-count-identical (e.g. seed42/topo1/episode0:
239 rows in attenuation_drift_logs vs 237 in attenuation_step3_logs, with the fired-event burst
at a different step number) -- they are two separate logging passes, not exact duplicates. Mixing
them within one episode's mean would risk combining events from what may functionally be two
different runs. attenuation_step3_logs/ is used as the SOLE source: it is the later, more complete
collection (matches the eventgraph_<band>/ per-step structure cx_step2_registration/ itself uses;
attenuation_drift_logs/ predates that infrastructure) and gives equal-or-better coverage in 2 of 3
bands. attenuation_drift_logs/ is not merged in, to keep every episode's mean built from one
internally-consistent logging pass.

BEHAVIOURAL RESIDUAL (verified equivalent to FORMULA B's cell-level behavioural_residual by exact
algebraic check, STEP 1 0.2/0.3 -- diffs on 6 sampled real cells were all ~1e-15, floating-point
noise, not approximation):
  residual_i = root_owned_static_cell - final_root_owned_count_i - root_owned_departures_i
  where root_owned_static_cell is read directly from the existing, committed 117-cell CSV
  (analysis/rq1a_regression_recovered_2026-08-07/rq1a_cells.csv, commit 418abc3) and
  final_root_owned_count_i / root_owned_departures_i are read per-episode from
  cx_step2_registration/eventgraph_<band>/s<seed>_<topo>/event_episode.jsonl (the same raw data
  FORMULA B and RQ1a's regression already use -- no new evaluation run).
  mean_i(residual_i) over a cell's change-arm episodes == that cell's committed behavioural_residual
  (exactly, by linearity of the mean -- this is what 0.3 verified).

JOIN KEY (verified, STEP 1 0.4): (seed, scenario_id, episode) -- identical column names AND
matching values on both sides (cx_step2_registration's event_episode.jsonl `run_id` field, e.g.
"10-15_seed42", matches the drift log's `run_id` for the same seed/scenario_id/episode exactly).

CONFOUND CHECK: partial correlation between att_strength and residual, controlling for
root_owned_departures_i (the episode's own mechanical-loss count -- the task's named
"event-severity proxy already present in the data", and literally the quantity already subtracted
out of the residual's own construction). Computed by residualizing both att_strength and residual
on root_owned_departures via OLS, then Pearson-correlating the two residual series.

SAFETY: this script only reads already-logged, already-existing files (attenuation_step3_logs/,
cx_step2_registration/, the existing rq1a_cells.csv). No training, no environment reset, no
checkpoint or encoder touched, no step()/encode()/reward path modified. Nothing beyond reading
existing files and computing statistics is run.

Usage: run from cyberbattle/agents/ (where the raw data lives):
  python ../../analysis/rq1c_mechanism_2026-08-07/compute_attenuation_residual_correlation.py
"""
import json
import os

import numpy as np
import pandas as pd

RQ1A_CSV = "../../analysis/rq1a_regression_recovered_2026-08-07/rq1a_cells.csv"
CHANGE_BASE = "cx_step2_registration/eventgraph_{band}/s{seed}_{topo}/event_episode.jsonl"
DRIFT_BASE = "attenuation_step3_logs/drift_{band}.csv"
BANDS = ["10-15", "30-40", "80-100"]
NBOOT = 10000
BOOT_SEED = 11
OUT_DIR = "../../analysis/rq1c_mechanism_2026-08-07"


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_drift_att_strength(band):
    """Return dict (seed, scenario_id, episode) -> list of valid per-event att_strength values."""
    path = DRIFT_BASE.format(band=band)
    per_ep = {}
    n_fired_total = 0
    n_valid_total = 0
    usecols = ["seed", "scenario_id", "episode", "change_fired", "delta_h_v_norm", "attenuation_ratio_full"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=500000):
        fired = chunk[chunk["change_fired"] == True]  # noqa: E712
        n_fired_total += len(fired)
        valid = fired[(fired["attenuation_ratio_full"].notna()) & (fired["attenuation_ratio_full"] > 0)]
        n_valid_total += len(valid)
        att = valid["delta_h_v_norm"] * (1.0 - 1.0 / valid["attenuation_ratio_full"])
        for seed, scen, ep, a in zip(valid["seed"], valid["scenario_id"], valid["episode"], att):
            key = (int(seed), int(scen), int(ep))
            per_ep.setdefault(key, []).append(float(a))
        # also record which (seed,scenario,episode) had at least one FIRED event (valid or not),
        # so we can distinguish "zero events" from "degenerate ratio" below.
        for seed, scen, ep in zip(fired["seed"], fired["scenario_id"], fired["episode"]):
            key = (int(seed), int(scen), int(ep))
            per_ep.setdefault(key, per_ep.get(key, []))  # ensure key exists even if empty list
    return per_ep, n_fired_total, n_valid_total


def pearson_ci(x, y, n_boot=NBOOT, seed=BOOT_SEED):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    rs = np.array([np.corrcoef(x[i], y[i])[0, 1] for i in idx])
    ci = (float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5)))
    # two-sided p-value via Fisher z-transform (standard, reference only per task instructions)
    if abs(r) >= 1.0:
        p = 0.0
    else:
        z = np.arctanh(r) * np.sqrt(n - 3)
        from scipy.stats import norm
        p = float(2 * (1 - norm.cdf(abs(z))))
    return r, ci, p, rs


def partial_corr(x, y, z, n_boot=NBOOT, seed=BOOT_SEED):
    """Partial correlation of x,y controlling for z: residualize x and y on z via OLS, then
    Pearson-correlate the residuals. Bootstrap CI resamples episodes and redoes the whole
    residualize-then-correlate procedure each time (not just resampling residuals)."""
    def resid_corr(xx, yy, zz):
        Z = np.column_stack([np.ones_like(zz), zz])
        bx, *_ = np.linalg.lstsq(Z, xx, rcond=None)
        by, *_ = np.linalg.lstsq(Z, yy, rcond=None)
        rx = xx - Z @ bx
        ry = yy - Z @ by
        return float(np.corrcoef(rx, ry)[0, 1])

    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    r = resid_corr(x, y, z)
    n = len(x)
    rng = np.random.default_rng(seed + 1)
    idx = rng.integers(0, n, size=(n_boot, n))
    rs = np.array([resid_corr(x[i], y[i], z[i]) for i in idx])
    ci = (float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5)))
    return r, ci


def main():
    rq1a = pd.read_csv(RQ1A_CSV)
    rows_out = []
    coverage_report = []

    for band in BANDS:
        band_cells = rq1a[rq1a["band"] == band]
        drift_per_ep, n_fired_total, n_valid_total = load_drift_att_strength(band)

        n_total_eps = 0
        n_zero_events = 0
        n_degenerate = 0
        n_included = 0

        for _, cell in band_cells.iterrows():
            seed, topo = int(cell["seed"]), int(cell["topo"])
            root_owned_static = cell["root_owned_static"]
            change_path = CHANGE_BASE.format(band=band, seed=seed, topo=topo)
            for r in load_rows(change_path):
                n_total_eps += 1
                key = (seed, topo, r["episode"])
                vals = drift_per_ep.get(key)
                if vals is None:
                    n_zero_events += 1
                    continue
                if len(vals) == 0:
                    n_degenerate += 1
                    continue
                att = float(np.mean(vals))
                residual = root_owned_static - r["final_root_owned_count"] - r["root_owned_departures"]
                rows_out.append({
                    "band": band, "seed": seed, "topo": topo, "episode": r["episode"],
                    "att_strength": att, "n_valid_events": len(vals),
                    "behavioural_residual_episode": residual,
                    "root_owned_departures": r["root_owned_departures"],
                    "final_root_owned_count": r["final_root_owned_count"],
                    "root_owned_static_cell": root_owned_static,
                })
                n_included += 1

        coverage_report.append({
            "band": band, "n_total_episodes": n_total_eps, "n_zero_events": n_zero_events,
            "n_degenerate_ratio": n_degenerate, "n_included": n_included,
            "n_fired_events_in_drift_log": n_fired_total, "n_valid_att_strength_events": n_valid_total,
        })

    df = pd.DataFrame(rows_out)
    df.to_csv(os.path.join(OUT_DIR, "attenuation_residual_episodes.csv"), index=False)

    cov_df = pd.DataFrame(coverage_report)
    print("=" * 78)
    print("EPISODE COVERAGE (per band)")
    print("=" * 78)
    print(cov_df.to_string(index=False))
    print(f"\nTotal episodes across all bands: {cov_df['n_total_episodes'].sum()} "
          f"(expect 4410); included in correlation: {cov_df['n_included'].sum()}")

    results = []
    print()
    print("=" * 78)
    print("RAW CORRELATION: att_strength vs behavioural_residual_episode, per band")
    print("=" * 78)
    for band in BANDS:
        sub = df[df["band"] == band]
        r, ci, p, _ = pearson_ci(sub["att_strength"], sub["behavioural_residual_episode"])
        print(f"  band {band}: n={len(sub)}  r={r:+.4f}  bootstrap 95% CI=[{ci[0]:+.4f}, {ci[1]:+.4f}]  p={p:.3e}")
        results.append({"band": band, "n": len(sub), "r_raw": r, "ci_raw_lo": ci[0], "ci_raw_hi": ci[1], "p_raw": p})

    print()
    print("=" * 78)
    print("CONFOUND CHECK: partial correlation controlling for root_owned_departures, per band")
    print("=" * 78)
    for i, band in enumerate(BANDS):
        sub = df[df["band"] == band]
        rp, cip = partial_corr(sub["att_strength"], sub["behavioural_residual_episode"], sub["root_owned_departures"])
        print(f"  band {band}: n={len(sub)}  partial_r={rp:+.4f}  bootstrap 95% CI=[{cip[0]:+.4f}, {cip[1]:+.4f}]")
        results[i]["r_partial"] = rp
        results[i]["ci_partial_lo"] = cip[0]
        results[i]["ci_partial_hi"] = cip[1]

    print()
    print("=" * 78)
    print("CROSS-CHECK ONLY (context, not validation) vs the old, unrecoverable analysis's")
    print("reported r = -0.295 / -0.418 / -0.400")
    print("=" * 78)
    old = {"10-15": -0.295, "30-40": -0.418, "80-100": -0.400}
    for res in results:
        print(f"  band {res['band']}: fresh r={res['r_raw']:+.4f}  old (unverifiable) r={old[res['band']]:+.4f}  "
              f"same sign: {(res['r_raw'] < 0) == (old[res['band']] < 0)}")

    pd.DataFrame(results).to_csv(os.path.join(OUT_DIR, "attenuation_residual_correlation_results.csv"), index=False)
    cov_df.to_csv(os.path.join(OUT_DIR, "attenuation_residual_coverage.csv"), index=False)
    print(f"\nWrote: {OUT_DIR}/attenuation_residual_episodes.csv "
          f"({len(df)} rows), attenuation_residual_correlation_results.csv, "
          f"attenuation_residual_coverage.csv")


if __name__ == "__main__":
    main()
