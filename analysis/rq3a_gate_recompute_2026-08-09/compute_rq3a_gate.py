"""Task RQ3A-RECOMPUTE STEP 1: rebuild the discovery-gate point prediction from the CX
diagnostic-condition data (cyberbattle/agents/cx_step2_registration/).

Background: TASK RQ2A-SOURCE-CHECK found two membership response-rate triples in the project's
written record (0.196/0.153/0.207 and 0.198/0.154/0.207) with no surviving computation anywhere
in the repo, and no RQ3(a) evidence card. This script is a FRESH computation, not an attempt to
reproduce either historical triple -- the definitions below were committed at the STEP 0 gate
BEFORE this script was run, specifically to avoid selecting whichever definition happens to match
a historical number (see TASK RQ3A-RECOMPUTE STEP 0 report for the full justification of each
choice against how this project already defines these quantities elsewhere).

DEFINITIONS (committed at STEP 0, not adjusted after seeing results):

(a) Membership response rate, per band: reuses compute_attenuation_analysis.py's own
    compute_response_rates() pipeline exactly --
      - load_and_filter() Stage 1: drop rows where touched_node_visible == False
        (compute_attenuation_analysis.py:504-506)
      - compute_response_rates()'s own filter: keep event_phase == 'immediate' only
        (compute_attenuation_analysis.py:649-653)
      - drop rows where change_drift_full is NaN, negative, or non-finite
        (compute_attenuation_analysis.py:661-670)
      - change_type == 'membership_leave' only (NOT membership_join -- matches this project's
        established ~43,000-event headline response set, evidence_taskQ.md Q1: "{property,
        membership_leave} x relevant x visible x immediate/attributed"; membership_join is
        explicitly excluded there, and the pipeline's own call site invokes compute_response_rates
        with change_type='membership_leave' specifically, compute_attenuation_analysis.py:1106-1114)
      - response_rate = count(change_drift_full > 0.0) / count(all rows surviving the above)
        i.e. tau=0.0, matching the guard_report key 'full_overall_response_rate_tau0' and the
        printed line "OVERALL response_rate (tau=0.0, all n_discovered pooled)"
        (compute_attenuation_analysis.py:677-679, 1131)

(b) Discovered share of the network, per band: mean(n_discovered / n_scenario) computed over the
    IDENTICAL filtered event population as (a) -- same rows, same denominator set. This is the
    only choice that makes the point-prediction comparison ("response rate should equal discovered
    share") apples-to-apples: both numbers come from the same events, not mismatched populations.
    n_discovered is already treated as a per-event (not per-episode) quantity elsewhere in this
    pipeline (compute_response_rates bins on raw per-event n_discovered via pd.cut, line 682).

(c) Property-change response rate, per band: identical pipeline to (a), with change_type=='property'
    substituted for 'membership_leave'. Same filter chain, same slice, same tau=0.0 threshold.

Per the METRIC AND NUMERICS CONVENTIONS in the task brief: an episode/cell with zero qualifying
events is EXCLUDED, not reported as 0. Every count (in / excluded / used) is reported per band and
per change type, including when the excluded count is exactly zero.

ADDITIONAL, CLEARLY LABELLED SIDE COMPUTATIONS (not the committed answer, reported separately per
STEP 1.4 and the STEP 0.6 discussion -- not used to select among candidate definitions):
  - single-node-only membership response rate (n_touched_nodes == 1), to test the dissertation's
    stated batch-event explanation for the ~2pp residual.
  - tau=1e-9 variant of the membership response rate, to check STEP 0.6's first candidate
    explanation for why two near-identical historical triples exist. This uses the SAME already-
    established function's own alternate default parameter (compute_response_rates is already
    called with taus=(0.0, 1e-9) in the existing pipeline) -- it is not a new definition invented
    to chase a match.
"""
import os
import sys
import numpy as np
import pandas as pd

AG = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents"
DATA_DIR = os.path.join(AG, "cx_step2_registration")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BANDS = ["10-15", "30-40", "80-100"]

# Currently-reported figures, for the comparison table only -- never used to select a definition.
REPORTED = {
    "membership_triple_1": [0.196, 0.153, 0.207],
    "membership_triple_2": [0.198, 0.154, 0.207],
    "discovered_share": [0.177, 0.131, 0.182],
    "property_control": [0.998, 0.987, 0.996],
}


def load_filtered(band):
    """Apply the committed filter chain: drop touched_node_visible==False, keep event_phase ==
    'immediate'. Returns (df_filtered, n_total, n_dropped_visibility, n_dropped_phase)."""
    path = os.path.join(DATA_DIR, f"drift_{band}.csv")
    df = pd.read_csv(path, low_memory=False)
    n_total = len(df)

    n_dropped_visibility = int((df["touched_node_visible"] == False).sum())  # noqa: E712
    df = df[df["touched_node_visible"] != False].copy()  # noqa: E712

    n_before_phase = len(df)
    df = df[df["event_phase"] == "immediate"].copy()
    n_dropped_phase = n_before_phase - len(df)

    return df, n_total, n_dropped_visibility, n_dropped_phase


def response_rate_for(df, change_type, tau=0.0, single_node_only=False):
    """compute_response_rates()'s own per-slice ('full') response-rate logic, tau threshold on
    change_drift_full, applied to change_type. Returns a dict of counts + rate, never a
    fabricated 0.0 for an empty population (NaN instead, per the numerics convention)."""
    sub = df[df["change_type"] == change_type].copy()
    n_events_of_type = len(sub)

    if single_node_only:
        sub = sub[sub["n_touched_nodes"] == 1]

    col = sub["change_drift_full"]
    nan_mask = col.isna()
    n_excluded_nan = int(nan_mask.sum())
    sub = sub[~nan_mask]
    col = sub["change_drift_full"]

    invalid_mask = (col < 0) | (~np.isfinite(col))
    n_excluded_invalid = int(invalid_mask.sum())
    sub = sub[~invalid_mask]

    n_used = len(sub)
    if n_used == 0:
        rate = float("nan")
    else:
        rate = float((sub["change_drift_full"] > tau).mean())

    return {
        "n_events_of_type_after_visibility_and_phase_filter": n_events_of_type,
        "n_excluded_nan_change_drift_full": n_excluded_nan,
        "n_excluded_negative_or_nonfinite": n_excluded_invalid,
        "n_used": n_used,
        "response_rate": rate,
    }, sub


def discovered_share_for(df, change_type):
    """mean(n_discovered / n_scenario) over the identical filtered population used for the
    membership_leave response rate (same rows: change_type == 'membership_leave', after the
    visibility + phase + NaN/invalid change_drift_full filters)."""
    _, sub = response_rate_for(df, change_type, tau=0.0, single_node_only=False)
    n_scenario_zero_or_missing = int((sub["n_scenario"].isna() | (sub["n_scenario"] == 0)).sum())
    valid = sub[~(sub["n_scenario"].isna() | (sub["n_scenario"] == 0))]
    if len(valid) == 0:
        share = float("nan")
    else:
        share = float((valid["n_discovered"] / valid["n_scenario"]).mean())
    return {
        "n_rows_considered": len(sub),
        "n_excluded_n_scenario_zero_or_missing": n_scenario_zero_or_missing,
        "n_used": len(valid),
        "discovered_share": share,
    }


def main():
    rows = []
    summary_lines = []
    for band in BANDS:
        df, n_total, n_drop_vis, n_drop_phase = load_filtered(band)
        summary_lines.append(
            f"[{band}] raw rows={n_total}, dropped touched_node_visible=False: {n_drop_vis}, "
            f"further dropped event_phase != immediate: {n_drop_phase}, "
            f"retained after visibility+phase filter: {len(df)}"
        )

        # (a) membership_leave response rate, tau=0.0 -- THE COMMITTED ANSWER
        ml_stats, ml_sub = response_rate_for(df, "membership_leave", tau=0.0)
        # (b) discovered share, same population as (a)
        ds_stats = discovered_share_for(df, "membership_leave")
        # (c) property response rate, tau=0.0, same pipeline
        prop_stats, _ = response_rate_for(df, "property", tau=0.0)

        # side computation: single-node-only membership response rate (batch-event test)
        ml_single_stats, _ = response_rate_for(df, "membership_leave", tau=0.0, single_node_only=True)
        # side computation: tau=1e-9 variant (STEP 0.6 candidate explanation check)
        ml_eps_stats, _ = response_rate_for(df, "membership_leave", tau=1e-9)

        for ct, stats in [("membership_leave", ml_stats), ("property", prop_stats)]:
            summary_lines.append(
                f"[{band}] change_type={ct}: events_of_type={stats['n_events_of_type_after_visibility_and_phase_filter']}, "
                f"excluded_nan={stats['n_excluded_nan_change_drift_full']}, "
                f"excluded_invalid={stats['n_excluded_negative_or_nonfinite']}, "
                f"used={stats['n_used']}, response_rate={stats['response_rate']:.3f}"
            )

        rows.append({
            "band": band,
            "membership_leave_events_in": ml_stats["n_events_of_type_after_visibility_and_phase_filter"],
            "membership_leave_excluded_nan": ml_stats["n_excluded_nan_change_drift_full"],
            "membership_leave_excluded_invalid": ml_stats["n_excluded_negative_or_nonfinite"],
            "membership_leave_used": ml_stats["n_used"],
            "membership_response_rate_tau0": round(ml_stats["response_rate"], 3),
            "discovered_share_n_used": ds_stats["n_used"],
            "discovered_share_excluded_n_scenario_zero_or_missing": ds_stats["n_excluded_n_scenario_zero_or_missing"],
            "discovered_share": round(ds_stats["discovered_share"], 3),
            "property_events_in": prop_stats["n_events_of_type_after_visibility_and_phase_filter"],
            "property_excluded_nan": prop_stats["n_excluded_nan_change_drift_full"],
            "property_excluded_invalid": prop_stats["n_excluded_negative_or_nonfinite"],
            "property_used": prop_stats["n_used"],
            "property_response_rate_tau0": round(prop_stats["response_rate"], 3),
            # side computations, clearly separated from the committed answer above
            "SIDE_membership_single_node_only_used": ml_single_stats["n_used"],
            "SIDE_membership_single_node_only_response_rate_tau0": round(ml_single_stats["response_rate"], 3),
            "SIDE_membership_response_rate_tau1e-9": round(ml_eps_stats["response_rate"], 3),
        })

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "rq3a_gate_recompute.csv")
    out_df.to_csv(out_path, index=False)

    print("\n".join(summary_lines))
    print()
    print(out_df.to_string(index=False))
    print(f"\nWritten: {out_path}")

    # Comparison table vs currently-reported figures -- no adjustment, plain report.
    print("\n=== COMPARISON vs currently-reported figures (no adjustment) ===")
    for i, band in enumerate(BANDS):
        r = rows[i]
        print(f"[{band}]")
        print(f"  membership response (a): fresh={r['membership_response_rate_tau0']:.3f}  "
              f"reported_triple_1={REPORTED['membership_triple_1'][i]:.3f}  "
              f"reported_triple_2={REPORTED['membership_triple_2'][i]:.3f}  "
              f"diff_vs_t1={r['membership_response_rate_tau0']-REPORTED['membership_triple_1'][i]:+.3f}  "
              f"diff_vs_t2={r['membership_response_rate_tau0']-REPORTED['membership_triple_2'][i]:+.3f}")
        print(f"  discovered share (b):    fresh={r['discovered_share']:.3f}  "
              f"reported={REPORTED['discovered_share'][i]:.3f}  "
              f"diff={r['discovered_share']-REPORTED['discovered_share'][i]:+.3f}")
        print(f"  property control (c):    fresh={r['property_response_rate_tau0']:.3f}  "
              f"reported={REPORTED['property_control'][i]:.3f}  "
              f"diff={r['property_response_rate_tau0']-REPORTED['property_control'][i]:+.3f}")
        print(f"  gap (a)-(b): fresh={r['membership_response_rate_tau0']-r['discovered_share']:+.3f}  "
              f"reported_gap_t1={REPORTED['membership_triple_1'][i]-REPORTED['discovered_share'][i]:+.3f}")
        print(f"  SIDE single-node-only membership response rate: {r['SIDE_membership_single_node_only_response_rate_tau0']:.3f}  "
              f"(n={r['SIDE_membership_single_node_only_used']})")
        print(f"  SIDE tau=1e-9 membership response rate: {r['SIDE_membership_response_rate_tau1e-9']:.3f}")


if __name__ == "__main__":
    main()
