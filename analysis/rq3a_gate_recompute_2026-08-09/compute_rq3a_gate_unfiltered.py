"""Task RQ3A-UNFILTERED: one addendum run of the discovery-gate point prediction, with the
touched_node_visible filter removed.

This is a COPY of compute_rq3a_gate.py (commit cf3092a), not an edit of it -- the original script
and its committed CSV (rq3a_gate_recompute.csv) are untouched, so both versions remain diffable and
the prior result stands on its own.

WHY THIS CHANGE, AND WHY IT IS NOT DEFINITION-SHOPPING (stated before the run, per the task brief):
TASK RQ3A-RECOMPUTE's STEP 0.2 committed, correctly, to compute_response_rates()'s own filter set
as written in source -- which begins by dropping touched_node_visible == False rows
(compute_attenuation_analysis.py:504-506). That filter restricts the population to changes on
ALREADY-DISCOVERED targets. Under Task CX's discovery gate, an already-discovered target's change
registers on the full/pooled slice by construction -- so on that population the point prediction
("response rate should equal discovered share, because discovered targets register at 1.000 and
undiscovered ones at 0.000") degenerates to 1.000 == 1.000 and cannot be tested at all. This is
exactly the STEP 1 result obtained (response rate 1.000 on all three bands, zero exclusions) --
not a failed reproduction, but a correct measurement of a different, gate-vacuous quantity.

The claim under test is about the POOLED population -- discovered and undiscovered targets
together. Testing it requires undiscovered targets left IN. So THE ONE CHANGE made here: do not
drop rows on touched_node_visible. Every other filter is identical, in the same order:
event_phase == 'immediate', then the NaN/negative/non-finite guard on change_drift_full.

This is principled (the cause was identified in source, not guessed), singular (one filter removed
for a stated mechanistic reason, not several tried), and falsifiable (the prediction below is
written down before the run). If the numbers come back far from the prediction, that is reported
as the finding, not adjusted toward it.

PREDICTION, STATED IN ADVANCE OF RUNNING THIS SCRIPT:
  - membership_leave response rate near 0.196, 0.153, 0.207
  - discovered share near 0.177, 0.131, 0.182
  - the two within roughly two percentage points of each other per band
  - property response effectively unchanged, still near 0.998, 0.987, 0.996 (property events in
    this dataset should be almost all visible-target already)
"""
import os
import numpy as np
import pandas as pd

AG = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents"
DATA_DIR = os.path.join(AG, "cx_step2_registration")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BANDS = ["10-15", "30-40", "80-100"]

REPORTED = {
    "membership_triple_1": [0.196, 0.153, 0.207],
    "membership_triple_2": [0.198, 0.154, 0.207],
    "discovered_share": [0.177, 0.131, 0.182],
    "property_control": [0.998, 0.987, 0.996],
}

# Previous task's visible-only counts (rq3a_gate_recompute.csv, commit bb47fa0), for item 5's
# side-by-side comparison -- copied as literal constants, not recomputed, so this file never
# silently drifts from that committed result.
PRIOR_VISIBLE_ONLY = {
    "10-15":   {"membership_leave_used": 2131, "property_used": 6996},
    "30-40":   {"membership_leave_used": 6479, "property_used": 25485},
    "80-100":  {"membership_leave_used": 10174, "property_used": 29353},
}


def load_unfiltered(band):
    """THE ONE CHANGE vs compute_rq3a_gate.py: do NOT drop touched_node_visible == False rows.
    Keep event_phase == 'immediate' only, same as before, same order."""
    path = os.path.join(DATA_DIR, f"drift_{band}.csv")
    df = pd.read_csv(path, low_memory=False)
    n_total = len(df)

    n_before_phase = len(df)
    df = df[df["event_phase"] == "immediate"].copy()
    n_dropped_phase = n_before_phase - len(df)

    return df, n_total, n_dropped_phase


def response_rate_for(df, change_type, tau=0.0, single_node_only=False):
    """Identical logic to compute_rq3a_gate.py's response_rate_for -- unchanged."""
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
    rates = {}
    for tau_val in (tau if isinstance(tau, (list, tuple)) else [tau]):
        rates[tau_val] = float((sub["change_drift_full"] > tau_val).mean()) if n_used else float("nan")

    frac_visible = float((sub["touched_node_visible"] == True).mean()) if n_used else float("nan")  # noqa: E712

    return {
        "n_events_of_type_after_phase_filter": n_events_of_type,
        "n_excluded_nan_change_drift_full": n_excluded_nan,
        "n_excluded_negative_or_nonfinite": n_excluded_invalid,
        "n_used": n_used,
        "rates": rates,
        "frac_touched_node_visible": frac_visible,
    }, sub


def discovered_share_for(sub):
    n_scenario_zero_or_missing = int((sub["n_scenario"].isna() | (sub["n_scenario"] == 0)).sum())
    valid = sub[~(sub["n_scenario"].isna() | (sub["n_scenario"] == 0))]
    share = float((valid["n_discovered"] / valid["n_scenario"]).mean()) if len(valid) else float("nan")
    return {
        "n_used": len(valid),
        "n_excluded_n_scenario_zero_or_missing": n_scenario_zero_or_missing,
        "discovered_share": share,
    }


def main():
    rows = []
    print("=== SAFETY CONFIRMATION ===")
    print("Reads only cx_step2_registration/drift_<band>.csv (already-logged data). No training, "
          "no environment reset, no new episode, no checkpoint/encoder touched, no step()/encode()/"
          "reward path modified.\n")

    for band in BANDS:
        df, n_total, n_drop_phase = load_unfiltered(band)
        print(f"[{band}] raw rows={n_total}, dropped event_phase != immediate: {n_drop_phase}, "
              f"retained (touched_node_visible filter REMOVED): {len(df)}")

        # Item 1: membership_leave response rate at tau=0.0 AND tau=1e-9, unfiltered population.
        ml_stats, ml_sub = response_rate_for(df, "membership_leave", tau=(0.0, 1e-9))
        # Item 2: discovered share over the same unfiltered membership_leave population.
        ds_stats = discovered_share_for(ml_sub)
        # Item 3: fraction touched_node_visible==True among membership_leave events, vs tau=0 rate.
        frac_vis_ml = ml_stats["frac_touched_node_visible"]
        rate_tau0_ml = ml_stats["rates"][0.0]
        rate_tau1e9_ml = ml_stats["rates"][1e-9]
        agreement = rate_tau0_ml - frac_vis_ml

        # Item 4: property response rate, unfiltered, + its own visible fraction.
        prop_stats, prop_sub = response_rate_for(df, "property", tau=(0.0,))
        rate_tau0_prop = prop_stats["rates"][0.0]
        frac_vis_prop = prop_stats["frac_touched_node_visible"]

        # Item 6: single-node-only block (batch-event test), unfiltered population.
        ml_single_stats, ml_single_sub = response_rate_for(df, "membership_leave", tau=(0.0, 1e-9), single_node_only=True)
        ds_single_stats = discovered_share_for(ml_single_sub)

        print(f"  [item 1] membership_leave: events_in={ml_stats['n_events_of_type_after_phase_filter']}, "
              f"excluded_nan={ml_stats['n_excluded_nan_change_drift_full']}, "
              f"excluded_invalid={ml_stats['n_excluded_negative_or_nonfinite']}, "
              f"used={ml_stats['n_used']}, response_rate(tau=0.0)={rate_tau0_ml:.3f}, "
              f"response_rate(tau=1e-9)={rate_tau1e9_ml:.3f}")
        print(f"  [item 2] discovered_share (same unfiltered membership_leave population): "
              f"{ds_stats['discovered_share']:.3f} (n_used={ds_stats['n_used']}, "
              f"excluded_n_scenario_zero_or_missing={ds_stats['n_excluded_n_scenario_zero_or_missing']})")
        print(f"  [item 3] frac(touched_node_visible==True) among membership_leave = {frac_vis_ml:.3f}  "
              f"vs response_rate(tau=0.0) = {rate_tau0_ml:.3f}  "
              f"agreement (rate - frac_visible) = {agreement:+.3f}")
        print(f"  [item 4] property: events_in={prop_stats['n_events_of_type_after_phase_filter']}, "
              f"used={prop_stats['n_used']}, response_rate(tau=0.0)={rate_tau0_prop:.3f}, "
              f"frac(touched_node_visible==True)={frac_vis_prop:.3f}")
        print(f"  [item 5] event counts -- unfiltered used: membership_leave={ml_stats['n_used']}, "
              f"property={prop_stats['n_used']}  |  prior visible-only used (bb47fa0): "
              f"membership_leave={PRIOR_VISIBLE_ONLY[band]['membership_leave_used']}, "
              f"property={PRIOR_VISIBLE_ONLY[band]['property_used']}")
        print(f"  [item 6, single-node only] membership_leave: used={ml_single_stats['n_used']}, "
              f"response_rate(tau=0.0)={ml_single_stats['rates'][0.0]:.3f}, "
              f"response_rate(tau=1e-9)={ml_single_stats['rates'][1e-9]:.3f}, "
              f"discovered_share={ds_single_stats['discovered_share']:.3f}, "
              f"frac_visible={ml_single_stats['frac_touched_node_visible']:.3f}")
        print()

        rows.append({
            "band": band,
            "membership_leave_events_in": ml_stats["n_events_of_type_after_phase_filter"],
            "membership_leave_excluded_nan": ml_stats["n_excluded_nan_change_drift_full"],
            "membership_leave_excluded_invalid": ml_stats["n_excluded_negative_or_nonfinite"],
            "membership_leave_used": ml_stats["n_used"],
            "membership_response_rate_tau0": round(rate_tau0_ml, 3),
            "membership_response_rate_tau1e-9": round(rate_tau1e9_ml, 3),
            "discovered_share": round(ds_stats["discovered_share"], 3),
            "discovered_share_n_used": ds_stats["n_used"],
            "frac_touched_node_visible_membership_leave": round(frac_vis_ml, 3),
            "item3_agreement_rate_minus_frac_visible": round(agreement, 4),
            "property_events_in": prop_stats["n_events_of_type_after_phase_filter"],
            "property_used": prop_stats["n_used"],
            "property_response_rate_tau0": round(rate_tau0_prop, 3),
            "frac_touched_node_visible_property": round(frac_vis_prop, 3),
            "prior_visible_only_membership_leave_used": PRIOR_VISIBLE_ONLY[band]["membership_leave_used"],
            "prior_visible_only_property_used": PRIOR_VISIBLE_ONLY[band]["property_used"],
            "SINGLE_NODE_membership_leave_used": ml_single_stats["n_used"],
            "SINGLE_NODE_response_rate_tau0": round(ml_single_stats["rates"][0.0], 3),
            "SINGLE_NODE_response_rate_tau1e-9": round(ml_single_stats["rates"][1e-9], 3),
            "SINGLE_NODE_discovered_share": round(ds_single_stats["discovered_share"], 3),
        })

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "rq3a_gate_recompute_unfiltered.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Written: {out_path}\n")

    print("=== COMPARISON vs prediction stated in advance (no adjustment) ===")
    for i, band in enumerate(BANDS):
        r = rows[i]
        print(f"[{band}]")
        print(f"  membership response tau=0.0: fresh={r['membership_response_rate_tau0']:.3f}  "
              f"tau=1e-9: fresh={r['membership_response_rate_tau1e-9']:.3f}  "
              f"predicted~t1={REPORTED['membership_triple_1'][i]:.3f}  predicted~t2={REPORTED['membership_triple_2'][i]:.3f}")
        print(f"  discovered share: fresh={r['discovered_share']:.3f}  predicted~{REPORTED['discovered_share'][i]:.3f}")
        print(f"  property response tau=0.0: fresh={r['property_response_rate_tau0']:.3f}  "
              f"predicted~{REPORTED['property_control'][i]:.3f}")


if __name__ == "__main__":
    main()
