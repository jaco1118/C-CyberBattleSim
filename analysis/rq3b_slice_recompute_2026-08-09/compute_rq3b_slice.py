"""Task RQ3B-RECOVER: rebuild the change-type comparison's max/min-slice response rates.

Same population, same filters, same anchor as TASK RQ3A-RECOMPUTE's compute_rq3a_gate.py
(commit cf3092a) -- that script reproduced the property FULL-slice response rate
(0.998/0.987/0.996) EXACTLY to three decimal places on all three bands using
cx_step2_registration/ with the visible/immediate filter set. This script reuses the IDENTICAL
filter chain unchanged (touched_node_visible != False, event_phase == 'immediate', NaN/negative/
non-finite guard, tau=0.0), applied to change_drift_max and change_drift_min instead of
change_drift_full. The only thing that changes is which drift column is thresholded.

Not a new script from scratch and not a redefinition -- see load_filtered() and
response_rate_for(), copied verbatim in structure from compute_rq3a_gate.py, generalized only to
accept a drift_col argument.

PREDICTION, STATED IN ADVANCE (per the task brief, quoted from the thesis's change-type comparison
section):
  - maximum slice, membership_leave: 0.925 / 0.729 / 0.407
  - maximum slice, property:         0.962 / 0.759 / 0.424
  - minimum slice, band 80-100:      property 0.411, membership_leave 0.284 (gap 12.7pp)
  - minimum slice, smaller two bands: the two change types close together, no separation of that size
"""
import os
import numpy as np
import pandas as pd

AG = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents"
DATA_DIR = os.path.join(AG, "cx_step2_registration")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BANDS = ["10-15", "30-40", "80-100"]

REPORTED = {
    "max_membership_leave": [0.925, 0.729, 0.407],
    "max_property": [0.962, 0.759, 0.424],
    "min_property_80_100": 0.411,
    "min_membership_leave_80_100": 0.284,
}

# From TASK RQ3A-RECOMPUTE's committed CSV (commit bb47fa0), full-slice event counts, for the
# side-by-side population-visibility comparison required by item 3. Copied as literal constants,
# not recomputed, so this script never silently drifts from that committed result.
PRIOR_FULL_SLICE = {
    "10-15":   {"membership_leave_used": 2131, "property_used": 6996},
    "30-40":   {"membership_leave_used": 6479, "property_used": 25485},
    "80-100":  {"membership_leave_used": 10174, "property_used": 29353},
}


def load_filtered(band):
    """IDENTICAL to compute_rq3a_gate.py's load_filtered -- unchanged."""
    path = os.path.join(DATA_DIR, f"drift_{band}.csv")
    df = pd.read_csv(path, low_memory=False)
    n_total = len(df)

    n_dropped_visibility = int((df["touched_node_visible"] == False).sum())  # noqa: E712
    df = df[df["touched_node_visible"] != False].copy()  # noqa: E712

    n_before_phase = len(df)
    df = df[df["event_phase"] == "immediate"].copy()
    n_dropped_phase = n_before_phase - len(df)

    return df, n_total, n_dropped_visibility, n_dropped_phase


def response_rate_for(df, change_type, drift_col, tau=0.0):
    """IDENTICAL filter logic to compute_rq3a_gate.py's response_rate_for -- generalized only to
    accept drift_col instead of hardcoding change_drift_full. Same NaN/negative/non-finite guard,
    same tau=0.0 threshold."""
    sub = df[df["change_type"] == change_type].copy()
    n_events_of_type = len(sub)

    col = sub[drift_col]
    nan_mask = col.isna()
    n_excluded_nan = int(nan_mask.sum())
    sub = sub[~nan_mask]
    col = sub[drift_col]

    invalid_mask = (col < 0) | (~np.isfinite(col))
    n_excluded_invalid = int(invalid_mask.sum())
    sub = sub[~invalid_mask]

    n_used = len(sub)
    if n_used == 0:
        rate = float("nan")
    else:
        rate = float((sub[drift_col] > tau).mean())

    return {
        "n_events_of_type_after_visibility_and_phase_filter": n_events_of_type,
        "n_excluded_nan": n_excluded_nan,
        "n_excluded_invalid": n_excluded_invalid,
        "n_used": n_used,
        "response_rate": rate,
    }


def main():
    print("=== SAFETY CONFIRMATION ===")
    print("Reads only cx_step2_registration/drift_<band>.csv (already-logged data). No training, "
          "no environment reset, no new episode, no checkpoint/encoder touched, no step()/encode()/"
          "reward path modified.\n")

    rows = []
    for band in BANDS:
        df, n_total, n_drop_vis, n_drop_phase = load_filtered(band)
        print(f"[{band}] raw rows={n_total}, dropped touched_node_visible=False: {n_drop_vis}, "
              f"further dropped event_phase != immediate: {n_drop_phase}, "
              f"retained: {len(df)}")

        row = {"band": band}
        for slice_name, col in [("max", "change_drift_max"), ("min", "change_drift_min")]:
            for ct in ["membership_leave", "property"]:
                stats = response_rate_for(df, ct, col, tau=0.0)
                print(f"  [{slice_name}] {ct}: events_of_type="
                      f"{stats['n_events_of_type_after_visibility_and_phase_filter']}, "
                      f"excluded_nan={stats['n_excluded_nan']}, "
                      f"excluded_invalid={stats['n_excluded_invalid']}, "
                      f"used={stats['n_used']}, response_rate={stats['response_rate']:.3f}")
                prefix = f"{slice_name}_{ct}"
                row[f"{prefix}_events_of_type"] = stats["n_events_of_type_after_visibility_and_phase_filter"]
                row[f"{prefix}_excluded_nan"] = stats["n_excluded_nan"]
                row[f"{prefix}_excluded_invalid"] = stats["n_excluded_invalid"]
                row[f"{prefix}_used"] = stats["n_used"]
                row[f"{prefix}_response_rate"] = round(stats["response_rate"], 3)

            gap_pp = (row[f"{slice_name}_property_response_rate"] - row[f"{slice_name}_membership_leave_response_rate"]) * 100
            row[f"{slice_name}_gap_property_minus_membership_pp"] = round(gap_pp, 2)
            print(f"  [{slice_name}] gap (property - membership_leave) = {gap_pp:+.2f}pp")

        row["prior_full_slice_membership_leave_used"] = PRIOR_FULL_SLICE[band]["membership_leave_used"]
        row["prior_full_slice_property_used"] = PRIOR_FULL_SLICE[band]["property_used"]
        print(f"  [full, from prior RQ3A-RECOMPUTE commit bb47fa0] membership_leave used="
              f"{PRIOR_FULL_SLICE[band]['membership_leave_used']}, property used="
              f"{PRIOR_FULL_SLICE[band]['property_used']}")
        print()

        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "rq3b_slice_recompute.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Written: {out_path}\n")

    print("=== COMPARISON vs prediction stated in advance (no adjustment) ===")
    for i, band in enumerate(BANDS):
        r = rows[i]
        print(f"[{band}]")
        print(f"  MAX slice: membership_leave fresh={r['max_membership_leave_response_rate']:.3f}  "
              f"predicted={REPORTED['max_membership_leave'][i]:.3f}  "
              f"property fresh={r['max_property_response_rate']:.3f}  "
              f"predicted={REPORTED['max_property'][i]:.3f}")
        print(f"  MIN slice: membership_leave fresh={r['min_membership_leave_response_rate']:.3f}  "
              f"property fresh={r['min_property_response_rate']:.3f}  "
              f"gap={r['min_gap_property_minus_membership_pp']:+.2f}pp")
    print(f"\n  MIN slice @ 80-100, predicted: property={REPORTED['min_property_80_100']:.3f}  "
          f"membership_leave={REPORTED['min_membership_leave_80_100']:.3f}  "
          f"predicted gap={100*(REPORTED['min_property_80_100']-REPORTED['min_membership_leave_80_100']):.1f}pp")


if __name__ == "__main__":
    main()
