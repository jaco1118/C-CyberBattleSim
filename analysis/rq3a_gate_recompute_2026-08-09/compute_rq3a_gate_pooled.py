"""Task RQ3A-POOLED: final run of the discovery-gate point prediction, on the FULL membership_leave
event population including fired-phase (unattributed / undiscovered-target) rows.

Separate file from compute_rq3a_gate.py (cf3092a) and compute_rq3a_gate_unfiltered.py (6445e39) --
neither is modified. TASK RQ3A-UNFILTERED's own crosstab showed event_phase=='immediate' is
perfectly collinear with touched_node_visible==True in this dataset for membership_leave/property,
with zero 'attributed' rows -- so the undiscovered-target population lives entirely in
event_phase=='fired' rows, which both prior scripts excluded (the first via the visibility filter,
the second via the event_phase=='immediate' filter that the task required keeping). This script
removes BOTH filters and works on the full membership_leave population.

Hand-computed check (from TASK RQ3A-UNFILTERED's own reported crosstab counts, stated in the task
brief BEFORE this script was written): attributed fraction = immediate / (immediate + fired) --
  10-15:  2131 / (2131+9881)  = 0.1774  vs reported "discovered share" 0.177
  30-40:  6479 / (6479+42816) = 0.1314  vs reported 0.131
  80-100: 10174/(10174+45861) = 0.1816  vs reported 0.182
Three exact-to-3-decimal matches. This script computes that fraction directly from the data (not
from the prior task's summary numbers) as item 3, and item 6 checks the assumption it rests on.

POPULATION: ALL membership_leave events in cx_step2_registration/, NO event_phase filter, NO
touched_node_visible filter. The ONLY filter applied is the NaN/negative/non-finite guard on
change_drift_full for the response-rate computation specifically (item 2) -- the attributed
fraction (item 3) does not depend on change_drift_full at all and uses the full unfiltered count.

PREDICTION, STATED IN ADVANCE:
  - attributed fraction: 0.177 / 0.131 / 0.182
  - pooled response rate at tau=0: 0.196 / 0.153 / 0.207 (roughly 2pp above attributed fraction)
  - the gap should shrink toward zero on single-node-only events (n_touched_nodes == 1)
If the response triple comes back far from this, report the miss and stop -- no filter chasing.
"""
import os
import numpy as np
import pandas as pd

AG = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents"
DATA_DIR = os.path.join(AG, "cx_step2_registration")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BANDS = ["10-15", "30-40", "80-100"]

REPORTED = {
    "attributed_fraction": [0.177, 0.131, 0.182],
    "response_triple_1": [0.196, 0.153, 0.207],
    "response_triple_2": [0.198, 0.154, 0.207],
}


def load_band(band):
    path = os.path.join(DATA_DIR, f"drift_{band}.csv")
    return pd.read_csv(path, low_memory=False)


def response_rate_block(sub, taus=(0.0, 1e-9)):
    """Apply ONLY the NaN/negative/non-finite guard on change_drift_full to `sub`, then compute
    the response rate at each tau. Returns counts + rates; never fabricates a rate for an empty
    population (NaN instead)."""
    n_in = len(sub)
    col = sub["change_drift_full"]

    nan_mask = col.isna()
    n_excluded_nan = int(nan_mask.sum())
    kept = sub[~nan_mask]
    col2 = kept["change_drift_full"]

    invalid_mask = (col2 < 0) | (~np.isfinite(col2))
    n_excluded_invalid = int(invalid_mask.sum())
    kept = kept[~invalid_mask]

    n_used = len(kept)
    rates = {}
    for tau in taus:
        rates[tau] = float((kept["change_drift_full"] > tau).mean()) if n_used else float("nan")

    return {
        "n_in": n_in,
        "n_excluded_nan": n_excluded_nan,
        "n_excluded_invalid": n_excluded_invalid,
        "n_used": n_used,
        "rates": rates,
    }, kept


def main():
    print("=== SAFETY CONFIRMATION ===")
    print("Reads only cx_step2_registration/drift_<band>.csv (already-logged data). No training, "
          "no environment reset, no new episode, no checkpoint/encoder touched, no step()/encode()/"
          "reward path modified.\n")

    rows = []
    for band in BANDS:
        df = load_band(band)
        ml = df[df["change_type"] == "membership_leave"].copy()
        prop = df[df["change_type"] == "property"].copy()

        # Item 1: crosstab of membership_leave events by event_phase x touched_node_visible.
        crosstab = pd.crosstab(ml["event_phase"], ml["touched_node_visible"], dropna=False)
        print(f"[{band}] item 1 -- membership_leave crosstab (event_phase x touched_node_visible):")
        print(crosstab.to_string())
        n_ml_total = len(ml)

        # Item 2: pooled response rate on the FULL membership_leave population (no phase/visibility
        # filter), NaN/negative/non-finite guard only.
        rr_stats, rr_kept = response_rate_block(ml, taus=(0.0, 1e-9))
        rate_tau0 = rr_stats["rates"][0.0]
        rate_tau1e9 = rr_stats["rates"][1e-9]

        # Item 3: attributed fraction = touched_node_visible==True share of the FULL population
        # (independent of change_drift_full / the NaN guard).
        attributed_fraction = float((ml["touched_node_visible"] == True).mean())  # noqa: E712
        hand_computed = {
            "10-15": 2131 / (2131 + 9881),
            "30-40": 6479 / (6479 + 42816),
            "80-100": 10174 / (10174 + 45861),
        }[band]
        # what fraction of PROPERTY events are fired-phase (tests the hand-computation's assumption)
        prop_fired_fraction = float((prop["event_phase"] == "fired").mean())

        # Item 4: gap
        gap_pp = (rate_tau0 - attributed_fraction) * 100

        # Item 5: same items 2-4, single-node only (n_touched_nodes == 1)
        ml_single = ml[ml["n_touched_nodes"] == 1]
        rr_single_stats, _ = response_rate_block(ml_single, taus=(0.0, 1e-9))
        rate_tau0_single = rr_single_stats["rates"][0.0]
        attributed_fraction_single = float((ml_single["touched_node_visible"] == True).mean())  # noqa: E712
        gap_pp_single = (rate_tau0_single - attributed_fraction_single) * 100

        # Item 6: fired-phase, non-visible membership_leave events -- drift distribution.
        fired = ml[(ml["event_phase"] == "fired") & (ml["touched_node_visible"] == False)]  # noqa: E712
        fired_col = fired["change_drift_full"]
        n_fired = len(fired)
        n_fired_nan = int(fired_col.isna().sum())
        fired_nonnull = fired_col.dropna()
        n_fired_nonnull = len(fired_nonnull)
        n_fired_exact_zero = int((fired_nonnull == 0).sum())
        n_fired_nonzero = int((fired_nonnull > 0).sum())
        fired_stats = {
            "n_fired": n_fired,
            "n_nan": n_fired_nan,
            "n_nonnull": n_fired_nonnull,
            "n_exact_zero": n_fired_exact_zero,
            "n_nonzero_gt0": n_fired_nonzero,
            "min_nonnull": float(fired_nonnull.min()) if n_fired_nonnull else float("nan"),
            "median_nonnull": float(fired_nonnull.median()) if n_fired_nonnull else float("nan"),
            "max_nonnull": float(fired_nonnull.max()) if n_fired_nonnull else float("nan"),
        }

        print(f"[{band}] item 2 -- pooled response rate: n_in={rr_stats['n_in']}, "
              f"excluded_nan={rr_stats['n_excluded_nan']}, excluded_invalid={rr_stats['n_excluded_invalid']}, "
              f"used={rr_stats['n_used']}, rate(tau=0.0)={rate_tau0:.4f}, rate(tau=1e-9)={rate_tau1e9:.4f}")
        print(f"[{band}] item 3 -- attributed_fraction(data)={attributed_fraction:.4f}  "
              f"hand_computed={hand_computed:.4f}  agree={abs(attributed_fraction-hand_computed) < 1e-6}  "
              f"property_fired_fraction={prop_fired_fraction:.4f}")
        print(f"[{band}] item 4 -- gap (tau0 rate - attributed_fraction) = {gap_pp:+.2f}pp")
        print(f"[{band}] item 5 -- single-node only: rate(tau=0.0)={rate_tau0_single:.4f}, "
              f"attributed_fraction={attributed_fraction_single:.4f}, gap={gap_pp_single:+.2f}pp, "
              f"n_used={rr_single_stats['n_used']}")
        print(f"[{band}] item 6 -- fired+non-visible membership_leave (n={n_fired}): "
              f"NaN={n_fired_nan}, non-null={n_fired_nonnull} (exact-zero={n_fired_exact_zero}, "
              f"nonzero>0={n_fired_nonzero}), drift[min/median/max] of non-null = "
              f"{fired_stats['min_nonnull']:.4f} / {fired_stats['median_nonnull']:.4f} / {fired_stats['max_nonnull']:.4f}")
        print()

        rows.append({
            "band": band,
            "membership_leave_total": n_ml_total,
            "immediate_count": int(crosstab.loc["immediate"].sum()) if "immediate" in crosstab.index else 0,
            "fired_count": int(crosstab.loc["fired"].sum()) if "fired" in crosstab.index else 0,
            "response_rate_n_in": rr_stats["n_in"],
            "response_rate_excluded_nan": rr_stats["n_excluded_nan"],
            "response_rate_excluded_invalid": rr_stats["n_excluded_invalid"],
            "response_rate_n_used": rr_stats["n_used"],
            "pooled_response_rate_tau0": round(rate_tau0, 4),
            "pooled_response_rate_tau1e-9": round(rate_tau1e9, 4),
            "attributed_fraction_from_data": round(attributed_fraction, 4),
            "attributed_fraction_hand_computed": round(hand_computed, 4),
            "property_fired_fraction": round(prop_fired_fraction, 4),
            "gap_pp": round(gap_pp, 2),
            "SINGLE_NODE_response_rate_tau0": round(rate_tau0_single, 4),
            "SINGLE_NODE_attributed_fraction": round(attributed_fraction_single, 4),
            "SINGLE_NODE_gap_pp": round(gap_pp_single, 2),
            "SINGLE_NODE_n_used": rr_single_stats["n_used"],
            "FIRED_NONVISIBLE_n": n_fired,
            "FIRED_NONVISIBLE_n_nan": n_fired_nan,
            "FIRED_NONVISIBLE_n_exact_zero": n_fired_exact_zero,
            "FIRED_NONVISIBLE_n_nonzero": n_fired_nonzero,
            "FIRED_NONVISIBLE_min": round(fired_stats["min_nonnull"], 4),
            "FIRED_NONVISIBLE_median": round(fired_stats["median_nonnull"], 4),
            "FIRED_NONVISIBLE_max": round(fired_stats["max_nonnull"], 4),
        })

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "rq3a_gate_recompute_pooled.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Written: {out_path}\n")

    print("=== COMPARISON vs prediction stated in advance (no adjustment) ===")
    for i, band in enumerate(BANDS):
        r = rows[i]
        print(f"[{band}]  pooled_response(tau=0)={r['pooled_response_rate_tau0']:.3f}  "
              f"pooled_response(tau=1e-9)={r['pooled_response_rate_tau1e-9']:.3f}  "
              f"predicted_t1={REPORTED['response_triple_1'][i]:.3f}  predicted_t2={REPORTED['response_triple_2'][i]:.3f}  "
              f"attributed_fraction={r['attributed_fraction_from_data']:.3f}  predicted={REPORTED['attributed_fraction'][i]:.3f}")


if __name__ == "__main__":
    main()
