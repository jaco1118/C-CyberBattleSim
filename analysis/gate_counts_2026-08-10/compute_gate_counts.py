"""Task GATE-COUNTS: exact counts for the undiscovered-target membership_leave events behind the
RQ3(a) discovery-gate passage, and a direct test of the co-occurrence contamination explanation.

Population: identical to compute_rq3a_gate.py's gate-passing filter (commit cf3092a), with the
visibility condition INVERTED to get the complement (undiscovered-target) population, not a fresh
filter set:
  - gate-passing: touched_node_visible != False  AND  event_phase == 'immediate'
  - undiscovered-target (this task): touched_node_visible == False
  (empirically, per TASK RQ3A-UNFILTERED's own crosstab, touched_node_visible==False is perfectly
  collinear with event_phase=='fired' for membership_leave/property in this data -- so inverting
  the visibility condition alone already isolates the complement population cleanly, without also
  needing to invert the phase condition.)

"No recorded observation shift" = a genuine NaN in change_drift_full (confirmed by direct row
inspection, STEP 0.3) -- change_drift_full is computed ONCE per step (h2->h3, in
_log_drift_rows, cyberbattle_env_compressed.py) BEFORE the per-event loop, so every row logged for
a given step carries the SAME step-level value regardless of which event that row represents. This
is the literal mechanism the contamination hypothesis in the printed passage describes: a
co-occurring change to a discovered node on the same step moves the pooled vector once, and every
event-row logged for that step (including an undiscovered-target one) inherits that same value.

PREDICTION, STATED IN ADVANCE:
  - fraction of undiscovered-target events with no drift value: near 50%
  - of those with a drift value, near 90% exactly zero
  - the non-zero remainder concentrated on multi-touch-or-multi-event steps, near-absent on
    single-touch, single-event steps
If item 4 shows the non-zero rows spread evenly across both groups, contamination does NOT hold
and that is reported as the finding, not chased toward a different explanation.
"""
import os
import numpy as np
import pandas as pd

AG = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents"
DATA_DIR = os.path.join(AG, "cx_step2_registration")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BANDS = ["10-15", "30-40", "80-100"]

REPORTED_ATTRIBUTED_FRACTION = [0.177, 0.131, 0.182]


def main():
    print("=== SAFETY CONFIRMATION ===")
    print("Reads only cx_step2_registration/drift_<band>.csv (already-logged data). No training, "
          "no environment reset, no new episode, no checkpoint/encoder touched, no step()/encode()/"
          "reward path modified.\n")

    rows = []
    for band in BANDS:
        path = os.path.join(DATA_DIR, f"drift_{band}.csv")
        df = pd.read_csv(path, low_memory=False)
        n_raw = len(df)

        # Step-level event count, ACROSS ALL change types, for the multi-event test in item 4.
        # "An event fired this step" = a row with event_phase in {fired, immediate} (the two phases
        # _log_drift_rows produces from dynamic_events; 'attributed' rows come from a separate,
        # unrelated logging pass -- deferred discovery-attribution, not a co-occurring pooled-vector
        # change at THIS step -- and 'no_change' rows represent zero events by definition).
        event_rows = df[df["event_phase"].isin(["fired", "immediate"])]
        events_per_step = event_rows.groupby(
            ["run_id", "seed", "scenario_id", "episode", "step"]
        ).size()
        multi_event_steps = set(events_per_step[events_per_step > 1].index)

        # --- Item 1: total / attributed / undiscovered, membership_leave ---
        ml = df[df["change_type"] == "membership_leave"].copy()
        n_total = len(ml)
        n_attributed = int((ml["touched_node_visible"] == True).sum())  # noqa: E712
        n_undiscovered = int((ml["touched_node_visible"] == False).sum())  # noqa: E712
        frac_attributed = n_attributed / n_total if n_total else float("nan")
        n_other_visibility = n_total - n_attributed - n_undiscovered  # should be 0; report if not

        # Restrict to the undiscovered-target population for items 2-5.
        und = ml[ml["touched_node_visible"] == False].copy()  # noqa: E712

        # --- Item 2: NaN vs has-a-value ---
        nan_mask = und["change_drift_full"].isna()
        n_nan = int(nan_mask.sum())
        n_has_value = int((~nan_mask).sum())
        frac_nan = n_nan / n_undiscovered if n_undiscovered else float("nan")

        with_value = und[~nan_mask].copy()

        # --- Item 3: exactly zero vs non-zero, exact equality ---
        zero_mask = with_value["change_drift_full"] == 0.0
        n_zero = int(zero_mask.sum())
        n_nonzero = int((~zero_mask).sum())
        frac_zero_of_valued = n_zero / n_has_value if n_has_value else float("nan")

        nonzero = with_value[~zero_mask].copy()

        # --- Item 4: the test. Per non-zero row, is its OWN event multi-touch (n_touched_nodes>1),
        # or does its STEP carry >1 event (co-occurrence with another event, e.g. a discovered-node
        # change firing on the same step)? Combined via OR, per the task's own framing.
        nonzero["is_multi_touch"] = nonzero["n_touched_nodes"] > 1
        step_keys = list(zip(nonzero["run_id"], nonzero["seed"], nonzero["scenario_id"],
                              nonzero["episode"], nonzero["step"]))
        nonzero["is_multi_event_step"] = [k in multi_event_steps for k in step_keys]
        nonzero["is_multi"] = nonzero["is_multi_touch"] | nonzero["is_multi_event_step"]

        # Same classification applied to the FULL with_value population (zero + non-zero), so the
        # non-zero RATE within each group (not just raw counts) can be reported.
        with_value = with_value.copy()
        with_value["is_multi_touch"] = with_value["n_touched_nodes"] > 1
        wv_keys = list(zip(with_value["run_id"], with_value["seed"], with_value["scenario_id"],
                            with_value["episode"], with_value["step"]))
        with_value["is_multi_event_step"] = [k in multi_event_steps for k in wv_keys]
        with_value["is_multi"] = with_value["is_multi_touch"] | with_value["is_multi_event_step"]
        with_value["is_nonzero"] = with_value["change_drift_full"] != 0.0

        single_group = with_value[~with_value["is_multi"]]
        multi_group = with_value[with_value["is_multi"]]
        n_single_total = len(single_group)
        n_single_nonzero = int(single_group["is_nonzero"].sum())
        rate_single = n_single_nonzero / n_single_total if n_single_total else float("nan")
        n_multi_total = len(multi_group)
        n_multi_nonzero = int(multi_group["is_nonzero"].sum())
        rate_multi = n_multi_nonzero / n_multi_total if n_multi_total else float("nan")

        # --- Item 5: largest non-zero drift value on an undiscovered-target row ---
        max_nonzero = float(nonzero["change_drift_full"].max()) if len(nonzero) else float("nan")

        print(f"[{band}]")
        print(f"  item 1: total membership_leave={n_total}, attributed(visible)={n_attributed}, "
              f"undiscovered={n_undiscovered}, other_visibility_value={n_other_visibility}, "
              f"frac_attributed={frac_attributed:.4f}")
        print(f"  item 2: undiscovered with NO drift value (NaN)={n_nan} ({100*frac_nan:.2f}%), "
              f"with a value={n_has_value} ({100*(1-frac_nan):.2f}%)")
        print(f"  item 3: of valued, exactly zero={n_zero} ({100*frac_zero_of_valued:.2f}%), "
              f"non-zero={n_nonzero} ({100*(1-frac_zero_of_valued):.2f}%)")
        print(f"  item 4: single-touch/single-event steps: n={n_single_total}, "
              f"non-zero={n_single_nonzero}, rate={rate_single:.4%}")
        print(f"          multi-touch-or-multi-event steps: n={n_multi_total}, "
              f"non-zero={n_multi_nonzero}, rate={rate_multi:.4%}")
        print(f"  item 5: largest non-zero drift on an undiscovered-target row = {max_nonzero:.4f} "
              f"(vs 0.0086 min observed on a genuine attributed membership_leave shift)")
        print()

        rows.append({
            "band": band,
            "ml_total": n_total, "ml_attributed": n_attributed, "ml_undiscovered": n_undiscovered,
            "ml_other_visibility_value": n_other_visibility, "frac_attributed": round(frac_attributed, 4),
            "undisc_nan": n_nan, "undisc_has_value": n_has_value, "frac_nan": round(frac_nan, 4),
            "valued_zero": n_zero, "valued_nonzero": n_nonzero,
            "frac_zero_of_valued": round(frac_zero_of_valued, 4),
            "single_group_n": n_single_total, "single_group_nonzero": n_single_nonzero,
            "single_group_nonzero_rate": round(rate_single, 6),
            "multi_group_n": n_multi_total, "multi_group_nonzero": n_multi_nonzero,
            "multi_group_nonzero_rate": round(rate_multi, 6),
            "max_nonzero_drift": round(max_nonzero, 4) if not np.isnan(max_nonzero) else None,
        })

    out_df = pd.DataFrame(rows)

    # Pooled row across all three bands.
    pooled = {
        "band": "POOLED",
        "ml_total": out_df["ml_total"].sum(),
        "ml_attributed": out_df["ml_attributed"].sum(),
        "ml_undiscovered": out_df["ml_undiscovered"].sum(),
        "ml_other_visibility_value": out_df["ml_other_visibility_value"].sum(),
        "frac_attributed": round(out_df["ml_attributed"].sum() / out_df["ml_total"].sum(), 4),
        "undisc_nan": out_df["undisc_nan"].sum(),
        "undisc_has_value": out_df["undisc_has_value"].sum(),
        "frac_nan": round(out_df["undisc_nan"].sum() / out_df["ml_undiscovered"].sum(), 4),
        "valued_zero": out_df["valued_zero"].sum(),
        "valued_nonzero": out_df["valued_nonzero"].sum(),
        "frac_zero_of_valued": round(out_df["valued_zero"].sum() / out_df["undisc_has_value"].sum(), 4),
        "single_group_n": out_df["single_group_n"].sum(),
        "single_group_nonzero": out_df["single_group_nonzero"].sum(),
        "single_group_nonzero_rate": round(out_df["single_group_nonzero"].sum() / out_df["single_group_n"].sum(), 6),
        "multi_group_n": out_df["multi_group_n"].sum(),
        "multi_group_nonzero": out_df["multi_group_nonzero"].sum(),
        "multi_group_nonzero_rate": round(out_df["multi_group_nonzero"].sum() / out_df["multi_group_n"].sum(), 6),
        "max_nonzero_drift": out_df["max_nonzero_drift"].max(),
    }
    out_df = pd.concat([out_df, pd.DataFrame([pooled])], ignore_index=True)

    out_path = os.path.join(OUT_DIR, "gate_counts_result.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Written: {out_path}\n")

    print("=== COMPARISON vs reported attributed fraction (no adjustment) ===")
    for i, band in enumerate(BANDS):
        print(f"  {band}: fresh={out_df.iloc[i]['frac_attributed']:.4f}  "
              f"reported={REPORTED_ATTRIBUTED_FRACTION[i]:.3f}")

    print("\n=== ITEM 6 -- does the pooled multi-vs-single non-zero rate support contamination? ===")
    p = out_df.iloc[-1]
    print(f"  single-touch/single-event non-zero rate: {p['single_group_nonzero_rate']:.4%}  (n={p['single_group_n']})")
    print(f"  multi-touch-or-multi-event non-zero rate: {p['multi_group_nonzero_rate']:.4%}  (n={p['multi_group_n']})")


if __name__ == "__main__":
    main()
