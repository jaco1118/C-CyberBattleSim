"""Task REPLAY-FLAG-CLOSEOUT: behavioural-signature check of allow_undiscovered_removal for
cx_step2_replay, the last dataset whose flag state rested on run_metadata alone.

Minimal extension of analysis/flag_ground_truth_2026-08-12/compute_flag_ground_truth.py's logic,
applied to cx_step2_replay/ only (the other three datasets were already settled in that task).

IMPORTANT precondition, checked before interpreting any number (per the task's explicit
instruction): cx_step2_replay is a FORCED-ACTION replay (CX_REPLAY=1), not a replayed-event log.
Confirmed from source (compute_attenuation_analysis.py:391-401): CX_REPLAY only substitutes the
AGENT's action (`action = _replay[_ri:_ri+1]` in place of `model.predict(state)`); the resulting
action is fed into `vec_env.step(action)` identically either way. grep -rn "CX_REPLAY" cyberbattle/
(no pathspec restriction) shows every other CX_REPLAY* reference is CX_REPLAY_PROBE, a read-only
diagnostic logger (cyberbattle_env_compressed.py:709,892-896) that never touches
maybe_apply_dynamic_step/_apply_dynamic_leave/_get_removal_eligible_nodes. So the environment's
own dynamic-leave/join mechanism runs on this run's OWN self.allow_undiscovered_removal exactly
as in any live rollout -- CX_REPLAY replays WHICH ACTIONS the agent takes, not WHICH NODES the
environment's stochastic dynamic-change process targets. The unknown_fraction statistic computed
here therefore IS a valid measurement of cx_step2_replay's own flag behaviour, not an echo of
cx_step2_registration's recorded events.

No experiment re-run. Reads only the already-on-disk cx_step2_replay/drift_<band>.csv.
"""
import os

import numpy as np
import pandas as pd

AGENTS_DIR = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BANDS = ["10-15", "30-40", "80-100"]
DATASET = "cx_step2_replay"

COLS = ["change_type", "n_scenario", "n_discovered_h2", "touched_node_visible"]


def main():
    print(f"=== REPLAY-FLAG-CLOSEOUT: behavioural signature for {DATASET} ===\n")
    rows = []
    pooled_used = 0
    pooled_visible = 0
    pooled_not_visible = 0
    for band in BANDS:
        path = os.path.join(AGENTS_DIR, DATASET, f"drift_{band}.csv")
        df = pd.read_csv(path, usecols=COLS, low_memory=False)
        leaves = df[df["change_type"] == "membership_leave"].copy()
        n_read = len(leaves)
        if n_read == 0:
            print(f"--- band {band}: n_rows_read=0 -> no events, undefined ---\n")
            rows.append(dict(dataset=DATASET, band=band, n_rows_read=0, n_rows_used=0,
                              n_rows_dropped_missing_col=0, n_visible=0, n_not_visible=0,
                              unknown_fraction="no events, undefined",
                              discfrac_median=None, discfrac_p5=None, discfrac_p95=None))
            continue

        missing_mask = leaves["touched_node_visible"].isna()
        n_dropped_missing = int(missing_mask.sum())
        used = leaves[~missing_mask]
        n_used = len(used)

        n_visible = int((used["touched_node_visible"] == True).sum())  # noqa: E712
        n_not_visible = int((used["touched_node_visible"] == False).sum())  # noqa: E712
        assert n_visible + n_not_visible == n_used

        unknown_fraction = (n_not_visible / n_used) if n_used else "no events, undefined"
        unknown_fraction_incl_missing = (n_not_visible / n_read)  # including missing rows in denom, per convention

        discfrac = (used["n_discovered_h2"] / used["n_scenario"]).replace([np.inf, -np.inf], np.nan).dropna()
        discfrac_median = float(discfrac.median()) if len(discfrac) else None
        discfrac_p5 = float(discfrac.quantile(0.05)) if len(discfrac) else None
        discfrac_p95 = float(discfrac.quantile(0.95)) if len(discfrac) else None

        uf_str = f"{unknown_fraction:.4f}" if isinstance(unknown_fraction, float) else unknown_fraction
        print(f"--- band {band} ---")
        print(f"  n_rows_read={n_read}  n_rows_dropped_missing_col={n_dropped_missing}  n_rows_used={n_used}")
        print(f"  n_visible={n_visible}  n_not_visible={n_not_visible}  unknown_fraction={uf_str}  "
              f"(incl.-missing-in-denom variant: {unknown_fraction_incl_missing:.4f})")
        print(f"  n_discovered_h2/n_scenario at leave-time: median={discfrac_median}, "
              f"p5={discfrac_p5}, p95={discfrac_p95}\n")

        rows.append(dict(dataset=DATASET, band=band, n_rows_read=n_read, n_rows_used=n_used,
                          n_rows_dropped_missing_col=n_dropped_missing,
                          n_visible=n_visible, n_not_visible=n_not_visible,
                          unknown_fraction=unknown_fraction,
                          unknown_fraction_incl_missing=unknown_fraction_incl_missing,
                          discfrac_median=discfrac_median, discfrac_p5=discfrac_p5, discfrac_p95=discfrac_p95))

        pooled_used += n_used
        pooled_visible += n_visible
        pooled_not_visible += n_not_visible

    pooled_uf = (pooled_not_visible / pooled_used) if pooled_used else "no events, undefined"
    pooled_uf_str = f"{pooled_uf:.4f}" if isinstance(pooled_uf, float) else pooled_uf
    print(f"=== {DATASET} POOLED: n_used={pooled_used}  n_visible={pooled_visible}  "
          f"n_not_visible={pooled_not_visible}  unknown_fraction={pooled_uf_str} ===\n")
    rows.append(dict(dataset=DATASET, band="POOLED", n_rows_read=pooled_used, n_rows_used=pooled_used,
                      n_rows_dropped_missing_col=None, n_visible=pooled_visible,
                      n_not_visible=pooled_not_visible, unknown_fraction=pooled_uf,
                      discfrac_median=None, discfrac_p5=None, discfrac_p95=None))

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "cx_step2_replay_signature.csv"), index=False)
    print("Done.")


if __name__ == "__main__":
    main()
