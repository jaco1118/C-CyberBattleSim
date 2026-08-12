"""Task FLAG-GROUND-TRUTH: establish the real allow_undiscovered_removal setting for
graphdepth_sweep_wide/ and rq2c_replay/ WITHOUT relying on run_metadata.

Mechanism under test: _get_removal_eligible_nodes (cyberbattle_env.py:520-524) bases the
removal candidate pool on self.discovered_nodes when allow_undiscovered_removal is False,
or the whole topology (self.environment.nodes()) when True. If the flag was OFF, every
membership_leave event's touched node was, by construction, already in discovered_nodes at
that moment -- and touched_node_visible (cyberbattle_env_compressed.py:1153-1154, tested
against h2.node_embeddings, a close proxy for discovered_nodes at that same step -- see
RUN_RECORD.md for the precise relationship) must then be True for every single leave event,
with zero exceptions. If the flag was ON on a partially-explored network, some leave events
should target undiscovered nodes, producing touched_node_visible=False rows.

This script computes, per dataset and band, over membership_leave events only:
  - n_rows_read: raw drift CSV rows read (change_type=='membership_leave')
  - n_rows_dropped_missing_column: rows where touched_node_visible is null/NaN
  - n_rows_used: n_rows_read - n_rows_dropped_missing_column
  - n_visible / n_not_visible / unknown_fraction = n_not_visible / n_rows_used
  - the escape-route check: distribution of n_discovered_h2 / n_scenario at the moment each
    leave event fired (median/p5/p95), so a 0% unknown_fraction can be judged informative
    (agent had NOT discovered most of the topology) or uninformative (agent had discovered
    nearly everything, so a 0% figure would follow even with the flag ON).

No experiment re-run. Reads only already-on-disk drift_<band>.csv files for the four datasets
compared. Datasets:
  - graphdepth_sweep_wide, rq2c_replay: DISPUTED (run_metadata under suspicion, see task).
  - attenuation_drift_logs: REFERENCE, flag OFF pinned independently of run_metadata --
    verified in this task (not taken on faith): commit 40dfc7c (2026-07-26) has zero
    occurrences of "allow_undiscovered_removal" anywhere in cyberbattle_env.py (the flag did
    not exist in the code at all), and _get_removal_eligible_nodes at that commit
    unconditionally returns `[node for node in self.discovered_nodes if ...]` with no branch.
    drift_10-15.csv's own mtime (2026-07-26 17:09:25) matches that commit's date, and no
    run_metadata_*.json file exists anywhere under attenuation_drift_logs/ (the metadata
    writer did not exist yet either) -- both independent corroborations that this data
    predates the flag, not just the git-log argument alone.
  - cx_step2_registration: COMPARISON POINT ONLY, not ground truth (its own basis is also
    run_metadata) -- reported for context, not treated as proof.
"""
import os

import numpy as np
import pandas as pd

AGENTS_DIR = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BANDS = ["10-15", "30-40", "80-100"]
DATASETS = ["attenuation_drift_logs", "cx_step2_registration", "graphdepth_sweep_wide", "rq2c_replay"]

COLS = ["change_type", "n_scenario", "n_discovered_h2", "touched_node_visible"]


def main():
    print("=== FLAG-GROUND-TRUTH: behavioural signature of allow_undiscovered_removal ===\n")
    rows = []
    for dataset in DATASETS:
        pooled_used = 0
        pooled_visible = 0
        pooled_not_visible = 0
        for band in BANDS:
            path = os.path.join(AGENTS_DIR, dataset, f"drift_{band}.csv")
            if not os.path.exists(path):
                print(f"--- {dataset} / band {band}: FILE NOT FOUND at {path} ---\n")
                rows.append(dict(dataset=dataset, band=band, n_rows_read=0, n_rows_used=0,
                                  n_rows_dropped_missing_col=0, n_visible=0, n_not_visible=0,
                                  unknown_fraction="no events, undefined",
                                  discfrac_median=None, discfrac_p5=None, discfrac_p95=None))
                continue
            df = pd.read_csv(path, usecols=COLS, low_memory=False)
            leaves = df[df["change_type"] == "membership_leave"].copy()
            n_read = len(leaves)
            if n_read == 0:
                print(f"--- {dataset} / band {band}: n_rows_read=0 -> no events, undefined ---\n")
                rows.append(dict(dataset=dataset, band=band, n_rows_read=0, n_rows_used=0,
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
            assert n_visible + n_not_visible == n_used, "unexpected non-boolean value in touched_node_visible"

            unknown_fraction = (n_not_visible / n_used) if n_used else "no events, undefined"

            # escape-route check: how much of the topology was already discovered when these
            # leave events fired (computed on the USED rows, i.e. excluding missing-column rows)
            discfrac = (used["n_discovered_h2"] / used["n_scenario"]).replace([np.inf, -np.inf], np.nan).dropna()
            discfrac_median = float(discfrac.median()) if len(discfrac) else None
            discfrac_p5 = float(discfrac.quantile(0.05)) if len(discfrac) else None
            discfrac_p95 = float(discfrac.quantile(0.95)) if len(discfrac) else None

            uf_str = f"{unknown_fraction:.4f}" if isinstance(unknown_fraction, float) else unknown_fraction
            print(f"--- {dataset} / band {band} ---")
            print(f"  n_rows_read={n_read}  n_rows_dropped_missing_col={n_dropped_missing}  n_rows_used={n_used}")
            print(f"  n_visible={n_visible}  n_not_visible={n_not_visible}  unknown_fraction={uf_str}")
            print(f"  n_discovered_h2/n_scenario at leave-time: median={discfrac_median}, "
                  f"p5={discfrac_p5}, p95={discfrac_p95}\n")

            rows.append(dict(dataset=dataset, band=band, n_rows_read=n_read, n_rows_used=n_used,
                              n_rows_dropped_missing_col=n_dropped_missing,
                              n_visible=n_visible, n_not_visible=n_not_visible,
                              unknown_fraction=unknown_fraction,
                              discfrac_median=discfrac_median, discfrac_p5=discfrac_p5, discfrac_p95=discfrac_p95))

            pooled_used += n_used
            pooled_visible += n_visible
            pooled_not_visible += n_not_visible

        pooled_uf = (pooled_not_visible / pooled_used) if pooled_used else "no events, undefined"
        pooled_uf_str = f"{pooled_uf:.4f}" if isinstance(pooled_uf, float) else pooled_uf
        print(f"=== {dataset} POOLED: n_used={pooled_used}  n_visible={pooled_visible}  "
              f"n_not_visible={pooled_not_visible}  unknown_fraction={pooled_uf_str} ===\n")
        rows.append(dict(dataset=dataset, band="POOLED", n_rows_read=pooled_used, n_rows_used=pooled_used,
                          n_rows_dropped_missing_col=None, n_visible=pooled_visible,
                          n_not_visible=pooled_not_visible, unknown_fraction=pooled_uf,
                          discfrac_median=None, discfrac_p5=None, discfrac_p95=None))

    out_df = pd.DataFrame(rows)
    out_df.to_csv(os.path.join(OUT_DIR, "flag_ground_truth_signature.csv"), index=False)
    n_undefined_cells = sum(1 for r in rows if r["unknown_fraction"] == "no events, undefined")
    print(f"n_cells_undefined (no events): {n_undefined_cells}")
    print("\nDone.")


if __name__ == "__main__":
    main()
