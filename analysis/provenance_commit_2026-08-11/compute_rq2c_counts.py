"""Task PROVENANCE-COMMIT, Item 2: the six uncommitted RQ2(c) counts, reproduced from raw files.

These six numbers (batch-excluded and episode-cluster counts per band) were previously computed
ad hoc in a chat session and reported without ever being saved to a script -- exactly the failure
mode this task family exists to close (R4). This script is the first committed reproduction.

Two STEP 0 notes carried forward as comments, not rediscovered by the next reader:
  - `_seed` is injected by the caller from the filename/run_id, NOT a field in the JSONL record
    itself (verified: the record keys are run_id/scenario/episode/step/... -- no `seed` key).
  - The record's own field is named `scenario`, not `scenario_id` (the drift CSV uses
    `scenario_id`; these are two different names for the same quantity in two different files).
"""
import json
import glob
import os
import re
import pandas as pd

AG = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents"
DRIFT_DIR = os.path.join(AG, "rq2c_replay")
JSONL_DIR = os.path.join(AG, "rq2c_replay", "rq2c")
OUT_DIR = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/analysis/provenance_commit_2026-08-11"
BANDS = ["10-15", "30-40", "80-100"]

TARGETS = {
    "batch": {"10-15": 136, "30-40": 168, "80-100": 254},
    "episode_clusters": {"10-15": 173, "30-40": 49, "80-100": 66},
    "total": {"10-15": 5101, "30-40": 1651, "80-100": 2962},
    "single": {"10-15": 4965, "30-40": 1483, "80-100": 2708},
}


def batch_counts(band):
    df = pd.read_csv(os.path.join(DRIFT_DIR, f"drift_{band}.csv"),
                      usecols=["change_type", "n_touched_nodes"], low_memory=False)
    sub = df[df["change_type"] == "membership_leave"]
    total = len(sub)
    single = int((sub["n_touched_nodes"] == 1).sum())
    batch = int((sub["n_touched_nodes"] > 1).sum())
    return total, single, batch


def episode_cluster_counts(band):
    # `_seed` (STEP 0 note): not a JSONL field -- parsed from the filename, matching the loader
    # convention already established for this dataset (rq2c_<band>_seed<seed>_<scenario>.jsonl).
    #
    # KEY BUG, found and fixed before this script's results were reported (not silently): the first
    # version of this function keyed episode_keys on (seed, scenario, episode) -- a 3-tuple -- which
    # gave 730/148/208, roughly 4x the target 173/49/66. The actual producing code
    # (compute_rq2c_action_divergence.py:154) keys its own `episode_changed` dict on
    # (seed, episode) ONLY -- a 2-tuple, with NO scenario component:
    #   s["episode_changed"][(r["_seed"], r["episode"])].append(ch)
    # Episode numbering restarts from 0 within each (seed, scenario) run, so this 2-tuple key
    # deliberately (or not -- the code does not say) collapses episodes across DIFFERENT
    # topologies that happen to share the same seed and episode number into ONE cluster. This is
    # reported as a real property of the bootstrap's own clustering unit, not smoothed over: BOTH
    # the buggy 3-tuple count and the corrected 2-tuple count are computed and reported below, so
    # the divergence itself is on the record.
    files = sorted(glob.glob(os.path.join(JSONL_DIR, f"rq2c_{band}_seed*_*.jsonl")))
    episode_keys_2tuple = set()   # (seed, episode) -- matches the actual code exactly
    episode_keys_3tuple = set()   # (seed, scenario, episode) -- the bug, kept for the record
    n_group_ii = 0
    n_group_i = 0
    files_zero_contrib = []
    for f in files:
        m = re.match(rf".*rq2c_{re.escape(band)}_seed(\d+)_", f)
        seed = int(m.group(1))
        n_before = len(episode_keys_2tuple)
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("excluded") is not None:
                    continue
                grp = r.get("group")
                if grp == "i":
                    n_group_i += 1
                elif grp == "ii":
                    n_group_ii += 1
                    # `scenario` (STEP 0 note): the record's own field name, not `scenario_id`.
                    episode_keys_2tuple.add((seed, r["episode"]))
                    episode_keys_3tuple.add((seed, r["scenario"], r["episode"]))
        if len(episode_keys_2tuple) == n_before:
            files_zero_contrib.append(os.path.basename(f))
    return (len(files), len(episode_keys_2tuple), len(episode_keys_3tuple),
            n_group_ii, n_group_i, files_zero_contrib)


def main():
    print("=== ITEM 2: RQ2(c) counts ===\n")
    rows = []
    for band in BANDS:
        total, single, batch = batch_counts(band)
        n_files, n_clusters_2tuple, n_clusters_3tuple, n_group_ii, n_group_i, zero_files = episode_cluster_counts(band)

        print(f"--- band {band} ---")
        print(f"  drift CSV: total membership_leave={total}  single-node(n_touched==1)={single}  "
              f"batch(n_touched>1)={batch}  (single+batch={single+batch}, check: {single+batch==total})")
        print(f"  target total={TARGETS['total'][band]} (diff {total-TARGETS['total'][band]:+d})  "
              f"target single={TARGETS['single'][band]} (diff {single-TARGETS['single'][band]:+d})  "
              f"target batch={TARGETS['batch'][band]} (diff {batch-TARGETS['batch'][band]:+d})")
        print(f"  JSONL source: {n_files} files contributing, {len(zero_files)} contribute zero group-ii episodes"
              + (f" ({zero_files})" if zero_files else ""))
        print(f"  group_i={n_group_i}  group_ii={n_group_ii}")
        print(f"  episode clusters, (seed,episode) 2-tuple [MATCHES the actual code exactly]: {n_clusters_2tuple}  "
              f"target={TARGETS['episode_clusters'][band]}  diff={n_clusters_2tuple-TARGETS['episode_clusters'][band]:+d}")
        print(f"  episode clusters, (seed,scenario,episode) 3-tuple [does NOT match the code -- kept for the record]: "
              f"{n_clusters_3tuple}  diff vs target={n_clusters_3tuple-TARGETS['episode_clusters'][band]:+d}")
        print()

        rows.append(dict(
            band=band, total_membership_leave=total, single_node=single, batch=batch,
            single_plus_batch_eq_total=(single + batch == total),
            target_total=TARGETS["total"][band], target_single=TARGETS["single"][band],
            target_batch=TARGETS["batch"][band],
            n_jsonl_files=n_files, n_jsonl_files_zero_contribution=len(zero_files),
            n_group_i=n_group_i, n_group_ii=n_group_ii,
            episode_clusters_2tuple_seed_episode=n_clusters_2tuple,
            episode_clusters_3tuple_seed_scenario_episode=n_clusters_3tuple,
            target_episode_clusters=TARGETS["episode_clusters"][band],
        ))

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "item2_rq2c_counts.csv"), index=False)
    print("Done.")


if __name__ == "__main__":
    main()
