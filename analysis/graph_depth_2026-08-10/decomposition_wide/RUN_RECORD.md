# Run record — GRAPH-DEPTH-WIDE STEP 3 (sweep) + STEP 4 (decomposition)

Date: 2026-08-11.

## STEP 3 — sweep

Exact command:
```
cd cyberbattle/agents
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0
RQ2C=1 LEG=1 YEG_DRIFT_DIR=graphdepth_sweep_wide \
  /cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/bin/python \
  compute_attenuation_analysis.py --manifest attenuation_manifest.yaml --collect
```
Inputs read: `attenuation_manifest.yaml` (the 15-checkpoint manifest, unchanged from the original
GRAPH-DEPTH sweep), the same 15 checkpoints' `checkpoint_250000_steps.zip` + `vecnormalize_*.pkl`
under each band's `run_folders` entries, the same `join_donor_pool_20_topologies` donor pool, the
same 8 topologies per band under `cyberbattle/data/env_samples/scalability_*`.
RNG: per-seed env seeding is intrinsic to `compute_attenuation_analysis.py`'s own harness (seeds
42/100/123/200/300, unchanged from the original sweep); `PYTHONHASHSEED=0` set for the join-donor
non-determinism (CA-2). No new/different seeding introduced for this task.
Output: `cyberbattle/agents/graphdepth_sweep_wide/` (NOT committed — bulk run data, 891MB, per this
project's standing convention against committing bulk run data; only derived summaries are
committed, per STEP 4 below).
Row counts in/out: reported in the STEP 3 completion message (total events / batch-excluded /
single-node per band: 1437/76/1361, 1923/415/1508, 3591/772/2819) and reproduced in STEP 4's own
output (below), computed independently by re-reading the raw JSONL.

## STEP 4 — decomposition

Exact command:
```
cd analysis/graph_depth_2026-08-10/decomposition_wide
/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/bin/python compute_graphdepth_decomposition_wide.py
```
Input paths read (all under `cyberbattle/agents/graphdepth_sweep_wide/`, none modified):
`leaveembed_10-15/*/*.jsonl` (21 files), `leaveembed_30-40/*/*.jsonl` (24 files),
`leaveembed_80-100/*/*.jsonl` (26 files).
RNG: none — the decomposition script performs no random sampling; every reported figure is computed
deterministically over the full logged population (no bootstrap in this task; STEP 4's spec did not
ask for one here, unlike TASK RQ3C-REBUILD's own separate regression work).

Row counts in/out at every stage, per band (ARTIFACT, reproduced from the script's own printed
output):

| band | total events | batch-excluded (n_touched!=1) | single-node | coverage-gate fail | used for ratio |
|---|---|---|---|---|---|
| 10-15 | 1437 | 76 | 1361 | 0 | 1361 |
| 30-40 | 1923 | 415 | 1508 | 0 | 1508 |
| 80-100 | 3591 | 772 | 2819 | 0 | 2819 |

Zero events excluded at the `no_survivors`, `N<=1`, or `direct==0-but-N>1` stages, at any band (see
`graphdepth_wide_report.md` item 1-3 for the full breakdown).

Outputs committed alongside this record: `compute_graphdepth_decomposition_wide.py` (script, already
committed `c87ef91`), `graphdepth_wide_summary.csv`, `graphdepth_wide_events_10-15.csv`,
`graphdepth_wide_events_30-40.csv`, `graphdepth_wide_events_80-100.csv`,
`graphdepth_wide_depth_distribution.csv`, `graphdepth_wide_report.md` (this run's narrative report).
