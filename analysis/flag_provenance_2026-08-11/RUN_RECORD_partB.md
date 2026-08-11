# Run record — Part B: does allow_undiscovered_removal explain the zero-degree finding

Date: 2026-08-11.

## Command
```
cd analysis/flag_provenance_2026-08-11
/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/bin/python compute_zero_degree_check.py
```

## Input paths (read-only)
- `cyberbattle/agents/graphdepth_sweep_wide/drift_<band>.csv` (independent, complete per-event log; not filtered by the leave-embedding logger's own guard).
- `cyberbattle/agents/graphdepth_sweep_wide/leaveembed_<band>/*/*.jsonl` (the leave-embedding log itself).
- `analysis/graph_depth_2026-08-10/decomposition_wide/graphdepth_wide_events_<band>.csv` (already-committed GRAPH-DEPTH-WIDE STEP 4 output, for the B4 cross-check).

## RNG
None.

## Row counts in/out at every stage

| band | drift CSV single-node leaves (total) | touched_node_visible=True | =False | =null | leaveembed logged | dropped by the guard |
|---|---|---|---|---|---|---|
| 10-15 | 1,361 | 1,361 | 0 | 0 | 1,361 | 0 |
| 30-40 | 1,508 | 1,508 | 0 | 0 | 1,508 | 0 |
| 80-100 | 2,819 | 2,819 | 0 | 0 | 2,819 | 0 |

Zero events dropped, at every band — the independent drift-CSV count matches the leaveembed log's own count exactly.

## Outputs (committed alongside this record)
`compute_zero_degree_check.py`, `partB_2x2_crosstab.csv`, `partB_b3_fraction.csv`, `partB_b4_medians.csv`, `partB_run_output.log`.

## Headline result
**The competing explanation is refuted, not confirmed.** Every single one of the 1,361/1,508/2,819 single-node membership_leave events in `graphdepth_sweep_wide` has `touched_node_visible==True` — none target an undiscovered node. B3: 0/617, 0/762, 0/2,196 zero-degree events had the departing node absent from the encoder's graph — the zero-degree population is genuinely discovered-but-structurally-isolated nodes, not undiscovered nodes registering as degree-0-by-absence. B4's restricted medians (0.7436/1.3575/1.2654) reproduce the already-published degree>0 figures (0.744/1.358/1.265) almost exactly (diff ≤0.0005), confirming no double-counting or population mismatch in the original GRAPH-DEPTH-WIDE analysis.

**Side finding, disclosed but not resolved further (outside this task's scope):** `graphdepth_sweep_wide` and `cx_step2_registration` use the identical checkpoint, identical `allow_undiscovered_removal`/`uncapped_join`/`patch_service_dynamic_enabled` flags (verified by diffing their `run_metadata_*.json`, which differ in only two fields: `vecnormalize_copied_to` and `code_commit`), yet `cx_step2_registration`'s own gate-counts data shows only 17.7/13.1/18.2% `touched_node_visible==True` for leave events while `graphdepth_sweep_wide` shows 100%. `graphdepth_sweep_wide`'s own `sweep_run.log` shows exactly 0 property events firing throughout (despite `patch_service_dynamic_enabled=True` in its metadata), while `cx_step2_registration`'s log shows property events firing in the hundreds — a real, unexplained behavioural difference between the two sweeps that does not trace to any flag difference found in this task. Both sweeps were launched with `RQ2C=1`, and the code commits differ (`8317571` vs `13bb2ee`), but no diff in the eligibility/discovery code between those commits was found. This does not affect the Part B answer above, which is independently and directly confirmed from `graphdepth_sweep_wide`'s own data, but it is flagged for whoever next touches this rather than left silent.

## Wipe test
Reproducible from the committed script, the already-committed `graphdepth_wide_events_<band>.csv`, and the already-on-disk `graphdepth_sweep_wide/` data (raw sweep output itself not committed, per this project's standing convention; script and output are).
