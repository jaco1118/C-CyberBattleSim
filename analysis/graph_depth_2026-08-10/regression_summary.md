# GRAPH-DEPTH STEP 2 regression result

Modelled on Task L's own STEP 2 regression (`run_L_regression.py`/`compare_L_regression.py`,
commit `c05a16a`). Fixed action sequence (`RandomState(12345)`), seeded env RNG (seed 42), 800
steps, both bands (30-40, 80-100), `drift_logging=True` throughout for every tag.

## First attempt (no `PYTHONHASHSEED` pinned)

| band | old vs new_off | old vs new_on |
|---|---|---|
| 30-40 | 482 differing cells | 494 differing cells |
| 80-100 | 1214 differing cells | 1214 differing cells |

Trajectory (obs/reward/done) was already byte-identical in every comparison at this stage.
Investigated before concluding anything: the differing columns were EXACTLY
`change_type, change_fired, event_id, step_fired, visibility_lag_steps, node_origin_is_join,
attenuation_ratio_{mean,max,min,full}, delta_h_v_norm` — precisely the column set `claims_audit.md`
CA-2 documents as affected by the pre-existing, un-caused-by-this-task `PYTHONHASHSEED`
join-donor-selection non-determinism (`cyberbattle_env.py:781`, donor set at `:189`). Nothing
outside that column set ever differed.

## Second attempt (`PYTHONHASHSEED=0`, per CA-2's own documented fix)

| band | old vs new_off | old vs new_on |
|---|---|---|
| 30-40 | **0** differing cells | **0** differing cells |
| 80-100 | **0** differing cells | **0** differing cells |

**OVERALL: PASS.** Both required comparisons close cleanly:
- **2a (flag OFF vs pre-change code): byte-identical** — drift CSV every column, trajectory
  (obs/reward/done) exact, both bands.
- **2b (flag ON, `leave_embedding_logging=True`, vs pre-change code): byte-identical** — same
  result. `leave_embedding_logging=True` does not perturb the agent's trajectory, the observation
  the policy receives, the reward, or any existing drift-log column, in either band tested.

This closes the exact gap the standing record flagged (only the logging-OFF path had ever been
compared against pre-instrumentation code; this task's own new flag, when ON, is now also proven
byte-identical to pre-instrumentation code, under `PYTHONHASHSEED=0`).

Raw run outputs (drift CSVs, trajectory npz, per-leg logs) are not committed here, per this
project's standing convention against committing bulk run data — `regression_results.csv`
(the second, passing attempt) is the committed artifact.
