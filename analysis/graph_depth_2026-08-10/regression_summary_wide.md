# GRAPH-DEPTH-WIDE STEP 2 regression result

Same harness as the original GRAPH-DEPTH STEP 2 regression (`run_graphdepth_regression.py` /
`compare_graphdepth_regression.py`, unmodified from commit `5461bc0`) run against the widened
logging code (`_log_leave_embeddings`, commit `1522b71`). Fixed action sequence
(`RandomState(12345)`), seeded env RNG (seed 42), 800 steps, both bands (30-40, 80-100),
`drift_logging=True` throughout for every tag, `PYTHONHASHSEED=0` set from the start (per CA-2,
the documented join-donor non-determinism the first GRAPH-DEPTH regression had to re-run to pick up).

| band | old vs new_off | old vs new_on |
|---|---|---|
| 30-40 | **0** differing cells | **0** differing cells |
| 80-100 | **0** differing cells | **0** differing cells |

**OVERALL: PASS**, first attempt (no re-run needed this time — `PYTHONHASHSEED=0` was applied from
the start, having already been established as necessary by the original GRAPH-DEPTH STEP 2 record).

- **2a (flag OFF vs pre-change code): byte-identical** — drift CSV every column, trajectory
  (obs/reward/done) exact, both bands.
- **2b (flag ON, widened `leave_embedding_logging=True`, vs pre-change code): byte-identical** —
  same result. The widened logging (every present node's embedding, not just the 2-hop subset,
  plus the new `hop_distance`/`departing_node_degree` metadata fields) does not perturb the agent's
  trajectory, the observation the policy receives, the reward, or any existing drift-log column, in
  either band tested.

Spot-checked the widened output directly (`regression_wide/30-40/le_new_on/leaveembed_new_on_30-40.jsonl`,
first record): `N=24`, `total_survivors=23`, `len(pre_embeddings)=23`, `len(post_embeddings)=23` —
full coverage confirmed (`len(pre)==N-1`, `len(post)==total_survivors`), where the superseded 2-hop
design would have logged only a handful of entries for the same event.

Raw run outputs (drift CSVs, trajectory npz, per-leg logs, leaveembed JSONL) are not committed here,
per this project's standing convention against committing bulk run data — `regression_results.csv`
is the committed artifact.
