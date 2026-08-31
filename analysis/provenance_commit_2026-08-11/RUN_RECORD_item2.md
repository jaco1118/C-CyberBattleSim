# Run record — Item 2: RQ2(c) counts

Date: 2026-08-11.

## Command
```
cd analysis/provenance_commit_2026-08-11
/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/bin/python compute_rq2c_counts.py
```

## Input paths (read-only)
- `cyberbattle/agents/rq2c_replay/drift_<band>.csv`.
- `cyberbattle/agents/rq2c_replay/rq2c/rq2c_<band>_seed<seed>_<scenario>.jsonl` (87 files total: 38 at 10-15, 23 at 30-40, 26 at 80-100).

## RNG
None.

## Bug found and fixed before any result was reported (see script comment and commit `489b7e9`)
First run keyed episode clusters on `(seed, scenario, episode)` — a 3-tuple — giving 730/148/208
against the target 173/49/66 (~4x too high). The actual producing code
(`compute_rq2c_action_divergence.py:154`) keys on `(seed, episode)` only. Fixed; both keys are
computed and reported in the committed output so the divergence is on the record, not silently
dropped.

## Row counts

| band | total membership_leave | single-node | batch | JSONL files | files contributing 0 new (seed,episode) keys | group_i | group_ii | episode clusters (2-tuple) |
|---|---|---|---|---|---|---|---|---|
| 10-15 | 5,101 | 4,965 | 136 | 38 | 28 | 1,176 | 3,789 | 173 |
| 30-40 | 1,651 | 1,483 | 168 | 23 | 13 | 101 | 1,382 | 49 |
| 80-100 | 2,962 | 2,708 | 254 | 26 | 16 | 27 | 2,681 | 66 |

"Files contributing 0 new keys" does not mean those files carry no data — it means every
`(seed, episode)` pair they contain was already introduced by an earlier-processed file for the
same seed (episode numbers restart per scenario, so this is expected under the 2-tuple key, not a
data-loss signal).

## Outputs (committed alongside this record)
`compute_rq2c_counts.py`, `item2_rq2c_counts.csv`, `item2_run_output.log`.

## Headline result
All six target numbers reproduce exactly under the corrected `(seed, episode)` key: 5,101/1,651/2,962 (total), 4,965/1,483/2,708 (single-node), 136/168/254 (batch), 173/49/66 (episode clusters) — every diff is 0.

## Wipe test
Reproducible from the committed script and the already-on-disk `rq2c_replay/` data (raw data itself not committed, per convention; script and output are).
