# Run record — Item 1: forced-replay extremal figures

Date: 2026-08-11.

## Command
```
cd analysis/provenance_commit_2026-08-11
/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/bin/python compute_replay_extremes.py
```

## Input paths (read-only)
- `cyberbattle/agents/cx_step2_replay/probe/probe_<band>_seed<seed>_<scenario>.jsonl` (117 files).
- `cyberbattle/agents/cx_step2_replay/drift_<band>.csv` (the authorised second source for quantity B, per Correction 2).

## RNG
None — deterministic read/join/count over already-existing files.

## Row counts in/out at every stage

| band | probe leave records | batch (n_touched>1) | single-node | discovered=1 | drift rows | join matched | join unmatched |
|---|---|---|---|---|---|---|---|
| 10-15 | 7,042 | 574 | 6,468 | 2,240 | 169,702 | 7,042 | 0 |
| 30-40 | 27,066 | 2,537 | 24,529 | 6,941 | 639,556 | 27,066 | 0 |
| 80-100 | 32,253 | 3,050 | 29,203 | 10,872 | 827,558 | 32,253 | 0 |

Quantity B, NaN-excluded counts (of events where the departing node held the relevant extreme): max slice 54/123/91, min slice 52/106/55, per band.

## Outputs (committed alongside this record)
`compute_replay_extremes.py`, `item1_quantity_A.csv`, `item1_quantity_B.csv`, `item1_join_diagnostics.csv`, `item1_run_output.log` (full console output).

## Headline result
`A_discovered` restricted to `departing_held_max` alone (not the max/min union) reproduces the thesis's 0.89/0.61/0.34 triple almost exactly: 0.886/0.609/0.344 (diffs +0.004/−0.001/+0.004). Quantity B lands at 0.995–1.000 at every band, both slices, confirming "approximately 1.000." Batch-event inclusion changes A by ≤0.003 everywhere (not material).

## Wipe test
Every number reproducible from the committed script and the already-on-disk `cx_step2_replay/` data (itself not committed — bulk run data, per this project's standing convention; the script and its output are).
