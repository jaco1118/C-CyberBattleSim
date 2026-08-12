# Run record — Task FLAG-GROUND-TRUTH

Date: 2026-08-12.

## Command
```
cd analysis/flag_ground_truth_2026-08-12
/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/bin/python compute_flag_ground_truth.py
```

## Input paths (read-only)
`cyberbattle/agents/{attenuation_drift_logs,cx_step2_registration,graphdepth_sweep_wide,rq2c_replay}/drift_<band>.csv`.
`cyberbattle/_env/cyberbattle_env.py` and `compute_attenuation_analysis.py` (source, read-only, STEP 0).
Raw session transcripts `~/.claude/projects/*/*.jsonl` (STEP 2, launch-command recovery).

## RNG
None — deterministic read/filter/aggregate over already-existing files.

## Row counts, every stage, every dataset/band (membership_leave events only)

| dataset | band | n_rows_read | n_dropped_missing_col | n_rows_used | n_visible | n_not_visible | unknown_fraction |
|---|---|---|---|---|---|---|---|
| attenuation_drift_logs | 10-15 | 5,321 | 0 | 5,321 | 5,321 | 0 | 0.0000 |
| attenuation_drift_logs | 30-40 | 21,445 | 0 | 21,445 | 21,445 | 0 | 0.0000 |
| attenuation_drift_logs | 80-100 | 28,088 | 0 | 28,088 | 28,088 | 0 | 0.0000 |
| attenuation_drift_logs | POOLED | 54,854 | — | 54,854 | 54,854 | 0 | 0.0000 |
| cx_step2_registration | 10-15 | 12,012 | 0 | 12,012 | 2,131 | 9,881 | 0.8226 |
| cx_step2_registration | 30-40 | 49,295 | 0 | 49,295 | 6,479 | 42,816 | 0.8686 |
| cx_step2_registration | 80-100 | 56,035 | 0 | 56,035 | 10,174 | 45,861 | 0.8184 |
| cx_step2_registration | POOLED | 117,342 | — | 117,342 | 18,784 | 98,558 | 0.8399 |
| graphdepth_sweep_wide | 10-15 | 1,399 | 0 | 1,399 | 1,399 | 0 | 0.0000 |
| graphdepth_sweep_wide | 30-40 | 1,672 | 0 | 1,672 | 1,672 | 0 | 0.0000 |
| graphdepth_sweep_wide | 80-100 | 3,127 | 0 | 3,127 | 3,127 | 0 | 0.0000 |
| graphdepth_sweep_wide | POOLED | 6,198 | — | 6,198 | 6,198 | 0 | 0.0000 |
| rq2c_replay | 10-15 | 5,101 | 0 | 5,101 | 5,101 | 0 | 0.0000 |
| rq2c_replay | 30-40 | 1,651 | 0 | 1,651 | 1,651 | 0 | 0.0000 |
| rq2c_replay | 80-100 | 2,962 | 0 | 2,962 | 2,962 | 0 | 0.0000 |
| rq2c_replay | POOLED | 9,714 | — | 9,714 | 9,714 | 0 | 0.0000 |

Zero rows dropped for missing `touched_node_visible` at every cell, every dataset — stated explicitly.
`n_cells_undefined` (no events) = 0 — every cell had events.

## Escape-route check (n_discovered_h2 / n_scenario at leave-event time)

| dataset | band | median | p5 | p95 |
|---|---|---|---|---|
| attenuation_drift_logs | 10-15 | 0.700 | 0.400 | 1.000 |
| attenuation_drift_logs | 30-40 | 0.735 | 0.438 | 0.917 |
| attenuation_drift_logs | 80-100 | 0.859 | 0.511 | 0.963 |
| graphdepth_sweep_wide | 10-15 | 0.700 | 0.400 | 1.000 |
| graphdepth_sweep_wide | 30-40 | 0.735 | 0.397 | 0.938 |
| graphdepth_sweep_wide | 80-100 | 0.857 | 0.475 | 0.953 |
| rq2c_replay | 10-15 | 0.700 | 0.400 | 1.000 |
| rq2c_replay | 30-40 | 0.733 | 0.444 | 0.917 |
| rq2c_replay | 80-100 | 0.859 | 0.513 | 0.964 |
| cx_step2_registration | 10-15 | 0.800 | 0.500 | 1.000 |
| cx_step2_registration | 30-40 | 0.833 | 0.212 | 0.968 |
| cx_step2_registration | 80-100 | 0.909 | 0.474 | 0.976 |

Median discovered-fraction is 70-86% across the disputed and reference datasets — never near 100%
at the median. The escape route (zero unknown-fraction being a trivial consequence of near-complete
discovery) does NOT apply: most leave events fired while a substantial minority (14-30% at median)
of the topology was still undiscovered, so a nonzero unknown-fraction had ample opportunity to
appear if the flag were ON, and it never did in `graphdepth_sweep_wide`/`rq2c_replay`.

## Headline result

`graphdepth_sweep_wide` and `rq2c_replay` reproduce `attenuation_drift_logs`'s (independently,
non-metadata, pre-flag-existence-pinned) flag-OFF signature EXACTLY: 0/6,198 and 0/9,714 unknown
across all three bands, matching `attenuation_drift_logs`'s 0/54,854 to the same zero. Their
escape-route profiles are near-identical to `attenuation_drift_logs`'s band-by-band (e.g. 0.700
median at 10-15 in all three), consistent with all three having run the same (non-relaxed)
configuration. `cx_step2_registration` shows a massively different signature (0.82-0.87 unknown
fraction) despite a similar discovered-fraction profile — ruling out "similar discovery dynamics"
as an explanation for the contrast and confirming the unknown-fraction statistic is doing real
discriminating work, not just reflecting exploration speed.

## Outputs (committed alongside this record)
`compute_flag_ground_truth.py`, `flag_ground_truth_signature.csv`, `flag_ground_truth_run_output.log`.

## Wipe test
Reproducible from the committed script and the already-on-disk drift CSVs for all four datasets
(raw data itself not committed, per this project's standing convention; script and output are).
