# Run record — Task METRIC-DEFINITIONS, Q2 quantitative check

Date: 2026-08-12.

## Command
```
cd analysis/metric_definitions_2026-08-12
/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/bin/python compute_snr_factor_check.py
```

## Input paths (read-only)
`cyberbattle/agents/attenuation_drift_logs/drift_<band>.csv` — confirmed as the "live 5-seed
grid" / reported population by matching row counts exactly against `evidence_taskT.md`'s Gate
row counts table (187,647 / 640,230 / 775,122 raw rows at 10-15/30-40/80-100).

## RNG
None — deterministic read/filter/aggregate.

## Row counts (after the same filters the pipeline applies before computing `snr`)
| band | n rows (membership_leave, visible, immediate/attributed phase, non-zero noise floor) |
|---|---|
| 10-15 | 1,612 |
| 30-40 | 6,764 |
| 80-100 | 10,282 |

## Headline result
`norm(h1)/norm(h2)`, on the population feeding the published SNR statistics:

| band | median | min | max | p5 | p95 |
|---|---|---|---|---|---|
| 10-15 | 0.9761 | 0.3896 | 1.7372 | 0.7790 | 1.0472 |
| 30-40 | 0.9945 | 0.3621 | 1.2369 | 0.8939 | 1.0116 |
| 80-100 | 0.9984 | 0.3941 | 1.1926 | 0.9151 | 1.0065 |
| pooled | 0.9968 | 0.3621 | 1.7372 | 0.8901 | 1.0101 |

Median is close to 1 (within 2.4% at the smallest band, within 0.2%/0.3% at 30-40/80-100) — the
central-tendency effect on the SNR *median* statistic is modest. The range is not: individual
events depart from 1 by up to ±60-75% (min 0.362, max 1.737, pooled). This is a distributional
statement about `norm(h1)/norm(h2)` itself, computed pooled per band (and pooled across bands) —
it is not restricted to the specific "near n_discovered=100" subset the published 0.492 LEVEL
figure bootstraps over, so it should be read as a general characterisation of the factor's size,
not a re-derivation of 0.492 itself.

Also reported: the code-as-written SNR (`change_drift_full/agent_drift_full`) vs the
"movement-ratio" reading with no `norm(h1)/norm(h2)` factor (`snr_code / (norm_h1/norm_h2)`,
algebraically `= norm(h3-h2)/norm(h2-h1)`), per band and pooled — see `snr_factor_check.csv`.

## Outputs (committed alongside this record)
`compute_snr_factor_check.py`, `snr_factor_check.csv`, `snr_factor_run_output.log`.

## Wipe test
Reproducible from the committed script and the already-on-disk `attenuation_drift_logs/` data
(raw data itself not committed, per this project's standing convention; script and output are).
