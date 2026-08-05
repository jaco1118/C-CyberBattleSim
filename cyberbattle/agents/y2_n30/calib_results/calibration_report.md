# Y3 (Y2-pilot N=30): degree calibration [ARTIFACT]

Target: N=50 pilot's ACTUAL ACHIEVED degree (mean 12.50, SD 1.67) -- not N=50's original ~10
extrapolated target -- since that's what N=50 actually trained at, and a genuine same-degree
comparison needs to match the degree actually used, not the pre-calibration intention.

3 probes (5 graphs each, N=30, SecureBERT-only, no-split, `--num_graphs 5` explicit on the CLI
this time -- avoiding N=50's num_graphs config-vs-CLI bug from the start):

| probe | knows_neighbor_probability_range | measured degree (5 graphs) | mean | SD |
|---|---|---|---|---|
| A | [0.08, 0.25] | 13.3, 7.8, 9.9, 9.3, 10.2   | 10.11 | 1.99 |
| B | [0.12, 0.30] | 11.9, 8.8, 11.2, 9.3, 7.3   | 9.70  | 1.86 |
| C | [0.15, 0.40] | 14.6, 12.0, 14.5, 14.0, 12.3 | 13.50 | 1.26 |

Non-monotonic across A->B (B's higher p gave a LOWER mean than A) -- read as sampling noise at
n=5/probe rather than a real inverse relationship, given C (highest p) gives the highest mean and
tightest spread, consistent with the expected direction overall.

**Selected: probe C's range, `knows_neighbor_probability_range: [0.15, 0.40]`** -- mean 13.50, SD
1.26, closest of the three to the 12.5 target and the tightest spread. Used for final generation.

## Final 5-seed topology set [ARTIFACT]

| seed | degree | folder |
|---|---|---|
| 42  | 14.73 | graphs_yN30_s42_2026-08-05_21-16-13 |
| 100 | 9.63  | graphs_yN30_s100_2026-08-05_21-16-40 |
| 123 | 14.10 | graphs_yN30_s123_2026-08-05_21-17-08 |
| 200 | 13.63 | graphs_yN30_s200_2026-08-05_21-17-46 |
| 300 | 9.63  | graphs_yN30_s300_2026-08-05_21-18-13 |

**mean = 12.35, SD = 2.51.** Very close to the target (N=50's actual achieved degree, 12.50) --
within 0.15 (~1.2%). SD is wider than the calibration probe's 1.26 (final-draw variance across
only 5 topologies), but within this project's established "1-3, not wildly variable" convention.
No material drift from target -- proceeding to training as planned.

Confirmed via subdir count (1 per seed, matching `--num_graphs 1` passed explicitly on the CLI
this time) that N=50's earlier num_graphs config-vs-CLI bug did not recur.
