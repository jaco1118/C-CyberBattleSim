# Y2-pilot-N50: degree calibration [ARTIFACT]

Empirical calibrate-then-verify pass (per Task Y's own methodology; STEP 0.2's degree~10 target
was an extrapolation, not a calibration -- this confirms it before committing to training).

3 probes (5 graphs each, N=50, SecureBERT-only, no-split, `--no_split -v 1`), measuring knows-graph
out-degree per generated candidate:

| probe | knows_neighbor_probability_range | measured degree (5 graphs) | mean | SD |
|---|---|---|---|---|
| A | [0.0, 0.1]  | 5.1, 4.5, 9.5, 7.6, 5.2   | 6.39  | 2.09 |
| B | [0.05, 0.15] | 11.1, 8.8, 9.8, 13.2, 9.3 | 10.46 | 1.78 |
| C | [0.1, 0.25]  | not measured (probe timed out under time pressure; B already hit target cleanly) |  |  |

**Selected: probe B's range, `knows_neighbor_probability_range: [0.05, 0.15]`** -- mean 10.46, SD
1.78, close to the target ~10 (STEP 0.2) with a tight spread (well within the "SD roughly 1-3, not
wildly variable" requirement). Matches Task Y's own experience of needing ~2 calibration
iterations, not one.

Used for the final 5-seed generation (`generation_configs/genconfig_yN50_s{42,100,123,200,300}.yaml`).
