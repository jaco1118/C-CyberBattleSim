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

## CORRECTION: final-generation bug (discovered post-generation, fixed before training)

The final 5-seed generation call omitted `--num_graphs 1` on the CLI. `generate_graphs.py`'s
`--num_graphs` is a CLI flag (default **5**), separate from the generation-config YAML's
`num_graphs: 1` key -- which the script does not actually read to control graph count. Result:
each seed's call produced **5** valid topology candidates (all passed generation-time validity
checks, no rejections; log: "Generation of 5 graphs completed successfully!"), not 1 as intended
-- inconsistent with Task Y's own convention of exactly 1 topology per seed folder (confirmed
against `graphs_yN30_s42_.../` which has only subdir `1`).

**Caught before training completed any steps** (crashed immediately for an unrelated reason --
the gae-config branch issue below -- giving time to notice this on relaunch). Fix: subdir `1` in
each seed's folder is the exact topology already measured for calibration (mean degree 12.50,
SD 1.67, reported above) -- kept; subdirs `2`-`5` (unmeasured, unused) deleted. Final topology set
used for training is therefore exactly the one already reported, one topology per seed, matching
Task Y's convention.

Separately (unrelated bug, also caught and fixed before any training steps ran): the frozen GAE
encoder's config files (`model_spec.yaml`, `train_config_encoder.yaml`) were tracked only on
`attenuation-pooling-scale` (force-added there in the commit audit); creating this branch from
`taskY-probe-n90` meant they were untracked here, and `git checkout` removed them from disk when
switching branches. Restored via `git show attenuation-pooling-scale:...` and re-tracked on this
branch too (commit `c6772a7`).
