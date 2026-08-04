# Task Y2-pilot — does a lower fixed degree fix Task Y's convergence problem? (N=50 pilot cell)

Tests whether Task Y's N=60/N=90 instability was driven by DEGREE (~20, fixed across all three of
Y's cells) rather than raw node count. One pilot cell first (N=50, degree well below Y's ~20)
before committing to the full 20/30/40 sweep. Branch `taskY2-pilot-n50`.

**Environment note:** this task ran under a hard environment time cutoff (2350 same day). Steps
below are reported as they land, per explicit instruction, rather than held for a single final
report.

## STEP 0 — verification [ARTIFACT, all 5 points confirmed as expected]

- **0.1** Generation script: `cyberbattle/env_generation/generate_graphs.py` (reusable, argparse),
  driven by per-cell YAML configs (node count + degree-controlling `knows_neighbor_probability_range`
  both config parameters, matching Task Y's own approach exactly).
- **0.2** Proposed target degree ~10 (extrapolated from Task Y's own N=30/60/90 floor/ceiling
  data: floor≈0.15×(N-1), ceiling≈0.80×(N-1); ~10 sits clear of both N=20's ceiling (~15.2) and
  N=50's floor (~7.4)). Explicitly flagged as an extrapolation, not a calibration — confirmed
  empirically in STEP 0.5/1 below before training.
- **0.3** Convergence criterion, quoted directly from `evidence_taskY.md`: *"Convergence rule (F4,
  from `evidence_taskF4.md`): metric `train/Root owned nodes`, 50k windows, per-seed within-band
  iff |Delta%|<5%, band CONVERGED iff mean|Delta%|<5% AND >=4/5 seeds within-band."*
- **0.4** Confirmed: no changes to `step()`/`encode()`/reward path — same pattern as Y-N30-N60
  itself (new topology config + standard `train_agent.py`, zero core code changes).
- **0.5** Seeds 42/100/123/200/300 (Task Y's own set). No collision: Task Y never used N=50; new
  topology folders (`graphs_yN50_s<seed>_*`) and run names (`yN50_s<seed>_stg<k>`) are structurally
  distinct from Y's `yN30_*`/`yN60_*`/`yN90_*`.

## Degree calibration [ARTIFACT — empirical, not the extrapolated value alone]

3 candidate probability ranges probed (5 graphs each, N=50): `[0.0,0.1]`→mean 6.39 (SD 2.09),
**`[0.05,0.15]`→mean 10.46 (SD 1.78, selected)**, `[0.1,0.25]` not probed (B already hit target
cleanly; time-boxed under the environment cutoff). Full detail:
`cyberbattle/agents/y2_n50/calib_results/calibration_report.md`.

**Final 5-seed topology set** (generated at the selected range, one topology per seed — see
correction below): degree 12.44 / 11.84 / 12.52 / 15.14 / 10.58 (seeds 42/100/123/200/300), **mean
12.50, SD 1.67**. Somewhat above the calibration probe's 10.46 mean (single-draw variance), but
still clearly "well below Task Y's ~20" (37.5% lower) and with a tight spread — accepted as-is
rather than re-calibrated, given the time constraint.

**Two bugs caught and fixed before any training steps counted** (both documented in
`calibration_report.md`):
1. `generate_graphs.py`'s `--num_graphs` is a CLI flag (default 5), NOT read from the config
   YAML's `num_graphs` key — produced 5 topologies/seed instead of 1, inconsistent with Task Y's
   own one-topology-per-seed convention. Fixed: kept subdir `1` (the exact topology already
   measured above), deleted subdirs `2`-`5`.
2. The frozen GAE encoder's config files (`model_spec.yaml`, `train_config_encoder.yaml`) were
   tracked only on `attenuation-pooling-scale` (a prior commit-audit fix); this branch, created
   from `taskY-probe-n90`, didn't have them tracked, and `git checkout` deleted them from disk on
   branch creation. Restored via `git show attenuation-pooling-scale:...`, re-tracked here too.

## Training setup [ARTIFACT]

`cyberbattle/agents/y2_n50/y2_base.yaml`: cloned from `y_n30n60/y_base.yaml` (same 250k-per-stage
schedule, `dynamic_mode=none`, same aggregations/hyperparameters), `checkpoints_save_freq`
tightened to 25000 (from 50000) for extra insurance against the environment cutoff. Orchestrator
`cyberbattle/agents/y2_n50/run_stage.sh` (adapted from `y_n30n60/run_stage.sh` — same F4
checkpoint-stopping rule, same thread-cap fix, same `--finetune_model` resume mechanism; differs
only in single-cell/glob-based topology lookup).

**Stage 1 (250k, all 5 seeds) launched 2026-08-04 21:50 BST.**

## HOW TO RESUME if the session/environment cuts off mid-run [ARTIFACT — read this first]

Everything except the in-progress training run is already committed on branch `taskY2-pilot-n50`
(topology configs, seeds, `y2_base.yaml`, `run_stage.sh`, this card). `checkpoints_save_freq=25000`
(tightened from Y-N30-N60's 50000 specifically for this) means at most 25k steps per seed are ever
at risk, not the whole 250k stage.

**To resume, for each seed:**
1. Find the latest surviving checkpoint:
   `ls -t logs/yN50_s<seed>_stg1_*/TRPO_x_control_SecureBERT/checkpoints/1/checkpoint_*_steps.zip | head -1`
2. Relaunch with `--finetune_model <that path, RELATIVE TO logs/>` — e.g.
   `train_agent.py --train_config y2_n50/y2_base.yaml --name yN50_s<seed>_resume --load_envs graphs_yN50_s<seed>_<ts> --load_seeds y2_n50/seeds/seeds_<seed> --finetune_model yN50_s<seed>_stg1_<ts>/TRPO_x_control_SecureBERT/checkpoints/1/checkpoint_<N>_steps.zip`
   (same resume mechanism used throughout this project for every staged run, e.g. Y-N60's
   stage-to-stage extensions). VecNormalize rebuilds fresh on resume — documented, accepted
   transient, not a bug.
3. If a seed has no checkpoint at all, relaunch it fresh (`--finetune_model NONE`) — nothing lost
   for that seed, just a normal cold start.
4. If stage 1 fully completes before a cutoff, the F4 check + verdict is already appended below;
   proceed to stage 2 (`run_stage.sh 2 ...`) the same way Y-N30-N60 did, only if stage 1 did NOT
   converge.

**Checkpoint state as of 2026-08-04 22:02 BST** (~40% through stage 1, all 5 seeds healthy, zero
errors): seed42 75k, seed100 100k, seed123 100k, seed200 75k, seed300 75k / 250k target.

## Stage 1 result (250k) [FINDING]

**NOT CONVERGED.** mean|Delta%|=7.09% (>=5%), 2/5 within band (need >=4). seed42 +0.82% YES,
seed100 +8.12% no, seed123 -11.29% no, seed200 -2.70% YES, seed300 +12.50% no (worst miss).

Unlike Task Y's own N=30 cell (which converged cleanly at this exact same first stage), this pilot
does NOT show the same fast-convergence pattern despite its much lower degree (~12.5 vs Y's ~20).
Stage 2 (250k->500k) launched immediately per the pre-agreed staged schedule, resuming from
stage-1 checkpoints. Verdict: `y2_n50/verdicts/stage1_N50.txt` (commit `758c383`).

<!-- Further stage results appended below as they land. -->
