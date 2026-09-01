# Dissertation artifact audit

Audit date: 2026-09-01 (Europe/London). This is a read-only provenance audit; no artifact was copied, moved, deleted, compressed, or committed.

## Frozen baseline

- Branch: `snr-abs-sensitivity` (tracking `myrepo/snr-abs-sensitivity`, ahead 0/behind 0).
- Commit: `8c2862ba2ba7424bb27e3ca67c5a9630a486bbde` (`Fix hardcoded path`, 2026-08-31T23:43:02+01:00).
- Worktree was not clean before this audit: `analysis/README.md` and `evidence_cards/README.md` were added to the index and modified; `analysis/uncertain_folders.zip` and `get_sizes.py` were untracked. These are pre-existing user work and are not part of the frozen commit.
- No submodules and no Git-LFS objects were found. `.git` occupies 395 MiB; the uncompressed `git archive HEAD` payload is 187,607,040 bytes (178.9 MiB). Tracked files total 186,713,983 bytes.
- Input checksums recorded before any copying: task text `26b104fe021814d0411401e1350883233e2a828e253939a1a6c663161e2aa8b9`; final dissertation text `9768133ac99d16b9a86a8f3933f2f8532a30c21c3be5eceb052d239fd8819bd7` (SHA-256).

## Final-thesis boundary

The final dissertation asks RQ1 (cost: conquest/scale, mechanical versus behavioural loss, node count versus neighbour count), RQ2 (encoder versus pooling, pooling ablation, view versus action set), and RQ3 (discovery gate, response by type, structure versus category, and loss concentration). It explicitly retires the absorbed/handled/blind classification, supersedes the first graph-depth analysis with the wide recomputation, replaces the original untraceable RQ3d figures with same-episode reranking, and treats cross-band behavioural trends as provisional because the main 15 policies do not meet the final reward-stability rule.

## Artifact classification

Sizes are actual local disk usage (du), except byte totals explicitly identified as manifest totals.

| Path | Classification | Size | Used by final thesis? | Reason | Action |
| --- | --- | ---: | --- | --- | --- |
| Git snapshot at commit above | MUST INCLUDE | 178.9 MiB payload | Yes | Source, simulator changes, configs, analysis scripts, committed CSVs, evidence cards and run records | Export the named commit, not the dirty worktree; omit `.git` |
| `analysis/` and `evidence_cards/` tracked content | MUST INCLUDE (inside snapshot) | 168.9 MiB | Yes | Final processed evidence and provenance | Do not duplicate outside `source/` |
| `cyberbattle/data/env_samples/scalability_{10_15,30_40,80_100}` selected IDs | MUST INCLUDE | included in 1.270 GiB subset below | Yes | Main 15-policy evaluation, graph depth, RQ2c, RQ3 | Include IDs 10-15 `{1,4,29,45,54,61,62,76}`, 30-40 `{3,15,19,32,44,50,64,92}`, 80-100 `{2,5,10,17,18,67,90,100}` |
| `cyberbattle/data/env_samples/join_donor_pool_20_topologies` | SHOULD INCLUDE | 249 MiB | Yes, as run input/confound | Exact donor input for main diagnostics; known same-band confound remains disclosed | Include intact |
| Final RQ1c topology folders (`graphs_yN30_*`, `graphs_yN60_*`, `graphs_yprobe_n90*`, low-neighbour `graphs_yN30_*_2026-08-05_*`) | MUST INCLUDE | included in 1.270 GiB subset | Yes | Crossed N=30/60/90 and fixed-N neighbour design | Include exact 20 folders named by `y_robustness/scripts/run_stage*.sh` |
| All selected topology folders above | MUST/SHOULD | 1,363,377,206 bytes (1.270 GiB) | Yes | Minimal topology and donor set | Preserve relative names and add checksums |
| Entire `env_samples/` | EXCLUDE | 20 GiB | No | Contains pilots, calibrations, deployment data, 200-250-node test, and superseded sets | Do not submit wholesale |
| `scrape_samples/default_data` | OPTIONAL | 553 MiB | No direct final-result dependency | Needed only to regenerate scenarios from scratch; exact generated scenarios suffice | Include only if licence/storage permits; document Shodan/NVD provenance |
| Main 15 checkpoint + VecNormalize pairs at 250k | MUST INCLUDE | 30,917,177 bytes (29.5 MiB) | Yes | Main attenuation, graph-depth, RQ2c and RQ3 runs cannot be regenerated exactly without trained policies | Include only `checkpoint_250000_steps.zip` and matching `checkpoint_vecnormalize_250000_steps.pkl`, plus each run's `train_config.yaml` |
| Final RQ1c checkpoint + VecNormalize pairs | MUST INCLUDE | about 31.5 MiB for the currently selected 16 pairs; exact list must be frozen | Yes | N=30/60/90 and low-neighbour robustness | Use paths in `y_robustness/scripts/run_stage.sh`, `run_stage_n60.sh`, and neighbour script; include final cumulative-stage checkpoint only |
| `evidence_cards/taskZ_raw/` | MUST INCLUDE (already tracked) | within snapshot (about 54 MiB final checkpoint pairs) | Yes | RQ2b ablation configs, final policies and evaluation CSVs | Retain; no second copy |
| `attenuation_drift_logs` | SHOULD INCLUDE | 618 MiB (manifest 647,334,698 bytes) | Yes | Raw basis for final response, drift, SNR and join-visibility figures | Include selected final raw dataset |
| `attenuation_analysis_output` | OPTIONAL | 5.0 MiB | Derived final diagnostics | Values can be recreated from raw logs and committed scripts | Include if convenient |
| `graphdepth_sweep_wide` | SHOULD INCLUDE | 891 MiB | Yes | Corrected RQ2a raw JSONL/embedding evidence | Include; exclude earlier `graphdepth_sweep` (126 MiB) |
| `cx_step2_registration` | SHOULD INCLUDE / SIZE-DEPENDENT | 5.8 GiB (manifest 6,135,739,168 bytes) | Yes | Raw RQ3a/b/c and 117-cell provenance | Include if submission limit permits; processed RQ3 tables are committed |
| `cx_step2_static` | SHOULD INCLUDE / SIZE-DEPENDENT | 6.4 GiB (manifest 6,760,971,422 bytes) | Yes | Matched static baseline for RQ1/RQ3 | Include with registration data if space permits |
| `rq2c_replay` final files | SHOULD INCLUDE | about 126 MiB after excluding two timestamped backup CSVs | Yes | Fresh seeded rollout for final RQ2c | Include manifests, final drift CSVs, `rq2c/`, summaries; exclude backups |
| `rq3d_data/change` plus static source | MUST INCLUDE or pin source commit | 131 MiB local change side | Yes | Same-episode final RQ3d | Package both arms from commit `3d8c9aa` on `attenuation-pooling-scale`, or include a bundle/reference; current local folder lacks the static arm |
| `y_robustness/out` | SHOULD INCLUDE / SIZE-DEPENDENT | 2.5 GiB | Yes | Raw crossed-design evaluation underlying final RQ1c table | Include selected final static/change CSVs; exclude calibration and smoke duplicates once exact rows are frozen |
| `cx_step2_replay` | OPTIONAL/HISTORICAL | 644 MiB | No final exclusive dependency | Diagnostic replay; final RQ2c used a fresh rollout | Exclude from minimal package |
| `attenuation_step3_logs` | OPTIONAL/HISTORICAL | 5.7 GiB | No final exclusive dependency | Headline distribution is superseded/covered by corrected analyses; stored actions are not faithful replay inputs | Exclude from minimal package |
| `attenuation_gate_archive` | OPTIONAL/HISTORICAL/SUPERSEDED | 621 MiB (manifest 650,551,023 bytes) | No exclusive final dependency found | Explicitly provisional, shared donor confound, membership-only; later recomputations support final claims | Do not include by default; never delete |
| `scrape_samples`, old logs, pilots, `tmp`, caches, `.claude`, virtual environments, `.git` | EXCLUDE | large | No | Regenerators, historical work, caches or machine state | Exclude |

## RQ/result provenance and completeness

Legend: Y = present; L = local/ignored; H = only another Git-history commit; P = processed only; N = absent/not established. “Checkpoint” includes VecNormalize.

| Result | Analysis code | Processed data | Raw data | Config | Topology | Checkpoint | Run record | Primary chain / status |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| RQ1a conquest + size, R²=0.338 | Y | Y | L | Y | L | L | Y | `rq1a_regression_recovered` <- CX registration/static -> main topology/checkpoint grid |
| RQ1b controlled 69-76%; 117-cell 48.2/31.9/12.9% | Y | Y | L | Y | L | L | Y | `rq1b_mech_split_scale` plus F-series/CX outputs; behavioural comparisons provisional |
| RQ1c matched N/degree nulls | Y | Y | L | Y | L | L | Y | node-count analyses + `y_robustness/out` -> final cumulative-stage policies/topologies |
| RQ2a exact two-hop cutoff and pooling gap | Y | Y | L | Y | L | L | Y | only `graph_depth.../decomposition_wide` is final; top-level first attempt excluded |
| RQ2b INFO/CAPACITY ablation | Y | Y | Y (tracked eval CSVs) | Y | Y/selected | Y (tracked Task-Z pairs) | Y | `three_lookups` + `taskZ_raw`; largest band inconclusive |
| RQ2c 72/88/95% retained action | Y | Y | L | L/Y | L | L | Y | fresh per-seed rollout in `rq2c_replay`; not forced replay of step3 actions |
| RQ3a 1.000/0.000 discovery gate; property 0.998/0.987/0.996 | Y | Y | L | Y | L | L | Y | `rq3a_gate_recompute`/CX registration; undiscovered property case untested |
| RQ3b maximum/minimum response gaps | Y | Y | L | Y | L | L | Y | `rq3b_slice_recompute` from CX; minimum-slice mechanism unresolved |
| RQ3c structure R²=.305 vs type .060 | Y | Y | L | Y | L | L | Y | `rq3c_rebuild`; raw registration/static needed for from-scratch recomputation |
| RQ3d 18.2/18.9/27.7%, overlap/ranks | Y | Y | H + partial L | Y | provenance in H | provenance in H | Y | final source is commit `3d8c9aa`; original 18/32/34 pipeline is retired |
| Main convergence 13.50/12.25/13.14%, 1/5 each | Y | Y | L training logs | Y | L | L | Y | final reward-based check; older denominator diagnostics are historical |
| Simulator correction/instrumentation validation | Y | Y | optional L | Y | L | L | Y | inherited property-visibility correction and logging checks; disclose hash-order join issue |
| Robustness-metric non-invariance | Y | Y | L | Y | L | L | Y | count-based metric is primary; ratio result is diagnostic |

## Secrets, privacy, paths, and suitability

- `cyberbattle/data_generation/config/auth.yaml` is correctly ignored and is absent. No `.env` or credential-named file was found. Do not create or package either.
- A pattern scan found token-handling code in `feature_extraction.py`, not a credential value. Re-scan the staged package before release and report paths only.
- Numerous committed scripts/run records contain machine-specific `/cs/student/...`, `claude_home`, scratch-branch, and Conda paths. They are provenance, not secrets, but prevent literal command replay; add portable path guidance rather than editing historical records.
- Generated topologies are synthetic. `scrape_samples/default_data` derives from Shodan/NVD and may carry database/licensing constraints; it is unnecessary when exact generated topologies are supplied.
- The repository README states no customer data. The deployment folders are irrelevant to this dissertation and should be excluded.

## Size summary

- Git snapshot: 187,607,040 bytes uncompressed (178.9 MiB).
- Selected topology/donor inputs: 1,363,377,206 bytes (1.270 GiB).
- Main 15 final checkpoint pairs: 30,917,177 bytes (29.5 MiB); final RQ1c pairs add roughly 31.5 MiB. Task-Z final pairs are already in the Git snapshot.
- Committed processed analysis/evidence: 177,158,433 bytes, already inside the snapshot (do not double-count).
- Selected raw final evidence if all recommended size-dependent groups are kept: approximately 17.0 GiB (`attenuation_drift_logs`, wide graph depth, CX registration/static, de-duplicated RQ2c, RQ3d, and `y_robustness/out`).
- Recommended package: approximately 18.5 GiB including source, selected topologies, checkpoints and selected raw evidence. A compact audit-first package omitting CX raw and `y_robustness/out` is about 3.2 GiB but cannot recompute RQ1/RQ3c/RQ1c from raw rows.
- All optional manifest raw artifacts add 21,104,488,898 bytes (19.65 GiB); optional scrape data adds 553 MiB and the gate archive is already part of that manifest total. Do not add the 142 GiB whole logs tree or the 20 GiB whole topology tree.
