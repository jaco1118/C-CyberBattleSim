# Missing or risky artifact findings

## Blocking before packaging

1. **The working tree is not a frozen submission state.** Two README files have staged and unstaged content, and two unrelated files are untracked. Decide whether the README additions belong in a new submission commit. Package only a named commit.
2. **Final RQ3d is not self-contained at HEAD.** Its analysis reads both arms from commit `3d8c9aa` on branch `attenuation-pooling-scale`; the local `rq3d_data` directory contains only `change`. Preserve both `change` and `static`, plus their commit identity/checksums, or include a Git bundle containing the source commit.
3. **No final package allow-list/checksum manifest exists yet.** `evidence_cards/artifact_manifest.tsv` is valuable but covers a historical 21.104 GB set, including superseded/optional groups, and does not cover all selected topologies and checkpoint pairs. Generate a new manifest after the user approves this plan.
4. **The exact final RQ1c checkpoint allow-list must be frozen.** The cumulative training is split across staged run folders; include the checkpoint at the final cumulative stage named by the final scripts, not every 5k intermediate file. Verify all pairs and record hashes before copying.

## Reproducibility risks and limitations

- Git alone lacks final raw data, exact generated topologies, and most trained checkpoints; it is not sufficient for reliable from-scratch reproduction.
- `rq3c_rebuild` explicitly says that wiping local CX registration/static data prevents its raw recomputation. The committed 117-cell CSVs permit auditing the regression, not rerunning the experiment.
- RQ2c is a fresh seeded rollout. Stored `attenuation_step3_logs/actions_*.npy` do not faithfully replay the original sweep and must not be presented as replay provenance.
- Join donor choice was not hash-pinned in the original gate; join drift/attenuation columns are not bit-reproducible. Headline leave/property results are not dependent on those join columns.
- The shared donor pool weakens large-band join perturbations; this known confound remains a limitation. The frozen gate archive is provisional and not exclusive final evidence.
- Main 15 policies do not meet the final reward-stability rule (only 1/5 seeds per band within tolerance). Cross-band behavioural trends are therefore provisional; arithmetic/two-hop architectural results are less affected.
- RQ1c nulls are underpowered and are absence of evidence, not evidence of no effect. Degree matching at N=90 is approximate.
- RQ2b at 80-100 is inconclusive; do not package it with a stronger claim.
- Property change on undiscovered nodes and connectivity change were not tested. The minimum-slice RQ3b mechanism and final RQ3d structural signature remain unresolved.
- The old RQ3d 18/32/34 figures and original structural-signature pipeline are superseded/unrecoverable; only the final 18.2/18.9/27.7 same-episode analysis should be labelled current.
- Several historical scripts contain absolute `/cs/student/...`, scratch and Conda paths. Preserve them as provenance but provide portable invocation wrappers/instructions in the artifact README.

## Security, privacy and licensing

- `auth.yaml` is ignored and absent; no `.env` or credential-named file was found. Never include credentials, and repeat a value-redacting secret scan on the final staging directory.
- Token-loading code exists but no embedded credential was identified by the audit pattern scan.
- Raw Shodan/NVD-derived `scrape_samples` is not required for reported results if exact scenarios are supplied. Treat it as optional pending licence/redistribution review.
- Deployment/emulated datasets are outside the final dissertation evidence chain and should be excluded, avoiding unnecessary privacy questions.

## Size risk

- The scientifically strongest selected package is about 18.5 GiB. The largest discretionary pair is CX registration/static (~12.6 GiB); `y_robustness/out` is another 2.5 GiB.
- If the university limit is below this, use a two-tier deposit: a ~3.2 GiB core package plus a checksummed external/raw-data deposit. Do not silently omit large raw data referenced by the provenance tables.
