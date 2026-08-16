This directory holds the checkpoints and evaluation data behind the CAPACITY/RAW/750k
three-arm pooling-ablation figures for the 30-40 and 80-100 bands (evidence_cards/evidence_taskZ.md).

Originally produced on an unmerged side branch: training/harness code in commits
0f3bdf0 and b72e3c5 (introduce taskZ_train.py, the ExtremalMask wrapper, and the
--extremal_mask flag). Neither commit is an ancestor of this repository's mainline HEAD
at the time this directory was added -- confirmed via
`git merge-base --is-ancestor 0f3bdf0 HEAD` / `b72e3c5 HEAD` (both false).

The scripts that read/analyse this data (taskZ_eval.py, z_step2_analyze.py) were
separately preserved from job-scratch in commit c05a16a ("Project-wide audit: preserve
85 at-risk scripts from job scratch"), which is ALSO not an ancestor of this branch's
HEAD. That commit's own message states the bulk output data (the CSVs and checkpoints
in this directory) was deliberately not copied to git at the time, "matches the
existing repo convention of keeping bulk artifacts off git" -- leaving this data
sitting only in an ephemeral job-scratch workspace
($CLAUDE_JOB_DIR/tmp/taskF1/{z_runs,z_step2_out,z750_runs,z750_eval}) with no git
history at all, the same failure mode that commit's own message says has already
destroyed other manifests once (F3, the original CX pipeline).

This directory is that recovery: the same raw data and scripts, copied byte-for-byte
(diff -rq confirmed identical) from job-scratch into this repository as new additions
on the current branch -- not a merge or cherry-pick of the old side-branch commits.

Recovered and reproduced exactly on 2026-08-16 (see git log for the exact commit(s)
that added this directory). The reproduction (script z750_analyze.py in scripts/,
written this session; z_step2_analyze.py's own output re-run against the CSVs here)
matched evidence_taskZ.md's originally-reported summary numbers exactly, at every
figure checked, including the one confidence interval that excludes zero (80-100
CAPACITY, fixed-relative, +1.068 [+0.356,+1.780], 5 seeds) and its non-replication at
750k (-0.320 [-1.867,+1.237]).

Contents:
  z_step2_out/          -- per-seed eval CSVs (static + fixed-rel + fixed-abs
                            membership conditions), 30-40 and 80-100, arms 1/2/3,
                            5 seeds each, 200 episodes/seed. 2.5 MB.
  z750_eval/             -- same, for the 750k-budget robustness re-run at 80-100
                            only (arms 1/2/3, 5 seeds). 1.6 MB.
  scripts/z_step2_analyze.py  -- original STEP 2 analysis (MDE + paired bootstrap
                            INFO/CAPACITY/RAW contrasts), reads z_step2_out/.
  scripts/z750_analyze.py     -- same method, adapted this session to read
                            z750_eval/ (750k re-run has no band loop -- 80-100 only).
  scripts/taskZ_eval.py, scripts/taskZ_train.py, scripts/launch_z_step2.sh
                          -- the original training/eval harness and launcher.

z_runs/ (114 MB) and z750_runs/ (57 MB) -- the actual trained model checkpoints
(.zip stable-baselines3 archives + VecNormalize .pkl files) -- were added in a
SEPARATE, later commit (both were individually under the ~200MB size-gate used to
decide this without stopping for confirmation; see git log for this directory for
the exact commit hash). If for any reason that second commit is missing, the eval
CSVs and the numbers derived from them are still safe in git, but the underlying
trained checkpoints themselves are not, and remain job-scratch-only.
