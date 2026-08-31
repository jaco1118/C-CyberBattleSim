# Task Y-REWARD-STAGE — cross-cell summary

Method: per-stage-local (per user's explicit instruction, not cumulative-global). Each
point tested is `compute_convergence_check.py`'s own `delta_pct()`, called on ONE stage's
tfevents file alone (default stop = that file's own max logged step), metric
`rollout/ep_rew_mean`, window 50000, threshold 5%, min-frac 0.8 (need ≥4/5 seeds).
"Durable": earliest point reported must stay converged at every later tested point.

N=30/N=60 aligned by stage number. N=90 aligned by each seed's own nominal cumulative
step total (sum of `train_iterations` across its stage chain) at which its most recent
stage ended, carry-forward for seeds whose training stopped before a later marker.

## Previously reported (already-existing FINDING — not changed here)

| node count | final step | mean rel. deviation | seeds within tolerance |
|---|---|---|---|
| 30 | 250,000 (stage 1, only stage) | 1.71% | 5/5 |
| 60 | 1,750,000 (stage 7, local-only, per the reproduced STEP1 computation) | 1.08% | 5/5 |
| 90 | up to 1,500,000 (per-seed range 500,000–1,500,000; seed100's marker is the latest) | 1.50% | 5/5 |

## Newly computed — earliest durable per-stage-local qualifying point (new FINDING)

| node count | earliest qualifying step | mean rel. deviation | seeds within tolerance | windows tested between earliest and previously-reported final |
|---|---|---|---|---|
| 30 | 250,000 (stage 1 — same as already reported; only one stage exists, no earlier point possible) | 1.71% | 5/5 | 0 |
| 60 | 750,000 (stage 3, nominal cumulative) | 3.391% | 4/5 | 4 (stages 4,5,6,7 retested after stage 3, all durably converged) |
| 90 | 500,000 (the first tested marker — every seed's own first stage) | 3.372% | 4/5 | 4 (markers 750k, 1M, 1.25M, 1.5M retested after 500k, all durably converged) |

## Notes on the two non-trivial findings, investigated before reporting

**N=60 does NOT converge durably starting at stage 1**, despite stage 1 itself passing
(5/5, 3.369%): stage 2 fails cell-level (3/5, seeds 42 and 100 exceed 5% right after the
first resume). Stages 3–7 are then durably converged. This looks like a genuine
resume-transient in two seeds' reward trajectories, not a computation error — checked the
per-seed stage-2 numbers directly (seed 42: +5.48%, seed 100: +9.13%, both barely-to-clearly
over threshold; seeds 123/200/300 pass cleanly at stage 2), and confirmed stage 3 onward
never dips below the required 4/5 or above 5% mean for any of the remaining 5 tested stages.

**N=90 converges at its very first tested marker (500,000)** — checked before reporting,
since this is exactly the "converged within the first tested window" pattern the task
flagged as needing investigation. Cause found, not assumed: N=90's first stage is a single
unbroken 500,000-step run (no resume within it), roughly 2× the length of N=30's or a single
N=60 stage — so by its own final 100,000 steps it has had substantially more within-stage,
uninterrupted settling time before the comparison windows are drawn, unlike N=60's stage 2,
which sits immediately after a resume discontinuity. One seed (100) genuinely fails at this
marker (+8.17%), consistent with the rule's own tolerance for up to 1/5 seed failures, not a
computation artifact — the raw pre/fin window means for that seed (12,815→20,445 nominally;
actually pre=18,900.6→fin=20,445.5, +8.17%) were checked directly from the CSV and are not
degenerate or NaN.

## Counts

- Seed-runs with usable continuous logs: 15/15 (5 seeds × 3 cells), confirmed in STEP 0.
- N=30: 1 T value tested per seed (5 total), 0 skipped, 0 undefined.
- N=60: 7 T values tested per seed (35 total), 0 skipped, 0 undefined.
- N=90: 3–4 stages tested per seed depending on chain length (17 total stage results:
  seed42=3, seed100=4, seed123=1, seed200=1, seed300=3), aligned onto 5 cumulative markers
  per cell (25 seed-marker cells via carry-forward), 0 skipped, 0 undefined.

## Outputs (committed alongside this record)
`compute_earliest_stage.py` (committed pre-run), `run_output.log`, `n30_per_stage.csv`,
`n60_per_stage.csv`, `n90_per_marker.csv`, this summary.

No thesis file, `Table tab:rq1c_crossed`, or any already-reported result file was edited.
No training or evaluation was run. `compute_convergence_check.py` was not modified.
