# Run record — N=60 robustness column, STEP 2 (main evaluation)

PROVENANCE BANNER: commissioned 2026-08-15, after the convergence criterion changed
from root-owned-node-count to episode reward (independent decision), which made the
N=60 cell newly eligible for Table IV.5. Nothing about this result was known or
estimated beforehand.

Date run: 2026-08-16. Script: `run_stage_n60.sh`, `CI_N60=5` (STEP 1's accepted
value). taskF2_eval.py unmodified.

## Design
5 seeds x {static, membership_matched} x 200 episodes = 2000 episodes, all 10 runs
launched concurrently (same design as the N=30/N=90 columns). Metric convention per
the task spec: robustness = mean(disturbed root_owned) / mean(matched undisturbed
root_owned), RAW count (not the root_owned/reachable ratio), paired within seed.

## Per-seed results
| seed | static root_owned mean | matched root_owned mean | robustness |
|---|---|---|---|
| 42  | 26.9300 | 16.7200 | 0.6209 |
| 100 | 28.3250 | 16.1200 | 0.5691 |
| 123 | 30.9350 | 18.1550 | 0.5869 |
| 200 | 29.8100 | 18.0400 | 0.6052 |
| 300 | 28.0650 | 17.0200 | 0.6064 |

Zero-undisturbed-mean seeds: 0 (no undefined ratios).

## Cell level
Mean robustness = 0.5977. SD across seeds = 0.0200.

## Episode counts
Launched: 2000 (5 seeds x 2 conditions x 200). Completed: 2000. Dropped: 0. Nothing
dropped at any seed/condition -- stated per the task's own instruction that this is a
result worth stating explicitly, not an assumed default.

## Achieved churn fraction, main run vs calibration
| seed | leave/ep (200-ep main run) | churn % of 60 |
|---|---|---|
| 42  | 27.58 | 45.97% |
| 100 | 27.94 | 46.57% |
| 123 | 28.89 | 48.15% |
| 200 | 28.33 | 47.22% |
| 300 | 28.79 | 47.98% |

Pooled mean leave/ep = 28.306 -> **churn = 47.18%**.

**This does not match STEP 1's calibration result.** STEP 1 (seed 42 only, 30
episodes, ci=5) measured 42.17% -- inside the declared 40.5-43.5% band, 0.17pp from
the 42.0% reference. The 200-episode main run at the same ci=5, same seed 42, reads
45.97% -- a swing of +3.80pp for that one seed alone between the 30-episode
calibration probe and the 200-episode main run. Pooled across all 5 seeds the main
run reads 47.18%, +5.18pp above STEP 1's calibration figure and +5.18pp above the
42.0% reference, OUTSIDE the +/-1.5pp tolerance band that was declared in advance for
accepting a calibration value. This mirrors the direction of the smaller, already-
disclosed gap between N=30's 30-episode probe (41.0%) and its own 200-episode full
run (42.0%) -- but here the gap is roughly five times larger. Reported as measured,
without re-calibrating or re-running: the task instruction was to run STEP 2 at the
STEP-1-accepted ci=5 and report the achieved churn fraction from the main run
alongside the calibration figure, which is what this section does. No conclusion is
drawn about which figure -- 42.17% or 47.18% -- should be treated as this column's
"true" churn rate, and no action is recommended.

## Wall-clock and orchestrator
Launched 2026-08-16 11:05:35 BST (all 10 processes backgrounded together). Fastest
completion 11:27:10 (21m35s after launch); slowest 11:37:03 (31m28s after launch).
Orchestrator: `[wait] done; failed=0` -- all 10 PIDs exited 0 individually (per-PID
exit logging, the one authorized addition to `run_stage_n60.sh`). No failures of any
kind reported, unlike the original N=30/N=90 batch's disclosed but unresolved
"failed=4".

## Outputs (committed alongside this record)
`score_static_seed<seed>_eval{static,membership_matched}.csv` and
`leaveown_static_seed<seed>_evalmembership_matched.csv` for all 5 seeds (10 + 5 = 15
files), plus the 10 `logs_run/n60_*.log` files and `run_orchestrator_n60.log`. NOT
committed: `drift_*.csv` (per-step logs, consistent with this directory's existing
bulk-artifact convention).

## Wipe test
Reproducible from `run_stage_n60.sh`, `CI_N60=5`, and the already-on-disk
checkpoints/topologies (not committed, per convention) -- modulo the stochastic
churn discrepancy documented above, which is itself part of what a re-run would need
to be checked against, not assumed away.
