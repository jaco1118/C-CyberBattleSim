# Run record — Task Y-EARLYCKPT STEP 1: robustness at the earliest durably-converged checkpoints

Date: 2026-08-20.

## Design
N=60 at stage 3 (local checkpoint 250000 = cumulative 750,000, the earliest durably-converged
reward point per Task Y-REWARD-STAGE), ci=7 (the change_interval already backing the
currently-reported N=60 figures, confirmed before running). N=90 at its static500k stage
(500,000 steps, single unbroken run, also its earliest durably-converged point), ci=4 (already
backing the currently-reported N=90 figures, confirmed before running). Same harness
(`taskF2_eval.py`, unmodified), same 200 static + 200 membership_matched episodes/seed, same
topologies. N=30 and the lower-neighbour N=30 cell untouched.

## Orchestrator
`run_stage_earlyckpt.sh`, 20 processes, launched together. `failed=0`, all 20 PIDs exit=0,
0 Tracebacks in any of the 20 logs, 200/200 episodes at every seed/condition.

## Investigation note (before reporting numbers, per the task's own instruction)
Seeds 123 and 200 never trained past static500k, so their "early checkpoint" IS the exact
same checkpoint already backing the currently-reported N=90 figures. Their new results
reproduced the currently-committed `y_robustness/out/n90/` values EXACTLY (static/matched
means bit-identical) -- confirmed directly from source, not assumed. This is expected:
`taskF2_eval.py:92-93` seeds `numpy`/`torch`/`random` deterministically from the seed argument,
so the same checkpoint + seed + topology + condition reproduces the identical 200 episodes.
This exact match is a clean validation that the new pipeline is computing correctly, not a
coincidence to be suspicious of.

## N=60 robustness (root_owned count, matched-undisturbed-paired)
| seed | static mean | matched mean | robustness |
|---|---|---|---|
| 42  | 25.1950 | 18.8650 | 0.7488 |
| 100 | 27.4800 | 18.4400 | 0.6710 |
| 123 | 26.7300 | 18.0450 | 0.6751 |
| 200 | 28.4950 | 18.3900 | 0.6454 |
| 300 | 24.6400 | 18.9400 | 0.7687 |

Cell mean = **0.7018**, SD = **0.0537**. (Currently-reported, stage 7/final: mean 0.6801, SD 0.0409.)

## N=90 robustness
| seed | static mean | matched mean | robustness |
|---|---|---|---|
| 42  | 29.4400 | 20.1000 | 0.6827 |
| 100 | 32.2100 | 21.4650 | 0.6664 |
| 123 | 25.9000 | 17.4950 | 0.6755 (identical to currently-reported -- same checkpoint) |
| 200 | 27.2600 | 18.8750 | 0.6924 (identical to currently-reported -- same checkpoint) |
| 300 | 28.2650 | 19.9400 | 0.7055 |

Cell mean = **0.6845**, SD = **0.0151**. (Currently-reported, mixed final-stage: mean 0.6833, SD 0.0258.)

## Achieved churn
| cell | per-seed leave/ep | pooled mean leave/ep | achieved churn | reference target | currently-reported achieved |
|---|---|---|---|---|---|
| N=60 stage3 | 24.09, 25.54, 25.84, 23.91, 25.61 | 24.998 | **41.66%** | 42.0% | 42.23% (0.23pp from target) |
| N=90 static500k | 38.51, 38.76, 36.91, 38.35, 38.55 | 38.216 | **42.46%** | 42.4% | 42.4% (0.4pp from N=30's 42.0%) |

Both land close to their respective targets -- N=60 stage3 is 0.34pp from 42.0% (vs the
final-checkpoint run's 0.23pp), N=90 static500k is 0.06pp from 42.4% (the two are the same
value here since 3/5 seeds' checkpoints are literally unchanged). Achieved churn depends
partly on policy behaviour (leave eligibility weighting interacts with which nodes the agent
holds), so a small drift from the final-checkpoint figures was expected; what's observed here
is a small drift, not a large one.

## Pairwise comparisons (N=30 unchanged; N=60/N=90 now the early-checkpoint figures)
Same bootstrap method as the existing `nodecount_ci_n60` comparison (seed-resampled, NB=10000,
RNG seed=11, sign = larger cell minus smaller cell).

| comparison | previous point diff | previous 95% CI | new point diff | new 95% CI | direction changed? | CI-includes-0 changed? |
|---|---|---|---|---|---|---|
| N=30 vs N=60 | -0.0255 (N60 lower) | [-0.0679,+0.0176] incl. 0 | **-0.0038** (N60 still lower, near zero) | [-0.0524,+0.0471] incl. 0 | no | no |
| N=30 vs N=90 | -0.0223 (N90 lower) | [-0.0574,+0.0117] incl. 0 | **-0.0210** (N90 still lower) | [-0.0526,+0.0071] incl. 0 | no | no |
| N=60 vs N=90 | +0.0032 (N90 slightly higher) | [-0.0355,+0.0424] incl. 0 | **-0.0173** (N90 now lower) | [-0.0615,+0.0253] incl. 0 | **yes -- sign flips** | no (both already included 0) |

No comparison's CI crosses from including zero to excluding it, or vice versa, in either
direction. The one directional change (N=60 vs N=90) flips a point estimate that was never
statistically distinguishable from zero either before or after -- both CIs are wide and both
include zero.

## Outputs (committed alongside this record)
`n60_stage3/score_*.csv`, `n60_stage3/leaveown_*.csv`, `n90_static500k/score_*.csv`,
`n90_static500k/leaveown_*.csv`, `run_stage_earlyckpt.sh` (committed separately, pre-run),
`run_orchestrator_earlyckpt.log`, the 20 `logs_run/*_stage3_*.log` /
`logs_run/*_static500k_*.log` files. NOT committed: `drift_*.csv` (bulk, per this directory's
existing convention).

## Not done in this task
`thesis_v3.tex`, `Table tab:rq1c_crossed`, and `Table tab:rq1c_robustness` were not edited.
N=30 and the lower-neighbour N=30 cell were not touched, run, or read for this comparison
beyond their already-committed `y_robustness/out/n30/` figures.
