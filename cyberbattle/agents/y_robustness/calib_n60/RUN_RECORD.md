# Run record — N=60 robustness column, STEP 1 (churn calibration)

PROVENANCE BANNER: commissioned 2026-08-15, after the convergence criterion changed
from root-owned-node-count to episode reward (a decision independent of this cell),
which made the N=60 cell newly eligible for Table IV.5. Nothing about this result was
known or estimated beforehand.

Date run: 2026-08-16.

## Reference and tolerance (declared in advance, per Amendment 1)
REFERENCE: N=30 cell's measured churn fraction, 42.0% (from the STEP 1.3 200-episode
full run, commit 9e9f518 -- distinct from the 41.0% figure the original calibration
used as its target at probe time, see note below).
TOLERANCE: +/-1.5 percentage points. Accept 40.5-43.5%.
Both values fixed before any N=60 trial was run.

## Procedure
Same harness (`taskF2_eval.py`, unmodified) and seed (42) used for N=30's and N=90's
own calibration probes. 30 episodes/trial (matching the larger of the episode counts
actually used in the precedent -- N=30's baseline probe and N=90's "confirmatory"
probe were both 30 episodes; other N=90 trials used 20 or 25). Checkpoint:
`logs/yN60_s42_stg7_2026-08-05_23-47-20/TRPO_x_control_SecureBERT`,
CKPT_STEP=250000. Topology: `graphs_yN60_s42_2026-08-03_17-15-34/1`.

Churn fraction = mean leave-events/episode / N, the same definition used for the
N=30/N=90 columns (verified against commit 9e9f518: N=30 12.61 leave/ep / 30 = 42.0%;
N=90 38.18 leave/ep / 90 = 42.4%). For N=60, denominator is 60.

## Incident during this run, disclosed
The first batch (ci=6,8,10,12) was launched as 4 concurrent processes all writing to
the SAME output path (`taskF2_eval.py`'s CSV name is a function of (agent_cond, seed,
eval_cond) only, not change_interval) -- a race condition of my own making. The
achieved-churn summary line for each trial was still captured correctly and
independently (each process's stdout went to its own dedicated log file, produced
before the CSV write), so no calibration number was lost, but the underlying CSVs
collided and only one, unidentifiable trial's file survived. That file was removed
(`UNIDENTIFIED_raced_trial.csv`, not committed). ci=6, 8, 10, and 12 were NOT
individually re-run with separate output dirs since none of the three admissible
candidates were among them once ci=5 and ci=7 (see below) were added -- their
achieved-churn figures below are taken from the surviving log lines, which are
independent of the CSV collision and were not affected by it. From the second batch
onward (ci=5, 6, 7), each trial was routed to its own subdirectory (`ci5/`, `ci6/`,
`ci7/`) to prevent recurrence; those three trials' CSVs are intact and committed.

**This explains a previously-unexplained gap in the N=90 precedent.** Task Q3 (STEP
0) recorded, without a known cause, that "only the final trial's CSV survives on
disk" for N=90's original ~7-trial calibration sweep (`calib/`, 2026-08-06
10:09:51-10:22:25 UTC). Those trials were launched sequentially, not concurrently
(one `taskF2_eval.py` CLI call at a time, per the session transcript), so there was
no race in that case -- but the identical root cause applies: the script names its
output by (agent_cond, seed, eval_cond) only, never by change_interval, so each
sequential trial's write silently overwrote the previous trial's identical-named
CSV. My concurrent version surfaced the same defect as a race; the original
sequential version surfaced it as silent overwrite. Same script, same clobbering,
now identified rather than merely observed.

## Full sweep (every integer tried)
| change_interval | leave/ep | churn (% of 60) | distance from 42.0% | in [40.5, 43.5]%? |
|---|---|---|---|---|
| 5  | 25.30 | 42.17% | +0.17pp  | **yes** |
| 6  | 25.87 | 43.12% | +1.12pp  | **yes** |
| 7  | 23.63 | 39.38% | -2.62pp  | no |
| 8  | 22.33 | 37.22% | -4.78pp  | no |
| 10 | 19.63 | 32.72% | -9.28pp  | no |
| 12 | 19.03 | 31.72% | -10.28pp | no |

Note: ci=5 achieved *less* churn than ci=6 (25.30 vs 25.87 leave/ep) -- the opposite of
the monotonic-decreasing-with-ci pattern the N=90 precedent showed cleanly at every
step. **Reported as observed, not re-run to smooth it out, and not treated as a
finding.** A plausible mechanism, UNVERIFIED: at short change_interval the pool of
currently-removable/eligible nodes may thin faster than it replenishes within a
300-step episode, so more frequent removal attempts do not translate one-for-one
into more realized removals -- a saturation effect at the aggressive end of the
sweep. This is offered only as a candidate explanation, not investigated further
(no additional probes run to test it) and not asserted as the cause; at n=30
episodes, single seed, ordinary sampling noise (frac0 across all 6 trials ranged
0.033-0.100, mean score 0.469-0.540) remains an equally live explanation.

## Result
**Two integers qualify: ci=5 (42.17%, +0.17pp) and ci=6 (43.12%, +1.12pp).**
**Accepted: ci=5, the closer of the two to the 42.0% reference.**
ci=6 also qualifies and is noted here so the choice isn't presented as unique.

Achieved-churn comparison across all three columns, so match quality is visible
rather than assumed equal:
- N=30: 42.0% (reference; 0.0pp by definition)
- N=90: 42.4% (0.4pp from reference; commit 9e9f518)
- N=60: 42.17% (0.17pp from reference) at ci=5 -- the tightest match of the three.

## Note on the reference figure itself
The 42.0% reference (this task's REFERENCE, per Amendment 1) is the N=30 column's
FULL 200-episode achieved figure. The ORIGINAL N=90 calibration (2026-08-06, commit
`1ab24df0f8b2857eeb909165ef3db3be8550d958` -- corrected here; an earlier draft of
this record mistakenly said "commit preceding 1ab24df") targeted a different,
earlier measurement of the same quantity: a 30-episode N=30 probe read 12.30
leave/ep = 41.0%, and N=90's ci=4 was accepted against THAT 41.0% target under an
explicit 2pp tolerance (both facts recovered from that commit's message, not
previously surfaced in this task's STEP 0 answer to Q3, which incorrectly stated no
tolerance was on record -- corrected here). The 1.0pp gap between the 30-episode
probe (41.0%) and the 200-episode full run (42.0%) for the SAME N=30 cell is itself
informative about probe-to-full-run noise at this sample size, consistent with the
non-monotonicity noted above.

## Tolerance comparison across the two calibrated columns
| column | reference target | tolerance declared | source |
|---|---|---|---|
| N=90 | 41.0% (30-ep N=30 probe) | +/-2.0pp | commit `1ab24df`, message text |
| N=60 | 42.0% (200-ep N=30 full run) | +/-1.5pp | Amendment 1, this task |

N=60's band is tighter (1.5pp vs 2.0pp) and anchored to the more stable of the two
available N=30 measurements (the 200-episode full run, not the 30-episode probe).
Both bounds are shown so the N=60 column isn't read as having been held to a looser
or unstated standard relative to the one N=90 was actually accepted under.

## Outputs (committed alongside this record)
`ci5/score_static_seed42_evalmembership_matched.csv`,
`ci5/leaveown_static_seed42_evalmembership_matched.csv`,
`ci6/score_...csv`, `ci6/leaveown_...csv`, `ci7/score_...csv`, `ci7/leaveown_...csv`,
and all 7 trial log files (`trial_ci*.log`). NOT committed: `drift_*.csv` (per-step
logs, ~5-20MB each, consistent with this project's existing bulk-artifact convention
for this directory). `run_stage_n60.sh` committed separately.

## Wipe test
Reproducible from the committed log files' recorded commands and the already-on-disk
checkpoint/topology (not committed, per convention). The CSVs in ci5/ci6/ci7 are the
actual per-episode outputs of those specific runs and are themselves the primary
record, not just reproducible byproducts, since taskF2_eval.py's eval is stochastic.
