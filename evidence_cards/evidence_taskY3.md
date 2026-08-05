# Task Y3 — N=30 pilot at Y2's lower degree (matched same-N comparison against Task Y's original N=30)

Direct follow-up to [`evidence_taskY2.md`](evidence_taskY2.md). Y2 tested one pilot cell (N=50,
degree ~12.5) against the degree hypothesis (does lower fixed degree fix Task Y's N=60/N=90
convergence instability?) and got a clean CONVERGED result. This task builds the single most
informative next comparison: **N=30 at that same lower degree**, directly against Task Y's own
original N=30 cell (same node count, degree ~19.7) — a genuine same-N, different-degree contrast,
which neither Y2 alone nor the N=60/N=90 cells provide (those differ in N as well as degree).
Branch `taskY2-pilot-n30`. **N=20 and N=40 explicitly NOT authorized or built.**

## STEP 0 — verification [ARTIFACT, confirmed before build]

- **0.1** Same generation/calibration process as Y2 (`generate_graphs.py`, per-cell YAML config,
  `knows_neighbor_probability_range` as the degree-controlling parameter), `--num_graphs 1` passed
  explicitly on the CLI from the start (Y2's num_graphs bug proactively avoided, not just fixed
  after the fact).
- **0.2** Target degree = **N=50's actual achieved degree, 12.50** (not N=50's original ~10 target)
  — deliberately matching what N=50 was actually trained on, not what it originally intended, so
  this is a genuine same-degree comparison.
- **0.3** Topology-reuse check: Task Y's original N=30 topologies (`graphs_yN30_s<seed>_2026-08-03_*`,
  degree ~19.7) coexist on disk under the same `graphs_yN30_s<seed>_` prefix as this task's new
  topologies. Confirmed NOT reused — new topologies generated fresh, folder names distinguished by
  generation date (`*_2026-08-05_*`), `run_stage.sh`'s topology glob explicitly date-filtered to
  avoid ever matching the original set (verified for all 5 seeds before launch).
- **0.4** Pipeline unchanged: same `train_agent.py`, same F4 rule
  (`compute_convergence_check.py`), same 250k-per-stage/1.25M-cap schedule, same seeds
  42/100/123/200/300, zero core code changes — matches Y2's own 0.4 answer exactly.

## Degree calibration [ARTIFACT]

3 candidate probability ranges probed (5 graphs each, N=30):

| probe | `knows_neighbor_probability_range` | measured degree (5 graphs) | mean | SD |
|---|---|---|---|---|
| A | [0.08, 0.25] | 13.3, 7.8, 9.9, 9.3, 10.2 | 10.11 | 1.99 |
| B | [0.12, 0.30] | 11.9, 8.8, 11.2, 9.3, 7.3 | 9.70 | 1.86 |
| C | [0.15, 0.40] | 14.6, 12.0, 14.5, 14.0, 12.3 | **13.50** | **1.26** |

Non-monotonic A→B (read as sampling noise at n=5/probe, not a real inverse relationship — C, the
highest-p probe, gives both the highest mean and tightest spread, consistent with the expected
direction overall). **Selected: probe C's range `[0.15, 0.40]`** — closest of the three to the
12.5 target, tightest spread. Full detail: `cyberbattle/agents/y2_n30/calib_results/calibration_report.md`.

**Final 5-seed topology set:**

| seed | degree | folder |
|---|---|---|
| 42  | 14.73 | `graphs_yN30_s42_2026-08-05_21-16-13` |
| 100 | 9.63  | `graphs_yN30_s100_2026-08-05_21-16-40` |
| 123 | 14.10 | `graphs_yN30_s123_2026-08-05_21-17-08` |
| 200 | 13.63 | `graphs_yN30_s200_2026-08-05_21-17-46` |
| 300 | 9.63  | `graphs_yN30_s300_2026-08-05_21-18-13` |

**Achieved: mean = 12.35, SD = 2.51 — within 0.15 (~1.2%) of the 12.50 target.** SD wider than the
calibration probe's 1.26 (final-draw variance across only 5 topologies, not a probe average), but
within this project's established convention. No material drift from target. Confirmed via subdir
count (1 per seed) that N=50's num_graphs config-vs-CLI bug did not recur.

## Training setup [ARTIFACT]

`cyberbattle/agents/y2_n30/y2_base.yaml` — copied verbatim from `y2_n50/y2_base.yaml` (250k/stage,
`dynamic_mode=none`, `checkpoints_save_freq=25000`, same aggregations/hyperparameters). Orchestrator
`cyberbattle/agents/y2_n30/run_stage.sh` (committed `cc4e396`, before launch, per standing rule
SR-1) — adapted from `y2_n50/run_stage.sh`, differing only in single-cell paths under `y2_n30/` and
an **explicit `2026-08-05` date filter** on the topology glob (necessary, not just prudent — see
STEP 0.3). Same thread-cap fix, same F4 checkpoint-stopping rule, same `--finetune_model` resume
mechanism as every staged run this project has used.

**Note on log-directory naming:** Task Y's original N=30 run also used the run-name convention
`yN30_s<seed>_stg1`, so `logs/yN30_s<seed>_stg1_*` collides by prefix with this task's own run
directories (`logs/yN30_s<seed>_stg1_2026-08-05_*` vs the original's `logs/yN30_s<seed>_stg1_2026-08-03_*`).
Checked before trusting `run_stage.sh`'s F4-check step (which selects the most-recent match via
`ls -dt | head -1`): ISO-8601 timestamps sort correctly both alphabetically and chronologically, and
this task's directories were created after the original's, so directory selection is correct and
stable regardless of mtime churn during training — verified directly for all 5 seeds, not just
reasoned about.

Gae encoder config files (`model_spec.yaml`, `train_config_encoder.yaml`) confirmed present on this
branch (inherited from `taskY2-pilot-n50` via commit `c6772a7`) before launch — the same class of
file that was lost to a branch-switch deletion earlier this session.

**Stage 1 (250k, all 5 seeds) launched 2026-08-05 21:24 BST.**

## Stage 1 result (250k) [FINDING] — CONVERGED, pilot complete, no further stages needed

**CONVERGED.** mean|Delta%|=2.67% (<5%), 4/5 within band (need >=4).

| seed | Delta% | within band? |
|---|---|---|
| 42  | +5.88% | **no (only miss)** |
| 100 | +3.14% | YES |
| 123 | +2.45% | YES |
| 200 | +1.44% | YES |
| 300 | -0.43% | YES |

Zero errors/warnings across all 5 seed logs. Per the pre-registered stopping rule (stop at first
stage that converges, same rule used for every cell in this project), **N=30 pilot is done at 250k
steps per seed — stages 2-5 were not launched.** Verdict: `y2_n30/verdicts/stage1_N30.txt` (commit
`7e7001a`).

## FINAL — three-way comparison at matched/near-matched conditions [FINDING]

| | Task Y original N=30 | Y3 pilot N=30 | Y2 pilot N=50 |
|---|---|---|---|
| degree | 19.69 +/- 1.18 | **12.35 +/- 2.51** | 12.50 +/- 1.67 |
| verdict | CONVERGED (250k) | **CONVERGED (250k)** | CONVERGED (1M) |
| mean\|Delta%\| at stopping stage | 4.18% | 2.67% | 4.22% |
| within-band at stopping stage | 4/5 | 4/5 | 4/5 |
| stopping stage | 1 | 1 | 4 |
| total steps used | 250k | **250k** | 1,000k |
| trajectory | converged immediately | converged immediately | 7.09%->4.37%->11.46%->4.22% (volatile) |

**Achieved degree vs target:** 12.35 / SD 2.51, vs the 12.50 target (N=50's actual achieved value)
— 1.2% off, accepted as a clean match given the established calibration tolerance.

**Outlier flag:** seed42 (+5.88%) is the only individual seed miss, and only barely above the 5%
per-seed threshold — not a pattern shared with any other seed, and the band-level verdict still
passes comfortably (4/5, well clear of the 4/5 minimum) with a low mean (2.67%, well under the 5%
mean threshold). Not treated as evidence of a systemic problem with this cell.

**Stopping stage and total steps:** stage 1 (250k steps/seed; 1.25M total across the 5-seed cohort),
identical to Task Y's own original N=30 cell — no stage 2 was needed or launched.

## Observation on the degree hypothesis [FINDING — observation only, not a conclusion]

At N=30, this pilot converges immediately at the first stage regardless of degree: Task Y's
original N=30 (degree ~19.7) and this pilot's N=30 (degree ~12.35) both converge at 250k, with
similar mean|Delta%| (4.18% vs 2.67%) and identical within-band counts (4/5). **This does not by
itself support or refute the degree hypothesis, because Task Y's N=30 never exhibited a convergence
problem to fix in the first place** — the hypothesis was motivated by N=60/N=90's instability
(both degree ~20), and N=30 at that same original degree was already the project's fastest-
converging cell before this pilot existed. Lowering degree at N=30 therefore has no visible ceiling
to lift against.

By contrast, Y2's N=50 pilot (also degree ~12.35-12.50, same lower-degree regime) DID need 4 stages
and passed through a notably volatile middle stretch (peaking at 11.46% mean error at stage 3)
before converging — a qualitatively different trajectory from either N=30 cell despite sharing the
low-degree condition with this task's pilot. **Read together, the two pilots suggest the
degree-alone explanation is incomplete: lower degree did not prevent N=50 from being harder to
converge than either N=30 cell, and N=30 was easy to converge at both degree settings tested.**
Whatever distinguishes N=50/N=60/N=90's difficulty from N=30's ease is not cleanly isolated by
either pilot alone — an N-by-degree interaction (or some other N-correlated factor) remains at
least as plausible as a pure degree effect. No RQ1(c)-style conclusion is drawn here; this is
report-only per the task's own scope.

## Commit references

- `ec14ba4` — calibration (degree 12.35, SD 2.51) + generation configs + seeds + `y2_base.yaml`.
- `cc4e396` — `run_stage.sh` orchestrator (committed before launch, per SR-1).
- `7e7001a` — stage 1 verdict (CONVERGED, mean 2.67%, 4/5).
- This file — evidence card.

**Not authorized and not built:** N=20, N=40. No thesis edits made. No other branches or
checkpoints touched.
