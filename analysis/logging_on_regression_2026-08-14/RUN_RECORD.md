# Run record — Task LOGGING-ON-REGRESSION

Date: 2026-08-14.

## STEP 0 summary (full answers given inline in the reply; not repeated in full here)
Single flag `drift_logging` (default `False`, `cyberbattle_env_compressed.py:103`). All three
snapshot-taking acts (h1/h2/h3) are entirely gated behind it; the two flag-only extra `encode()`
calls (`_drift_snapshot_fresh`, h2/h3 fallback branches) draw no RNG, mutate no module state
(encoder confirmed `.eval()`-only throughout the evaluation pipeline, no `.train()` call anywhere
reachable from it), and write only to local variables never read by the agent-facing code path.
Baseline commit: `7cdfb2b` (parent of `40dfc7c`, the commit that introduced `drift_logging`).
Existing 2000-step/2-seed harness: `analysis/recovered_scripts_2026-08-04/drift_regression_check_v2.py`
(git-archives `7cdfb2b`, `drift_logging=False` on both sides, synthetic random actions). This task
reuses its repo-root-parameterized git-archive technique but drives the rollout with a real,
reported TRPO checkpoint instead.

## Commands
```
cd analysis/logging_on_regression_2026-08-14
N_STEPS=2000 SEEDS="42 100 123" bash run_all.sh
```
plus a standalone diagnostic-control run (`run_side.py <repo_root> B_off <seed> 2000 <out>`, three
seeds) and a `compare_sides.py` invocation over the B_off/B pair.

## Checkpoint used
`cyberbattle/agents/logs/trpo_250k_tuned_compressed_band10-15_seed42_2026-07-26_11-56-51/TRPO_x_control_SecureBERT/checkpoints/1/checkpoint_250000_steps.zip`
+ its `checkpoint_vecnormalize_250000_steps.pkl` — band 10-15, seed 42. This is one of the 15
manifest checkpoints (`attenuation_manifest.yaml`) behind every reported attenuation figure in this
project — the same checkpoint used throughout FLAG-GROUND-TRUTH, REPLAY-FLAG-CLOSEOUT, and
METRIC-DEFINITIONS. Topology: `cyberbattle/data/env_samples/scalability_10_15/1` (real manifest-band
topology). Policy actions: **stochastic** (`deterministic=False`), matching the regime that actually
produced Chapter IV's figures — this project has already established that a deterministic policy on
these checkpoints barely explores and fires ~0 leave events, so a deterministic-only check would not
be representative, and stochastic sampling is also the more stringent test of RNG-stream identity
between the two sides.

## A bug found and fixed before any result was reported

The first full run (2000 steps × 3 seeds) showed a large divergence starting at step 88–225
(seed-dependent): identical chosen actions on both sides, but Side A kept a small negative reward
and `done=False` while Side B jumped to `reward≈3999` and `done=True`, with `n_discovered`/
`n_root_owned` collapsing to 1/0 (an episode reset). Root cause, found via a diagnostic control
(`B_off`: current HEAD, `drift_logging=False`, everything else identical to `B`) run **before**
touching Side A's kwargs: `B_off` vs `B` showed **zero divergence** across all 2000 steps × 3
seeds — ruling out the instrumentation immediately. `A` vs `B_off` diverged at the exact same steps
as `A` vs `B`, proving the cause was specific to Side A's construction, not to any current-HEAD
code. Traced to this script's own `SIDE_A_ALLOWED_TRAIN_CONFIG_KEYS` whitelist: it omitted
`winning_reward`, `losing_reward`, and `stop_at_goal_reached` (this checkpoint's
`train_config.yaml` sets `winning_reward: 4000`, `stop_at_goal_reached: true`, confirmed by direct
grep) — so Side A silently fell back to the constructor defaults (`0`, `False`) and never
registered a goal-reached win the way Side B correctly did on the identical action. **Not a
difference between the pre-instrumentation and current-HEAD environment code — a bug in this
comparison script**, fixed by rebuilding the whitelist exhaustively from both commits' actual
`__init__` signatures (`git show 7cdfb2b:cyberbattle/_env/cyberbattle_env.py` /
`cyberbattle_env_compressed.py`, read in full) rather than a hand-picked subset. The environment,
the instrumentation, and the flag default were not touched.

## Row counts (both runs, all seeds)

| comparison | seed | Side X rows | Side Y rows | items differing | obs bitwise-equal |
|---|---|---|---|---|---|
| A (7cdfb2b) vs B (HEAD, drift_logging=True) | 42 | 2000 | 2000 | 0/10 | 2000/2000 |
| A vs B | 100 | 2000 | 2000 | 0/10 | 2000/2000 |
| A vs B | 123 | 2000 | 2000 | 0/10 | 2000/2000 |
| B_off (HEAD, drift_logging=False) vs B | 42 | 2000 | 2000 | 0/10 | 2000/2000 |
| B_off vs B | 100 | 2000 | 2000 | 0/10 | 2000/2000 |
| B_off vs B | 123 | 2000 | 2000 | 0/10 | 2000/2000 |

Zero rows dropped, zero step-count mismatches, on both comparisons, all seeds — stated explicitly.
The 10 exactly-checked items per step: `source_node`, `target_node`, `vulnerability`, `outcome`,
`min_distance_action`, `reward`, `done`, `cumulative_reward`, `n_discovered`, `n_root_owned`. All
zero differing on both comparisons, all seeds. Observation vectors: bitwise-equal (`np.array_equal`,
no tolerance applied anywhere) on 2000/2000 steps, both comparisons, all seeds.

## STEP 4 (divergence diagnosis)
Not applicable to the final, corrected result — nothing diverged. The one divergence that did occur
(the buggy first run) is fully diagnosed above, including the first-diverging-step detail
(step 105/225/88 per seed, full action/reward/done state on both sides), and traced to its exact
cause in this script rather than the environment.

## Wall-clock and cost
Full `run_all.sh` (git-archive + Side A × 3 seeds + Side B × 3 seeds + compare, 2000 steps each,
CPU-only, `torch.set_num_threads(1)` per side): **1m28s**, corrected run (1m26s, first buggy run).
`B_off` control (3 seeds × 2000 steps): comparable, ~1m. Total for this task: under 5 minutes of
compute, no GPU, no training, evaluation rollouts only on one already-existing checkpoint.

## Bottom line
**(a) IDENTICAL WITH LOGGING ON**, established by:
- `analysis/logging_on_regression_2026-08-14/comparison_report.txt` (Side A, commit `7cdfb2b`,
  vs Side B, current HEAD with `drift_logging=True`): zero differing on all 10 exactly-checked
  items and bitwise-equal observations, on 2000 steps × 3 seeds (6,000 step-comparisons).
- `analysis/logging_on_regression_2026-08-14/control_vs_B_comparison_report.txt` (`drift_logging=False`
  vs `drift_logging=True`, same current-HEAD commit — the most direct possible isolation of the
  instrumentation's own effect, holding every other line of code fixed): zero differing, same scope.

**The thesis may state that the instrumentation does not change the agent's behaviour when
enabled** — now established by direct paired comparison, not merely inferred from source reading,
and confirmed on two independent comparisons (against the true pre-instrumentation baseline, and
against the same commit with the flag off), so the corrected-script concern above cannot be
mistaken for the substantive answer.

## Outputs (committed alongside this record)
`run_side.py`, `compare_sides.py`, `run_all.sh`, `comparison_report.txt`,
`control_vs_B_comparison_report.txt`, `full_run_console.log`, this record. The per-step `.pkl`
artifacts (~19MB × 9 files) are NOT committed, per this project's standing convention against
committing bulk run data — they are regenerable from the committed scripts in under 5 minutes.

## Wipe test
Reproducible from the committed scripts alone: `run_all.sh` re-extracts `7cdfb2b` via `git
archive` (no dependency on any uncommitted state) and re-runs both sides against the already-
on-disk, already-committed-elsewhere checkpoint and topology files.
