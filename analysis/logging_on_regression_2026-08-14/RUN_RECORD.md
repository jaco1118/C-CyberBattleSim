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

---

## Amendment 2 — B1/B2/B3 (2026-08-14, same day, appended after the accepted result)

The bottom line above stands unrevised. This section closes three items that were not reported:
whether dynamic change was active in the accepted run (so the busiest part of the instrumentation
— attribution rows, leave-embedding logging, the per-event loops — was genuinely exercised, not
skipped), the h2/h3 branch counts, and which RNG generators exist and whether the 6,000-step
action agreement bears on them.

### B3 — every RNG the environment and policy draw from during a step (source only, no run)

`grep -n "random\.\|np\.random\.\|numpy\.random\.\|torch\.rand\|RandomState\|\.sample(\|\.choice(\|\.shuffle("`
over `cyberbattle_env.py`, `cyberbattle_env_compressed.py`, `cyberbattle_env_switch.py`:

- **Python's global `random` module** — by far the heaviest consumer: node/starter selection at
  reset (`:163,246`), the dynamic-leave and dynamic-join Bernoulli/batch mechanism in full
  (`cyberbattle_env.py:623-716`, the code already read in earlier tasks this session), and several
  outcome-processing paths (vulnerability/service selection, `:804,924,927,943,947,969,1020,
  1340,1341,1412,1422`) reached from inside the agent's own action resolution.
- **`numpy.random`'s global state** — the Poisson/choice draws inside the same dynamic-change
  mechanism (`:630,634,708,712`), `RandomSwitchEnv._switch_environment`'s
  `np.random.choice(self.envs_ids)` (`cyberbattle_env_switch.py:75`, fires on every episode reset
  even with a single env in the list), and `__balance_action_space_by_outcome`'s
  `np.random.choice(len(actions), self.sample_subset_samples, replace=False)`
  (`cyberbattle_env_compressed.py:1423`) — confirmed ACTIVE for this checkpoint
  (`sample_subset_samples: 100` in `train_config.yaml`, not the disabled `False` default), and
  reached from the action-space rebuild path that fires on the same
  `action_changed_graph`/`nodes_changed` triggers as the h2/h3 snapshot branches.
- **`torch`'s global RNG** — consumed by the policy's own stochastic action sampling
  (`model.predict(obs, deterministic=False)`, standard SB3 distribution sampling; not located in
  the environment files at all).
- **A separate, independently-gated diagnostic** (`_rq2c_probe`, `cyberbattle_env_compressed.py:
  932-967`) explicitly snapshots and restores all three RNG states (`random`, `numpy`, `torch`)
  around its own two extra `model.predict()` calls, specifically so it "consumes no net RNG"
  (its own comment) — gated on `RQ2C=1` / `self._rq2c_model` being attached, neither of which this
  task's runs ever set. Confirmed inert here, and self-neutralizing by design when it does run.
- `env.action_space.sample()` (`cyberbattle_env_compressed.py:1565-1566`, `sample_random_action`)
  is a standalone helper never called from `step()` — not reached in a normal rollout.

**What the 6,000-step agreement covers, and what it doesn't:** Python's `random` module,
`numpy.random`, and `torch`'s RNG are each a single process-wide global stream, not a separate
instance per code path — the dynamic-change mechanism, the action-space rebalancing, and the
policy's stochastic sampling all draw from the *same* three streams the two flag-only encode
calls would have drawn from, had they drawn from anything. STEP 0.3 already established from
source that they draw from none of them. The empirical result adds a second, independent line of
evidence for exactly that: with `deterministic=False`, the policy consumes from `torch`'s global
RNG on *every single step*, so if either flag-only encode call had consumed even one draw from any
of these three streams, the very next stochastic action, Bernoulli leave check, or action-space
rebalance would have drawn a different number on Side B than on Side A/B_off, and the two
trajectories would have desynchronised at the first such call — not stayed aligned for 6,000
consecutive steps across three seeds. The 6,000-step agreement is therefore not just an action
match; it is itself evidence, independent of the source read, that no flag-only code path consumes
from any generator this run exercises. It does **not** bear on the `_rq2c_probe` generator usage
(inert here by construction) or on any generator only reached by code this run never executed.

### B1/B2 — dynamic-event and branch counts (script: `run_side_diagnostics.py`, committed `91b52e6`)

Read-only supplementary counters over the identical Side B construction already accepted (same
checkpoint, topology, seeds, 2000 steps, `drift_logging=True`) — not a re-comparison; the
trajectory is deterministic given (seed, checkpoint, config, code), so this reproduces the
already-validated run and simply reads more off it. B1 uses `info["change_type"]`, a field
`StepInfo` already carries unconditionally every step (`cyberbattle_env_compressed.py:765`) that
the original run captured to a local variable and then discarded rather than saved. B2 wraps (not
replaces) the bound `_drift_snapshot_from_cache`/`_drift_snapshot_fresh` methods from outside the
class to record call order; with `drift_sample_rate=1` (this run's config), `log_this_step` and
`need_h1_h2` are both True on every step (`:587,592`), so h1, h2, and h3 are captured every step in
that fixed order (h1 always cache, h2 second, h3 third) — verified directly (`steps_with_exactly_3_
snapshot_calls = 2000/2000` on every seed, `steps_with_other_call_count = 0` on every seed).

| seed | dynamic events (all `membership_leave`) | h2 cached | h2 fresh | h3 cached | h3 fresh |
|---|---|---|---|---|---|
| 42 | 53 | 620 | 1380 | 53 | 1947 |
| 100 | 43 | 540 | 1460 | 43 | 1957 |
| 123 | 57 | 662 | 1338 | 57 | 1943 |

Internal consistency check: `h3_cached` equals the dynamic-event count exactly on every seed
(53/43/57 both places) — expected by construction (h3 reads cache only when `nodes_changed` is
truthy, i.e. exactly when a dynamic event fired) and confirms the call-order classification is
correct, not merely plausible-looking.

**B1 — dynamic change was ACTIVE.** `dynamic_mode: both` in this checkpoint's own
`train_config.yaml` (`:41`, independently re-confirmed here by direct grep), and 43-57 real
`membership_leave` events fired per seed over 2000 steps. No `membership_join` or `property`
events fired in this window (consistent with this project's own established findings elsewhere
this session: join events are comparatively rare and capped, and `patch_service_dynamic_enabled:
false` for this checkpoint makes property events structurally impossible) — so the accepted
comparison exercises the `membership_leave` per-event logging path (attribution rows where
applicable, leave-embedding logging, the loops over `dynamic_events`) on every seed, but does not
speak to `membership_join`'s or `property`'s per-event logging path specifically.

**B2 — every fresh branch fired, substantially, on every seed.** `h2_fresh` (1338-1460) and
`h3_fresh` (1943-1957) both fire on the large majority of steps, every seed — neither flag-only
branch is untested.

### Decision, per the amendment's own rule
"If B1 shows dynamic change WAS active and B2 shows every fresh branch fired: nothing further is
needed. Report the counts and stop." Both conditions hold. No further run performed. The one
scope note above (this run's evidence covers `membership_leave`'s per-event path; it does not
directly speak to `membership_join`'s or `property`'s) is reported rather than silently extended
to cover paths this run did not exercise.

### Outputs (this addendum)
`run_side_diagnostics.py`, `diagnostics_console.log`.
