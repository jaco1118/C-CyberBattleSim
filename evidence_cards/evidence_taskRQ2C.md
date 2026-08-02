# Task RQ2C-1 — does the chosen action follow the VIEW, or only the ACTION SET

Branch `attenuation-pooling-scale`. Answers RQ2(c): when a single-node membership change moves the
agent's view, does its chosen action move with it, or does behaviour change only because the action
it wanted was removed from the candidate set? Method: at each single-node `membership_leave` event,
compare the policy's preferred action **before** vs **after** the change, partitioned by whether that
preferred action survives in the post-change candidate set.

> **BANNER:** RQ2C: single-node membership-leave events only; batch events excluded; 80-100 band agent
> is not confirmed converged (see Task F4); group (i) events have no choice-changed metric by construction.

> **SCOPE (read before citing any number):** measured on **fresh episodes** from a **per-seed-seeded
> STOCHASTIC** rollout of the trained policy — the exploratory regime that actually produces
> `membership_leave` events (the DETERMINISTIC policy discovers ~nothing and fires **0** leaves, so it
> cannot measure RQ2(c) at all). Same checkpoints/seeds/bands and the same standard attenuation config
> as the headline sweep (`patch_service` OFF, no CX_DIAG relaxation). These are **NOT** the literal
> reported headline episodes (the stored `attenuation_step3_logs` actions are not faithfully replayable
> — see **claims_audit CA-3**), nor the stochastic-action-selection headline numbers themselves. The
> **counterfactual pre/post predicts are `deterministic=True`** (noise-free before/after comparison —
> that is where "no stochastic sampling" matters). Reproducible: torch/np/random seeded per seed +
> `set_num_threads(1)` + `PYTHONHASHSEED=0`.

## Method (as implemented)

Per single-node `membership_leave` event, computed LIVE during the rollout (env
`cyberbattle_env_compressed.py::_rq2c_probe`, gated on `RQ2C=1`, default-off):
1. `candidate_set_pre` / `candidate_set_post` = the env's own action-embedding set immediately BEFORE /
   AFTER the churn (reused, never re-enumerated — no RNG, no state mutation).
2. `chosen_pre` = `find_closest_action_embedding(predict(obs_pre))` snapped to `candidate_set_pre`;
   `chosen_post` likewise on `candidate_set_post`. Both predicts `deterministic=True`.
3. **Identity** = `(source, target, vulnerability_ID, type(outcome).__name__)` — verified **collision-free**
   over 400 real candidate sets (0 distinct action_keys ever collapsed to one identity tuple).
   - `chosen_pre ∉ candidate_set_post` → **GROUP (i)** "preferred action removed" (no choice-changed metric
     by construction — the original choice cannot be reselected).
   - `chosen_pre ∈ candidate_set_post` → **GROUP (ii)** "action available in both"; `changed = (chosen_post ≠ chosen_pre)`.
   - Degenerate guards: empty pre set → `no_candidate_pre`; empty post → `no_candidate_post` (both **0** observed).
- Secondary: `emb_dist = ‖predict(obs_post) − predict(obs_pre)‖` (the view movement), group (ii) only.
- Primary FINDING per band: group-(ii) divergence rate = n_changed / n_group_ii, with an **episode-clustered
  bootstrap 0.95 CI** (episode = resampling unit; the project convention — events within an episode are
  autocorrelated), reported **per band AND per seed**.

## Data provenance (frozen snapshot 2026-08-02 ~23:15) [ARTIFACT]

Live rollout, 3 concurrent per-band jobs, `RQ2C=1 PYTHONHASHSEED=0`, writing `rq2c_replay/rq2c/rq2c_<band>_seed<seed>_<scenario>.jsonl`.
**The run was WALL-CLOCK-TRUNCATED at the session deadline**, not run to the full 400-ep/seed budget
(the big bands do not early-stop — `membership_join` never reaches its 200-event target — so they would
run the full 2000 episodes/band; they were stopped after seed 42):

| band | seeds completed | leave records | status |
|---|---|---|---|
| 10-15 | **42, 100, 123, 200, 300 (all 5)** | 4965 | **complete** (all 5 seeds finished) |
| 30-40 | 42 only | 3146 | **single-seed, partial** (killed mid-seed-42) |
| 80-100 | 42 only | 2058 | **single-seed, partial** (killed mid-seed-42) |

**Consequence for significance:** the project's standing rule is that the SEED is the unit for any
across-condition claim. That is satisfied only for **10-15** (5 seeds, tight agreement). **30-40 and
80-100 are single-seed point estimates** — their numbers are provisional and the cross-band SCALING
trend is **suggestive, not seed-supported** at the two larger bands until their remaining seeds are run.

## RESULTS [ARTIFACT counts; FINDING = divergence rate]

Implausible-value check (mandatory) **PASSED**: rates are neither all-0 nor all-1; group (i) is
non-empty in every band (the identity/membership test is live, not silently matching); `emb_dist > 0`
in **100%** of group-(ii) events in every band (the view genuinely moves); **0** excluded events.

| band | n_group_i | n_group_ii | n_changed | **rate group-ii (FINDING)** | boot 0.95 CI | emb_dist mean/median |
|---|---|---|---|---|---|---|
| 10-15 (5 seeds) | 1176 | 3789 | 1062 | **0.280** | [0.264, 0.296] | 2.81 / 1.19 |
| 30-40 (seed 42) | 197 | 2949 | 307 | **0.104** | [0.092, 0.117] | 0.91 / 0.44 |
| 80-100 (seed 42) | 18 | 2040 | 101 | **0.050** | [0.038, 0.060] | 0.45 / 0.13 |

Per-seed (10-15 only has the full set): 0.235 / 0.273 / 0.278 / 0.331 / 0.284 (seeds 42/100/123/200/300)
— consistent, spread 0.235–0.331. 30-40 seed42 = 0.104; 80-100 seed42 = 0.050.

## FINDING — behaviour follows the ACTION SET, and does so more with scale [FINDING]

When the agent's preferred target **survives** a membership change — the overwhelming majority of leaves
(group-ii share = 76% / 94% / 99% at 10-15 / 30-40 / 80-100) — its chosen action **usually stays the
same**: the group-(ii) divergence rate is **0.280 / 0.104 / 0.050**, i.e. the choice is unchanged in
**72% / 90% / 95%** of surviving-preferred cases. So **behaviour predominantly follows the action set
(candidate-set membership), not the view**, and this dominance **strengthens monotonically with network
size**. The view is **not inert** — every CI excludes 0, and at the smallest band ~28% of surviving-action
events do change choice — but its behavioural influence **shrinks sharply with scale** (0.28 → 0.10 → 0.05).

The action-set channel itself also collapses with scale: group (i) "preferred action removed" is
**23.7% / 6.3% / 0.9%** of all single-node leaves — at 80-100 a departing node is almost never the target
the policy currently wants, so the leave rarely even removes the preferred action. Net at 80-100: a
membership leave changes the agent's action in only ~6.5% of cases total (0.9% removal + 5.0%×99% view),
despite the view moving in 100% of them — a near-complete decoupling of view movement from behaviour at scale.

**RQ2(c) answer:** the change moves behaviour **primarily through the action set, not the view**, and the
view's residual influence **diminishes with scale** — consistent with the attenuation story (larger pools
swallow more of the per-node change before it reaches behaviour). Weight-bearing at 10-15 (5 seeds);
provisional single-seed at 30-40/80-100 pending their remaining seeds.

## Caveats
- **30-40 / 80-100 are single-seed (seed 42), wall-clock-truncated** — provisional; re-run the remaining
  4 seeds/band to make the scaling claim seed-supported.
- 80-100 agent not confirmed converged (Task F4) — its low divergence partly reflects a weaker policy.
- group (i) carries no choice-changed metric by construction (original choice unavailable to reselect).
- Uses the disclosed action-embedding **staleness** (Task L) as-is: the group assignment is by candidate
  IDENTITY (the tuple), so staleness does not affect it; it only affects the numeric snap distance, an
  existing disclosed property of the system under test.
- The stored headline actions are **not** faithfully replayable (CA-3); this is a fresh seeded rollout.

## Files [ARTIFACT] — all to be committed
- `cyberbattle/_env/cyberbattle_env_compressed.py` — `_rq2c_probe` + pre/post capture in `step()` (gated
  `RQ2C=1`, default-off). **Byte-identical regression PASS**: 0 differing cells vs tag
  `env-baseline-2026-08-01` (1b42a2c), both seeds × 2000 steps, `drift_logging=False`, flag OFF.
- `cyberbattle/agents/compute_attenuation_analysis.py` — attaches model/vecnorm; extends per-seed seeding
  to `RQ2C`; trajectory stays stochastic-seeded (gated).
- `cyberbattle/agents/compute_rq2c_action_divergence.py` — NEW analysis script (this table).
- `cyberbattle/agents/rq2c_replay/rq2c_action_divergence_table.md` — the output table.
- Raw: `cyberbattle/agents/rq2c_replay/rq2c/*.jsonl` (54 files; large dir, manifested not committed if it
  matches the project's excluded-artifact convention).
- `evidence_cards/claims_audit.md` CA-3 — the replay-fidelity scope correction.

Reproduce (per band): `env PYTHONHASHSEED=0 YEG_DRIFT_DIR=<out> RQ2C=1 RQ2C_DIR=<out>/rq2c python
compute_attenuation_analysis.py --manifest <band manifest> --collect`, then
`python compute_rq2c_action_divergence.py --input-dir <out>/rq2c --out <table>.md`.
