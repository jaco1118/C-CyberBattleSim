# Task R — can the existing 250k agents be resumed to 300k?

Verification only. No training, no runs, no files modified. Source- and log-quoted.

## Recommendation

**Technically possible, but NOT worth it for the dynamicity replication — backed by a flat 30-40
learning curve.** The replication band (30-40) has CONVERGED by 250k (final-50k Δ = −0.14 root-owned,
noise), so 50k more produces agents indistinguishable from the ones we have and would not move the
−0.262 slope. And the K half of the "budget" difference is IMMATERIAL at 30-40 (episode length is 300
at both K=25 and K=10), so matching the budget would not remove a real difference there anyway.
Separately (and more seriously), the **80-100 agents were still climbing at 250k** — a finding worth
disclosing independent of this exercise.

## 1. Is a faithful resume possible? [ARTIFACT]

Saved checkpoint dir contains two artefacts: `checkpoint_250000_steps.zip` and
`checkpoint_vecnormalize_250000_steps.pkl`.
- **(a) model .zip contents:** `policy.pth` (policy + value network params), `policy.optimizer.pth`,
  `pytorch_variables.pth`, `data`, version/system info.
- **(b) VecNormalize running stats: SAVED** (`checkpoint_vecnormalize_*.pkl`, `save_vecnormalize=True`
  in the CheckpointCallback). This is the one most likely to be missing; it is present. A resumed run
  loads it and continues with the same observation normalisation — **positive**.
- **(c) env/sampler RNG: NOT recoverable.** The environment/topology-sampler/dynamic-change RNG is
  process-global (`set_seeds` at training start), not part of the SB3 checkpoint; a resumed process
  starts a fresh RNG. Consequence: for a single-topology run (F1/F2/single-topo dynamicity) the
  topology is trivially the same; for a multi-topology run (gate, RandomSwitchEnv over 8) the extra
  50k revisits topologies/scenarios in a DIFFERENT but statistically-equivalent sequence — a valid
  continuation in distribution, not a bit-identical one.
- **(d) optimiser state: SAVED and restored.** `policy.optimizer.pth` is present; TRPO's value
  function uses Adam (`m.policy.optimizer` = Adam) whose state is restored, while the policy update
  uses conjugate gradient + line search (no persistent momentum state). So optimiser continuity holds
  — **positive**.

Verdict: a faithful-enough resume IS possible (VecNormalize + optimiser restored, LR constant); the
only discontinuity is the fresh env RNG, immaterial for single-topology and distributionally fine for
multi-topology.

## 2. Is the learning rate scheduled? [ARTIFACT]

`algo_config.yaml` trpo: `learning_rate_type: constant`, `learning_rate: 0.0001`; the frozen
`train_config.yaml` confirms `learning_rate_type: constant`. `train_agent.py` applies a
`linear_schedule` ONLY when `learning_rate_type == "linear"`, else a constant. **Constant — no
schedule to restart. Point closes, positive.**

## 3. Had the agents converged by 250k? (existing tensorboard logs, no runs) [FINDING]

`train/Root owned nodes` (the control metric), mean over 200–250k vs the preceding 175–200k window,
5 seeds/band:

| band | 175–200k mean (sd) | 200–250k mean (sd) | Δ | verdict |
|---|---|---|---|---|
| 30-40 | 15.80 (1.33) | 15.66 (1.01) | **−0.14** | **CONVERGED — flat** |
| 80-100 | 22.93 (3.79) | 25.69 (0.96) | **+2.75 (~12%)** | **STILL CLIMBING** (4/5 seeds up) |

**FINDING (stated plainly, not softened):** at 30-40 the curve is flat over the final 50k — a further
50k changes nothing measurable and would produce identical agents. At 80-100 the curve is still
rising (+2.75 root-owned, ~12%, with the between-seed sd tightening 3.79→0.96), so **the 80-100 F2/F3
results were produced by agents that had not fully converged at 250k.** This is a more serious finding
than the budget discrepancy and is reported as such.

## 4. What would it cost? [ARTIFACT]

Median training throughput from the original runs' tensorboard `time/fps`: **30-40 = 249 fps**,
**80-100 = 89 fps**. 50k timesteps: 30-40 = 50000/249 ≈ 201s ≈ 3.4 min/run → **~17 min for 5 runs**;
80-100 = 50000/89 ≈ 562s ≈ 9.4 min/run → **~47 min for 5 runs**. Cheap at both bands (<1h).

## 5. K is the larger discrepancy — and it is IMMATERIAL at these bands [FINDING]

- **(a)** K enters via `proportional_cutoff_reached`: truncate when
  `num_iterations >= self.proportional_nodes * self.proportional_cutoff_coefficient`
  (`cyberbattle_env.py:986`), checked BEFORE the `episode_iterations` cap. For control,
  `proportional_nodes = ownable_count`. Effective cap = `min(ownable × K, episode_iterations=300)`.
- **(b) Where K binds — the prior finding is REVERSED.** Measured ownable (reachable column):
  **30-40 topo44 = 32** (constant), **80-100 = ~80** (73–92). At K=25: 32×25=800 and 80×25=2000, both
  ≥300, so **300 binds and K does NOT bind at 30-40 or 80-100.** K binds only at 10-15 (ownable
  ~10–15 → 250–375, below 300 for the smaller topologies). The task's prior "K binds at 30-40 and
  80-100 but not 10-15" is backwards.
- **(c) K=25 → 10 effect on episode length (measured):** 30-40 topo44 (ownable 32): 32×10=320 ≥300 →
  **300, no change**. 80-100 (ownable ~80): 800 ≥300 → **300, no change**. 10-15 (ownable ~10):
  10×10=100 <300 → K binds → episodes shorten to ~100 (from ~250). So K=10 changes episode length
  ONLY at 10-15.
- **(d)** Could K=10 be adopted without invalidating comparability with this work's OWN results?
  At 30-40 and 80-100 — the replication and F-series bands — K=10 yields the SAME 300-step episodes as
  K=25, so it is the SAME task and **comparability is preserved** (it could be adopted with no effect
  there). It would only shorten 10-15 episodes, and 10-15 is not used for the dynamicity replication
  or the F-series.

**Consequence:** at the replication bands the K difference is not a *realised* difference (both give
300-step episodes), so the "budget" divergence reduces to the raw timestep count (250k vs 300k), and
at 30-40 that extra budget changes nothing (converged, point 3). Matching K to 10 is free at these
bands. This NARROWS the list of divergences (K dissolves at these bands; single-topology training and
reward reweighting remain) but does not change what the replication can claim: it stays a
shape/direction comparison, not a clean single-factor test, because ≥2 structural differences remain.

## 6. What would have to be re-run downstream? [FINDING]

- **If 300k checkpoints REPLACED the 250k ones:** every evaluation built on those agents must be
  re-run — F1-R (30-40), F2 (80-100), F3 (all sub-analyses), and both Condition B arms. That is
  essentially the entire F-series eval grid (~hundreds of cells), far exceeding the training cost in
  point 4.
- **But the central findings rest on INTERNAL comparisons** (same agent evaluated with and without a
  change, at an identical budget), which need only within-comparison consistency, not any particular
  absolute budget. **So it is coherent to resume ONLY the dynamicity-replication arm to 300k, leaving
  F1-R/F2/F3 on their 250k agents, and disclosing the two budgets separately.** The cleanest such arm
  is the multi-topology gate agents (JOB 1 Condition B), which are SEPARATE from the F1/F2 agents used
  by F1-R/F2/F3 — resuming them touches nothing else; re-running only the 25-cell multi-topology
  Condition B eval (~1–2h) would suffice. The single-topology Condition B (F1) shares agents with
  F1-R, so it should stay at 250k (or defer to the multi-topology arm). No reason this decomposition
  fails.
- **However**, point 3 shows the 30-40 agents are converged, so even the resumed dynamicity arm would
  produce a slope indistinguishable from −0.262 — which is why the recommendation is "not worth it".
