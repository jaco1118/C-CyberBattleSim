# Condition B on the multi-topology gate checkpoints (JOB 1)

Numbers and provenance only. No thesis wording. Evaluation only — no training.

> **PROVISIONAL (donor-pool confound, Task G pending):** not directly relevant here (Condition B
> has no join events), carried for consistency across the F-series outputs.

## Purpose / provenance

Isolates the single biggest known difference between this setup and the released study:
training across many topologies vs one. F1's single-topology specialists degrade under the
released study's own dynamicity condition with a fitted slope −0.262 [−0.304, −0.216], where that
study reported +0.09 for the control goal. Here the EXISTING band-30-40 gate checkpoints (5 agents,
each trained across the band's 8 grid topologies — no retraining) are run under the same Condition
B sweep, on those same 8 topologies (`grid_topologies_30-40/{1..8}`, RandomSwitchEnv switching
every 5 episodes per the gate manifest).

- Defender: `ExternalRandomEvents` at per-node intervention probabilities pn ∈ {0.01, 0.10, 0.25,
  0.50}; pn=0 attaches no defender (undisturbed baseline). Control goal, **stochastic** eval.
- Capture: F1-validated terminal method (`switch_env.root_owned_nodes` / `.reachable_count` /
  `current_env._episode_count` at the terminal step before reset; never `get_statistics()` after
  done; never join on the post-reset counter).
- **Episode budget: 200 episodes per cell** (same as F1's Condition B). 5 agents × 5 pn = 25 cells,
  all exit 0, 200/200 episodes each. **Zero dropped.**
- Checkpoints: gate `trpo_250k_tuned_compressed_band30-40_seed{42,100,123,200,300}` final
  (250k-step) checkpoints + their VecNormalize.
- **Firewall defect retained for fidelity** (as in F1): `firewall_change_add`'s outgoing branch
  tests membership against `.incoming` not `.outgoing` (`static_defender.py:136`). Terranova's
  published dynamicity results used it; kept unchanged so this replicates his mechanism.

## (b) Undisturbed baseline (pn = 0) per gate agent, on its own 8 topologies [ARTIFACT]

| gate seed | n | mean | median | frac_zero | min | max |
|---|---|---|---|---|---|---|
| 42 | 200 | 0.7780 | 0.784 | 0.000 | 0.286 | 1.000 |
| 100 | 200 | 0.7250 | 0.733 | 0.005 | 0.000 | 0.966 |
| 123 | 200 | 0.7050 | 0.714 | 0.000 | 0.069 | 0.966 |
| 200 | 200 | 0.7625 | 0.784 | 0.005 | 0.000 | 0.966 |
| 300 | 200 | 0.7120 | 0.733 | 0.010 | 0.000 | 0.966 |

**FINDING: the multi-topology gate agents are NOT near the floor** — undisturbed baselines
0.705–0.778 (mean ~0.74), comparable to the F1 specialist's ~0.72 on topology 44 (stochastic).
This is the deciding check: the earlier "gate policy ~0.005 on topology 44" figure was a
deterministic-eval artifact; under stochastic eval on their own topologies these agents are
competent. A slope reported below therefore reflects real degradation, not "nothing to lose".

## (a) Score vs pn, and fitted slope [ARTIFACT / FINDING]

Per-agent mean score (ARTIFACT):

| gate seed | pn=0 | pn=0.01 | pn=0.10 | pn=0.25 | pn=0.50 |
|---|---|---|---|---|---|
| 42 | 0.778 | 0.551 | 0.351 | 0.356 | 0.336 |
| 100 | 0.725 | 0.525 | 0.347 | 0.320 | 0.322 |
| 123 | 0.705 | 0.492 | 0.297 | 0.301 | 0.291 |
| 200 | 0.762 | 0.528 | 0.372 | 0.332 | 0.362 |
| 300 | 0.712 | 0.495 | 0.318 | 0.302 | 0.299 |

**FITTED slope (score vs pn over {0.01, 0.10, 0.25, 0.50}, cluster-bootstrap over seeds):**

| agents | slope | bootstrap 95% | undisturbed baseline |
|---|---|---|---|
| **multi-topology (gate)** | **−0.3043** | **[−0.3218, −0.2833]** | 0.705–0.778 (mean ~0.74) |
| single-topology (F1 specialist) | −0.2620 | [−0.3037, −0.2160] | ~0.72 (topology 44) |

**FINDING: the multi-topology agents degrade under the defender at least as steeply as the
specialists — not flat.** The two slope intervals overlap (gate −0.322…−0.283 vs specialist
−0.304…−0.216 overlap in −0.304…−0.283), so the two are statistically similar, with the gate
point estimate slightly steeper. Both show the same saturating shape: a sharp drop from pn=0 (~0.74)
to pn=0.10 (~0.33), then a plateau (~0.30–0.36) through pn=0.50; the linear slope is dominated by the
0→0.10 drop (fit over nonzero pn to match F1's method). **Multi-topology training did NOT confer
robustness to this defender.**

## Interpretation (bounds, per instruction 1.6)

This converts the largest known confound into a measured result: training breadth (multi- vs
single-topology) does **not** explain the gap between our agents' degradation (slope ≈ −0.26 to
−0.30) and the released study's reported +0.09. Both of OUR OWN agent families degrade steeply and
similarly. **This is NOT presented as refuting the released study** — the reward configuration and
training budget still differ from it. The comparison isolates one factor between two of our own
agents, which is narrower and defensible. Comparison to +0.09 is SHAPE and DIRECTION only (both our
families are negative; the released figure is ~flat/positive); reward configs differ, no ratio.

Filter stages: 25 cells in, 25 out, 200 episodes each, 0 dropped, 0 non-zero exits. Baseline
zero-score fraction 0.000–0.010 (a handful of failed episodes, does not compress the ~0.74 mean).

> **PROVISIONAL (donor-pool confound, Task G pending):** carried for consistency; no join events
> in Condition B.
