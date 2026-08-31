# Task F4 — train to convergence, re-test the central finding

STEP 0 (plan + convergence criterion, fixed IN ADVANCE of any training). No training yet — reported
at the gate for approval. Numbers/provenance only.

> **PROVISIONAL (donor-pool confound, Task G pending):** join-related numbers inherit the ~2.2x
> weaker-pool caveat.

## Why (one line)

The central finding (larger band loses proportionally less: count robustness 0.874/0.775 at 80-100
vs 0.665 at 30-40) compares a CONVERGED 30-40 agent against a NOT-converged 80-100 agent (Task R
point 3). An undertrained agent may not have locked onto specific attack paths and so may lose less
to disruption — a confound sitting directly on the headline number. This task removes it by measurement.

## 0.1 Convergence criterion (FIXED IN ADVANCE) [ARTIFACT]

Metric: `train/Root owned nodes` (the control-goal count, the same metric as the headline). Window
N = **50k timesteps**. Comparing the final 50k window to the immediately preceding 50k window, a band
is **CONVERGED iff (i) the across-seed MEAN |Δ%| < 5% AND (ii) ≥ 4 of 5 seeds have |Δ%| < 5%.**

Justification against the 30-40 noise floor (the band already known converged). Matched 50k windows
(150–200k vs 200–250k), per seed:
- **30-40 (converged):** Δ% = −0.3 / −18.2 / +4.5 / +3.1 / −0.3; mean **−2.3%**; **4/5 seeds within 5%**
  (seed100 −18.2% is a single noisy outlier). → PASSES the criterion.
- **80-100 (not converged):** Δ% = +23.9 / +1.7 / +15.6 / −10.1 / +48.4; mean **+15.9%**; **1/5 within 5%.**
  → FAILS.

The per-seed metric is noisy (30-40 per-seed consecutive-window |Δ%| median 3.8%, p90 16.3% at
N=50k), which is why the criterion uses the across-seed mean as the primary check and tolerates 4/5
(not 5/5) seeds. 5% is the tightest threshold the converged band reliably clears while the climbing
band clearly fails.

## 0.2 How far 80-100 must go + hard ceiling [ARTIFACT]

80-100 across-seed mean Root-owned per 50k window: 13.60 → 17.88 → 22.58 → 22.93 → 25.82
(increments +4.28, +4.70, +0.35, +2.88). Decelerating but the final-window increment is still +2.88
(**+12.6%**), far above the 5% criterion. Extrapolating the deceleration, the increment should drop
below 5% (~+1.3/50k at a level of ~26–30) around **400–500k**. **Target budget 500k; hard ceiling
750k** — if 80-100 still fails the criterion at 750k, STOP and report (deeper undertraining is itself
the finding), do not run open-ended.

## 0.3 Budget rule + cost [ARTIFACT] — recommend option (b)

**Option (b): both large bands to the same 500k budget** (retains the flat 30-40 curve as measured
evidence the extra budget changed nothing there, rather than asserting it). Cost via **resume +250k**
from the existing 250k static checkpoints (fps 249 at 30-40, 89 at 80-100):
- 30-40: 250k/249 ≈ 16.7 min/run × 5 ≈ **1.4 h**
- 80-100: 250k/89 ≈ 46.8 min/run × 5 ≈ **3.9 h**
- (Fresh-500k alternative for reference: 30-40 ~2.8 h, 80-100 ~7.8 h.)

Option (a) (per-band budget to criterion) invites the objection that the larger band got more
compute; (b) avoids it at ~+1.4 h. **Proceed with (b) via resume.**

## 0.4 Third band 10-15 (turns two points into three) [ARTIFACT]

No single-topology static specialists exist at 10-15 (only gate multi-topology agents), so train **5
FRESH single-topology static agents** to 500k, seeds→topologies: 42→scalability_10_15/61 (10 nodes),
100→54 (11), 123→62 (10), 200→29 (12), 300→4 (10). Cost: 500k/459 fps ≈ 18 min/run × 5 ≈ **1.5 h**
(cheapest band; droppable if the schedule tightens).

**Band-specific K caveat (carried into the card, K unchanged):** at 10-15 the proportional cutoff is
NOT inert — episode cap = min(ownable×K, 300); with 10–12 node topologies (ownable ≲ node count),
ownable×25 = 250–300, so K binds and episodes run ~250–275 steps rather than the full 300 that 30-40
and 80-100 get. The 10-15 agents therefore train on a shorter episode budget; this is a property of
the band, not a change to K, and must be disclosed when the third point is used.

## 0.5 Resume mechanics for a longer extension [ARTIFACT]

Confirmed (Task R + re-checked): checkpoint carries `policy.pth`, `policy.optimizer.pth` (TRPO's
critic Adam state) and `checkpoint_vecnormalize_*.pkl` (all restored); `learning_rate_type: constant`
(no schedule to restart). The F1 (30-40) and F2 (80-100) static checkpoints are present. **The static
agents trained with `dynamic_mode=none`** (confirmed), so there is NO dynamic-change RNG — the only
resume discontinuity is a fresh starter-node/action-sampling RNG, which is minimal. **A single resume
is faithful.** Repeated resumes compound discontinuities, so the plan resumes ONCE to 500k and reports
the criterion check; a second resume (→750k) only if a band fails at 500k, disclosed. 10-15 is fresh
(no prior checkpoint). If maximum faithfulness (one continuous run) is preferred, the fresh-500k cost
above (~2×) applies.

## Total cost (option b + 10-15, resume for the two large bands)

30-40 ~1.4 h + 80-100 ~3.9 h + 10-15 fresh ~1.5 h ≈ **~6.8 h** (concurrency will reduce wall-clock).
Existing 250k checkpoints are RETAINED — they are the earlier point in the comparison, not superseded.

## GATE

Reported 0.1–0.5. **Criterion fixed above in advance.** Awaiting approval before STEP 1.

## STEP 1.1 — threshold RECALIBRATION on the eval signal (BEFORE the 500k verdict) [ARTIFACT]

The F4 resume runs emit no tensorboard, so the pre-registered `train/Root owned nodes` signal is
reconstructed by EVALUATING checkpoints (static, N=60 episodes) at the window boundaries. Because the
signal changed (training rollout curve → checkpoint eval), the 5% threshold was re-derived on the eval
signal using the SAME reference data the training-curve criterion used, BEFORE looking at any 80-100
500k number. Windows 150/200/250k, preceding [150,200] vs final [200,250].

- **30-40, known CONVERGED (noise floor):** per-seed window Δ% = −1.8/+4.2/+0.1/+3.6/−0.4 → mean|Δ%|
  **2.0%**, max **4.2%**, **5/5 within 5%**.
- **80-100 @250k, known NOT-CONVERGED:** per-seed Δ% = +15.5/+4.9/+5.0/+1.1/+1.1 → mean|Δ%| **5.5%**,
  **3/5 within 5%** (largely driven by seed42 +15.5%).

**Findings:** (1) the eval signal is QUIETER than the training curve on the converged side — 30-40 is
5/5 within 5% (the training curve needed the 4/5 tolerance for a −18.2% outlier seed). (2) 5% still
SEPARATES on the full two-part criterion: 30-40 CONVERGED (mean 2.0%<5% AND 5/5 within); 80-100@250k
NOT-CONVERGED (mean 5.5%≥5% AND 3/5 within). Margin is thinner than on the training curve (80-100
not-convergence is +5.5% on eval vs +12–16% on training, and mostly one seed).

**Decision: KEEP the 5% threshold + 4/5 rule** (validated on the eval signal: converged max 4.24% <
5% < not-converged mean 5.53%; it's the classification-safe choice — below the not-converged mean).
Reported here BEFORE the 500k verdict below, so the order is auditable.

## STEP 1.2 — CONVERGENCE VERDICT at 500k (checkpoint-eval, 5% criterion, N=60) [FINDING]

Windows 400/450/500k; preceding [400,450] vs final [450,500]; per band across 5 seeds.

| band | across-seed mean\|Δ%\| | seeds within 5% | verdict |
|---|---|---|---|
| **30-40** | 1.1% | 5/5 | **CONVERGED** |
| **80-100** | 6.4% | 1/5 | **NOT CONVERGED** |
| **10-15** | 1.0% | 5/5 | **CONVERGED** |

Per-seed window Δ%: 30-40 = −1.7/+1.2/−0.7/+0.8/−1.2 (all flat). 10-15 = +2.2/+0.4/+1.7/+0.1/−0.5
(all flat). **80-100 = +9.5/−6.8/+9.2/−0.9/+5.8 — 4/5 seeds >5%, mixed sign, net still rising (signed
mean +3.4%), high inter-seed instability.** Not a noise artifact (eval-signal converged floor is
1.1%/2.0%, max 4.2%).

**FINDING (plain): the 80-100 band did NOT converge even at 500k.** Per the pre-registered plan
(0.2: hard ceiling 750k, "second resume to ≤750k if 80-100 still fails at 500k, disclosed"), the next
step is a SECOND resume of the five 80-100 runs 500k→750k, then re-apply this same check. If 80-100
still fails at 750k, STOP and report (deeper undertraining is itself the finding). 30-40 and 10-15 are
converged and need no further training. Downstream (F4 STEP 2/3/4, D3 STEP 2.2, Task H's use of 80-100)
must wait for the 80-100 checkpoint to settle at 750k (or the fail-at-750k stop).

## STEP 1.3 — 80-100 RE-CHECK at 750k (second resume) → STOP [FINDING]

Windows 650/700/750k. Per-seed window Δ%: −3.1 / +4.8 / +2.1 / **+7.9** / **+8.8** → **mean|Δ%| = 5.4%,
3/5 within 5% → NOT CONVERGED at 750k.** Genuinely still rising (not noise): seeds 200/300 final-window
increments +13.1% / +16.2%; root_owned grew materially over the 500k→750k resume (seed200 ~37.8→43.2,
seed300 ~34.8→37.2). Clears the eval-signal noise floor (~2%).

**FINDING (headline, stated plainly): the 80-100 band does NOT reach the convergence criterion even at
750k — ~3× into the released study's budget region — and is still measurably improving.** Per the
pre-registered ceiling (0.2: "if 80-100 still fails at 750k, STOP and report; do not run open-ended"),
training STOPS here. The 250k F2/F3 results and the F4 500k results were produced by agents that had not
converged; and unlike 30-40 (converged) and 10-15 (converged), the large band's undertraining CANNOT be
removed within a feasible budget. This means the central-finding confound F4 set out to eliminate is
only *partially* removable: 30-40 vs 80-100 will remain a converged-vs-not-fully-converged comparison
even after maximal feasible training.

**Consequence for downstream (needs a decision, not assumed):** the best-available "as converged as
feasible" 80-100 checkpoint is 750k. F4 STEP 2/3/4 (re-run F2/F3 cells + the 250k-vs-converged
comparison) and D3 STEP 2.2 (80-100 substitution) would use the **750k** 80-100 checkpoints, disclosed
as not-fully-converged (still +5.4%/rising), against the CONVERGED 30-40 (500k) and 10-15 (500k). 30-40
and 10-15 need no further training. Reported to the user for direction before running STEP 2+.
