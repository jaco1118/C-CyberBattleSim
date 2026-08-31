# Task F3 — relative-churn sensitivity, and who actually leaves

Numbers and provenance only. No thesis wording. Analysis on existing agents + short instrumented
characterisation evals (frozen policy, no training).

> **PROVISIONAL (donor-pool confound, Task G pending):** `membership_join` draws from a shared
> donor pool ~2.2x weaker at the large band; every join-related number inherits this caveat.

## STEP 0 findings (reported before the STEP 1 sweep, per the GATE)

### 0.1 — how the departing node is chosen (quoted from source) [ARTIFACT]

Two independent parts, both in `cyberbattle/_env/cyberbattle_env.py`, **identical config at both
bands**:

**Eligibility filter** (`_get_removal_eligible_nodes`), verbatim:
```python
return [node for node in self.discovered_nodes
        if self.get_node(node).status == model.MachineStatus.Running
        and node != self.starter_node and node != self.source_node
        and node != self.target_node and node != getattr(self, "interest_node", None)]
```
→ only **discovered + running + non-protected** nodes are eligible; an undiscovered node can never
leave.

**Weighting** (`_apply_dynamic_leave`, `dynamic_degree_weighting: true`), verbatim:
```python
degrees = {n: (self.access_graph.degree(n) if n in self.access_graph else 0) for n in eligible}
weights = {n: 1.0 / (1.0 + degrees[n]) for n in eligible}
...
probabilities = {n: min(self._DYNAMIC_P_MAX, target_rate * weights[n] / weight_sum) for n in eligible}
```
→ **inverse-degree bias: low-degree nodes are preferentially removed.** The Poisson batch removal
uses the same weights. **Not** weighted by node value, ownership, or discovery-beyond-eligibility
(value weighting exists only for *join*). Floor: `max(dynamic_min_alive_nodes=5,
ceil(0.5·num_nodes))`.

**FINDING: the selection rule is degree-biased (low-degree) and discovery-gated, and is IDENTICAL
across bands.** No rule-level bias differs between bands. Whether the rule's *effect* differs is 0.2
/ 0.3.

### 0.2 — who actually left, per band [ARTIFACT / FINDING]

Instrumented stochastic membership eval (natural rate), 5 replicates/band, 60 episodes each
(30-40: F1 static agents on topo44; 80-100: F2 static agents on their topos). Pre-step snapshot of
ownership/discovery/degree/value (nodes are pruned on leave). **Zero dropped.**

| quantity | 30-40 (3852 leave events) | 80-100 (4662 leave events) |
|---|---|---|
| **was_owned_frac** | **0.503** (per-seed 0.478–0.520, sd 0.020) | **0.231** (per-seed 0.150–0.292, sd 0.052) |
| discovered-not-owned frac | 0.489 | 0.760 |
| neither disc nor owned | 0.008 | 0.008 (≈0 structural; residual = same-step discovery snapshot timing) |
| departed degree mean / median | 52.85 / 62 | 106.66 / 148 |
| departed value mean / median | 52.67 / 52 | 53.58 / 55 |

**FINDING (GATE-TRIGGERING): the already-owned fraction of departing nodes is materially higher at
30-40 (0.503) than 80-100 (0.231) — a robust ~2.2× difference.** At 30-40 ~half of all departures
are of nodes the agent already owned (a mechanical loss no policy could prevent); at 80-100 only
~23% are. Departures are ~100% discovered at both bands (confirming the eligibility filter). Node
value of departures ≈ typical at both bands (no value bias, consistent with 0.1). The **rule is
identical** (0.1); this difference arises because the agent OWNS a larger fraction of the network at
30-40 (static ceiling 0.727 vs 0.372), so a degree-weighted, ownership-blind departure is far more
likely to hit an owned node there.

**Consequence for the F2 opposite-directions result:** a substantial part of the higher membership
cost at 30-40 (+0.0614 vs +0.0111 at 80-100) is attributable to more OWNED nodes departing
(mechanical/arithmetic loss), not to the change being more perceptible or more important. This is a
simpler partial explanation than "harder to perceive AND less important as networks grow," and the
relative-churn sweep (STEP 1) would not resolve it — matching the *number* of departures does not
match their *ownership composition*.

### 0.3 — eligible pool vs whole network degree [ARTIFACT]

| band | whole-net degree mean/median | eligible-pool degree mean/median (at first leave) | departed degree mean/median |
|---|---|---|---|
| 30-40 | 58.35 / 62 | 59.89 / 62 | 52.85 / 62 |
| 80-100 | 148.23 / 152 | 155.10 / 152 | 106.66 / 148 |

**FINDING:** the eligible pool's degree distribution ≈ the whole network's at both bands (discovery
does not strongly bias by degree). Departed nodes have lower degree than the eligible pool at both
bands (the inverse-degree weighting), and the gap is larger at 80-100 (departed 106.7 vs eligible
155.1) than 30-40 (52.9 vs 59.9) — the degree bias bites somewhat harder on the denser large-band
graphs. These are dense graphs (median degree 62 at 30-40, 152 at 80-100).

### 0.4 — achievable leave rate for the relative-churn condition [ARTIFACT]

Measured under the trained F2 policy (seed42/topo5, 85 nodes; floor=43, so ≤~42 leaves/ep possible):
`change_interval=10 → 24.23`, `8 → 27.37`, `6 → 33.00` leave/ep. **`change_interval=8` gives 27.37
leave/ep = 32.2% fractional churn**, essentially matching 30-40's ~32% (11.2/34). Achievable and
sustained (well below the floor cap). Target ~28/ep is reachable — no shortfall.

### Fractional-churn summary (the controlled quantity) [ARTIFACT]

| condition | leave/ep | nodes | fractional churn |
|---|---|---|---|
| 30-40 natural | ~11.2 | 34 | ~32.9% |
| 80-100 fixed-absolute (F2) | ~15.7 | 85 | ~18.5% |
| 80-100 fixed-relative (proposed, ci=8) | ~27.4 | 85 | ~32.2% |

## GATE status: STOPPED before STEP 1, pending user decision

Per the GATE ("if 0.1 or 0.2 shows a selection bias that differs between bands, stop and report;
that finding outranks the sweep"): **0.2 shows the already-owned departure fraction differs
materially between bands (0.50 vs 0.23)** — a downstream consequence of the identical
ownership-blind rule interacting with the agent's differing ownership level. Stopped and reported;
user directed the ownership-split re-analysis below FIRST (Option 3), then the sweep.

## Ownership split — mechanical vs behavioural decomposition of the membership cost

**Method note / negative result (ARTIFACT).** The naive cross-episode OLS (score ~ owned_dep +
unowned_dep) is INVALID here: owned-departure count is endogenous to episode quality (a better
episode owns more, so more owned nodes are available to depart), giving positive coefficients
(b_owned=+0.020) and higher scores at higher owned-departure counts — reverse causation, not a
mechanical loss. Reported as a documented failure; not used.

**Valid method (ARTIFACT).** Instrumented each departure's INSTANTANEOUS arithmetic score impact
(`remove_node_dynamic` wrapped per-instance, no schema/behaviour change): delta = root_owned/ownable
just after vs just before that single removal. **Only ROOT-owned departures lower the score;
un-owned and owned-non-root departures shrink the denominator and RAISE it.** Per-episode net
arithmetic displacement = Σ deltas. cost = mean(static) − mean(membership); behavioural = cost −
arithmetic loss. Cluster-bootstrap over 5 replicates. 100 ep × 5 replicates/band (mech run);
6375 (30-40) + 7847 (80-100) removals. Zero dropped.

| quantity | 30-40 | 80-100 |
|---|---|---|
| per-removal Δ: owned-ROOT / owned-nonroot / un-owned | −0.0181 / +0.0131 / +0.0094 | −0.0096 / +0.0033 / +0.0016 |
| root-owned departures per episode | 5.39 | 2.69 |
| **total cost** | +0.0635 [+0.052,+0.075] | +0.0112 [+0.006,+0.017] |
| **ARITHMETIC (mechanical) loss** | **+0.0205 [+0.016,+0.025] = 32%** | **+0.0022 [+0.002,+0.003] = 19%** |
| — owned-root loss offset by denom-shrink gain | +0.097 offset by +0.077 | +0.026 offset by +0.024 |
| **BEHAVIOURAL residual** | **+0.0430 [+0.031,+0.056] = 68%** | **+0.0091 [+0.004,+0.014] = 81%** |

**FINDING: the `was_owned` fraction (0.2) OVERSTATES the mechanical channel.** The owned-root
arithmetic loss is largely (~79%) offset by the denominator-shrink gain from all the other
departures, so the net mechanical loss is only 32% of the cost at 30-40 and 19% at 80-100 — a
MINORITY at both bands. The gate concern (that the 30-40 cost is "arithmetic rather than
perception") is not borne out: most of the cost is behavioural at both bands.

**FINDING: the F2 opposite-directions result SURVIVES the ownership decomposition.** The behavioural
component — the only part a perception failure could affect — is +0.0430 at 30-40 vs +0.0091 at
80-100 (both intervals exclude 0), i.e. it FALLS with scale (~4.7×) just as the total cost does,
while attenuation RISES (0.2/0.3, 3.4). The mechanical channel does not explain away the opposite-
directions pattern. The behavioural cost is not near zero at either band, so this is not a clean
RQ3 null — there is a real, perception-relevant cost that falls with scale, opposite to attenuation.
**Caveat:** the arithmetic deltas are instantaneous (removal-time); behavioural = cost − arithmetic
is approximate (the two compound over the trajectory).

**The quantity to compare against the attenuation curve is the BEHAVIOURAL residual** (mechanical
loss is independent of perception).

## STEP 1 relative-churn sweep (80-100 at ci=8) [ARTIFACT / FINDING]

Evaluation only, F2 static agents, stochastic, 200 ep × 5 replicates. `change_interval=8`.
**Achieved: 27.43 leave/ep = 32.3% fractional churn** (target ~32%, matching 30-40); was_owned_frac
0.253; zero dropped.

Three conditions side by side (control metric = root_owned/reachable):

| condition | leave/ep | churn % | cost (static−cond) | robustness | behavioural residual | leave max/min RR |
|---|---|---|---|---|---|---|
| 30-40 natural | ~11.2 | ~32.9% | +0.0614 | 0.916 | +0.043 (68%) | 0.813 / 0.827 |
| 80-100 fixed-absolute (F2) | ~15.7 | ~18.5% | +0.0111 | 0.970 | +0.0091 (81%) | 0.644 / 0.657 |
| 80-100 fixed-relative (ci=8) | 27.4 | 32.3% | **−0.0110** [−0.018,−0.003] | **1.030** | −0.0167 (breaks down) | 0.661 / 0.664 |

**FINDING (the sweep does not adjudicate — it exposes a metric artifact):** under fixed-relative
churn the total cost is **negative** (membership score 0.3827 > static 0.3717, robustness 1.03).
This is the denominator artifact the ownership split identified: at 32% churn a large fraction of
`reachable` (ownable_count) is removed, and since root_owned/reachable rises when the denominator
shrinks, heavy churn arithmetically INFLATES the score. The behavioural decomposition also flips
sign (−0.017) because the instantaneous per-removal deltas do not capture the PERSISTENT denominator
reduction over the rest of the episode — the decomposition is valid at moderate churn (fixed-
absolute) but breaks down at high churn. So the score metric is **not churn-robust** at this
fraction, and the fixed-relative condition cannot cleanly say whether cost "rises" or "falls".

**How to read the result (per the pre-stated framing):** neither churn condition isolates
attenuation. Of the two anticipated outcomes — "cost falls under BOTH conditions" (robust
opposite-directions) or "cost RISES under relative churn" (disturbance size drove F2) — **neither
holds**: under fixed-relative churn the cost does not rise, it goes negative via the denominator
artifact. The evidence therefore does NOT independently confirm the F2 opposite-directions result
at equal fractional disturbance, and does NOT overturn it either. What it establishes is narrower
and, arguably, more important: **the control metric inflates under heavy churn, so total-cost
comparisons across churn conditions are contaminated, and the cleanest available signal remains the
behavioural component under the milder fixed-absolute condition** (where it is +0.043 at 30-40 vs
+0.0091 at 80-100 — falling with scale, opposite to attenuation). The per-slice leave response at
32% churn (max 0.66 / min 0.66) is close to the fixed-absolute 80-100 figure (0.64/0.66) — more
events, similar per-event perceptibility.

## Summary — what F3 established

1. **0.1:** leave selection is discovery-gated + inverse-degree-weighted, identical at both bands
   (not value- or ownership-weighted).
2. **0.2/ownership split:** the `was_owned` between-band difference (0.50 vs 0.23) OVERSTATES the
   mechanical channel; the directly-measured arithmetic loss is a minority at both bands (32% /
   19%), because non-root departures shrink the denominator and offset the owned-root loss. The
   behavioural (perception-relevant) component is majority and falls with scale (+0.043 → +0.0091),
   so the F2 opposite-directions result survives the mechanical confound under fixed-absolute churn.
3. **STEP 1:** at equal fractional churn (32%) the control metric inflates via the denominator, so
   the relative-churn condition cannot cleanly confirm or overturn the result; it exposes that
   root_owned/reachable is not churn-robust. Reported as a documented limitation, not a positive or
   negative result. No trend fitted through the points.

> **PROVISIONAL (donor-pool confound, Task G pending):** `membership_join` draws from a shared donor
> pool ~2.2x weaker at the large band; every join-related number inherits this caveat.

---

# Appendix — recomputation on the churn-invariant measure (ROOT-OWNED COUNT)

Analysis only, no new runs. Metric here = **root-owned COUNT** (numerator only, NOT divided by
reachable), which removes the denominator artifact that contaminated the ratio metric at high churn.
Robustness = count(change)/count(static) formed WITHIN a band (units cancel). **Every row below is
labelled with its metric. The ratio metric root_owned/reachable is retained ONLY for the Condition B
replication (comparability with the released study); it is NOT mixed with the count metric here.**

> **PROVISIONAL (donor-pool confound, Task G pending):** join-related numbers inherit the ~2.2x
> weaker-pool caveat.

## Four conditions on the COUNT metric [ARTIFACT / FINDING]

| condition (churn %) | metric | static count | change count | cost (count) | robustness (count) |
|---|---|---|---|---|---|
| 30-40 fixed-absolute (~33%) | root-owned COUNT | 23.26 | 15.48 | +7.79 [7.44, 8.14] | **0.665** [0.643, 0.684] |
| 80-100 fixed-absolute (~18%) | root-owned COUNT | 29.67 | 25.92 | +3.74 [3.26, 4.22] | **0.874** [0.864, 0.883] |
| 80-100 fixed-relative (32.3%) | root-owned COUNT | 29.67 | 23.00 | +6.66 [5.05, 8.00] | **0.775** [0.757, 0.800] |

**FINDING: on the count metric all three change conditions have POSITIVE cost** (the agent ends
with fewer root-owned nodes under churn) — the negative-cost / robustness>1 result the ratio metric
produced at 32% churn was a pure denominator artifact and is gone.

## Mechanical / behavioural split on the COUNT metric [FINDING]

On counts a root-owned departure is a full −1 with NO offsetting denominator gain, so mechanical =
root-owned departures per episode; behavioural = cost − mechanical.

| condition | mechanical (dep/ep) | % of cost | behavioural (count) | behavioural % of STATIC count |
|---|---|---|---|---|
| 30-40 fixed-abs (~33%) | 5.39 [5.20, 5.58] | **69%** | +2.40 [1.93, 2.93] | 10.3% |
| 80-100 fixed-abs (~18%) | 2.69 [2.26, 3.07] | **72%** | +1.06 [0.81, 1.28] | 3.6% |
| 80-100 fixed-rel (32.3%) | 5.06 [4.31, 5.71] | **76%** | +1.61 [0.65, 2.39] | 5.4% |

**FINDING: the split changes materially and the count version SUPERSEDES the ratio version for the
scale/churn comparison (RQ2/RQ3).** On the ratio metric mechanical was a MINORITY (32% at 30-40, 19%
at 80-100) because non-root departures shrank the denominator and offset ~79% of the owned-root
loss. That offset was itself a denominator artifact: on the count metric it disappears entirely and
**mechanical is the MAJORITY at all three conditions (69–76%)**. Behavioural is real but a minority
(24–31% of cost; intervals exclude 0). The ratio-based 32%/19% mechanical fractions are superseded
here and retained only where the ratio metric itself is used (Condition B).

## Does the relative-churn condition now adjudicate? [FINDING]

**Yes — on the count metric it adjudicates, and the opposite-directions result is robust to
equalising disturbance as a fraction of the network.**

At EQUAL fractional churn (30-40 fixed-absolute ~33% vs 80-100 fixed-relative 32.3%):
- count robustness: 0.665 (30-40) vs **0.775** (80-100) — the larger network is MORE robust, not
  less, even when the same fraction of nodes churns.
- behavioural cost as a fraction of static count: 10.3% (30-40) vs **5.4%** (80-100) — behavioural
  cost FALLS with scale at equal fractional disturbance.
- within 80-100, raising churn 18%→32% raised behavioural 3.6%→5.4% (more disturbance → more cost,
  as expected), but 80-100 at 32% still sits below 30-40 at 33%.

This is the "cost falls under BOTH churn conditions" outcome named in the pre-stated reading — a
STRONGER claim than F2 alone: the behavioural (perception-relevant) cost falls with scale whether
disturbance is held fixed in absolute terms or as a fraction of the network, while attenuation of
the change signal rises (leave max/min response 0.81/0.83 → 0.64/0.66). Disturbance size is NOT what
drove the F2 result. (Two/three points only — no trend fitted, no extrapolation.)

**Metric labelling:** every number in this appendix uses root-owned COUNT (churn-invariant). The
ratio metric root_owned/reachable is used only in the Condition B replication section and in F1-R/F2
where comparability, not churn-invariance, is the priority.

> **PROVISIONAL (donor-pool confound, Task G pending):** carried into this appendix.
