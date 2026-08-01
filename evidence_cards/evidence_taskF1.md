# Task F Pass 1 — does the lost signal change what the agent does?

Numbers and provenance only. No thesis wording.

> **PROVISIONAL (donor-pool confound, Task G not yet run):** `membership_join` events draw from a
> shared donor pool ~2.2x weaker at the large band. Every join-related number here inherits this
> caveat. This banner is repeated at the foot of the card.

## Provenance

- Code: no repo code changes in this task. HEAD = `e714f53` (Task D: property-change wiring).
  All F1 scripts (`taskF1_train.py`, `taskF1_eval.py`, `taskF1_analyze.py`, CHECK/diag scripts)
  live in the job scratch dir, not the repo.
- Topology: `scalability_30_40/44` (34 nodes, 565 vulnerabilities; = gate grid slot 3). Single
  topology by design (both agents face identical difficulty).
- Agents: 10 TRPO checkpoints, 250k steps, tuned config verbatim from the gate's own 30-40
  seed42 `train_config.yaml`. STATIC (dynamic_mode=none, patch off) vs ADAPTED (dynamic_mode=both,
  patch on) × seeds 42/100/123/200/300. Item-3 check: static vs adapted policy params differ for
  all 5 seeds (max|Δ| 0.29–0.34 across 17 tensors, none identical) — comparisons not vacuous.
- **Episode budget: 200 episodes per cell, every cell** (Condition A: 2 agents × 5 seeds × 3 eval
  conditions = 30 cells = 6000 episodes; Condition B: static × 5 seeds × 4 pn = 20 cells = 4000
  episodes). All 50 cells exit 0, 200/200 episodes each. **Zero skipped, zero crashes.**

## Evaluation-mode decision (with justification on record)

**Evaluation is STOCHASTIC (`predict(deterministic=False)`).** Justification, on record:
- **FINDING (architecture — retained, not discarded):** DETERMINISTIC (greedy) evaluation
  collapses this continuous-action Gaussian policy. The mean action maps through cosine-nearest to
  a repeated unproductive action, so under deterministic static-eval **2 of 10 cells discover
  exactly 1 node every episode** (static-seed42, adapted-seed200) and the rest sit at the root%
  floor. Deterministic static-eval root% per seed (ARTIFACT, N=40):

  | agent | seed | mean | max | frac_zero | discovered (mean) |
  |---|---|---|---|---|---|
  | static | 42 | 0.000 | 0.000 | 1.00 | 1.0 |
  | static | 100 | 0.009 | 0.0625 | 0.73 | 20.9 |
  | static | 123 | 0.006 | 0.0625 | 0.85 | 21.1 |
  | static | 200 | 0.010 | 0.0312 | 0.68 | 18.1 |
  | static | 300 | 0.005 | 0.0312 | 0.85 | 20.3 |
  | adapted | 42 | 0.000 | 0.000 | 1.00 | 5.3 |
  | adapted | 100 | 0.005 | 0.0312 | 0.85 | 20.0 |
  | adapted | 123 | 0.002 | 0.0312 | 0.95 | 11.3 |
  | adapted | 200 | 0.000 | 0.000 | 1.00 | 1.0 |
  | adapted | 300 | 0.000 | 0.000 | 1.00 | 12.1 |

  A reader who evaluates deterministically and sees zeros should find this documented, not
  conclude the agent does not work.
- Stochastic evaluation is principled, not chosen for nicer numbers: (a) it is what TRPO's
  objective optimises (return is an expectation over the policy's action distribution), and (b) it
  is the inherited/released convention — SB3 `predict()` defaults `deterministic=False` and every
  eval predict call in the released code (`test_utils.py:60,243,275`, `callbacks.py:267`) uses that
  default with no override. Condition B (replication) is therefore also stochastic.

## Score-capture method (CHECK 1-validated)

SB3 vec envs auto-reset on `done`; `get_statistics()` read after `done` returns the freshly-reset
episode. **ARTIFACT (CHECK 1, 20 episodes):** the naive post-reset read recorded root_owned=0 on
every episode the agent actually rooted a node (3/20), agreeing with the correct read only when
the true terminal value was also 0; the post-reset episode counter was +1 on 20/20. Correct method
used throughout: manual stepping loop, at the terminal step BEFORE our own reset read
`switch_env.root_owned_nodes` / `.reachable_count` / `.discovered_nodes` (cached in
`cyberbattle_env_switch.py:108-110` before any reset) and `current_env._episode_count` (matches the
drift-row `episode` field). **CONTROL score = root_owned_nodes / reachable(ownable)_count.**
CHECK 2 (ARTIFACT): topology resets to pristine (34 nodes / 565 vulns) at the start of all 20
episodes both conditions; within-episode erosion confirmed — no run-level confound.

## Join integrity (2.3)

**ARTIFACT: zero join failures.** For every (agent, membership/property) cell: score episodes =
1000, drift episodes = 1000, score-without-drift = 0, drift-without-score = 0, joined = 1000. Join
key = (seed, scenario_id, episode).

## STEP 2 / Condition A

### Score distribution per (agent, eval condition), 1000 episodes each (ARTIFACT)

| eval cond | agent | min | median | mean | max | frac_zero | per-seed mean (between-seed sd) |
|---|---|---|---|---|---|---|---|
| static | static | 0.000 | 0.750 | 0.727 | 0.906 | 0.005 | 0.704–0.742 (0.017) |
| static | adapted | 0.469 | 0.750 | 0.734 | 0.906 | 0.000 | 0.711–0.757 (0.020) |
| membership | static | 0.000 | 0.696 | 0.666 | 0.895 | 0.024 | 0.606–0.696 (0.037) |
| membership | adapted | 0.250 | 0.683 | 0.680 | 0.875 | 0.000 | 0.659–0.691 (0.013) |
| property | static | 0.000 | 0.719 | 0.710 | 0.906 | 0.015 | 0.660–0.730 (0.029) |
| property | adapted | 0.438 | 0.719 | 0.717 | 0.875 | 0.000 | 0.694–0.736 (0.018) |

### STATIC vs ADAPTED contrast — the "mattered" test (bootstrap 0.95, pooled episodes; causal weight lives HERE)

| eval cond | mean(adapted−static) | CI95 | verdict | per-seed diff (between-seed sd) |
|---|---|---|---|---|
| static | +0.0073 | [+0.0002, +0.0148] | MATTERED (adapted>static) | −0.015…+0.019 (0.014) |
| membership | +0.0146 | [+0.0049, +0.0246] | MATTERED (adapted>static) | −0.037…+0.083 (0.044) |
| property | +0.0077 | [−0.0011, +0.0166] | **NOT ESTABLISHED (CI includes 0)** | −0.035…+0.045 (0.029) |

**FINDING:** membership change costs the STATIC agent a small but bootstrap-significant amount
(+0.0146 vs the ADAPTED ceiling). For property, whether it costs anything is **not established**
(CI includes 0) — reported as "not established", not "did not matter". (A small +0.0073
static-eval difference also clears 0, i.e. the ADAPTED agent is marginally better even with no
change firing; the change-specific effect is the membership excess over this baseline.) The
membership eval condition bundles leave+join, so this contrast is for membership-as-a-whole.

### Response rate at τ=0 (primary cross-type metric) + per-slice drift (within-type), pooled 5 seeds

Filter: relevant + touched_node_visible + event_phase∈{immediate,attributed}. Counts at each filter
stage given per cell (example, static/membership/leave: raw 11198 → relevant 7749 → +visible 7749
→ +phase 7749; static/property: raw=relevant=visible=phase=14849).

| change type | agent | n_events | slice | resp_rate(τ=0) (between-seed sd) | drift mean [CI95] |
|---|---|---|---|---|---|
| membership_leave | static | 7749 | full | 1.000 (0.000) | 0.0723 [0.0708, 0.0737] |
| membership_leave | static | 7749 | mean | 1.000 (0.000) | 0.1081 [0.1062, 0.1100] |
| membership_leave | static | 7749 | max | 0.813 (0.021) | 0.0651 [0.0636, 0.0666] |
| membership_leave | static | 7749 | min | 0.827 (0.018) | 0.0635 [0.0620, 0.0651] |
| membership_leave | adapted | 8167 | full | 1.000 (0.000) | 0.0734 [0.0719, 0.0748] |
| property | static | 14849 | full | 1.000 (0.000) | 0.00114 [0.00083, 0.00150] |
| property | static | 14849 | max | 0.718 (0.020) | 0.00109 [0.00078, 0.00145] |
| property | adapted | 14982 | full | 1.000 (0.000) | 0.00045 [0.00042, 0.00050] |

**FINDING:** at τ=0 every membership_leave and property event moves the mean/full slice
(resp_rate 1.000); under max/min pooling ~0.71–0.83 do. **Property drift is ~65× smaller than
membership_leave drift** (full-slice 0.0011 vs 0.072) — perceptible but vanishing, consistent with
Task D (property perturbs one node's vulnerability sub-vector; leave removes a whole node
embedding). Adapted-vs-static drift magnitudes are within-type comparable and near-identical for
leave; property drift is ~2.5× smaller for the adapted agent (0.00045 vs 0.00114).

### membership_join — COVERAGE, not attenuation (numeric investigation, reported before use)

**Investigated because the first pass showed resp_rate 0.000 with NaN drift.** Cause (ARTIFACT,
pooled 5 seeds): every join `fired` row has `touched_node_visible=False` (5323 static / 5298
adapted) — the joined node is invisible until the agent independently discovers it (Task D: joins
start undiscovered). Fire-time response rate = **0.024–0.027** (≈0). The later `attributed`
(discovered) rows (441 static / 496 adapted) carry `change_drift = NaN` by schema (attribution
rows log visibility-lag, not h2→h3 change drift). **FINDING: a join does not move the pooled
observation when it fires — this is the COVERAGE / discovery-limit mechanism (the node is not yet
visible), structurally distinct from pooling ATTENUATION, and is reported separately per the
Task L rule (do not merge coverage with attenuation).** membership_join is therefore NOT
classified via the perceived/BLIND rule below.

### per-episode drift↔score correlation (associational only — numeric investigation)

**NOTE (stated, not implied): a cross-episode correlation is associational; the causal claim rests
on the STATIC-vs-ADAPTED matched-condition contrast above, not on this correlation.**

| agent/eval | n_ep | pearson (all) | spearman (all) | pearson (excl. score=0) | spearman (excl.) |
|---|---|---|---|---|---|
| static/membership | 975 | −0.140 | −0.117 | −0.140 | −0.117 |
| adapted/membership | 1000 | −0.118 | −0.120 | −0.118 | −0.120 |
| static/property | 1000 | **−0.705** | −0.099 | **−0.013** | −0.057 |
| adapted/property | 1000 | −0.055 | −0.069 | −0.055 | −0.069 |

**FINDING (investigated before presenting):** the strong static/property pearson −0.705 is an
artifact of 15 zero-score episodes (1.5%). On those, the agent discovered very few nodes, so a
tiny-graph pooled observation has small norm and the *relative* property drift is mechanically ~280×
larger (mean drift 0.140 on score=0 episodes vs 0.0005 on score>0). Excluding them, pearson →
−0.013 and spearman is −0.057 — **no monotonic drift→score relationship.** −0.705 is NOT presented
as a finding. The genuine (small) negative associations are membership (−0.12 to −0.14, stable
with/without zeros).

### Classification per change type (ABSORBED / BLIND / HANDLED)

Rule: perceived(event) = change_drift_full>0 at τ=0 on the filtered events; mattered(change type) =
the STATIC-vs-ADAPTED contrast for the matching eval condition. **HANDLED operationalized at EPISODE
granularity** (per-step score not captured; see limitation): perceived events whose ADAPTED agent
held a near-ceiling terminal score.

| change type | n (static, filtered) | perceived | mattered? | classification |
|---|---|---|---|---|
| membership_leave | 7749 | 7749 (100%) | MATTERED | **HANDLED = 7749** (perceived; residual cost +0.0146 vs ceiling) |
| membership_join | (coverage) | — | MATTERED (membership) | **not classified by this rule — COVERAGE mechanism** (see above) |
| property | 14849 | 14849 (100%) | NOT ESTABLISHED | **perceived (100%) but effect-on-score NOT ESTABLISHED** (τ=0 drift 0.0011; contrast CI includes 0) |

**No BLIND-via-attenuation events found:** every leave and property event is perceived (τ=0), so
there are 0 "not perceived & mattered" attenuation-blind events on this 34-node topology. The one
imperceptible mechanism (join at fire time) is coverage, not attenuation.

**Limitation (stated):** HANDLED's definition ("perceived, dips, then recovers within the episode")
needs a within-episode score trajectory; the score is captured once per episode (terminal). HANDLED
counts here mean "perceived, with the retrained agent at near-ceiling terminal score", not a
verified within-episode dip-and-recover. A per-step ownership log + re-run would be needed for the
literal definition.

## STEP 3 / Condition B (replication) — static agent, control score vs defender pn

ExternalRandomEvents defender (start/stop service, allow/block firewall rule, per node per attacker
step at prob pn), stochastic eval, 1000 episodes per pn.

| pn | mean score | CI95 | per-seed mean (sd) |
|---|---|---|---|
| 0.01 | 0.4645 | [0.4529, 0.4759] | 0.363–0.530 (0.068) |
| 0.10 | 0.2933 | [0.2850, 0.3011] | 0.208–0.355 (0.055) |
| 0.25 | 0.2794 | [0.2717, 0.2869] | 0.207–0.334 (0.047) |
| 0.50 | 0.2915 | [0.2840, 0.2992] | 0.235–0.349 (0.042) |

**FITTED slope (control score vs pn): −0.262, cluster-bootstrap-over-seeds 95% [−0.304, −0.216].**

**FINDING:** the defender degrades the static agent's control score. Shape is non-linear/saturating
— a sharp drop from pn=0.01 (0.46) to pn=0.10 (0.29), then a plateau (0.28–0.29 through pn=0.50);
the linear slope is dominated by the 0.01→0.10 drop. **Comparison to Terranova's reported +0.09 for
the control goal is SHAPE and DIRECTION only** (reward configurations differ; absolute scores and
any ratio are not comparable): the sign here is **negative** (measurable degradation), which
**differs in direction from the reported +0.09** (no measurable degradation).

### Firewall defect retained for replication fidelity (disclosed, not fixed)

`firewall_change_add`'s outgoing-rule branch tests membership against `.incoming` instead of
`.outgoing` (`cyberbattle/_env/static_defender.py:136`). Terranova's published dynamicity results
were produced with this same code; it is retained deliberately so Condition B replicates his
mechanism rather than a corrected variant. Recorded here per instruction.

## Filter-stage counts (reporting requirement)

Per cell, pooled 5 seeds (raw → relevant → +visible → +phase{immediate,attributed}):
- static/membership/leave: 11198 → 7749 → 7749 → 7749
- static/membership/join: 5764 → 441 → 441 → 441 (all attribution rows; see coverage note)
- static/property: 14849 → 14849 → 14849 → 14849
- adapted/membership/leave: 11583 → 8167 → 8167 → 8167
- adapted/membership/join: 5794 → 496 → 496 → 496
- adapted/property: 14982 → 14982 → 14982 → 14982
Episodes: 200/cell in, 200/cell out, 0 dropped, 0 crashed, 0 join failures.

> **PROVISIONAL (donor-pool confound, Task G not yet run):** `membership_join` events draw from a
> shared donor pool ~2.2x weaker at the large band. Every join-related number here inherits this
> caveat.

---
---

# Re-analysis on the within-agent axis (Task F1-R)

Numbers and provenance only. No new runs — recomputed from the same 50-cell Pass 1 data
(eval_out/{score,drift}_*.csv). No thesis wording.

> **PROVISIONAL (donor-pool confound, Task G pending):** `membership_join` draws from a shared
> donor pool ~2.2x weaker at the large band; every join-related number inherits this caveat.

**Why this re-cut.** (a) Attenuation (RQ3) is a SCALE mechanism (mean-pool shift ≈ (h̄−h_k)/(N−1)
when a node leaves), so comparing change TYPES at fixed N does not test attenuation — the type axis
serves RQ1 (coverage). (b) The Pass 1 between-agent contrast (adapted−static) is contaminated: under
STATIC eval with nothing firing, adapted already beats static by +0.0073 [+0.0002,+0.0148], so ~half
the reported membership effect exists when nothing happens. The clean measure uses ONE agent —
evaluated with change vs without — and is automatically fair across change types.

## STEP 0 — what is stored

**0.1 (ARTIFACT): absolute per-condition per-episode scores ARE stored** (`score_*.csv`:
`root_owned,reachable,n_discovered,won,score`), recoverable without re-running. Per-agent/eval/seed
means (200 episodes each; **zero dropped**), root_owned%:

| agent | eval | seed means (42/100/123/200/300) | pooled mean |
|---|---|---|---|
| static | static | 0.704/0.733/0.741/0.742/0.715 | 0.727 |
| static | membership | 0.606/0.693/0.681/0.696/0.652 | 0.666 |
| static | property | 0.660/0.731/0.719/0.730/0.708 | 0.710 |
| adapted | static | 0.711/0.753/0.757/0.727/0.724 | 0.734 |
| adapted | membership | 0.689/0.687/0.691/0.659/0.675 | 0.680 |
| adapted | property | 0.706/0.736/0.734/0.694/0.717 | 0.717 |

(min/max/median per cell in the raw output; e.g. static/static/seed42 min=0.000 max=0.875 med=0.719.)

**0.2 (ARTIFACT): per-slice response rate at τ=0 — MAX/MIN are the attenuation-bearing slices**
(mean/full = 1.000 by construction, carry no attenuation info):

| agent | change type | full | mean | max (between-seed sd) | min (sd) |
|---|---|---|---|---|---|
| static | membership_leave | 1.000 | 1.000 | 0.813 (0.021) | 0.827 (0.018) |
| static | property | 1.000 | 1.000 | 0.718 (0.020) | 0.713 (0.021) |
| adapted | membership_leave | 1.000 | 1.000 | 0.799 (0.015) | 0.817 (0.013) |
| adapted | property | 1.000 | 1.000 | 0.711 (0.027) | 0.712 (0.015) |

The 30-40 max-slice response rate (0.72–0.81) is in line with the gate's per-band figure at 30-40
(84.0) and sits between the gate's 10-15 (98.5) and 80-100 (43.0) — i.e. comparable, as required.
membership_join reported separately (fire-time resp_rate 0.024–0.027, visible@fire=0.000) — COVERAGE,
not attenuation.

**0.3 (ARTIFACT): per-episode join table exists on disk.** `eval_out/score_*.csv` (30 files × 200
rows) join exactly to `eval_out/drift_*.csv` (30 files) on (seed,scenario_id,episode): static/
membership 1000, static/property 1000, adapted/membership 1000, adapted/property 1000 — **0 unjoined**.

**0.4 (ARTIFACT): zero-score episodes** (compressed dynamic range / less reliable mean):
static/static 5/1000 (0.005); static/membership 24/1000 (0.024); static/property 15/1000 (0.015);
adapted (all three) 0/1000 (0.000).

**0.5 (ARTIFACT): change-events per episode:** static — membership_leave 11.198, membership_join
5.323, property 14.849; adapted — 11.583, 5.298, 14.982.

## STEP 1 — within-agent contrast (primary). cost(change) = mean(score | STATIC eval) − mean(score | CHANGE eval)

**1.1 STATIC agent (FINDING):**
- cost(membership) = **+0.0614** CI95 [+0.0507, +0.0718] — excludes 0. Per-seed 0.098/0.041/0.060/0.046/0.063 (between-seed sd 0.022).
- cost(property) = **+0.0174** CI95 [+0.0083, +0.0269] — excludes 0. Per-seed 0.043/0.003/0.022/0.013/0.006 (sd 0.016).

**1.2 ADAPTED agent (FINDING):**
- cost(membership) = **+0.0541** CI95 [+0.0475, +0.0608] — excludes 0. Per-seed 0.021/0.066/0.067/0.067/0.049 (sd 0.020).
- cost(property) = **+0.0170** CI95 [+0.0105, +0.0233] — excludes 0. Per-seed 0.005/0.016/0.024/0.032/0.008 (sd 0.011).

**FINDING: both change types impose a real within-agent cost on BOTH agents** (all four intervals
exclude 0) — much larger and cleaner than the contaminated between-agent contrast. Membership costs
~0.055–0.061 root_owned% per episode; property ~0.017.

**1.3 Difference-in-differences (FINDING) — the part training could have avoided:**
DiD = cost(change)|static − cost(change)|adapted.
- DiD(membership) = **+0.0073** CI95 [−0.0049, +0.0196] — **NOT ESTABLISHED** (includes 0).
- DiD(property) = **+0.0004** CI95 [−0.0112, +0.0119] — **NOT ESTABLISHED** (includes 0).

**FINDING: training on the change did NOT demonstrably reduce its cost for either change type.** The
adapted agent pays essentially the same within-agent cost as the static agent. The membership DiD
point estimate (+0.0073) equals the STATIC-eval baseline gap from 0.—exactly the contamination that
inflated the Pass 1 between-agent claim.

**1.5 Ratio robustness(change) = mean(CHANGE) / mean(STATIC eval) (FINDING — the cross-band-comparable quantity):**
static: membership 0.9156 [0.9017,0.9295], property 0.9761 [0.9636,0.9888]. adapted: membership
0.9264 [0.9177,0.9352], property 0.9768 [0.9683,0.9857].

**1.6 Cost per event (FINDING; ASSUMES effects add linearly across events — labelled assumption; does
not replace per-episode):** static — membership 0.0614/11.198 = **+0.00548/event**, property
0.0174/14.849 = **+0.00117/event**; adapted — membership **+0.00467/event**, property **+0.00113/event**.

## STEP 2 — classification, recomputed (within-agent measure, identical for every change type)

perceived reported PER SLICE (0.2, not collapsed); cost = within-agent contrast (1.1, static agent).

| change type | perceived full/mean/max/min | cost (static agent) | classification |
|---|---|---|---|
| membership_leave | 1.000/1.000/0.813/0.827 | +0.0614 [+0.0507,+0.0718] | perceived + real cost → HANDLED-candidate |
| property | 1.000/1.000/0.718/0.713 | +0.0174 [+0.0083,+0.0269] | perceived + real cost → HANDLED-candidate |
| membership_join | coverage (fire-time ~0, invisible node) | — | COVERAGE, reported separately (not attenuation) |

**FINDING: 0 BLIND, 0 ABSORBED on this band** — both attenuation-relevant change types are perceived
(max/min slices move) AND carry a cost whose interval excludes 0, so neither is "not perceived".
**HANDLED is NOT MEASURABLE with the present instrumentation** (score captured once per episode; a
within-episode dip-and-recovery needs per-step ownership logging). Not approximated — reported as not
measurable.

## STEP 3 — what this band can/cannot support

**3.1 (FINDING): the 30-40 band has enough dynamic range to detect a cost, and one is detected.**
STATIC-eval ceiling (score available to lose): static 0.7270, adapted 0.7343. Largest observed cost:
static +0.0614 = **8.4% of ceiling**; adapted +0.0541 = **7.4% of ceiling**.

**3.2 No extrapolation.** This band is one point.

**3.3 (FINDING; NOT a causal basis) drift↔score correlation, exclusion rule = drop score==0 episodes (0.4):**
static/membership ALL n=975 pear −0.140 spear −0.117 | excl-zero n=975 pear −0.140 spear −0.117.
static/property ALL n=1000 pear −0.705 spear −0.099 | excl-zero n=985 pear −0.013 spear −0.057.
adapted/membership −0.118/−0.120 (unchanged). adapted/property −0.055/−0.069 (unchanged).
**This correlation is NOT the basis of any causal claim.** The causal weight is the within-agent cost
(STEP 1); a cross-episode correlation confounds everything that differs between episodes — as the
static/property −0.705 → −0.013 collapse (15 failed tiny-graph episodes) demonstrates. Reported for
completeness and as a documented negative result only.

## Figures SUPERSEDED by this re-analysis

1. **Pass 1 "membership MATTERED" (between-agent adapted−static = +0.0146 [+0.0049,+0.0246])** →
   SUPERSEDED. The correct within-agent DiD(membership) = +0.0073 [−0.0049,+0.0196] is **NOT
   ESTABLISHED**: training on membership change did not demonstrably reduce its cost.
2. **Pass 1 "property NOT ESTABLISHED (adapted−static +0.0077 [−0.0011,+0.0166])"** → SUPERSEDED as a
   *statement about the change's effect*: property imposes a real within-agent cost on both agents
   (+0.0174 / +0.0170, intervals exclude 0); only the *training benefit* (DiD +0.0004) is not
   established.
3. **Pass 1 static-eval baseline "MATTERED (+0.0073 [+0.0002,+0.0148])"** → reinterpreted: this is the
   contaminating baseline gap, not a change effect; it is exactly why the between-agent contrast
   overstated membership.
4. **Pass 1 classification (membership_leave HANDLED=7749 via the "mattered"/between-agent test;
   property "perceived but effect not established")** → SUPERSEDED by the within-agent-cost basis in
   STEP 2 above.
5. **Pass 1 response-rate reporting (mean/full = 1.000 only)** → SUPERSEDED/extended: the informative
   max/min slices are 0.71–0.83 (0.2), the attenuation-bearing figures.

The Pass 1 Condition-B replication (defender slope −0.262) and the coverage finding for
membership_join are unaffected by this re-cut and stand.

> **PROVISIONAL (donor-pool confound, Task G pending):** `membership_join` draws from a shared donor
> pool ~2.2x weaker at the large band; every join-related number inherits this caveat.
