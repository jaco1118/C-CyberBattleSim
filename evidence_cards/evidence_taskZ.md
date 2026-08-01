# Task Z — three-arm pooling ablation (STEP 0 gate)

Read-only + one pre-flight assertion (~1,200 steps, no training). Reporting 0.1–0.4 and STOPPING.

## 0.1 CLAIMS AUDIT — the "69/72/76% arithmetic share" [FINDING — the premise is mislabelled; the numbers are sound]

**The two statements are reconcilable (different files), but the task's premise — "69/72/76 across three
bands" — is WRONG: they are two bands under three CHURN CONDITIONS, and there is no 10-15 figure.**

**Exact computation, per figure** (`evidence_cards/evidence_taskF3.md:237-239`, Appendix "recomputation on
the ROOT-OWNED COUNT metric"): each % = **mechanical ÷ total-count-cost**, where mechanical = **root-owned
departures per episode = Σ `was_root`** and total-count-cost = mean(static root_owned) − mean(change
root_owned). Verified: 5.39/7.79 = 0.692, 2.69/3.74 = 0.719, 5.06/6.66 = 0.760.

| figure | condition (NOT a band) | mechanical (dep/ep) | count-cost | % |
|---|---|---|---|---|
| 69% | **30-40, fixed-absolute churn (~33%)** | 5.39 [5.20,5.58] | 7.79 | 0.692 |
| 72% | **80-100, fixed-absolute churn (~18%)** | 2.69 [2.26,3.07] | 3.74 | 0.719 |
| 76% | **80-100, fixed-relative churn (32.3%)** | 5.06 [4.31,5.71] | 6.66 | 0.760 |

**Data & source:** `was_root` per departure is logged by `taskF3_mech_eval.py` → CSVs
`f3_mech_out/mech_{30-40,80-100}_seed*.csv` and `f3_rel_out/mech_80-100rel_seed*.csv` (verified present:
**10 files at 30-40, 15 at 80-100**); aggregated + the %/CIs computed in `taskF3_mech_analyze.py`.
Counted exactly, event by event — **not estimated**.

**Reconciliation with the Task-N gate report:** Task N (N2.2) examined the `leaveown_*` file, which logs
**`was_owned`** and exists **only at 80-100**, and correctly said the arithmetic component "cannot be
counted at 30-40 *from that file*". The 69/72/76 series uses a **different file** — the F3 **mech CSVs**,
which log **`was_root`** and **DO exist at 30-40**. So both statements are true; they are about different
records. **Which is the right arithmetic component?** For the root-owned COUNT cost it is **`was_root`**
(removing a *root*-owned node subtracts exactly 1 from root_owned) — so the F3 mech CSV is the correct
instrument and it covers both bands; `was_owned` (owned at any privilege) would over-count. Task N's
statement is narrowly about the `leaveown` file, not a claim that 30-40 arithmetic is uncountable in
general.

**What must change in the draft:** wherever the thesis presents 69/72/76 as an arithmetic-share series
**"across three bands" (implying 10-15/30-40/80-100)**, that is a **mislabel** — it is
**30-40 fixed-abs / 80-100 fixed-abs / 80-100 fixed-rel** (two bands, one band twice, no 10-15). The
numbers are correctly computed; only the axis label is wrong. (This is the cheap catch the gate was placed
to make before compute is spent.)

## 0.2 TRAINING ESTIMATE — the reading [ARTIFACT]

**"Per arm" in the Task-N figures means the FULL FIVE-SEED SET, not one seed.** N5.5 was
`30-40 ≈ 55 min/run × 5 ≈ 4.6 h/arm` and `80-100 ≈ 87 min/run × 5 ≈ 7.3 h/arm` (this box's measured fps
76 / 48 at 250k). So:
- **Total, SEQUENTIAL:** 3 arms × (4.6 h + 7.3 h) = **~35.7 h** (30 runs).
- **Total, 5-way CONCURRENT** (as Task F4 ran, 5 procs = 20 threads < 32 cores, GPU idle): each 5-seed cell
  finishes in ~one run's time (≈55 min at 30-40, ≈87 min at 80-100); 6 cells ⇒ ~**7–9 h wall-clock**.

**Either way it is affordable before 15 August** (the 5-seed reading is the correct one; even the
sequential 36 h fits, and concurrency brings it to ~one day-part). Historical hardware (fps 249/89) would
be ~2–3× faster still.

## 0.3 ARM-3 PRE-FLIGHT ASSERTION — PASS [FINDING], verbatim

Implemented the Task-N fix: substitute the 128 extremal dims (graph_embeddings `[64:192]` = max+min slices;
mean = `[0:64]`, next_escalation = `[192:256]`) to a fixed constant **AFTER `normalize_obs`**. 1,200 steps
on the 30-40 F1 static agent, verbatim output:
```
(a) steps sampled: 1200 ; substituted dims per step: 128 (=128 extremal)
(b) distinct values across all 153600 substituted entries: 1  -> PASS (==1)
(c) the constant's value: [0.0]
(d) 64 mean dims: max |arm1 - arm3| over all steps = 0.000e+00  -> PASS (==0)
```
**(b) exactly ONE distinct value** across 153,600 entries — bit-exactly constant, not "99% constant".
**(c) constant = 0.0.** **(d) the 64 mean dims are unaffected — max |arm1−arm3| = exactly 0.** The gate
condition (>1 distinct value ⇒ STOP) is **NOT triggered**. Arm 3 is a valid control **provided the
substitution is applied post-normalisation** (as here); the same substitution applied *before* VecNormalize
during training-time stat updates is the failure mode Task N flagged and is deliberately avoided.

## 0.4 ARM-2 IS CONFIGURATION-ONLY [FINDING]

Setting `graph_embeddings_aggregations: [mean]` yields a **64-dim pooled** observation. The observation
space is **derived, never hard-coded to 192** — every location checked scales with
`len(graph_embeddings_aggregations)` and `node_embeddings_dimensions`:
- `observation_space` graph-box shape: `cyberbattle_env_compressed.py:142` (node-goal) and **`:150`
  (control goal)** — `node_embeddings_dimensions × len(graph_embeddings_aggregations) + node_embeddings_dimensions`
  (= 64×3+64 = 256 now; → 64×1+64 = 128 under `[mean]`).
- pooled-vector assembly: `:411-424` (loops over `graph_embeddings_aggregations`).
- `observation_space` Dict + discrete: `:156-160` (`len(self.discrete_features)`).
- The `× 2` shapes at `:133` and `:1211` are the **ACTION** space, not the observation — unaffected.
- train_config `policy_kwargs.net_arch` does **not** reference the obs dim; SB3 infers the first-layer input
  size from the observation space, so it auto-adjusts.

**Caveat (from N5.4):** the frozen **encoder** is per-node (64-d) and layout-agnostic, but the **trained
policy and its saved VecNormalize stats assume the current 256 layout**, so Arm 2 needs a **fresh policy +
fresh VecNormalize** (a retrain), not a config flip on the existing agents. Same for Arm 3 (it keeps 256 but
still trains its own policy).

## GATE

0.1–0.4 reported. **The one serious item is 0.1: the 69/72/76 series is mislabelled in the draft as
three bands when it is two-bands-three-conditions with no 10-15 point — correct the label (the numbers are
sound).** 0.3 passed bit-exact (Arm 3 buildable via post-normalisation substitution). 0.2 reading: per-arm =
5 seeds, total ~7–9 h concurrent, affordable. 0.4: Arm 2 config-only + retrain. **No training started;
awaiting acceptance before STEP 1.**

**STEP 0 ACCEPTED (2026-07-31).** Three notes from the user carried into STEP 1: (1) 0.1 recorded closed in
`claims_audit.md` (CA-1) — figures correct, relabel is a writing task, not done here. (2) Run at 5-way
concurrency; report actual wall clock vs the 7–9 h estimate. (3) **Arm 1 is RETRAINED FRESH** under this
harness (not reused from the reported F1/F2 runs) so 1.2 tests the harness, not fresh-vs-old.

## STEP 1 — training the three arms (LAUNCHED) [ARTIFACT]

**Harness `taskZ_train.py`** mirrors `taskF1_train.py`/`taskF2_train.py` STATIC single-topology TRPO 250k
**exactly** (same tuned config template = gate 30-40 seed42 `train_config.yaml`, same frozen encoder, same
`RandomSwitchEnv → DummyVecEnv(Monitor) → VecNormalize → TRPO`, same `set_seeds`, `dynamic_mode=none`,
patch off). Verified F1 and F2 drivers differ **only** in topology path + naming (`diff`: 6 hunks, all
topo/name), so one harness reproduces both bands' Arm 1. The **only** per-arm differences:
- Arm 1: `graph_embeddings_aggregations=[mean,max,min]` → obs graph_embeddings **256**.
- Arm 2: `graph_embeddings_aggregations=[mean]` → **128** (the one config key).
- Arm 3: 256, with `ExtremalMask(VecEnvWrapper)` zeroing `graph_embeddings[64:192]` to 0.0 **after**
  VecNormalize (the 0.3 fix). Inner VecNormalize sees raw obs; mask re-applied at eval.

**Pre-launch smoke (512-step override, all three arms, seed 42, 30-40/44):** all three construct, train,
and checkpoint without error. Saved-VecNormalize obs dims verified: **arm1=256, arm2=128, arm3=256** (inner
VecNorm at full dim — mask is outside, so its stats are unmasked, correct). Drift logging OFF (pure
instrumentation; does not touch the training RNG stream).

**Grid:** 3 arms × 2 bands × 5 seeds = **30 runs**, 250k steps each. Topologies match the reported runs:
30-40 = `scalability_30_40/44` (all seeds); 80-100 = per-seed {42:5, 100:100, 123:18, 200:2, 300:67}.
Seeds {42,100,123,200,300}. **Launched 2026-07-31 00:43** at **MAXJOBS=5** (24 cores, GPU 16 GB free/idle);
`launch_z.sh`. Arm order 1→2→3 so the Arm-1 reproduction check (1.2) lands first. Runs under
`z_runs/z_arm{A}_{band}_seed{SEED}/`.

**COMPLETE (2026-07-31 04:51).** All 30 runs exit 0, all 30 final 250k checkpoints present. **Wall clock =
00:43:41 → 04:51:03 = ~4 h 07 m** (vs the 7–9 h estimate — faster; measured ~30 min/run at 30-40, ~45 min/run
at 80-100, at MAXJOBS=5). The 0.2 estimate was conservative; concurrency + higher-than-benchmarked fps beat it.

## STEP 1.2 — reproduction check: Arm 1 reproduces at BOTH bands [FINDING]

Eval harness `taskZ_eval.py`, static, root-owned COUNT, 200 ep/seed. To isolate the TRAINING harness (per
the user's note 3), both the fresh Arm 1 and the OLD reported F1/F2 checkpoints were evaluated with the SAME
`taskZ_eval` (NEW vs OLD), and OLD was cross-checked against the reported numbers (REP). **OLD == REP
exactly** (`taskZ_eval` reproduces the reported eval bit-for-bit on the old checkpoints), so NEW−OLD is a
pure training-harness comparison. No errors, 20/20 CSVs.

| band | NEW (fresh Arm 1) | OLD = REP (reported) | \|NEW−REP\| | between-seed SD (ref) | verdict |
|---|---|---|---|---|---|
| 30-40 | 23.760 | 23.263 | 0.497 | 0.542 | **REPRODUCES** (within spread) |
| 80-100 | 29.513 | 29.667 | 0.154 | 5.699 | **REPRODUCES** (within spread) |

Per-seed (NEW−REP): 30-40 = {+0.94, +0.42, −0.09, +0.41, +0.81}; 80-100 = {−0.92, −0.28, +0.04, +0.04,
+0.35} — all small, and 80-100 is near-identical per seed (each seed = its own topology). **The ablation
harness is validated; Arms 2 and 3 can be trusted.** (1.2 gate PASSED — proceed permitted, not forced.)

## STEP 1.3 — cost of a 750k run of all three arms at 80-100 (reported, NOT launched) [ARTIFACT]

Measured 80-100 250k ≈ **45 min/run** at MAXJOBS=5. Linear in steps → 750k ≈ **135 min/run**. All three arms
× 5 seeds = **15 runs**; at 5-way concurrency = 3 sequential waves × ~135 min ≈ **~6.5–7 h wall clock**.
**Affordable before 15 August with room to spare.** Per the spec, this decision is taken separately — **NOT
launched.** (Rationale for offering it: 80-100 does not converge at 250k, but the arm-difference measurement
is valid between equally-undertrained arms; a 750k set would test whether the information/capacity effect
changes with more budget, disclosed as a separate question.)

## GATE (STEP 1)

1.2 reported: **Arm 1 reproduces at both bands within seed spread** (30-40 Δ=0.50<0.54; 80-100 Δ=0.15<5.70);
harness validated. 1.3 reported: 750k-all-arms-80-100 ≈ 7 h, affordable, **not launched** (separate
decision). **No STEP 2 measurement computed yet — awaiting go, and awaiting the change-condition choice for
2.1 (the difference is measured under "change"; the exact membership rate is a methodological fork worth
confirming before the 2.3 headline is produced).**

## STEP 2 — the measurement (pre-registered: null + power FIRST, primary = fixed-RELATIVE) [FINDING]

Change conditions locked by the user before any number was computed: **fixed-relative PRIMARY** (30-40 CI=20,
80-100 CI=8), **fixed-absolute SECONDARY** (both CI=20); primary stands if they disagree. Eval: 3 arms × 2
bands × 5 seeds, static + both change conditions, 200 ep/seed (75 cells, 139 min).

### 2.4 NULL + POWER (reported BEFORE any difference)
Static between-seed spread (the null threshold), root-owned COUNT:

| band | arm1 SD | arm2 SD | arm3 SD | MDE (single-arm SD) | MDE as % of static mean |
|---|---|---|---|---|---|
| 30-40 | 0.267 | 0.747 | 0.147 | **0.267 nodes** | **1.1%** |
| 80-100 | 6.060 | 3.818 | 3.744 | **6.060 nodes** | **20.5%** |

The asymmetry is by DESIGN (Task E 0.5): 30-40 is 5 seeds on ONE topology (tight null); 80-100 is 5 seeds on
5 DIFFERENT topologies (huge null, static means 19.8–35.1). The paired arm1−arm3 static-difference SD (which
cancels the shared-topology variance) is **0.150** (30-40) and **2.862** (80-100) — reported alongside as the
tighter, paired noise floor; the conclusions below hold under **either** threshold.

### 2.1 / 2.2 — counts and the three differences (PRIMARY = fixed-relative)

| band | arm1 change | arm2 change | arm3 change | INFO (a1−a3) | CAPACITY (a2−a3) | RAW (a1−a2) |
|---|---|---|---|---|---|---|
| 30-40 | 15.71 | 15.64 | 15.51 | **+0.208** [−0.43,+0.85] | +0.136 [−0.92,+1.04] | +0.072 |
| 80-100 | 22.88 | 22.61 | 21.54 | **+1.338** [−0.02,+2.72] | +1.068 **[+0.36,+1.78]** | +0.270 |

(Fixed-absolute SECONDARY agrees directionally: INFO 30-40 +0.208, 80-100 +1.327; CAPACITY 80-100 +1.073
[−0.12,+2.46]. No primary/secondary disagreement.)

### 2.3 — THE ANSWER: information effect 30-40 vs 80-100
Point estimates: **30-40 = +0.21 nodes, 80-100 = +1.34 nodes → LARGER at 80-100, OPPOSITE to the expected
"larger at 30-40."** Reported plainly, not softened. **But this direction is NOT established** — see the
null labelling: 30-40's effect is at its (tight) floor and 80-100 is deep in noise.

### Null labelling (pre-registered EFFECT-ABSENT vs UNDERPOWERED)
- **30-40 → EFFECT ABSENT.** The band is well-powered (MDE = 1.1% of score); the information effect is
  **+0.21 nodes (0.9%), below the 0.27 floor, CI includes 0.** A ~1%+ effect would have been detected; none
  was. **At 30-40 the agent derives no measurable behavioural value from the extremal channels — removing
  them (arm3) costs nothing measurable, and the capacity control (arm2) is likewise flat.**
- **80-100 → UNDERPOWERED.** MDE = 6.06 nodes (**20.5%** of score) because of the 5-topology design; the
  observed info effect (+1.34, 4.5%) and even the capacity effect are far below it. **The null carries NO
  information — this is NOT evidence that the extremal channels stopped mattering at scale**, per the
  pre-registered rule. (One caveat disclosed, not a renegotiation: the CAPACITY effect's *paired* bootstrap
  CI [+0.36,+1.78] excludes 0, because the paired difference cancels the topology variance that inflates the
  single-arm MDE. By the pre-registered single-arm-SD criterion it is still "no measurable effect"; the
  criterion is not renegotiated, but the paired signal is flagged as the one thing a tighter, topology-matched
  design might resolve.)

### STEP 2 verdict
**The pre-registered hypothesis (information effect larger at 30-40, shrinking with scale) is NOT supported.**
30-40 shows **no measurable information OR capacity effect** (well-powered null); 80-100 is **underpowered by
design** and cannot adjudicate. The point estimates lean the *opposite* way (larger at 80-100) but
unreliably. **250k is primary; the 80-100 underpower motivates the pre-registered 750k robustness run** (does
more budget, or the topology-matched precision, surface an effect?) — launched next per the standing
pre-registration, PYTHONHASHSEED unaffected here (Task Z uses per-seed processes; determinism not required for
these count means).

## STEP 2 CORRECTION — the pre-registration used UNPAIRED variance for a PAIRED design [reported per user, (a)/(b)/(c)]

**(a) PRE-REGISTERED criterion + verdict — PRIMARY, binding, not renegotiated.** MDE = single-arm static
between-seed SD. **30-40: EFFECT ABSENT, well-powered** (MDE 0.267 = 1.1%; info +0.21, CI incl 0). **80-100:
UNDERPOWERED, null carries no information** (MDE 6.06 = 20.5%; info +1.34 below floor). This stands.

**(b) PAIRED analysis — NOT PRE-REGISTERED** (all three arms face identical networks per seed, so the paired
difference cancels the between-topology variance; noise floor = between-seed SD of the *paired static
difference*):

| band | effect | fixed-REL (primary) | fixed-ABS | paired MDE | verdict |
|---|---|---|---|---|---|
| 30-40 | INFO a1−a3 | +0.208 [−0.43,+0.85] | +0.208 [−0.43,+0.85] | 0.150 | above floor but **CI incl 0** |
| 30-40 | CAPACITY a2−a3 | +0.136 [−0.92,+1.04] | +0.136 [−0.92,+1.04] | 0.703 | below floor |
| 80-100 | INFO a1−a3 | +1.338 [−0.02,+2.72] | +1.327 [−0.47,+3.14] | **2.862** | **below floor** (even paired) |
| 80-100 | CAPACITY a2−a3 | +1.068 **[+0.36,+1.78]** | +1.073 [−0.12,+2.46] | 1.162 | at floor, **primary CI excl 0** but fixed-ABS incl 0 |

**What the paired estimator changes and does NOT change:** it tightens the 80-100 noise floor from 6.06 to
2.86, but the **information effect is still not measurable at either band** (30-40 CI includes 0; 80-100 +1.34
< paired floor 2.86). The **only** near-signal is the **80-100 CAPACITY effect** (arm2 mean-only *beats* arm3
extremal-zeroed by ~1 node, primary CI excludes 0) — but it sits at its paired floor and does **not** replicate
under fixed-absolute, so it is suggestive, not robust. **The substantive conclusion is unchanged: no robust
information effect; the hypothesis is not supported.**

**(c) DISCLOSURE.** The pre-registration specified an **unpaired** between-seed spread as the null threshold;
the design is **paired** (identical networks across arms), so the paired difference is the correct estimator.
This was an **error in the pre-registration, disclosed here rather than corrected after the fact** — both
analyses are reported (a = binding primary; b = paired, not pre-registered) so the reader can judge. The
paired result is **not** presented as the headline and is **not** buried.

## STEP 3 SCOPE ADD — do PROPERTY-change cost figures exist? YES [reported per user; not run]

Property change has **no arithmetic component** (it removes nothing the agent owns → the count does not fall
mechanically), so its cost is **100% behavioural** — a cleaner behavioural test than membership. **Cost +
robustness figures under property change alone already exist, at BOTH bands:**
- **30-40:** `eval_out/score_{static,adapted}_seed*_evalproperty.csv`; `evidence_taskF1.md:289,324` —
  cost(property, ratio) = +0.0174 [+0.0083,+0.0269] (excludes 0), robustness ~0.976.
- **80-100:** `f2eval_out/score_static_seed*_evalproperty.csv`; `evidence_taskF2.md:158-159` —
  cost(property, ratio) +0.0174/+0.0161, robustness 0.976→0.957.
These are on the **ratio** metric. The churn-invariant **COUNT-metric** property cost is **derivable from the
same existing CSVs (they carry `root_owned`) — NO new sweep needed**: e.g. 30-40 count cost =
mean(static)−mean(property) = **+0.556 nodes** (per-seed [1.39,0.09,0.70,0.40,0.20]), and it is 100%
behavioural by construction. So the behavioural property-cost decomposition is available at both bands from
existing data; only a re-analysis (not an eval sweep) is required.

## 750k ROBUSTNESS RUN — caveat on record [ARTIFACT]

The 750k-at-80-100 run (resume 250k→750k, 3 arms × 5 seeds, PYTHONHASHSEED=0) tests **whether undertraining
hides an effect**. It **CANNOT fix the power problem at 80-100**: the MDE there is driven by
**between-topology variance** (5 distinct topologies per band), not by training budget — more steps cannot
shrink a between-network spread. If 750k still shows no effect, that rules out undertraining as the
explanation but leaves the underpower (a design/topology-variance limitation) intact.

## ADDENDUM (analysis-only) — re-examining Task N gate N2 against the mech CSVs [FINDING]

N2 concluded band 30-40 "cannot count the arithmetic component" because `leaveown_*` exists only at 80-100.
0.1 showed the **mech CSVs** carry `was_root` at **both** bands. Re-examined against the mech CSVs (schema
read only — `taskF3_mech_eval.py` + headers of `f3_mech_out/mech_{30-40,80-100}_seed*.csv`, 5 files/band;
**no analysis run**). Mech-CSV columns (identical both bands): `band, seed, episode, was_owned, was_root,
score_before, score_after, delta`. Each row = **one departure event** (written by the monkeypatched
`remove_node_dynamic`).

**(a) Is `was_root` per-event or per-episode, each band?** **Per departure EVENT, at BOTH bands** — one row
per removed node, with an `episode` column for grouping. Not pre-aggregated. So N2's pessimism about 30-40
is **too strong for the arithmetic COST component**: the per-event root-owned arithmetic *is* countable at
30-40 from the mech CSV.

**(b) Do the mech CSVs carry / allow reconstruction of per-episode STRUCTURAL summaries?** (same at both
bands — identical schema):
- **max degree among the episode's departing nodes** — **NOT available.** No degree column, and **no
  `node_id`**, so departures cannot even be joined back to the topology to look degree up. Not
  reconstructable from the CSV.
- **# departures above a degree threshold** — **NOT available** (same reason: no degree).
- **count of change events by type** — **PARTIAL / effectively NO.** Removal *count* per episode = row count
  per episode (available), and splittable by `was_owned`/`was_root` (an ownership "type"). But the mech CSV
  logs **only departures** (the `remove_node_dynamic` patch); **joins and property-change events are not
  recorded at all**, so "change events by type" in the intended sense (leave/join/property) is **not**
  obtainable.

**(c) Is N2 branch (ii) — episode cost regressed on episode structural summaries — available at BOTH bands
from the mech CSVs?** **NO, at neither band.** The **cost** side (LHS) *is* available at both bands
(per-episode mechanical displacement = Σ`delta` per episode; plus `mechscore_*` per-episode score). But the
**structural regressors** (RHS) are absent from the mech CSV at **both** bands (no degree, no node_id, only
departures). So the mech CSV **fixes the band asymmetry for the arithmetic-share computation** (was_root at
both bands — the CA-1 numbers stand) but **does not enable branch (ii)**: regressing cost on structure still
needs a logging re-run that records per-departure degree (and ideally node_id + change type). That re-run is
a **separate task**, not begun here.

**Net correction to N2:** N2 was right that branch (ii) is unavailable, but for the **wrong reason** — it is
not a 30-40-only gap fixable by finding the right file; the structural predictors are missing at **both**
bands. What *was* over-pessimistic in N2 is the narrower claim that 30-40 cannot count the arithmetic COST —
the mech CSV shows it can.

## STEP 1.3 — 750k ROBUSTNESS at 80-100 (resume 250k→750k, PYTHONHASHSEED=0) [FINDING]

Does more budget surface an effect that 250k undertraining hid? Eval: 3 arms × 5 seeds @80-100, static +
fixed-rel + fixed-abs, 200 ep/seed, 45 cells, 102 min. All 15 checkpoints at 750k; wall clock 5.0 h.

**Static (750k):** arm means 35.4 / 35.0 / 35.3 (vs 250k ~29.5 — the agents trained further); between-seed SD
4.66 / 3.13 / 5.28; paired static-diff SD arm1−arm3 = 1.72, arm2−arm3 = 2.98.

| condition | effect | 750k | (250k reference) |
|---|---|---|---|
| fixed-REL (primary) | INFO a1−a3 | **−0.64** [−2.19, +0.92] | +1.34 [−0.02, +2.72] |
| fixed-REL (primary) | CAP a2−a3 | **−0.32** [−1.88, +1.24] | +1.07 **[+0.36, +1.78]** |
| fixed-ABS | INFO a1−a3 | −0.07 [−1.47, +1.37] | +1.33 [−0.47, +3.14] |
| fixed-ABS | CAP a2−a3 | −0.02 [−2.16, +2.08] | +1.07 [−0.12, +2.46] |

**Result: at 750k ALL effects are ~0 with CIs including 0, below both the single-arm (5.28) and paired (1.72/
2.98) floors.** Two things move from 250k: (1) the information effect **collapses from +1.34 to −0.64** — the
250k "opposite-direction" point estimate was noise, not a real reversal; (2) **the one 250k near-signal — the
capacity effect (+1.07, primary CI excluded 0) — does NOT replicate at 750k (−0.32, CI includes 0).** So it
was an undertraining/noise artifact, not a real capacity effect.

**Verdict:** **undertraining did NOT hide an effect — more budget makes the three arms MORE similar (all
converge to ~35 static / ~27 under change), not less.** The pre-registered hypothesis (extremal channels earn
their place; an information/capacity effect that the 250k null missed) is **not supported at either budget.**
Caveat on record (as pre-registered): 750k **cannot** fix the 80-100 power problem — the MDE there is driven
by between-topology variance (5 distinct topologies), not training budget — but here the point estimates are
~0 regardless, so even a fully powered test would most likely find nothing. **250k remains PRIMARY; 750k is a
robustness check and it confirms the null.**
