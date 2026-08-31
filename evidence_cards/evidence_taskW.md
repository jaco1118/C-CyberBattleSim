# Task W — repair the RQ1 classification (STEP 0 gate)

Analysis only, reading existing data. Reporting 0.1–0.5 and STOPPING for acceptance. Source-quoted.

## GATE VERDICT (one paragraph, up front)

**The premise "all three faults are fixable in analysis on existing data" does NOT hold — two of the
four repair steps are not computable without re-running, which this task forbids.** Specifically: STEP 4
(discovery decomposition) is fully computable; STEP 1 (contamination-free baseline) is computable **only
on the F-series** data (which has static twins) and **NOT on the gate data that produced the quoted
0.492/−0.804** (the gate ran membership only, no static condition); **STEP 2 (counterfactual relevance)
is NOT computable on any existing data** — the static runs log only per-step **counts**, never node
identities, so "did the agent discover/own/act on node v" cannot be looked up, and even if it could,
episode alignment between static and dynamic runs breaks after the first episode (random starter drawn
from the same RNG stream the change draws consume). STEP 3's perception axis is computable but its
six-cell table needs the (blocked) STEP 2 relevance axis. **Recommend: STOP.** Repairing FAULT 1
(relevance) and de-contaminating the actual quoted figures (FAULT 3 on the gate) both require new eval
runs with per-node-identity logging and a static arm — out of scope here.

---

## 0.1 Matched static twins [ARTIFACT]

**F-series (F1/F2/F3, single-topology): twins EXIST and join.** For each seed the static, membership and
property cells share the SAME seed and SAME scenario (`taskF1_eval.py`/`taskF2_eval.py` load one fixed
topology per seed: 30-40 = `scalability_30_40/44` all seeds; 80-100 = one topology per seed). They join
on **`(seed, scenario_id, episode)`** (the key `taskF1R_reanalyze.py` already uses:
`merge(..., on=["seed","scenario_id","episode"])`). Confirmed present: `eval_out/{drift,score}_static_seed*_eval{static,membership,property}.csv`.

**GATE (Task T, the source of 0.492/−0.804): NO static twin.** The gate drift logs
(`attenuation_gate_archive/2026-07-26_trpo_5seed_gate/attenuation_drift_logs/drift_{10-15,30-40,80-100}.csv`)
contain **only** `change_type ∈ {membership_join, membership_leave}` — there is **no static condition**
run for the gate. So the contamination-free baseline STEP 1 asks for **cannot be built for the quoted
figures**; the F-series static twins are single-topology and are NOT matched twins for the
multi-topology gate. **This alone blocks de-contaminating 0.492/−0.804 in analysis.**

## 0.2 Node identity across runs [ARTIFACT — passes literally, but the data blocks STEP 2]

Node identifiers are **consistent, not re-issued**: node names come from the scenario `.pkl`
(`Node_0…Node_N`, fixed), loaded identically in the static and dynamic runs of the same scenario; only
dynamically-joined nodes get fresh `JoinNode_*` ids (dynamic-only, irrelevant to a leave counterfactual).
So the literal 0.2 test (same id ⇒ same node) passes.

**BUT the counterfactual is not computable, because per-node identities are never LOGGED for the static
run.** The static drift/score CSVs record only **counts** — full column set has `n_discovered`,
`n_discovered_h{1,2,3}`, `root_owned`, `reachable`, `n_touched_nodes`, … and **no `source_node`,
`target_node`, or `node_ids`** (verified on `drift_static_seed42_evalstatic.csv` and its score CSV).
There is therefore no way to ask "after step t in the static twin, did the agent discover/own/act on
node **v**" — the identity of what the static agent discovered/owned/acted-on is not in the data. STEP 2
as written is not computable from existing data. (For `membership_join`, STEP 2 is additionally
undefined **in principle**: the joined node does not exist in the static twin at all.)

## 0.3 Is agent_drift decomposable? [ARTIFACT — YES, STEP 4 is possible]

`_log_drift_rows` forms it as the relative drift of the pooled vector across the agent's own action
(`cyberbattle_env_compressed.py:815`):
```
815  agent_drift_full = self._rel_drift(h1.combined, h2.combined)   # h1 = pre-action, h2 = post-action
```
Every drift row also carries **`n_discovered_h1` and `n_discovered_h2`** (pre- and post-action discovered
counts). A step **discovered ≥1 node iff `n_discovered_h2 > n_discovered_h1`**, so agent_drift can be
split into discovery-steps vs no-discovery-steps directly from the logged columns. **STEP 4 is
computable.** (Note: this attributes by *whether discovery happened*, not by node identity — sufficient
for STEP 4, which only needs the discovery/no-discovery split.)

## 0.4 Checkpoint provenance of 0.492 / −0.804 [ARTIFACT]

From `evidence_taskT.md:294-298`: SNR (= same-step `change_drift_full / agent_drift_full`) at ~100
discovered nodes = **0.492 [0.385, 0.632]**, log-log slope vs `n_discovered` = **−0.804 [−0.842,
−0.765]**, membership_leave. Produced by the **5-seed TRPO multi-topology attenuation-gate agents**
(`cyberbattle/agents/logs/trpo_250k_tuned_compressed_band{10-15,30-40,80-100}_seed{42,100,123,200,300}`,
**250k** steps), via `compute_attenuation_analysis.py` on the gate drift logs. **These are NOT the F1/F2
single-topology agents that Task F4 retrained** (F4 touched only `runs/trpo_250k_F1_static_*` and
`f2_runs/trpo_250k_F2_static_*`). So **the gate figures are unaffected by F4 and remain valid on their
250k gate checkpoints** — no recompute needed on provenance grounds. (They are, separately, blocked from
the STEP-1 de-contamination by 0.1.)

## 0.5 Episode alignment [ARTIFACT — holds ONLY for episode 0]

**NO — episode k of a static run and episode k of the matched dynamic run do not begin from the same
initial state for k ≥ 1.** The starter node is drawn randomly every episode from Python's **global
`random`** stream (`cyberbattle_env.py:236`):
```
236  self.starter_node, _ = list(self.environment.nodes(data=True))[random.randrange(len(self.environment.nodes))]
```
(`random_starter_node: true` in every train_config). The dynamic run's per-step change draws consume the
**same** global stream (`_apply_dynamic_leave`, `cyberbattle_env.py:605` `random.random()`, plus the
`numpy.random` batch draws), and the eval harness seeds **once** before the episode loop
(`taskF1_eval.py`: `np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)`), never per
episode. So after episode 0 the two runs' RNG states diverge and their episode-k starters differ.
Only **episode 0** shares a starter. The starter is **not logged** (no `source_node` in the drift CSV,
and `reachable` is starter-invariant here — constant 32 at the dense 30-40 topology — so it cannot even
proxy the starter), so the misalignment **cannot be repaired by starter-matching** from existing data.
STEP 2.3 anticipated *within-episode* divergence after the first change; this is a more fundamental
*cross-episode* misalignment that invalidates episode-index matching beyond episode 0.

---

## What IS vs ISN'T computable (summary for the decision)

| step | needs | computable on existing data? |
|---|---|---|
| STEP 1 (contamination-free baseline) | static twin + per-step `agent_drift_full` + `n_discovered` | **F-series: YES** (match on `n_discovered`, no episode-alignment needed). **Gate (0.492/−0.804): NO** — no static condition (0.1). |
| STEP 2 (counterfactual relevance) | static twin's per-node discover/own/act on node **v** + episode alignment | **NO** — identities not logged (0.2); alignment broken (0.5); undefined for join. |
| STEP 3 (perception axis) | `change_drift_full` + STEP-1 threshold (perception); STEP-2 (relevance) for the 6-cell | perception: **YES**; 6-cell table: **blocked by STEP 2**. |
| STEP 4 (discovery decomposition) | `n_discovered_h{1,2}` + `agent_drift_full` | **YES** (0.3). |

## Things noticed while reading (unsolicited)

1. **The quoted figures' own baseline cannot be de-contaminated in analysis at all** — not because of a
   subtle statistical issue but because the gate never ran a static arm (0.1). FAULT 3's fix as written
   (a static-twin baseline) is therefore not an analysis fix for the gate; it is a re-run.
2. **STEP 2's relevance repair is blocked by a logging gap, not a conceptual one.** The counterfactual is
   well-defined; the data simply never recorded which nodes the static agent touched. A one-line addition
   (log `source/target` or the discovered/owned id sets per step) in a *future* eval would make it
   computable — but that is a re-run, forbidden here.
3. **FAULT 2's own evidence is already in hand and supports the review:** the mean/full slice responds to
   100% of events (`evidence_taskT.md`), i.e. change_drift is essentially never exactly zero on a
   discovered-node change, so "ABSENT" (STEP 3.3) will indeed be empty for every current condition — the
   strict form of blindness cannot occur in conditions that only change discovered nodes. This is
   confirmable now and does not need STEP 2.
4. **The SNR's non-scale-invariance (FAULT 3's urgent point) is checkable now without the repair:** the
   numerator (`change_drift_full`) and denominator (`agent_drift_full`) are both logged per event in the
   gate CSVs, and STEP 4's discovery split (0.3) is available on them, so **STEP 4.4's dilution-expectation
   check is computable on the gate data even though STEP 1/2 are not** — if the decision is to run
   anything, STEP 4 is the piece that stands on its own.

**GATE: reported 0.1–0.5. Recommend STOP** (STEP 2 not computable; STEP 1 not computable for the quoted
figures). STEP 4 (and STEP 4.4) is the one part fully supported by existing data if a partial run is
wanted. Awaiting acceptance.

---

# STEP 4 — Is the SNR slope an artefact of the denominator? [RAN, per user request]

**PROVENANCE:** 5-seed TRPO attenuation-gate agents `trpo_250k_tuned_compressed_band{10-15,30-40,80-100}_seed{42,100,123,200,300}` at **250k** (NOT F4-retrained). Data: `attenuation_gate_archive/2026-07-26_trpo_5seed_gate/attenuation_drift_logs/drift_*.csv`. membership_leave; SNR = `change_drift_full / agent_drift_full`; zero-agent-noise-floor fraction = **0.660** (excluded, matches Task T's 0.664). Discovery-step = `n_discovered_h2 > n_discovered_h1` (0.3). Zero dropped beyond the stated filters.

## 4.1 Discovery prevalence and its effect on agent_drift, per band [FINDING]

| band | discovery-step frac | mean agent_drift (discovery) | mean agent_drift (no-disc) | ratio |
|---|---|---|---|---|
| 10-15 | 0.058 | 0.976 | 0.0467 | 20.9× |
| 30-40 | 0.106 | 1.022 | 0.0266 | 38.5× |
| 80-100 | 0.229 | 1.048 | 0.0195 | 53.7× |

The review's premise is **confirmed at the all-steps level**: discovery-step prevalence grows ~4× with band (5.8%→22.9%) and a discovery step's agent_drift is 20–54× a no-discovery step's — so the agent_drift denominator's *composition* genuinely varies with scale.

## 4.2 / 4.3 SNR slope, OLD (all) vs NEW (no-discovery denominator) [FINDING]

| method | OLD (all leave events) | NEW (no same-step discovery) |
|---|---|---|
| median-SNR-per-n_discovered-bin OLS (Task-T-style) | slope **−0.810** (ref −0.804 ✓) | slope **−0.832** |
| per-event log-log OLS | slope −0.737 [−0.766,−0.706] | slope −0.781 [−0.811,−0.752] |

The median-bin OLD slope **−0.810 reproduces Task T's −0.804**, validating the method. **The slope SURVIVES** the discovery exclusion — it does **not** flatten; both methods *steepen* it slightly. Reason: **only 957/18,658 leave events (5.1%) coincided with same-step discovery**, so the SNR denominator is already ~discovery-free. **The reported attenuation trend is NOT substantially a discovery effect.** (The 4.1 contamination lands on non-leave-event steps, which the SNR does not use.)

## 4.4 Scale-invariance — numerator/denominator dilution decomposition [FINDING]

Under pure 1/N dilution, numerator and denominator each have log-log slope −1 vs n_discovered, so the **ratio's expected slope = 0** (scale-invariant). Measured (median-bin):

- numerator `slope(change_drift_full)` = **−1.461** — the change signal dilutes **FASTER** than 1/N
- denominator `slope(agent_drift_full, no-disc)` = **−0.567** — the noise baseline dilutes **SLOWER** than 1/N
- implied SNR slope = −1.461 − (−0.567) = **−0.894** ≈ measured NEW −0.832 ✓

**The −0.81 SNR decline is neither dilution (which predicts 0) nor discovery (survives the split): it is the GAP between a change signal that over-dilutes (−1.46) and an agent-noise baseline that under-dilutes (−0.57).**

## STEP 4 verdict [FINDING]

**"Is the slope an artefact of the denominator?" — Two-part answer.** (i) It is **NOT a discovery artefact**: excluding discovery steps does not flatten it (survives, even steepens; only 5.1% of leave events were contaminated). (ii) But the SNR *ratio* **is** confounded by its denominator in a different way — `agent_drift` is a slow-diluting, non-1/N baseline (slope −0.567), so the SNR is not the scale-invariant quantity FAULT 3 expected, and it conflates real change-attenuation with the denominator's behaviour. **The cleaner measure is the numerator alone: `change_drift_full` attenuates at slope −1.461 vs n_discovered — real, and steeper than 1/N.** So the underlying attenuation is genuine and strong; the SNR ratio *understates* it (the declining denominator partly offsets it) and should be replaced by the numerator drift, not the ratio, for the attenuation claim. This is consistent with FAULT 3's charge that the same-step `agent_drift` denominator is unsuitable — confirmed here by direct decomposition. (STEP 1–3 remain not computable on existing data per the STEP 0 gate; only STEP 4 was run.)

---

# STEP 5 — the dilution reference, MEASURED [RAN, per user request]

**PROVENANCE:** gate agents `trpo_250k_tuned_compressed_band*_seed*` @250k; 54,832 membership_leave events (n_discovered≥2). Absolute per-slice drift = `change_drift_slice × norm_h2_slice`; log-log slope vs `n_discovered` via **mean-absolute-drift-per-integer-bin** (mean INCLUDES zero/silent events, so extremal silence is captured); bootstrap 95% CI over events. Zero dropped beyond stated filters.

## 5.1 The reference is ≈ −1.0 after all [ARTIFACT]

`||h_bar − h_k||` is **NOT directly loggable** — only `||h_k||` (`delta_h_v_norm`) and `||h_bar||` (`norm_h2_mean`) are stored; the difference-norm needs the cross term `h_bar·h_k`, which is absent. It can be recovered only *as* `abs_mean × (N−1)`, so measuring its slope and checking the mean-slice slope against it is circular. Independent component evidence:

| quantity | slope vs n_discovered |
|---|---|
| `||h_bar||` (norm_h2_mean) | **−0.259 [−0.264, −0.253]** (mean-slice norm shrinks ~1.8× over the range — matches the "~2×" note) |
| `||h_k||` (delta_h_v_norm) | −0.100 [−0.107, −0.092] |
| `||h_bar − h_k||` (recovered) | a = **−0.007 [−0.022, +0.007]** ≈ 0 |

**Despite `||h_bar||` and `||h_k||` each shrinking, their difference `||h_bar − h_k||` is ~N-independent (a ≈ 0), so the dilution reference a−1 ≈ −1.0. The assumed −1.0 is validated** (the directly-measured mean-slice absolute slope is −1.09, ~0.09 steeper — an aggregation gap between mean-per-bin-of-products and the direct recovery; both put the reference at −1.0 to −1.1, not far from −1.0). **FAULT's worry that the reference is materially ≠ −1.0 does NOT hold.**

## 5.2 Absolute change-drift slopes, all slices [FINDING]

| slice | absolute slope vs n_discovered |
|---|---|
| mean | **−1.092 [−1.107, −1.079]** |
| max | −0.882 [−0.911, −0.865] |
| min | −0.927 [−0.981, −0.906] |
| full | **−0.891 [−0.917, −0.874]** |

## 5.3 Decomposition restated on measured quantities [FINDING — reverses the earlier one]

| quantity | value | earlier framing |
|---|---|---|
| empirical mean-slice (dilution) reference | **−1.092** | assumed −1.0 |
| full-vector absolute slope | **−0.891** | (STEP 4 used the *relative* −1.461) |
| **extremal contribution = full − mean** | **+0.200** | inferred −0.46 |

**The full-vector absolute drift declines SLOWER than the mean slice, not faster** — the extremal slices *offset* the mean's dilution rather than steepening it. Mechanistically the full concatenated vector's magnitude is dominated by the max/min slices (whose norms do not shrink — see 5.5), so `slope(full) ≈ slope(max/min) ≈ −0.89`, shallower than the mean's −1.09.

## 5.4 Consistency check — it FAILS, reported not explained away [FINDING]

Per-slice response rates (this data): mean 1.000/1.000/1.000; max 0.986/0.840/**0.429**; min 0.985/0.828/**0.361** (reproduces the given 98.5/84.0/43.0 and 98.5/82.8/36.1). Slopes: mean −1.092, **max −0.882, min −0.927**. **The extremal slopes are SHALLOWER than the mean, not steeper — the check FAILS.** The premise "a slice silent on a growing share of events must decline steeper" is wrong here: the mean-over-events absolute max/min drift ≈ response_rate × per-response magnitude, and the per-response magnitude is an *elementwise extreme of a single node*, which does **not** dilute at 1/N the way an average does. So even as the response rate falls (0.99→0.43), the surviving magnitude stays large and the net absolute-drift decline (−0.88) is shallower than the mean's 1/N decline (−1.09). The inconsistency is real and its cause is that extremal magnitudes are not 1/N-diluting.

## 5.5 Relative vs absolute [FINDING]

| slice | relative slope | absolute slope |
|---|---|---|
| mean | −0.846 [−0.860, −0.834] | −1.092 |
| max | −1.022 [−1.054, −1.005] | −0.882 |
| min | −1.041 [−1.097, −1.019] | −0.927 |
| full | −0.988 [−1.013, −0.971] | −0.891 |
| full (median-per-bin, STEP-4 method) | −1.545 [−1.568, −1.522] | — |

On the RELATIVE basis the ordering flips (max/min steeper than mean), because relative drift's denominator is the slice norm and those norms are themselves N-dependent in *opposite directions*: `slope(norm_mean) = −0.26` (shrinks) vs implied `slope(norm_max) ≈ +0.14` (the max-pool norm GROWS with N). Relative drift therefore conflates the physical signal with the slice-norm's N-dependence. **The decomposition should be built on ABSOLUTE drift** — it is the physical signal magnitude and the reference derivation (`||h_bar−h_k||/(N−1)`) is itself absolute; the relative basis silently mixes in the norm slopes.

## VERDICT (one sentence)

**The earlier −0.46 attribution to the extremal slices is WITHDRAWN** — it compared a *relative* full-vector slope (−1.461, median-per-bin) against an *absolute* mean-pooling reference (−1.0), incompatible bases; restated consistently on absolute drift, the dilution reference is ≈ −1.0 to −1.09 (validated), and the extremal slices contribute **+0.20** (they make the full vector decline *slower*, and dominate it, because elementwise extremes do not dilute at 1/N) — the opposite sign to what was inferred.
