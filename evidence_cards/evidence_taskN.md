# Task N — the six gates (read-only audit)

No runs, no training, no modification. Six independent premise-checks. Branches reported, not selected.

---

## GATE N1 — does degree predict ownership? [decides RQ3(b)/RQ1(d) wording]

**N1.4 (availability first):** degree is **NOT logged per event**. The drift CSV has no node_id/degree; the
`leaveown_*` side-CSV has `node_id` + `was_owned` but no degree. Degree is recomputable as the **static
access-graph degree** via `leaveown.node_id` → scenario `.pkl` `access_graph.degree`. The **discovered-graph
degree at the moment is NOT recoverable** (the discovered graph is not logged — established in Tasks W/P).
`leaveown` exists only for **80-100** (30-40 membership eval wrote none — Task X), so N1 is answerable at
**80-100 only, on static degree**.

**N1.1 / N1.3 [FINDING] (80-100, static access-degree, n=15,539 leave events, zero dropped):**
- corr(access-degree, was_owned) = **+0.350**.
- mean degree | owned = **156.9** (sd 20.8, n=3,603) | not-owned = **94.5** (sd 79.6, n=11,936).

**N1.2:** `was_owned` vs `discovered`: every leave node is discovered by construction (eligibility
`_get_removal_eligible_nodes` filters `discovered_nodes`, `cyberbattle_env.py:511-517`), so the discovered
indicator is **constant True** and carries nothing.

**Bearing on the decision (not selected):** a **moderate-to-strong positive** relation (owned departing
nodes are ~1.66× the degree of unowned) — under the gate's own rule this points to **restricting RQ3(b) to
the behavioural residual**. Caveat that must travel with it: this is *static* degree at 80-100 only; the
per-moment discovered-graph degree, which is what the mechanism would use, is not recoverable from logs.

---

## GATE N2 — is loss attributable to individual events? [decides RQ1(d)/RQ3(b) answerability]

**N2.1 cost quantities [ARTIFACT]:** cost exists **only at EPISODE level**. `score_*csv` columns:
`root_owned, reachable, n_discovered, won, score` — one row per (agent_cond, seed, scenario_id, episode).
No per-event or per-step cost field anywhere.

**N2.2 / N2.3 arithmetic component [FINDING]:** attributable **exactly, event by event, but only where
`leaveown` exists (80-100)**. `leaveown_*` logs `was_owned` per departing node (`taskF2_eval` snapshots
`owned_nodes` before the step, checks each leave node against it) — so an owned departure = −1 to
root_owned, **counted, not estimated**. Per-band owned-departure count (80-100): **3,603 of 15,539 leave
events** (23.2%). **At 30-40 the arithmetic component cannot be counted** (no leaveown) — it would have to be
estimated, and the gate warns that any estimation error lands in the behavioural residual.

**N2.4:** the behavioural residual (episode cost − arithmetic) is **episode-level only** — the arithmetic
component is a per-episode *sum* of owned departures; the residual cannot be attributed to single events.

**N2.5 [ARTIFACT]:** 200 episodes/band/seed × 5 seeds = 1,000 episodes/band. Events/episode (80-100
immediate): ~14 leave, ~2.8 join (capped at 3), ~15 property.

**N2.6 middle option [FINDING]:** for each episode, from existing logs —
- **max degree among departed nodes:** ✔ constructible (80-100) via leaveown node_id → static access-degree.
- **count of departures above a degree threshold:** ✔ same source (80-100).
- **summed propagation magnitude across the episode:** ✗ **NOT available** — propagation is not logged (the
  drift CSV logs change_drift, not per-event propagation; propagation only exists in the offline Task-P
  probe, on proxy graphs, not linked to logged events).
- **count of change events of each type:** ✔ from the drift CSV (`change_type`, `event_phase=='immediate'`).

**Branches (reported, not selected):**
- (i) event-level cost attribution — **does NOT exist** (N2.1/N2.4).
- (ii) episode-level cost + N2.6 episode summaries — **partially constructible at 80-100** (degree summaries
  and event-type counts yes; summed propagation no; 30-40 lacks the leaveown to build even the degree
  summary). So (ii) is available for a **degree/event-count** episode summary at 80-100, **not** for a
  propagation summary and **not** at 30-40.
- (iii) disturbance-concentration (per-event change drift) — **fully available** (drift CSV), but it is the
  perception question, not cost.

---

## GATE N3 — are the pre/post observation vectors saved? [decides RQ2(c): half-day vs re-run]

**N3.1 / N3.2 [FINDING]:** the pre- and post-change pooled vectors are formed as `_DriftSnapshot`
(`cyberbattle_env_compressed.py:67`, fields `node_embeddings, combined, slices, n_discovered`) and held **in
memory only**. `_build_drift_row` persists **norms and scalar relative-drifts** (`norm_h1/h2/h3`,
`change_drift_full`, `norm_*_slices`, …) — **the full observation vectors are NOT written to disk.** So the
paired counterfactual **cannot be run offline from logs**; the vectors must be regenerated → **an evaluation
sweep**. RQ2(c) is therefore a re-run, not a half-day task, unless logging is added.

**N3.3 [ARTIFACT]:** the pooled 192 dims (3 aggregations × 64) are **NOT** the whole policy input. The
observation is a **Dict** `{"graph_embeddings", "discrete_features"}`; `graph_embeddings` =
`node_embeddings_dimensions × |aggregations|` (192) **+ next_escalation_target (64)** (+ interest_node (64)
for node-goals) (`obs shape :142-151`), and `discrete_features` = `[len(discovered), len(owned)]`
(`create_discrete_features`). All of these change with the graph at a change event, so a counterfactual that
swaps the *whole* post-change observation is valid; one claiming to vary "only the pooled part" would be
inaccurate.

**N3.4:** policy checkpoints are on disk and **loadable for inference alone** (`TRPO.load(ckpt, device=cpu)`
+ `VecNormalize.load`) without a training environment — done routinely in every eval script.

**N3.5 candidate set [FINDING]:**
- (a) built in `create_continuous_action_space` (`:972`) from `running_owned_nodes × running_discovered_nodes
  × vulnerabilities_embeddings_per_node_type`. **Neither the set nor its inputs are logged per event**, so it
  is **NOT recoverable offline** at a change moment (before or after) without re-encoding the graph — i.e.
  only inside a re-run. So of the three quantities the gate wants, only the first (policy output on a given
  observation) is computable offline; the two involving the candidate set need the re-run.
- (b) the snap is `find_closest_action_embedding` (`:1064`), a **pure `np.argmin` over cosine distance** of
  the policy vector against `action_embeddings` — no hidden state, reproducible offline given the set (Task V
  quoted it in full).
- (c) TRPO `MultiInputPolicy` output is a **DiagGaussian distribution** (continuous action space); its
  parameters (mean, log_std) are recoverable via a policy forward pass on a given observation, so a
  divergence between two output distributions **is** computable — provided the observation is available (N3.1
  says it is not, offline).

**N3.6 [FINDING]:** no-change episodes ARE logged with drift detail (the `static` condition logs
`agent_drift_full` = step-to-step pooled movement with no change), so a **scale baseline exists as a scalar**
(ordinary step-to-step drift). But it is logged as **norms/scalars, not vectors** — same limitation as N3.1
— so the *distributional* baseline the counterfactual needs is not offline-available either.

**Net:** vectors not saved, candidate set not recoverable → **one evaluation sweep with added logging is
required** (log h2/h3 vectors + the candidate set at each change event). Rough cost on this box: it is the
existing membership/static eval grid (already ~30–40 min/5-seed cell at 250k here) plus vector serialisation
— order **a few hours** for both bands, dominated by disk I/O of the vectors, not compute. Not run.

---

## GATE N4 — can the generator hold mean degree fixed while node count varies? [decides the highest-value experiment]

**N4.1 [FINDING]:** the connection probability is a **PARAMETER, not a constant**. `generate_network.py`:
```
102  knows_neighbor_probability_range=None,
114  if knows_neighbor_probability_range is None:
115      knows_neighbor_probability_range = [0.2, 0.3]
190  knows_neighbor_probability = scale_probability_range_with_score(knows_neighbor_probability_range, map_confidentiality_impact_to_real(vulnerability['confidentiality_impact']))
191  for node_id in nodes_graph.nodes:
194      if random.random() < knows_neighbor_probability:  # per ordered pair
```
So the experiment is **not blocked at source** — `p` is settable.

**N4.2:** call path — the config key `knows_neighbor_probability_range` flows through
`generate_graphs.py` into `generate_network`; **already exposed** in every `generation_config_*.yaml`
(`knows_neighbor_probability_range: [0.2, 0.8]`, verified in Task H).

**N4.3 [FINDING]:** the model is a **directed per-ordered-pair Bernoulli** graph (each recon-bearing node
links to each other node w.p. `p·impact_scaling`) — an **Erdős–Rényi-like** knows-graph, so mean knows-out-
degree ≈ `p_eff·(N−1)` and **`p ∝ 1/(N−1)` holds mean degree fixed** — and Task H's measurement supports it
(degree/N ≈ 0.62–0.65 constant across bands, i.e. degree grows linearly, the ER signature). **Caveat that
must be checked before relying on it:** `p` is scaled per-vuln by `map_confidentiality_impact_to_real` (line
190), so the *effective* p is a distribution, not a constant; the ER relation holds in expectation but the
constant is ~0.62–0.69 (Task H), not the nominal mid-range 0.5 — so the target `p` must be **calibrated
empirically** (generate, measure degree, adjust), not set analytically.

**N4.4 [FINDING]:** `knows_neighbor_probability` affects **only the knows/discovery edges** (hence the
derived access/reachability graph). Vulnerability assignment, service counts (`num_services_range`), node
values (`value_range`), and property distributions are **independent parameters** and are **NOT** forced to
change — so no new confound there. **But** two coupled effects: (i) lowering `p` at large N reduces
reachability, and the acceptance thresholds `minimum_{knows,access,dos}_connectivity_threshold: 0.5`
(rejection sampling) may reject more sparse large-N graphs → generation may need more attempts or a relaxed
threshold (which would itself be a disclosed change); (ii) holding *degree* fixed is the goal, but it also
fixes reachability, which is the very thing that normally scales — that is intended, not a confound.

**N4.5 [ESTIMATE]:** generation is dominated by a one-time load (553 MB NVD data + SecureBERT model, ~several
min) then per-graph work; Task H's single 225-node graph took ~29 min almost entirely in that one-time load.
Amortised over 15 graphs (3 sizes × 5), estimate **~45–75 min total** (load once + ~1–3 min/graph, more at
larger N and under rejection retries). Not generated.

---

## GATE N5 — mean-channel-only arm + constant-replacement arm buildable? [decides the 3-arm ablation]

**N5.1 [ARTIFACT]:** the pooled vector is assembled at `cyberbattle_env_compressed.py:411-424`:
```
411  for agg_type in self.graph_embeddings_aggregations:   # config = [mean, max, min]
413/417/419  graph_embeddings.append(np.average/np.min/np.max(embeddings_array, axis=0))
424  observation_embedding = np.concatenate(graph_embeddings)   # 3 x 64 = 192
```

**N5.2 [FINDING]:** the layout is **derived, not hard-coded in many places** — it flows from one config key
`graph_embeddings_aggregations` (`[mean,max,min]`) × `node_embeddings_dimensions` (64), and the
`observation_space` shape (`:142-151`) derives from the same. So **Arm 2 (mean only)** = set
`graph_embeddings_aggregations: [mean]` → 64 dims, one config edit. **But** the trained policy and its saved
VecNormalize stats assume 192, so Arm 2 requires a **new policy + new VecNormalize (retrain)** — not a config
flip on the frozen agents.

**N5.3 the silent-break risk — CONFIRMED present [FINDING]:** VecNormalize **does** sit between observation
and policy (`taskF1_eval.py:140,154` `VecNormalize.load` + `normalize_obs`; training uses it too).
- (a) expression: `np.clip((obs − obs_rms.mean) / np.sqrt(obs_rms.var + self.epsilon), −clip_obs, clip_obs)`,
  **epsilon = 1e-8**, clip_obs = 10 (SB3 defaults, verified).
- (b) for a channel with variance **exactly 0** and value = its running mean: `(c−c)/sqrt(0+1e-8) = 0` → a
  **constant 0** ⇒ Arm 3 is the control it should be. **But** if the substituted channels carry any
  floating-point residue (var = ε_fp ≪ 1e-8, value ≠ mean), the output ≈ residue/√(1e-8) = residue·10⁴ →
  **structureless amplified noise**, and Arm 3 would receive 128 dims of noise, plausibly perform WORSE than
  the 64-dim Arm 2, and support the opposite conclusion. **The risk is real and depends entirely on whether
  the substitution is bit-exactly constant before VecNormalize.**
- (c) mitigations available: the obs is a **Dict**, so VecNormalize can be told which keys to normalise
  (`norm_obs_keys`) — but the 128 extremal dims live *inside* the `graph_embeddings` key, so per-key
  exclusion won't isolate them. The clean path is to **apply the constant substitution AFTER
  `normalize_obs`** (post-normalisation), in the same wrapper that already calls `normalize_obs` in the
  eval/train loop, so VecNormalize never sees the substituted dims. That path exists and is where to build it.
- (d) **pre-flight assertion to build into the ablation task (not run here):** over a sample of ≥1,000 steps,
  print `min`, `max`, and `unique-count` of each of the 128 substituted post-normalisation dimensions and
  **assert every one is a single constant** (max−min = 0 exactly); abort training if any dimension varies.

**N5.4 [FINDING]:** the **frozen encoder is per-node (64-d output)** and does **not** assume the 192 layout —
it is unaffected by the pooling choice. The **policy and the saved VecNormalize statistics DO** assume 192,
so Arm 2 needs both regenerated (i.e. a fresh train + fresh VecNormalize); Arm 3 keeps 192 so it reuses the
layout but still needs its own trained policy.

**N5.5 [ESTIMATE, this box's measured fps]:** 30-40 ≈ 76 fps, 80-100 ≈ 48 fps (Task P/H). At the reported
**250k** budget: 30-40 ≈ 55 min/run × 5 ≈ **4.6 h/arm**; 80-100 ≈ 87 min/run × 5 ≈ **7.3 h/arm**. Three arms
× two bands ≈ **~36 h** sequential (far less under concurrency; historical hardware ~2–3× faster). Not run.

**Branch:** Arm 3 is **buildable** provided the substitution is applied post-normalisation (N5.3c) with the
pre-flight assertion (N5.3d); without that discipline it breaks silently.

---

## GATE N6 — two-hop coverage per removal trial? [decides a new sub-question]

**N6.1 [ARTIFACT]:** the Task-P probe graphs are **deterministically reproducible** (fixed `random.seed(0)`,
deterministic BFS/DFS trees over fixed scenario `.pkl`s), though **not saved** — each trial is revisitable by
re-running the (cheap, forward-pass-only) probe.

**N6.2:** per trial, 1-hop / 2-hop node counts and fractions are **computable** — the probe already uses
`nx.single_source_shortest_path_length` for hop distance (Task P STEP 3.1); reusable as-is.

**N6.3 [FINDING] — two-hop coverage, mean AND variance, on the ACCESS graph (dense, the "60%" note's basis),
5 scenarios/band, cheap (BFS only, no encoder):**

| band | 1-hop cov mean | 2-hop cov mean | 2-hop cov VARIANCE (sd) |
|---|---|---|---|
| 10-15 | 0.803 | 0.803 | 0.078 (0.279) |
| 30-40 | 0.690 | 0.702 | **0.095** (0.308) |
| 80-100 | 0.820 | 0.824 | 0.070 (0.265) |

**Mean coverage is high everywhere (~0.70–0.82)** — it confirms the "≈60% one-hop" note (and cannot by itself
explain a band-specific effect). **Variance is non-monotone and peaks at 30-40 (0.095), lower at both
extremes** — and it **tracks the degree-correlation** (+0.66/**+0.79**/+0.39, also peaking at 30-40). So the
*variance*-carries-the-effect hypothesis is **plausible and directionally supported**, not the mean. **Strong
caveat:** this is coverage on the **dense access graph**, whereas the probe's propagation ran on the **sparse
DFS-tree** (Task P) — different graph — so the coverage figures and the propagation correlation are on
different structures and the link is **suggestive, not established**.

**N6.4:** the probe kept **only the summary** correlations, not per-trial records on disk — but since the
probe is reproducible, the correlations **can be re-derived with 2-hop coverage added as a variable** by
re-running it (cheap). Currently only summaries exist.

**N6.5 [FINDING]:** the same computation is available on the **probe's graphs** but **cannot be linked to the
real logged change events** — the discovered graph's shape is **not logged at leave events** (Tasks W/P). So
the sub-question is answerable **on the probe only**, and that boundary must be stated: it characterises the
encoder's behaviour on representative graphs, not the specific graphs the logged response rates came from.

**Branch:** computable **on the probe** (cheap re-run, forward passes only) → the sub-question can be added,
scoped to the probe; it **cannot** be linked to logged events without logging the discovered-graph shape.

---

## One cross-gate note (unsolicited)

Three of the six gates (N1 discovered-graph degree, N2.6 summed propagation, N3 obs vectors + candidate set,
N6.5 real-event coverage) fail for the **same single reason**: the per-event **node identities and the
discovered-graph structure are never written to the drift log** (only counts/norms/scalars are). A single
future logging addition — the departing/affected node ids, the evolving_visible_graph edge list, and the
h2/h3 vectors at each change event — would unblock N1 (real degree), N2.6 (propagation summary), N3 (offline
counterfactual), and N6.5 (link to logged events) at once. That is a re-run, out of scope here, but it is one
change, not four.
