# Task CX — the fully crossed (change-type × discovery-status) design

Goal: identify the claim "whether a change registers is decided by discovery status, not change
type" by MEASURING all six cells of the 3×2 (removal/addition/property × discovered/undiscovered),
rather than filling the empty cells by the construction argument (which Appendix B shows already
failed once for property change). Zero-shot, no retraining; static-trained agents.

All file:line references are to branch `attenuation-pooling-scale` as of 2026-08-01.

## STEP 0 — what gates what, and what breaks (READ-ONLY) [ARTIFACT/FINDING]

### 0.1 Removal eligibility — the discovered predicate and its downstream assumers [ARTIFACT]

`cyberbattle/_env/cyberbattle_env.py:510-518`, called from `_apply_dynamic_leave` (`:565`):
```
def _get_removal_eligible_nodes(self):
    return [
        node for node in self.discovered_nodes
        if self.get_node(node).status == model.MachineStatus.Running
        and node != self.starter_node
        and node != self.source_node
        and node != self.target_node
        and node != getattr(self, "interest_node", None)
    ]
```
Predicates, all enforced here: **(1) `node in self.discovered_nodes`** — the discovered predicate, the
one to relax; **(2)** status `Running`; **(3)–(6)** not starter/source/target/interest (protected roles).

**Relaxing (1)** means drawing the candidate list from a broader set (running nodes of
`self.environment.nodes()`) instead of `self.discovered_nodes`. Downstream code that might assume the
removed node was discovered — each checked:
- **shared purge** `remove_node_common` (`:707-726`): `if node_id in self.discovered_nodes:` / `if
  node_id in self.owned_nodes:` — **guarded** (`:712,714`); count decrements guarded on the
  `shortest_paths_starter_*` snapshot (`:716-724`). Safe for an undiscovered node.
- **compressed visible-graph purge** `remove_node_dynamic` (`cyberbattle_env_compressed.py:342-360`) →
  `remove_node_evolving_visible_graph` (`:291-293`): `if node_id in self.evolving_visible_graph.nodes():`
  — **guarded**; `self.node_embeddings.pop(node_id, None)` and the action/edge filters are pop-with-
  default / comprehensions — **safe** if the node was never in those structures.
- **degree weighting** (`:592`): `self.access_graph.degree(n) if n in self.access_graph else 0` —
  **guarded**; an undiscovered node is an original topology node so it IS in `access_graph`.
- **drift snapshot / relevance / event logger:** the drift h2→h3 pair is computed on
  `evolving_visible_graph`; an undiscovered node is absent from it, so its removal legitimately produces
  **zero** movement (the predicted result). The event logger records `changed_node_discovered` per event
  (designed for both states); the relevance flag keys off discovered status → an undiscovered removal is
  flagged not-perceived.
- **score:** win-condition denominators only decrement via the guarded snapshot check above.
→ **allow_undiscovered_removal looks non-crashing and correctly guarded throughout.**

### 0.2 Join rate and its cap [ARTIFACT]

`_apply_dynamic_join` (`cyberbattle_env.py:640-700`), `_spawn_node_from_pool` (`:768-799`). **Two** caps:
- **Hard per-episode budget** (`:648-650`): `remaining_budget = self.dynamic_max_joins_per_episode -
  len(self._dynamic_joined_this_episode)` … `if remaining_budget <= 0 or not
  self.dynamic_join_donor_pool: return []`. **`dynamic_max_joins_per_episode = 3`** in these configs
  (train_config.yaml; env default `:74`). This is the binding limit on join event COUNT.
- **Alive ceiling** `_get_dynamic_ceiling` (`:546-550`): `min(ceiling, self.num_nodes +
  self.dynamic_max_joins_per_episode)` → alive node count ≤ **N+3**.
- **Rate anchor** (`:663`): `target_rate = ramp * (1.0 / self.dynamic_join_rate_interval)` =
  1/20 per eligible parent per step.
- **Donor pool** finite; `_spawn_node_from_pool` returns `None` when exhausted (`:779-780`).

**The "300/change_interval" ceiling is NOT the join cap.** `episode_iterations=300`,
`change_interval=20` → 300/20 = **15** is the legacy *property* change cadence (one attempt every
`change_interval` steps over a 300-step episode, `:449`), unrelated to joins. The join cap here is
`dynamic_max_joins_per_episode = 3`. **`uncapped_join` must relax BOTH** the per-episode budget (`:648`)
**and** the `N + dynamic_max_joins_per_episode` clamp in `_get_dynamic_ceiling` (`:550`), or joins stay
pinned at N+3 alive.

### 0.3 Property targeting — requires discovered, and is currently OFF [ARTIFACT]

`_patch_random_vulnerability` (`:894-911`) and `_disable_random_service` (`:913-931`), both:
```
running_nodes = [node for node in self.discovered_nodes
                 if self.get_node(node).status == model.MachineStatus.Running]
...
node_id = random.choice(running_nodes)
```
Target is drawn from **`discovered_nodes` ∩ Running** — requires discovered. Relaxing = draw from
`environment.nodes()` running instead. **Two caveats:**
1. Property only fires when `patch_service_dynamic_enabled and num_iterations % change_interval == 0`
   (`:449`). **`patch_service_dynamic_enabled = false` in every reported run** — this is why STEP-3
   reported 0 property events ("structurally impossible"). Both property cells require this **enabled**
   in the diagnostic condition.
2. Relaxing touches `update_node_dynamic` — see 0.4.

### 0.4 What might break, per relaxation [FINDING]
- **allow_undiscovered_removal — no named failure.** Every downstream purge/read is guarded (0.1). An
  undiscovered removal is a clean no-op on the visible graph and yields the predicted zero own
  contribution.
- **uncapped_join — no crash, but may not actually uncap.** Bounded by donor-pool exhaustion (handled,
  returns `None`) and the alive-ceiling `N+3` clamp (`:550`) unless that clamp is also relaxed. The
  Local/Global action-space headroom pre-sized from `dynamic_max_joins_per_episode` (`:542-545`) is
  **not** a hazard here — the compressed env uses a continuous action space, no fixed join headroom to
  overflow.
- **allow_undiscovered_property — NAMED CRASH: `KeyError` at
  `cyberbattle_env_compressed.py:297`.** `update_node_dynamic` (`:368-369`) →
  `update_node_evolving_visible_graph` (`:296-297`): `self.evolving_visible_graph.nodes[node_id].update(
  {'x': ...})` is **unguarded**; the comment (`:363-367`) explicitly relies on the node "always already
  in evolving_visible_graph," true only for a *discovered* target. Patching an undiscovered node → the
  node is absent from the visible graph → KeyError. The relaxation must add an
  `if node_id in self.evolving_visible_graph` guard (mirroring remove's `:292`) so an undiscovered
  property change is a legitimate no-op on the observation.

### 0.5 Expected cell counts, one band, ~400 ep/seed × 5 seeds (risk, not sampling) [ARTIFACT]
Reference = STEP-3 `membership_leave` counts (discovered-only, current rule): **5,121 / 21,388 /
27,998** at 10-15/30-40/80-100. Undiscovered fraction ≈ (N − n_discovered)/N ≈ **48% / 35% / 22%**
(n_discovered 6.3/22.6/70.5 vs N≈12/35/90).
- **Removal cells:** `allow_undiscovered_removal` keeps total removals ≈ constant (rate 1/change_interval
  is normalised over the larger pool), split by the undiscovered fraction → undiscovered-removal ≈
  **2.5k / 7.5k / 6.2k**; discovered-removal the complement. Both amply reportable.
- **Addition cells:** `uncapped_join` lifts joins from ~3/ep to donor/ceiling-bounded, so
  undiscovered-addition (spawn, own-contribution 0) is plentiful. **addition-DISCOVERED is the at-risk
  thin cell** — it needs a joined node to actually be *discovered* by the agent via the injected
  Reconnaissance, which OI-1/RQ2(c) flag as rare-to-never ("may never perceive them"). Report the risk;
  do not stratify-sample.
- **Property cells:** with the mechanism enabled, ~15 attempts/ep capped, cutoff-shortened to ~5–7/ep →
  thousands/seed, split by the undiscovered fraction → both cells reportable.
→ **Only addition-discovered is at material risk of being too thin; flagged, not engineered around.**

### 0.6 Cost [ARTIFACT]
One diagnostic condition, one band, 5 seeds, no retraining ≈ **1.5–2.5 h wall clock**. Basis: the STEP-3
sweep ran 3 bands × 5 seeds in ~4 h (~1.3 h/band); `uncapped_join` + enabled property add re-encodes per
extra change, so round up. GPU near-idle (CPU-bound), fits alongside other jobs.

**Whole-task estimate:** STEP 0 done (this); STEP 1 ≈ half a day coding + 30–60 min regression compute
(byte-identical, flags off, both bands — the gate); STEP 2 ≈ 1.5–2.5 h compute **but queued behind Task
L STEP 3 (done), the OI-1 probe re-run, and the RQ2(c) counterfactual** (not yet run); STEP 3 ≈ 30 min
table. ≈ **1 day of active work; calendar time gated by the OI-1 + RQ2(c) queue.**

**GATE: STEP 0 reported. STOPPING. No code written.**

## STEP 1 — three flags (default off) + the guard, PROVEN inert [FINDING, 2026-08-01]

### 1.1 — what was added (8 edits = 7 in base env + 1 in compressed)
All default to pre-CX behaviour; each relaxation is a **filter on the candidate list** (1.1a), so the
flag-off path reconstructs the exact original set.

| # | file | line | edit |
|---|---|---|---|
| 1 | `cyberbattle_env.py` | 77 | signature: `allow_undiscovered_removal/uncapped_join/allow_undiscovered_property=False` |
| 2 | `cyberbattle_env.py` | 126 | the 3 `self.` assignments |
| 3 | `cyberbattle_env.py` | 517 | `_get_removal_eligible_nodes`: `base = environment.nodes() if flag else discovered_nodes` |
| 4 | `cyberbattle_env.py` | 560 | `_get_dynamic_ceiling`: `if uncapped_join: return ceiling` (drops the N+max_joins clamp — 1.1b(i) limit 2) |
| 5 | `cyberbattle_env.py` | 662 | `_apply_dynamic_join`: uncapped budget = donor-pool headroom (1.1b(i) limit 1) |
| 6 | `cyberbattle_env.py` | 914 | `_patch_random_vulnerability` widened base |
| 7 | `cyberbattle_env.py` | 936 | `_disable_random_service` widened base |
| 8 | `cyberbattle_env_compressed.py` | 366 | `update_node_dynamic`: `if node_id in evolving_visible_graph` guard (1.1b(ii), in-scope) |

### 1.2 — byte-identical regression on Task L's VERIFIED recipe [FINDING]
**Instrument correction (recorded so it never recurs):** the first attempt used the full attenuation
**sweep**, which is NOT byte-identical run-to-run — same-code control gave 5148 vs 4970 drift rows (identical
20/20 episode + 0/0 skip counts → trajectory divergence, not structural). Primary cause: the sweep sets **no
per-run seed** (unseeded RNG diverges from step 1); `deterministic=False` predict + unpinned threads compound
it. The correct instrument is Task L's recipe (`drift_regression_check_v2.py`): fixed action sequence
`RandomState(42)`, seeded `random`/`numpy`/`torch`, `set_num_threads(1)`, `dynamic_mode=both`+`patch_service`
(`change_type=mixed`), 800 steps, `drift_logging=True`.

**Sides compared:** OLD = reconstructed **pre-CX** (`env_PRECX.py`/`compressed_PRECX.py`, proven = working tree
minus exactly the 8 edits above; §Step-5 below) swapped into the tree; NEW = **CX-flags-off**
(`env_CX.py`/`compressed_CX.py` = current tree). Topologies `scalability_30_40/1`, `scalability_80_100/1`.

| band | obs mism | reward | done | drift CSV differing cells | property events (pre-CX = CX-off) | leave | join |
|---|---|---|---|---|---|---|---|
| 30_40 | **0** | **0** | **0** | **0 / 47,226** | **37 = 37** | 27=27 | 0=0 |
| 80_100 | **0** | **0** | **0** | **0 / 67,932** | **35 = 35** | 45=45 | 0=0 |
| 30_40 (+donor pool) | **0** | **0** | **0** | **0 / 47,685** | **36 = 36** | 30=30 | **10=10** |

**`old == new_off`, zero differing cells, identical trajectories at both bands.** The differing-cell count is
**0** verbatim. Property fired **35–37×/band** (far from zero), so the widened `_patch/_disable` base was
**exercised** and the `update_node_dynamic` **guard condition was evaluated** on every property change — Step-3's
"the property path runs" confirmation is measured, not argued. **SCOPE — the guard is NOT yet verified.** With
flags off, property only ever targets the **starter**, which is always in `evolving_visible_graph`, so only the
guard's **TRUE branch** ran; its **FALSE branch** — the actual protective function, a no-op instead of a
KeyError on an *undiscovered* target — **has never executed** and runs for the first time in STEP 2 under
`allow_undiscovered_property`. The donor-pool run additionally fired **10 joins**, exercising the
`_get_dynamic_ceiling` relaxation branch — also byte-identical.

### 1.3 — RNG consumption unchanged when flags off [FINDING]
Each relaxation is `A if flag else B`; Python evaluates only the `else` branch, which is the **exact original
object** (`self.discovered_nodes`) in the same order, so the candidate-list length — hence every downstream
`random.random()`/`random.choice`/`numpy` draw — is consumed in the identical sequence. `uncapped_join`'s off
branches compute the original budget/ceiling expressions verbatim; the compressed guard is always-true when off
(discovered targets are in the visible graph). Empirically confirmed by the 0-cell trajectory+drift result.

### Step-5 baseline proof + reproducibility finding [FINDING]
No **committed** pre-CX baseline existed: prior tasks left both env files (and the `compute_*` scripts)
uncommitted-modified — **a project-wide reproducibility exposure: there was no named version of the environment
that produced the reported figures.** Reconstruction proven faithful directly: CX-backup ≡ working tree (empty
diff); `git diff --no-index` current→pre-CX shows only the 8 Task-CX blocks; pre-CX ≠ CX; all parse; round-trip
pre-CX + edits = CX byte-for-byte. **Remediated:** committed the environment code AND the analysis scripts —
10 files: the 2 env files, 5 `compute_*` + `compile_appendix_data.py`, `event_graph_logger.py`, a generation
config — as `1b42a2c`, tagged **`env-baseline-2026-08-01`**. Verified: no `.py` analysis script remains
uncommitted (`compute_*` all clean). Bulk artifacts (step3 logs 5.7G, gate archive 621M, outputs) excluded;
`evidence_cards/` committed separately (`git log` for the SHA).

**SCOPE OF THE TAG (do not overstate):** the tag is byte-identical to the pre-CX tree **only on the STEP 1
regression configuration** — 800 steps, `RandomState(42)` actions, two topologies (`scalability_30_40/1`,
`scalability_80_100/1`) — **not** the full attenuation sweep. The **reported figures were produced by the
pre-CX tree, which is the tag minus the 8 flagged edits** (all default-off); byte-identity on that config is
the evidence that the flags are inert, not a claim that the tag reproduces the full-sweep figures bit-for-bit.

### STANDING CITATION CONVENTION [record, applies project-wide]
Committing does not stop **line drift**: appends are safe, but any correction inserted *above* a cited line
silently moves it, and a `file:line` citation still resolves — to the wrong line — with no error or warning.
Therefore **every figure/claim cited anywhere (thesis, cards, log) carries the file, the line, AND the quoted
line itself**, so the citation stays greppable after drift. This convention is retroactive-safe: re-locate any
stale citation by grepping its quoted text.

**GATE: STEP 1 reported — regression PASS (0 differing cells, both bands). STOPPING.** STEP 2 remains queued
behind Task L STEP 3 (done), the OI-1 probe re-run, and the RQ2(c) counterfactual.
