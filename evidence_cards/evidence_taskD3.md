# Task D3 — vulnerability substitution as a property condition

Implements the substitution D2 showed to be expressible, as an ADDITIONAL property subtype beside the
existing removal-only condition. Removal-only is untouched; nothing already run is invalidated.

> **DONOR-POOL PROVISIONAL BANNER:** substitution draws a donor vulnerability from the current
> scenario's own nodes (0.1), NOT from the join donor pool, so it does **not** inherit the ~2.2x
> weaker-pool caveat. Stated explicitly because it is a different pool from Task C/join.

## STEP 0 — confirmation (D2 accepted, not re-derived). Reporting 0.1–0.4, then implementing.

### 0.1 Catalogue draw [ARTIFACT]

The scenario-wide catalogue is `self.vulnerabilities_embeddings`, built by
`create_vulnerabilities_embeddings` from **this env instance's own nodes only**
(`..._compressed.py:1125-1129`, `for node in self.environment.nodes`). Each env is a single scenario, so
the catalogue is **single-scenario by construction** — no cross-scenario global pool, no
different-size-scenario mixing. The draw is therefore from the **same distribution the node's existing
vulnerabilities came from** (the same scenario's nodes).

**Refinement (necessary, not a new pattern):** `vulnerabilities_embeddings` stores only
`{vuln_ID: embedding}` — not the full `VulnerabilityInfo` (results / outcome / port) needed to install a
*working* substitute. So the donor must be a full `VulnerabilityInfo` copied from another node in the
same scenario. This reuses the **widened-donor tiering of `_synthesize_recon_vulnerability`** (Task C,
`cyberbattle_env.py:838-878`): prefer discovered nodes, then any node in `self.environment.nodes()`,
`sorted()` for determinism. No new pool is invented.

### 0.2 Refresh path — reaches BOTH channels [ARTIFACT]

Node feature (`mean_vulnerabilities_embedding`): `_apply_legacy_dynamic_change` already calls
`self.update_node_dynamic(touched_node)` (`cyberbattle_env.py:491`) → compressed
`update_node_evolving_visible_graph` → `get_node_feature_vector` → `convert_node_info_to_observation`,
which pools `self.vulnerabilities_embeddings[vuln]` over `node_info.vulnerabilities` (`:497-505`). The
donor's `vuln_ID` is a scenario node's vuln, already in the catalogue, so `x` reflects the swap.

Action space: `refresh_vulnerabilities_embeddings_for_node(node)` (`:1155`) rebuilds the node's
`vulnerabilities_embeddings_per_node_type` entry from its **current** vulns (drops removed, adds new);
then the dynamic-change gate re-encode (`nodes_changed` truthy → `:641 create_continuous_action_space()`,
which reads `vulnerabilities_embeddings_per_node_type[target][type]` at `:1014`) rebuilds the actions.

**Reachable from the property hook without restructuring: YES** — mirror
`_synthesize_recon_vulnerability`'s guarded call
`if hasattr(self, "refresh_vulnerabilities_embeddings_for_node"): self.refresh_vulnerabilities_embeddings_for_node(node)`
inside the new `_substitute_random_vulnerability`; the `update_node_dynamic` call is already in place.
This is exactly the node-join path. **This refresh call is added ONLY on the substitution branch, so the
removal-only path (patch/service) stays byte-identical.**

### 0.3 Single event, single event_id [ARTIFACT]

`maybe_apply_dynamic_step` appends **one dict per change** to `_last_dynamic_events`; `_log_drift_rows`
loops `dynamic_events` and emits **one row (and at most one event_id) per dict** (`:826-860`). A
substitution appends exactly one `{"change_type":"property_substitution","node_ids":[node_id]}` → one
row, one logical event, `n_touched_nodes=1`. For a **discovered** node (visible at h2) the row is
`event_phase="immediate", event_id=None` (`:849-850`), so it never touches the pending-node counter —
the b008aef undercount/collision defect was specific to **pending** (not-yet-visible) nodes sharing the
counter, which a visible property event does not enter. Visibility statistics count by event dict, so a
substitution is counted **once**, not as removal+addition. Confirmed the path supports it.

### 0.4 Degenerate cases — intended behaviour stated BEFORE implementing [ARTIFACT]

- **Node has exactly one vulnerability:** remove it (node transiently at 0 vulns) → add donor → 1 vuln.
  Valid substitution, **proceed**. The transient empty set is safe: `convert_node_info_to_observation`
  guards `if len(node_info.vulnerabilities) > 0` (`:498`) and the swap+refresh are applied together
  before any re-encode.
- **Draw returns a vuln the node already has (incl. the just-removed id):** **NO-OP → not counted.** The
  donor search **excludes every vuln_ID currently on the node (and the removed one)**; if no distinct
  donor exists anywhere in the scenario, the function fires nothing and **returns `None`** (exactly like
  `_patch_random_vulnerability` returns None when a node has no vuln) → no event appended → the step logs
  the `event=None` "no_change" sanity row, **never a counted change event**. A no-op substitution is thus
  logged as a no-op, never silently counted.

**GATE cleared — implementing STEP 1.**

## STEP 1 — implementation + verification [done]

**Code (additive only; removal-only path byte-identical):**
- `cyberbattle_env.py` `_apply_legacy_dynamic_change`: new `if self.change_type == "substitute"` branch
  inserted **before** the unchanged `patch`/`service`/`mixed` lines, tagging one event
  `{"change_type":"property_substitution","node_ids":[n],"removed_vuln":...,"added_vuln":...}`.
- `cyberbattle_env.py` `_substitute_random_vulnerability`: pick a running discovered node with ≥1 vuln;
  find a donor vuln from the same-scenario catalogue (widened tiering: discovered nodes → any node,
  sorted) whose id is new to the node **and actionable** (has a canonical outcome, via the compressed
  per-node catalogue — inert in base env); remove one current vuln, add a deep-copied donor; refresh
  embeddings; invalidate the node's action cache; return `(node, removed, added)` or `None` (no-op).
- `cyberbattle_env_compressed.py` `_invalidate_action_cache_for_node`: clears ONLY the node's
  `processed_pairs` marks so the dynamic-gate rebuild re-processes its pairs and adds the CURRENT vulns
  (the added one included). It deliberately does **not** delete any action key — the removed vuln's
  now-stale key is left in place exactly as the removal-only condition leaves it (selecting it fails at
  exploit time, NoVulnerability). Called ONLY on the substitution branch. Without the `processed_pairs`
  clear the gate rebuild skips cached pairs and the added vuln never enters the action space (a gap D2
  did not surface).
- `cyberbattle_env_compressed.py` `find_closest_action_embedding`: a **degenerate-state guard** — if
  `action_embeddings` is empty, return a benign invalid action (`NoVulnerability` at exploit, no state
  change) so the episode runs to its normal cutoff instead of crashing cdist on a 1-D empty array.
  Inert for every condition that never empties the action space (static/membership/property never do, so
  **F1/F2/F3 are unaffected**); reached only via substitution swapping away a node's last productive
  vuln when a single node is reachable.

**Bug hunt (recorded, because it changed the fix and matters for reading the cost).** The first
action-space handling deleted the node's action keys and rebuilt; at 200-episode scale this could leave
the action space **empty** (when the removed vuln was the sole surviving action and the added vuln was
filtered out — e.g. a self-loop LateralMove/CredentialAccess), crashing `find_closest`. Root cause
confirmed by instrumented repro (owned=1, discovered=1, all of the one node's actionable vulns
self-loop-filtered). Corrected to (i) never delete keys (space can't empty) + (ii) the empty guard for
the genuinely-degenerate case. **Stranding is negligible: ~2% of episodes (4/200 at seed42) ever hit
the empty guard, and removing them barely moves the mean (0.588 non-stranded vs 0.587 overall).** So
the substitution cost below is a genuine policy-disruption effect, not a stranding artifact.

**Design note (donor = actionable subset):** the donor is restricted to vulns that carry a canonical
outcome (i.e. that actually enter the action space), so a substitution is a genuine
capability-for-capability exchange, not a capability→non-capability trade. Still same-scenario, same
distribution (its actionable subset). Documented because it narrows the draw from "any embedded vuln"
to "any actionable vuln".

**STEP 1.5 verification (F1 static seed42 / topo44, change_type="substitute", ci=20):**
- **(a) byte-identical regression:** `git diff cyberbattle_env.py` is purely additive — the shared
  removal path (`patch`/`service`/`mixed` dispatch) has zero changed bytes; the new branch only
  executes for `change_type=="substitute"`. Functional confirmation: a `change_type="patch"` run still
  fires 30 removal ("property") events with unchanged drift magnitude (mean 0.00059). **PASS.**
- **(b) null control:** 955 no-change rows, `max|change_drift_full| = 0.00e+00` (exactly zero). **PASS.**
- **(c) positive control:** 45 `property_substitution` rows, **non-zero fraction = 1.000** (mean
  change_drift_full 0.00144, max 0.00605), **no NaN**, **all `n_touched_nodes==1`** (single-event, 0.3).
  **PASS.**
- **(d) action-space:** across 45 events — added vuln in the node's action catalogue **all True**
  (usable capability present); removed vuln gone from `node.vulnerabilities` **all True** and gone from
  the node catalogue **all True** (authoritative removal); **no NaN**; **0 failing events**. The removed
  vuln's *stale action key* may persist (removed_in_actions ~0.36) — this is **expected and matches the
  removal condition** (the key is caught at exploit time); keeping it is what prevents the empty-action
  crash. **PASS.**

All four controls pass.

## STEP 2.1 + STEP 3 — band 30-40, property substitution [COMPLETE]

Config: 5 F1 static agents (250k, topo44), stochastic eval, 200 episodes/seed, `change_type=substitute`,
ci=20. **Achieved event rate: 15.00 substitution events/episode** (removal 14.85/ep) — matched in KIND
and frequency; zero episodes dropped, all 5 seeds = 200 episodes.

### Cost & robustness (both metrics named, bootstrap 0.95 over episodes, between-seed sd) [FINDING]

| metric | condition | robustness (cond/static) | cost | seed-spread sd |
|---|---|---|---|---|
| **COUNT (root_owned)** [primary] | removal | 0.976 [0.963, 0.989] | 0.556 owned [0.26, 0.86] | 0.023 |
| | **substitution** | **0.869 [0.850, 0.887]** | **3.043 owned [2.62, 3.49]** | 0.065 |
| **RATIO (root/reach)** [secondary] | removal | 0.976 [0.963, 0.990] | 0.017 [0.008, 0.027] | 0.023 |
| | **substitution** | **0.869 [0.851, 0.887]** | **0.095 [0.082, 0.109]** | 0.065 |

**Difference (substitution cost − removal cost): +2.49 root-owned [95% CI 1.96, 3.02] (COUNT); +0.078
[0.062, 0.095] (RATIO). Both CIs EXCLUDE 0.** All 5 seeds individually show substitution < removal.
**Substitution costs ≈5.5× more than removal** (robustness drop 13.1% vs 2.4%).

### Per-slice response rate at tau=0 (max/min = attenuation-bearing) [FINDING]

| condition | full | mean | **max** | **min** |
|---|---|---|---|---|
| property removal | 1.000 | 1.000 | **0.718** (sd .020) | **0.713** (sd .021) |
| **property substitution** | 1.000 | 1.000 | **0.741** (sd .015) | **0.736** (sd .017) |

Substitution's response rate is **slightly HIGHER** than removal's (0.74 vs 0.72), not lower — the agent
perceives substitution events at least as well as removals. So the larger cost is **not** a
perception/attenuation deficit.

### Absolute & relative change-drift (within change type, 30-40) [ARTIFACT]

- removal: relative 0.00114, absolute (|Δh_v|) 0.0045
- **substitution: relative 0.00360, absolute 0.0154** — ≈**3× larger** drift than removal (swapping an
  embedding perturbs the node feature more than deleting one).

### Reading the result [FINDING]

**Substitution and removal DIFFER, and substantially — this is NOT a "no difference" result.** The agent
responds not merely to *losing* an exploitable capability but to *which* capability replaced it:
substitution both (i) perturbs the observation ~3× more and (ii) presents an unfamiliar capability, and
costs the agent ≈5.5× more than a pure removal, at a slightly HIGHER perception (response) rate. The
extra cost is genuine, not a perception gap and not the stranding artifact (~2% of episodes). Zero-score
fraction (pooled): static 0.005 / removal 0.015 / substitution 0.076.

### STEP 2.2 (band 80-100) — HELD

Per the task's sequencing note, the 80-100 substitution cell is NOT run yet: Task F4 may replace the
80-100 checkpoints (convergence re-training). Will run once the F4 80-100 checkpoint decision is
settled, against whichever checkpoints are adopted (250k or 500k), with the removal baseline on the same
checkpoints.
