# Task D — Property Change Observability: Fix + Verification

Numbers and provenance only. No thesis wording.

## Commit

Implementation: uncommitted at time of writing this card, to be committed as a single commit
immediately after. File/line references below are to the working tree at that commit.

Files changed:
- `cyberbattle/_env/cyberbattle_env.py`
- `cyberbattle/_env/cyberbattle_env_compressed.py`
- `cyberbattle/agents/config/train_config.yaml` (`patch_service_dynamic_enabled: False -> True`)

## STEP 0 answers (given before implementation, reproduced for the record)

**0.1 — Is there a wired service path distinct from vulnerability-removal, already wired?**
No. `change_type` ∈ {`"patch"`, `"service"`, `"mixed"`} (`cyberbattle_env.py:467-476`) all converge on
`_apply_legacy_dynamic_change`, whose only effect was appending to `_last_dynamic_events`
(`:478`, pre-fix) — a side channel never consumed by `maybe_apply_dynamic_step`'s return value
(pre-fix: `removed + joined` only). No firewall-specific mechanism exists in this file's history
(`git log --all -S"firewall" -- cyberbattle/_env/cyberbattle_env.py` returns empty). A firewall
mechanism exists only in `static_defender.py`'s `ExternalRandomEvents`, unrelated to
`patch_service_dynamic_enabled`. Neither `"patch"` nor `"service"` was wired — proceeded as a
plumbing fix.

**0.2 — Is surfacing the node id into the return value sufficient on its own?**
No. Two independent gaps: (a) the gate-firing gap (`maybe_apply_dynamic_step` not returning the
touched node), and (b) the stale-cache gap — `encode()` (`cyberbattle_env_compressed.py:367-380`)
reads whatever is already cached in `evolving_visible_graph.nodes[node]['x']` via
`from_networkx(graph)`; it never recomputes per-node features. The only refresh path is
`update_node_evolving_visible_graph` (`:280-281`), whose sole pre-fix call site
(`update_evolving_visible_graph_after_step:957`) is gated on the agent's own action outcome.
Both parts were required (Amendment 1).

**0.3 — Is a patched-to-zero node distinguishable from an unseen node?**
Initial STEP 0 answer claimed the `visible` field (`convert_node_info_to_observation`,
`cyberbattle_env_compressed.py:497`) reliably distinguishes them. **Correction, found during the
STEP 2 empirical check (below): this overstated the case.** `node_info.visible` is a distinct
flag from `discovered_nodes` membership — it defaults to `False` (`model.py:332`) and is only
set `True` by an explicit per-node `Discovery`-outcome action targeting that specific node
(`attacker_actions.py:252,472`), not by ordinary reconnaissance/discovery into
`discovered_nodes`. Empirically, even the starter node (owned from step 0) has `visible=False`
in a fresh reset (verified directly). So in practice, most discovered nodes — patched-to-zero or
not — carry `visible=0` and zero-forced `firewall_config_array`/`listening_services_*` arrays
regardless of vulnerability state; the `visible` flag does not reliably distinguish them.

This does **not** reopen the STOP clause, for a different reason than originally stated: an
undiscovered/"unseen" node is never added to `evolving_visible_graph` at all
(`add_node_evolving_visible_graph` is only called for nodes already in `discovered_nodes`), so it
never contributes a competing node embedding for the agent to confuse with a real one — there is
no side-by-side aliasing in the encoded graph. The only case where two already-discovered nodes
converge to an identical representation is "patched-to-zero" vs "naturally zero-vulnerability,
discovered" — which is the **correct** representation (both are, in truth, nodes with zero
exploitable vulnerabilities), not a defect. What actually matters for Task D — whether a single
node's own representation changes when its vulnerabilities are removed — is answered directly by
the pre/post-patch deltas below, which are real and non-zero.

**0.4 — Do membership leave/join already fire the gate correctly? Any other change_type sharing the gap?**
Yes, correctly wired (`_apply_dynamic_leave`/`_apply_dynamic_join`, `cyberbattle_env.py:529,612`,
append to `removed`/`joined`, which were already returned pre-fix). Structurally they don't need
a cache-refresh fix: join sets a fresh `x` at add-time via `add_node_dynamic` ->
`add_node_evolving_visible_graph`; leave deletes the node from the graph entirely via
`remove_node_dynamic` -> `remove_node_evolving_visible_graph` (`:331`), so `encode()` naturally
reflects its absence either way. No other `change_type` exists beyond `patch`/`service`/`mixed`;
all three share the identical pre-fix gap.

## Ordering (Amendment 3)

Reported before implementation, confirmed unchanged after:
- h1 — `cyberbattle_env_compressed.py:542`, before `step_attacker_env` (agent's action).
- h2 — `:595`/`:597`, after the agent's own action + its own re-encode.
- `maybe_apply_dynamic_step()` — `:622`, strictly after h2; calls `_apply_legacy_dynamic_change`
  first (property), then `_apply_dynamic_leave`/`_apply_dynamic_join` (membership) — all three
  in the same h2->h3 window.
- h3 — `:658`/`:660`, after `maybe_apply_dynamic_step()` returns and its re-encode gate runs.

The new `update_node_dynamic` hook is invoked from inside `_apply_legacy_dynamic_change`, called
at the top of `maybe_apply_dynamic_step` — the same h2->h3 window membership's graph mutations
already occupy. No new snapshot window was created.

**FINDING (empirical, post-implementation, smoke-test data, n=18 property rows):**
`norm_h2 != norm_h3` in 18/18 property rows (drift lands in the h2->h3 pair, as designed).
`norm_h1 != norm_h2` in 9/18 of those rows (a concurrent agent-driven re-encode happened the same
step) — orthogonal and non-contaminating, since `change_drift_full` is computed strictly from
`h2`/`h3` node embeddings via `_node_delta_vector` (`:735-747`), never from `h1`.
**Property drift lands in the same snapshot pair membership uses — comparable, as required.**

## Implementation (STEP 1, Amendments 1 & 2)

`cyberbattle_env.py`:
- Added `update_node_dynamic(self, node_id): pass` — base no-op hook, mirroring
  `add_node_dynamic`/`remove_node_dynamic`'s existing pattern.
- `_apply_legacy_dynamic_change` now calls `self.update_node_dynamic(touched_node)` when a node
  was touched, and returns `touched_node` (or `None`).
- `maybe_apply_dynamic_step` captures that return value into `touched` and returns
  `touched + removed + joined` (previously `removed + joined`).

`cyberbattle_env_compressed.py`:
- Added `update_node_dynamic(self, node_id): self.update_node_evolving_visible_graph(node_id)` —
  override placed alongside `remove_node_dynamic`/`add_node_dynamic`. No discovered/visible guard
  needed: the touched node is always already in `evolving_visible_graph` by this point in the
  step (backfilled by `update_evolving_visible_graph_after_step`'s discovered-node loop, which
  runs earlier in the same `step()` call, `:944-946`).

No new refresh mechanism created; no membership-handling code touched; `change_interval`/reward/
training config untouched apart from the single flag flip below.

## Byte-identical regression (MANDATORY)

Baseline: pre-Task-D commit `0b06f61` extracted fresh via `git archive` (same pattern as Task C's
corrected regression). Config: `patch_service_dynamic_enabled=False`, `drift_logging=False`,
`dynamic_mode="both"`, `change_interval=20`, `change_type="mixed"`, topology
`local_baseline_single_topology/1`, 2000 steps, seeds 12345 and 54321.

```
=== seed 12345 ===
PASS: all 2000 steps byte-identical (seed=12345, drift_logging=False, patch_service_dynamic_enabled=False)
  _apply_legacy_dynamic_change calls (this process): 0, update_node_dynamic calls: 0
=== seed 54321 ===
PASS: all 2000 steps byte-identical (seed=54321, drift_logging=False, patch_service_dynamic_enabled=False)
  _apply_legacy_dynamic_change calls (this process): 0, update_node_dynamic calls: 0
```

**Zero-execution counter: `_apply_legacy_dynamic_change` and `update_node_dynamic` both called 0
times in both 2000-step runs** — expected and definitional, since the call site itself is gated
on `patch_service_dynamic_enabled` (`maybe_apply_dynamic_step`, guard before the call). Default
path (`False`) is provably unchanged.

## STEP 2 — property drift is now real

Config: `patch_service_dynamic_enabled=True`, `dynamic_mode="both"`, `change_interval=20`,
topology `scalability_10_15/1`, 400 steps, seed 1 (identical to Task C's smoke test).

**ARTIFACT: no crashes, 400/400 steps completed.**

| change_type | rows | change_drift_full mean | std | min | max | exactly-zero count |
|---|---|---|---|---|---|---|
| property | 18 | 0.001237 | 0.001332 | 0.000163 | 0.005170 | 0/18 |
| membership_leave | 9 | 0.241846 | 0.143363 | 0.068867 | 0.446985 | 0/9 |
| membership_join | 0 | — | — | — | — | — |

Per-slice (`change_drift_mean/max/min`), property rows: mean=0.001265, max-slice mean=0.001113,
min-slice mean=0.001249 (all std/min/max on the same order as `change_drift_full`, no slice
degenerate). No nulls in `agent_drift_full`/`change_drift_full`/`norm_h1`/`norm_h2`/`norm_h3` for
any property or membership_leave row.

**FINDING: `change_drift_full` is no longer exactly 0.0 for any property event (0/18, was 18/18
pre-fix).** All 18 property rows have `event_phase="immediate"` (property changes are always
seen the same step they fire, unlike `membership_join` which can lag).

**FINDING: property drift magnitude (mean 0.00124) is ~195x smaller than membership_leave (mean
0.242) on this topology/step budget.** Expected direction: property perturbs one node's local
`mean_vulnerabilities_embedding` slice by removing one vulnerability from a per-node mean;
membership_leave removes an entire node's embedding from the pooled graph outright — a
structurally larger perturbation, especially under mean/min pooling over a small (10-node)
topology. Not claimed to be an artifact or a bug; recorded as-is per instruction not to weigh in
on magnitude comparisons beyond reporting them.

Zero skipped: 0 crashes, 0 dropped drift rows across 400 steps.

## STEP 2 — 0.3 empirical check

Topology `scalability_10_15/1`, target node `Node_9` (23 vulnerabilities at reset), direct calls
to `update_node_dynamic` after deleting one vulnerability at a time (exercising the exact
production hook, not a re-derived approximation):

| transition | max abs delta | L2 delta | unchanged? |
|---|---|---|---|
| 2 vulns retained (23->22) | 0.001340 | 0.009620 | No |
| 1 vuln retained (22->21) | 0.000953 | 0.007784 | No |

Both transitions produce a real, non-trivial, non-zero change to the node's own cached feature
vector — this is the core deliverable of Task D, confirmed directly on the raw vector, not just
via the aggregate drift number.

`visible` flag check, run on two separately-forced patched-to-zero nodes to capture both real
states of the flag:

- **Starter node** (`scalability_10_15/1`, `visible=True` in this run — visibility was actually
  acquired on it): forced to 0 vulnerabilities, then compared its real dict against the same
  dict with `visible` toggled to `False` (never written back to the live node). Result: `visible`
  field differs (1 vs 0), `listening_services_running_array` differs (`[1,0,...,0]` vs all-zero,
  since that array is gated on `visible`), `mean_vulnerabilities_embedding` identical in both
  (all-zero either way, since that field is gated on vulnerability count, not `visible`).
  **When `visible=True` genuinely holds, a patched-to-zero node is clearly distinguishable** —
  both the flag itself and the visible-gated arrays differ.
- **`Node_9`** (same topology, `visible=False` in this run — visibility was never acquired on it):
  the same toggle comparison produced `visible` field 0 in both the real and force-`False` cases
  (no difference), and `mean_vulnerabilities_embedding` identical in both (also all-zero for a
  different reason here — the field wasn't forced to zero-vuln in this particular probe, but the
  point stands: the `visible` field itself carried no signal since it was already `False`).

**Correction to the original 0.3 answer: `visible` distinguishes a patched-to-zero node from its
own "unseen" counterfactual only when visibility has actually been acquired on that node — which
is not the default state (see the 0.3 correction above; `visible` defaults to `False` and
requires an explicit per-node `Discovery`-outcome action).** For the majority of discovered nodes
in a typical episode (`visible=False`), the flag carries no distinguishing signal. This is not a
new stop condition: as reasoned above, actual unseen (undiscovered) nodes never enter
`evolving_visible_graph` at all, so they never compete for an embedding slot with a discovered
patched-to-zero node — the only real-world collision is between two already-discovered,
already-zero-vulnerability nodes (one naturally, one via patch), which is a correct, not a
defective, convergence.

## STEP 2 — SATURATION

Topology: `scalability_80_100/100` (grid_topology_id_map.json: band `80-100`, slot `4` -> original
topology id `"100"`). Verified: **88 nodes total, 53 with exactly 1 vulnerability** (matches the
number cited in the task instructions exactly) — full distribution:
`{1: 53, 3: 2, 5: 9, 9: 2, 10: 2, 11: 1, 14: 1, 15: 2, 23: 2, 34: 2, 39: 2, 48: 1, 61: 1, 64: 5, 69: 1, 71: 2}`.

Config: `dynamic_mode="none"` (isolates the property mechanism from membership leave/join, per
instruction), `patch_service_dynamic_enabled=True`, `change_type="patch"`, `change_interval=20`,
seed 1.

**ARTIFACT — patchable pool (topology-wide, >=1 vulnerability, ground truth):**
- Episode start: 88/88 nodes patchable (all start with >=1 vulnerability).
- After 15 patch events (300 steps): 81/88 patchable — **7 nodes exhausted to exactly zero
  vulnerabilities** (7/15 = 47% of these first 15 events drove their target to zero, consistent
  with 53/88 ≈ 60% of nodes starting at exactly 1 vulnerability — high probability any random
  pick lands on a 1-vuln node early).
- 14 distinct nodes touched across 15 events (1 node patched twice: `Node_62`, 4 vulns -> 3).
- Discovered-node count reached 88/88 (full topology) well within the 300-step window, so the
  "among discovered" patchable count tracks the topology-wide count exactly once recon
  saturates — recon does not meaningfully lag the patch mechanism on this topology/step budget.

**FINDING: no monotonic within-episode decline in per-event drift magnitude detected.**
`change_drift_full` for the 15 events, in order: `0.000224, 0.007637, 0.009101, 0.000008,
0.000076, 0.019473, 0.000002, 0.006349, 0.000048, 0.000049, 0.000572, 0.000595, 0.000426,
0.000601, 0.000344`. Spearman correlation (event order vs. `change_drift_full`): **-0.039**
(effectively zero). Drift magnitude is driven by *which* node/vulnerability-count is hit (a
1-vuln node exhausted to zero produces a proportionally larger perturbation to its own mean
embedding than removing one of many vulnerabilities from a high-count node), not by episode
position — recorded as-is, no claim of a stationarity guarantee beyond this window. Per Task D's
scope, this is reported for the record ahead of Task F, not acted on here (no change to
`change_interval` or reward/training config).

## LOCAL env — record only, not fixed

**One line, for the record: LOCAL shares the identical pre-Task-D gap, unfixed.** It caches `x`
in `evolving_visible_graph` (`cyberbattle_env_local.py:264-269`) and only refreshes it via
`update_node_evolving_visible_graph`, gated on the agent's own action outcome
(`cyberbattle_env_local.py:430`); it does not override the new `update_node_dynamic` hook (only
`CyberBattleCompressedEnv` does, per this task's explicit scope), so a patched node's feature
vector in Local's own observation (`step():` reads `evolving_visible_graph.nodes[...]['x']`
directly, `cyberbattle_env_local.py:~403-406`) remains stale exactly as Compressed's did
pre-fix. Not fixed in this task; report only.

## Config change

`cyberbattle/agents/config/train_config.yaml:43`:
`patch_service_dynamic_enabled: False -> True`. `train_config_static.yaml` (the `dynamic_mode:
none` static-baseline sibling config) was deliberately left untouched — its purpose is a
no-dynamics baseline, and flipping this flag there was not requested and would contradict that
config's role.
