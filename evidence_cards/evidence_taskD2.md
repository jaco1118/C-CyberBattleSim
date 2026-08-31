# Task D2 — is DynPen-style property SUBSTITUTION implementable here?

Verification only. No implementation, no runs, no files modified. Source-quoted. Numbers/refs only.

## Verdict

**Partially.** The literal DynPen `<os, port, service>` triple substitution is NOT fully
representable (no OS field; no new-service-creation operation). But substitution at the level the
agent actually acts on — the node's exploitable **vulnerability set** — IS expressible with existing
machinery (the node-join / recon-synthesis add-vulnerability path + the Task-D action-space rebuild).
So removal-only was a design choice for the agent-relevant substrate, and a genuine platform
limitation only for the OS and new-service components.

## 1. Is there a catalogue to draw from? [ARTIFACT]

| component | scenario-wide source | verdict |
|---|---|---|
| ports | `ports_list = list(set(overall_ports))` (`generate_network.py:239`), attached to the Model (`model.py:401` `self.ports_list`) | YES |
| services | per-node `NodeInfo.services: List[ListeningService]` (`model.py:308`); enumerable scenario-wide by iterating nodes; `ports_list` is derived from them | YES (per-node source, aggregatable) |
| vulnerabilities | `vulnerabilities_embeddings` dict, built from ALL nodes at construction (`cyberbattle_env_compressed.py:1125-1129`), each with its 768-d embedding | YES (scenario-wide, with embeddings) |
| **os** | **none** — `NodeInfo` has no os/operating_system field; only `tag` ("tag representing the category", `model.py:305`), set to `category` at generation (`generate_network.py:228`) | **NO** |

A pool of ports + services + vulnerabilities is available scenario-wide. The `<os, ...>` component
of the triple has no representation.

## 2. Can a capability be ADDED, not only removed? [ARTIFACT]

| operation | status | source |
|---|---|---|
| enable a service that was off | EXISTS (sets `running=True`) — but enables a PRE-EXISTING service, does not create one | `static_defender_actions.py:158-164` (`start_service`) |
| add a NEW service/port to an existing node | DOES NOT EXIST — no code appends a `ListeningService` to an existing node mid-episode (join spawns whole nodes); a new service would also need a precomputed `feature_vector` | (absence) |
| add a vulnerability to a node's list | EXISTS but unused for property change — `_synthesize_recon_vulnerability` adds a vuln, and `refresh_vulnerabilities_embeddings_for_node` (`cyberbattle_env_compressed.py:1155`) tops up the catalogue; currently used only by the join/recon path | `cyberbattle_env.py` `_synthesize_recon_vulnerability`; `..._compressed.py:1155` |
| change a node's OS field | DOES NOT EXIST — no os field to change | (absence; `model.py:295-335`) |

The existing property operations are removal-only: `_patch_random_vulnerability` (deletes a vuln)
and `_disable_random_service` (`service.running=False`).

## 3. Would an added capability be reachable by the agent? [ARTIFACT — the crux]

**Feature vector (encoder): YES.** An added vulnerability enters `mean_vulnerabilities_embedding`
(`..._compressed.py:497-503`); an enabled/added service enters the `listening_services_*` arrays —
both provided the node's cached feature vector `x` is refreshed via `update_node_evolving_visible_graph`,
which the Task-D property hook (`update_node_dynamic`) already calls.

**Action space: YES — this is the point that most plausibly fails but does not.** An added
vulnerability enters the continuous action space via `refresh_vulnerabilities_embeddings_for_node`
(`:1155`, rebuilds the node's `vulnerabilities_embeddings_per_node_type` entry from its CURRENT
vuln list — dropping removed, adding new) followed by `create_continuous_action_space`, which is
rebuilt on the dynamic-change gate (`nodes_changed = maybe_apply_dynamic_step(); if nodes_changed:
... create_continuous_action_space()`, `:622-625`, the Task-D wiring). **This is exactly what the
node-join mechanism already does** for a joined node's vulnerabilities. Constraint: the vuln's
outcome must be one of the 10 canonical outcomes (`canonical_labels`, `:1102-1104`; otherwise
skipped — `if outcome_embedding is None: continue`, `:1141`). A substitute drawn from the scenario
catalogue already has a valid outcome, so this is not a blocker.

**Caveat (a one-line gap, not a blocker):** the current property hook `update_node_dynamic`
refreshes `x` but does NOT call `refresh_vulnerabilities_embeddings_for_node`; a substitution
implementation would add that one call so the added vuln reaches the action-space catalogue (join
already does this via `add_node_dynamic`).

## 4. What would it cost? (2 and 3 are positive for the vulnerability-level substitution) [ESTIMATE]

- **Files:** `cyberbattle_env.py` — add a `_substitute_random_property` path + a `change_type="substitute"`
  branch in `_apply_legacy_dynamic_change`; and one `refresh_vulnerabilities_embeddings_for_node`
  call from the compressed property hook. ~1–2 files.
- **Embedding source:** draw the substitute vuln from the scenario catalogue (`vulnerabilities_embeddings`
  / another node's vuln), reusing the Task-C widened-donor pattern — no new NLP/embedding computation.
- **Re-verify:** byte-identical regression against a fresh baseline under the exact config (as in
  Task D, since the change-application path changes); a smoke test that the substituted vuln reaches
  the observation (non-zero `change_drift_full`) AND the action space (appears in `action_embeddings`,
  is not a NaN/degenerate `cdist` pick — the Task-C failure mode); confirm removal-only property is
  unchanged when the flag is off.
- **Estimate:** ~half a day to a day.
- **Nothing already run is invalidated.** Substitution would be an ADDITIONAL `change_type`;
  removal-only property (both bands) stays and is NOT superseded.

## 5. The honest negative — what CANNOT be expressed, and the exact blocker [FINDING]

Full `<os, port, service>` triple substitution is not representable, blocked at **point 1 and 2**,
specifically:
- **OS:** `NodeInfo` (`cyberbattle/simulation/model.py:295-335`) has **no operating-system field** —
  only `tag` ("tag representing the category", `model.py:305`), which is read nowhere in the
  simulation (no `.tag` logic in `attacker_actions.py` / `cyberbattle_env.py`) and does not appear
  in the feature vector (`convert_node_info_to_observation` references no os/tag). So "changing the
  OS" has neither a field to change nor an effect on the agent.
- **New service/port:** there is no operation to introduce a new `ListeningService`/port to an
  existing node — only enable/disable of pre-existing services (`static_defender_actions.py:146-164`).

**But none of 1/2/3 blocks the agent-relevant substitution (the node's exploitable vulnerability
set), which is what the continuous action space and the reward are built on.** That is expressible
with existing machinery: a vulnerability can be added mid-episode and made to reach both the
observation and the action space via the same `refresh_vulnerabilities_embeddings_for_node`
(`..._compressed.py:1155`) + action-space-rebuild (`:625`) path the node-join mechanism uses.

**Defensible thesis statement:** the literal DynPen triple substitution is a genuine platform
limitation for the OS component (no field) and the new-service component (no add operation); but
substitution of the agent's exploitable capability (vulnerabilities) was a design choice, not an
impossibility — the add-and-reach machinery already exists and is used by node-join.
