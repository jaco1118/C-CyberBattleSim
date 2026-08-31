# Task R2 STEP 0 — is reachability change a third mechanism, or a relabelling of property change?

Read-only source audit. Nothing run, trained, generated, or modified except this card. Ran alongside the
Task Z training (not touched). Every claim carries file:line + a quoted line.

## SHORT ANSWER

**There IS a field that alters which nodes the agent can reach from which, without altering the target's
services or vulnerabilities: the per-node FIREWALL configuration** (`FirewallConfiguration.{incoming,
outgoing}`, a list of per-port ALLOW/BLOCK rules). So reachability is **not** determined entirely by the
target's own services/vulnerabilities — a reachability change is **not merely a relabelling of property
change in the behavioural (DO) sense.** BUT it enters the encoder through the **same channel** as property
change (the endpoint node's feature vector), and — decisively for the build decision — **changing it is NOT
reachable through the existing property-change machinery; it requires new environment code.** Per the fixed
decision rule that is the **"field exists but requires new environment code → not built"** row. I report the
evidence and stop; I do not take the decision. Separately, **all four stated reasons HOLD** (R2.5).

---

## R2.1 — HOW IS REACHABILITY DETERMINED?

The code path deciding whether the agent can act on a target from a source it holds is
`exploit_remote_vulnerability` (`cyberbattle/simulation/attacker_actions.py:92-197`). It reads, in order:

| # | what it reads | line | quoted | mutable at runtime? |
|---|---|---|---|---|
| 1 | source **owned** (`agent_installed`) | `:109` | `if not source_node_info.agent_installed:` | yes (ownership flag) |
| 2 | target **discovered** | `:115` | `if target_node_id not in self._discovered_nodes:` | yes (grows via Reconnaissance) |
| 3 | source & target **running** | `:121,:127` | `if source_node_info.status != model.MachineStatus.Running:` | yes |
| 4 | target **vulnerabilities** | `:133` | `if vulnerability_id in target_node_info.vulnerabilities:` | **yes** (property change patches/substitutes) |
| 5 | target **privilege_level** vs required | `:143` | `if not target_node_info.privilege_level >= vulnerability.privileges_required:` | yes |
| 6 | target **listening on the port** (services) | `:161` | `target_node_is_listening = vulnerability.port in [i.name for i in target_node_info.services if i.running]` | **yes** (service change) |
| 7 | **source firewall.outgoing** allows port | `:170-172` | `if not source_node_info.defense_evasion and not self.__is_passing_firewall_rules(source_node_info.firewall.outgoing, vulnerability.port)` | **yes** (firewall) |
| 8 | **target firewall.incoming** allows port | `:180-182` | `if not target_node_info.defense_evasion and not self.__is_passing_firewall_rules(target_node_info.firewall.incoming, vulnerability.port)` | **yes** (firewall) |
| 9 | success rate (random) | `:190` | `if random.random() >= vulnerability.rates.successRate:` | — (stochastic) |

Mapping to the task's checklist:
- **target services / open ports** — READ (`:161`). Lives in `NodeInfo.services` (`model.py:308`); mutable
  (`_disable_random_service`, `cyberbattle_env.py:913-931`, sets `service.running=False`).
- **target vulnerabilities** — READ (`:133`). `NodeInfo.vulnerabilities` (`model.py:314`); mutable
  (`_patch_random_vulnerability`, `cyberbattle_env.py:894-911`).
- **credentials the agent holds** — **NOT READ.** There is no credential store; the only "access"
  precondition is `agent_installed` on the source (`:109`). `CredentialAccess` (`model.py:79-80`
  `class CredentialAccess(VulnerabilityOutcome): """Access the target node using some credentials found"""`)
  is an **outcome label** that owns the target on the same path as `LateralMove`
  (`cyberbattle_env.py:1034` `elif isinstance(outcome, model.CredentialAccess) or isinstance(outcome, model.LateralMove):`).
- **firewall configuration, incoming/outgoing** — READ (`:170-188`). See R2.2. Mutable, and — the key
  point — **not** touched by the property-change machinery (R2.4).
- **explicit edge / adjacency structure outside the nodes** — **NOT a runtime reachability gate.** The
  exploit path above consults no adjacency structure; gating is discovered + firewall + services on the
  target. The encoder's `edge_index` is the agent's **traversal history**, not a reachability gate
  (`cyberbattle_env_compressed.py:284-324`; and the ground-truth graph is edge-cleared at construction,
  `generate_network.py:309`; `reachable_count` derives from the **frozen** `access_shortest_paths`,
  `cyberbattle_env.py:1088-1100`, which is a scoring quantity, not an exploit precondition). This confirms
  Task E 0.1/0.4 against source.
- **anything else** — running status (`:121,:127`), `privilege_level` (`:143`), success-rate roll (`:190`).

**So reachability from source→target = source owned + target discovered + both running + target has the
vuln + privilege + target listening on the port + firewall (source-out AND target-in) allows the port +
success roll.** Items 7–8 (firewall) are a source→target gate that is **separate from** the target's
services/vulnerabilities.

## R2.2 — DOES A FIREWALL CONFIGURATION EXIST?

**R2.2.1 — exists. YES.**
- `model.py:258-266` `@dataclass class FirewallRule: ... port: PortName / permission: RulePermission / reason: str = ""`
- `model.py:277-284` `@dataclass class FirewallConfiguration: ... outgoing: List[FirewallRule] ...; incoming: List[FirewallRule] ...`
- `model.py:316` `firewall: FirewallConfiguration = field(default_factory=FirewallConfiguration)` (a field on `NodeInfo`).
- `model.py:253-256` `class RulePermission(Enum): ALLOW = 0 / BLOCK = 1`.

**R2.2.2 — populated non-default, varies between nodes. YES.** `generate_network.py:222-256`: a
`FirewallConfiguration` is built with rules (`:222`), deep-copied onto each node (`:232`
`firewall=copy.deepcopy(firewall_conf)`), then rules are set **a-posteriori and per-node-randomly** to
BLOCK/ALLOW: `:246` `if random.random() < firewall_rule_incoming_probability:` → `:247`
`...firewall.incoming[index].permission = RulePermission.BLOCK` (and the outgoing analogue `:253-256`).
Because the draw is per node per rule, permissions **vary between nodes** (Task E measured 45 incoming +
45 outgoing rules across the 34-node 30-40 topo). `firewall_rule_incoming/outgoing_probability` default
`0.2` (`model.py:358-359`).

**R2.2.3 — read at runtime. YES.** `attacker_actions.py:560-569`
`def __is_passing_firewall_rules(self, rules, port_name): for rule in rules: if rule.port == port_name: if
rule.permission == model.RulePermission.ALLOW: return True else: return False; return True` — called from
the exploit path at `:170-172` (source outgoing) and `:180-182` (target incoming); a BLOCK returns
`ActionResult(... outcome=model.FirewallBlock(...))` (`:178,:188`) instead of owning the node. It is a
consulted mechanism, not a dead field.

**R2.2.4 — changing it alters reachability WITHOUT altering services or vulnerabilities. YES.** The
permission field (`FirewallRule.permission`, `model.py:264`) is orthogonal to `NodeInfo.services`
(`model.py:308`) and `NodeInfo.vulnerabilities` (`model.py:314`). Flipping an incoming rule ALLOW→BLOCK on
a port the target already listens on makes the next remote exploit return `FirewallBlock` (`:180-188`)
while the target's services list and vulnerabilities dict are untouched. **This is the whole question, and
the answer is yes.** (Coupling caveat: a firewall rule only *matters* for a port that carries a
service, and is only *observable* — R2.3 — when that port maps to a represented service index; but the
permission value itself is a distinct axis from service existence.)

## R2.3 — DOES THE ENCODER SEE IT?

**R2.3.1 — the node features the frozen encoder consumes.** Built by
`convert_node_info_to_observation` (`cyberbattle_env_compressed.py:461-522`), flattened by
`get_node_feature_vector` (`:264-269` `node_features_dict = self.convert_node_info_to_observation(...)`;
`flatten_dict_with_arrays(...)`), stored as node attribute `x` on `evolving_visible_graph` (`:273`
`self.evolving_visible_graph.add_node(node_id, x=self.get_node_feature_vector(node_id))`; refreshed `:281`),
and fed to the encoder via `data = from_networkx(graph)` in `encode()` (`:382`). The feature keys
(`:507-522`): **`firewall_config_array`**, `listening_services_running_array`, `visible`, `persistence`,
`data_collected`, `data_exfiltrated`, `defense_evasion`, `reimageable`, `privilege_level`, `status`,
`value`, `sla_weight`, `listening_services_fv_array`, `mean_vulnerabilities_embedding`.

**R2.3.2 — is the firewall among them? YES**, as `firewall_config_array`
(`cyberbattle_env_compressed.py:462-476`, returned at `:508`): `:468-471`
`for config in node_info.firewall.incoming: permission = config.permission.value; if
self.get_service_index(config.port, node_info) != -1 and ... < self.max_services_per_node:
firewall_config_array[self.get_service_index(config.port, node_info)] = permission` (and the outgoing half
`:472-476`). **Conditions on observability** (`:466`): only `if node_info.visible`, and only for ports that
map to a service index `< max_services_per_node`. So a firewall change IS encoder-visible, provided the
node is visible and the port is service-mapped (the "double structural-zero hazard" of Task E 0.2).

**R2.3.3 — N/A.** The field IS among the encoder's features; it is not a negative result. (A firewall
change is therefore a legitimate *candidate* — it survives R2.1–R2.3.)

## R2.4 — IF A CANDIDATE SURVIVES, HOW WOULD IT BE CHANGED? (it does)

**R2.4.1 — could the EXISTING property-change mechanism change the firewall with config + no new code?
NO.** The property-change dispatch is `_apply_legacy_dynamic_change` (`cyberbattle_env.py:476-506`):
`:481` `if self.change_type == "substitute":`, `:493` `if self.change_type == "patch": touched_node =
self._patch_random_vulnerability()`, `:495` `elif self.change_type == "service": ... _disable_random_service()`,
`:497` `elif self.change_type == "mixed":`. **None of the four subtypes touch the firewall**, and
`cyberbattle_env.py` **never references `firewall` at all** (grep: no matches in that file). A firewall
change would attach as a **new branch** in this elif chain (`:495-497`) dispatching to a **new method**
`_change_random_firewall_rule` — structurally a parallel of `_disable_random_service`
(`:913-931`: pick a running discovered node → pick a rule → flip `permission` → return `node_id`). **That
is new environment code (~15-25 lines), not a configuration entry.**

**R2.4.2 — would drift instrumentation, the relevance flag, and event logging handle it unmodified? YES,
conditional on the new code doing two things** (both of which are part of the new code, not free):
(i) the new method must **append** `{"change_type": "reachability", "node_ids": [endpoint]}` to
`self._last_dynamic_events` (the same shape as `:504` `{"change_type": "property", "node_ids":
[touched_node]}`), which the generic drift logger and relevance path consume by node id, agnostic to the
type string; and (ii) it must call `update_node_dynamic(endpoint)` (`cyberbattle_env_compressed.py:352-353`
→ `update_node_evolving_visible_graph`, which recomputes `x` — including `firewall_config_array` — so the
re-encode in `step()` reflects the flip). Without (ii) the drift is a structural zero (the exact
property-bug lesson, Task E 0.2). The **relevance flag** `_is_event_relevant` is change-type-agnostic
(node-id based), so it needs no change. **Net: the instrumentation is type-agnostic and would not need
modification — but the two calls above are new code.**

**R2.4.3 — work estimate.** Configuration: ~0 (one `change_type` string). **New environment code**:
`_change_random_firewall_rule` + dispatch branch + event tag ≈ 15-25 lines, plus a positive-control test
(flip a visible, service-mapped port; assert non-zero `change_drift_full`) and a matched-rate check. The
new-code portion is the whole risk: an unaudited change-mutation method.

## DECISION-RULE EVIDENCE (I do not take the decision)

- Field exists: **YES** (firewall).
- Encoder sees it: **YES** (`firewall_config_array`).
- Changes through the **existing** property-change machinery with **no new environment code**: **NO** —
  requires a new method + dispatch branch (R2.4.1).

→ The applicable row is **"Field exists but requires new environment code → not built."** The reason the
rule gives — no time to audit new environment code before 15 August, and an unaudited change mechanism is
the most likely source of a false headline — applies directly here. **Reported; decision not taken.**

---

## R2.5 — AUDIT OF THE FOUR STATED REASONS (reported separately, per instruction)

The four source-verified reasons the draft gives for DynPen's connectivity category not being reproducible
are recorded canonically in `evidence_cards/evidence_taskE.md` (headline + naming rule `:228-232`:
"*no credential set `Cr`, no subnet/connection-set, no ownership loss on sever, no node discovery on add*").
Each re-checked against source now (not from recollection):

| # | reason (as stated) | verdict | settling source line |
|---|---|---|---|
| 1 | **No credential set `Cr`** — "credentials" is only an outcome label, no cached-credential store | **HOLDS** | `model.py:79-80` `class CredentialAccess(VulnerabilityOutcome): """Access the target node using some credentials found"""` (an outcome label); no credential store exists (grep `credential_store/cached_credential/...` in `simulation/`+`_env/` → **none**); `cyberbattle_env.py:1034` owns the target on the same path as `LateralMove` |
| 2 | **No subnet / connection-set** — reachability is strictly per-edge, not set-based | **HOLDS** | `NodeInfo` fields (`model.py:294-338`) contain **no** `subnet`/`zone`/connection-list field; reachability is per-(source,target,port) via firewall+services+discovered (R2.1) |
| 3 | **Severing does NOT revoke ownership** — no ownership-loss mechanical term | **HOLDS** | `agent_installed` is set **False at exactly one runtime site**: `static_defender_actions.py:49` `node_info.agent_installed = False` (defender `reimage_node`). All other assignments: `generate_network.py:231` (init), `static_defender_actions.py:66`/`attacker_actions.py:82` (set True). **No firewall path touches `agent_installed`** |
| 4 | **Adding cannot discover an undiscovered node** | **HOLDS** | `discovered_nodes.append` occurs only at `cyberbattle_env.py:213` (starter) and `:1030`, and `:1030` is inside `if isinstance(outcome, model.Reconnaissance):` (`:1021`). A firewall `allow` produces no `Reconnaissance` outcome, so it cannot add a node to `discovered_nodes` |

**All four HOLD. No draft paragraph is falsified by the source; the four-reason paragraph stands.**

**One correction to flag (not a reason failing — a scope-of-claim issue).** The R2 premise reports the
draft as saying connectivity change "**has no analogue here**" / "**cannot be reproduced**." That blanket
phrasing is **too strong** relative to what the source supports and to Task E's own card: the four reasons
justify only that **DynPen's connectivity *category* is not faithfully reproducible**, not that no
reachability mechanism exists. The core R2 question answers **YES** — the firewall is a field that alters
reachability separately from services/vulnerabilities. `evidence_taskE.md` already carries the correct,
non-overclaiming framing (`:220-232`: a firewall-based "**reachability**" change **is** implementable —
observable via `firewall_config_array`, behaviourally real as sever+restore — but was **not built**, and
must be named "reachability," **never** "connectivity"). **Action for the manuscript (report only, I did
not edit it):** wherever the prose states connectivity/reachability change "has no analogue" or "cannot be
reproduced" without the DynPen-category qualifier, narrow it to Task E's precise wording. This is a
wording/scope correction, not one of the four reasons failing — the four are correct as stated.

## GATE

R2.1–R2.4 (mechanism) reported: a reachability field (firewall) exists, gates reachability separately from
services/vulns, is encoder-visible, but requires **new environment code** to change → decision-rule row
**"not built"** (decision not taken). R2.5 (four-reason audit) reported separately: **all four HOLD**; the
only correction is a manuscript scope-of-claim narrowing ("no analogue"/"cannot be reproduced" → "DynPen's
connectivity *category* not faithfully reproducible"), which does not depend on any experiment. **Nothing
implemented, trained, run, or modified except this card; Task Z untouched. Report and stop.**
