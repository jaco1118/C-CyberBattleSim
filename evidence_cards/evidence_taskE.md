# Task E — connectivity change (revision 2): FINDING — category not fully reproducible

**Status: CLOSED at STEP 0. No implementation was done and none will be.** This card is a
source-verified account of why DynPen's connectivity change category is **not fully reproducible on
this platform**, and it goes into the thesis as such — a limitation of the platform relative to a
published change taxonomy, not a blocked or abandoned engineering task. Every line citation is
retained. Nothing below was prototyped, trained, or run.

**Decision (recorded):** a firewall-based condition was rejected because (i) DynPen's connectivity
category has **neither** of its two components here (0.3), and (ii) the only behaviourally-real
firewall primitive enters the observation through the **endpoint node's feature vector** — the *same
channel* as node-property change (0.1) — so at the level the pooling operation acts on it would not be
distinguishable from the property condition already reported, while claiming coverage of a category it
does not reproduce.

> **DONOR-POOL PROVISIONAL BANNER:** any join/discovery-adjacent number here inherits the ~2.2x
> weaker-pool caveat (Task G pending). No join numbers are produced in STEP 0.

## Headline (one paragraph, so the gate decision is on the table before the detail)

A connectivity change is **implementable in a narrowed, approximated form that is both observable and
behaviourally real, via FIREWALL rule flipping** (`static_defender_actions.override_firewall_rule`),
which is enforced at exploit time (`attacker_actions.exploit_remote_vulnerability:170-188`) and enters
the encoder through the endpoint node's `firewall_config_array` feature (`..._compressed.py:462-476`).
It is **NOT an edge change in the encoder's sense** (see 0.1) and it reproduces **neither of DynPen's
two defining components faithfully**: (i) DynPen's credential set `Cr` has **no analogue** (there is no
cached-credential store — "credentials" here is only an outcome label), and (ii) DynPen's subnet /
connection-set resampling `con'=sample(Num,r)` has **no subnet abstraction and no per-node
connection-list**; per-node connectivity is gated by ~1.3 firewall ports, not by membership of a set of
n nodes. Two behavioural consequences DynPen relies on also do not occur: severing connectivity **does
NOT revoke ownership** (0.4c → the mechanical term of 3.6 is absent), and adding connectivity **cannot
reveal an undiscovered node** (that is the join/Reconnaissance path, a different change type). The task
is therefore **runnable but only as a disclosed approximation** (firewall sever + restore, magnitude
scaled network-wide), not as DynPen's connectivity event. This is a gate decision, not a defect to
work around.

---

## 0.1 Does the encoder see the adjacency at all? [ARTIFACT]

**What the encoder receives.** `encode()` builds `data = from_networkx(self.evolving_visible_graph)`
and calls `self.graph_encoder(data.x, data.edge_index, data.vulnerabilities_embeddings)`
(`..._compressed.py:382-388`). So it receives **node features `x`, an `edge_index` (adjacency), and
edge features (`vulnerabilities_embeddings` = `edge_attr`)** — not node-features-only, not a
placeholder.

**Are the edges the simulated topology's edges?** They are the edges of `evolving_visible_graph`, a
`nx.DiGraph` whose edges are added **only by `add_edge_evolving_visible_graph(source,target,vuln_key)`**
(`:284-324`), called from `update_evolving_visible_graph_after_step` on a **successful agent
traversal** (`:969`). So `edge_index` is the agent's **exploit/traversal history graph** (directed,
"the agent moved source→target via this vuln"), **not** the ground-truth reachability graph and **not**
a placeholder star/chain/complete graph. `nodes_graph.clear_edges()` is called at construction
(`generate_network.py:309`), so the ground-truth `network` itself carries no edges at runtime; ground
truth reachability lives in the frozen `knows_graph`/`access_graph`/`dos_graph` (0.4).

**Do edge features exist and reach the first layer? VERIFIED (prior belief confirmed).** Encoder spec
(`gae/logs/default/SecureBERT/model_spec.yaml` + `train_config_encoder.yaml`) is 2 layers:
`[0] NNConv (out=64, NN_channels=16), [1] GCNConv (out=64)`. **NNConv is the first layer and consumes
`edge_attr`** (`gae/model.py:48-57,75-77`); `edge_feature_vector_size=768`. So edge features do enter,
at layer 0. Message passing is **2 hops deep** (two conv layers) — shallow, directly relevant to 1.6
and to the "how far does a connectivity change propagate" reading.

**THE LOAD-BEARING NUANCE (does not trip the STOP, but reframes the whole task).** The implementable
connectivity primitive (firewall) **does not manifest as an edge change** in this encoder:
- `edge_index` (traversal edges): unaffected by a firewall flip — the agent's past traversals are
  unchanged, and a not-yet-traversed potential edge was never in the graph.
- `edge_attr` at NNConv: these are **exploited-vulnerability embeddings**, not connectivity/permission
  metadata — also unaffected.
- **Node feature `x`:** `firewall_config_array` (`:462-476`) DOES change when a firewall rule flips —
  **but only `if node_info.visible`** and only if the port maps to a service index
  `< max_services_per_node`.

So the encoder **is not** node-features-only (it has real edges + edge features), and the STOP
condition in 0.1 ("node features only, or a placeholder adjacency") is **not** met. But the connectivity
change we can actually make is observable **exclusively through the endpoint's node feature**, ≤2-hop
propagated, **not** through `edge_index`. The task's "edge change" framing does not map onto this
encoder's edges. Reported prominently rather than smoothed over.

**Not touching the encoder.** No re-wiring proposed; the encoder is frozen and shared by every run.

## 0.2 The re-encode trigger [ARTIFACT]

Two independent triggers in `step()`:
1. **Agent-action window (h1→h2):** `action_changed_graph = self.action_changes_evolving_visible_graph(outcome)`;
   if true (or a static defender acted) → re-encode + action-space rebuild (`:567-595`).
2. **Dynamic-mutation window (h2→h3):** `nodes_changed = self.maybe_apply_dynamic_step()` (`:630`); **if
   `nodes_changed` is truthy** → `self.node_embeddings, self.observation = self.encode(...)` + rebuild
   (`:633-641`). `maybe_apply_dynamic_step` returns `touched + removed + joined` (`cyberbattle_env.py:463`).

**Exact condition a connectivity change must report:** it must be a new branch in
`maybe_apply_dynamic_step` that **returns the affected endpoint node IDs** (so `nodes_changed` is
non-empty and the re-encode fires) **AND** appends a `{"change_type":"connectivity", "node_ids":[...]}`
entry to `self._last_dynamic_events` (so drift logging at `:665-676` picks it up).

**The property-bug lesson (0.2 "bit us before"), restated for connectivity.** Returning the node fires
the re-encode, but the re-encode recomputes `x` from `evolving_visible_graph.nodes[n]['x']`, which is a
**cached** feature vector. Property change logged **exactly-zero** drift until Task D added
`update_node_dynamic(touched_node)` (`cyberbattle_env.py:490-491`) to **refresh that cache**. A
connectivity change has the identical hazard **and one more**: even after calling
`update_node_dynamic(endpoint)` to refresh `x`, the drift is **still zero unless the flipped port maps
to a represented service index and the node is `visible`** (`:466-476`). So the 1.7(b) positive control
must flip a firewall rule on a **service-mapped, visible** node and confirm non-zero
`change_drift_full`, or it is a structural zero exactly like the property bug.

**Undiscovered endpoint:** if an endpoint is not in `evolving_visible_graph` (undiscovered), refreshing
its `x` is a no-op / KeyError risk; per 1.3 the change must log only discovered endpoints and flag the
undiscovered ones (the connectivity analogue of the join visibility artefact).

## 0.3 Which DynPen component is implementable here? [ARTIFACT]

| primitive | exists? | source | changes DO / SEES / both |
|---|---|---|---|
| cached credentials granting access to another node; scenario-wide pool `Cr` | **NO** | no credential store anywhere in `simulation/` or `_env/`; `CredentialAccess` (`model.py:79`) is just an **outcome label** that owns the target identically to `LateralMove` (`attacker_actions.py:330-345`) | — (cannot be built; there is nothing to substitute) |
| subnet / zone making a SET of nodes mutually reachable | **NO** | no `subnet`/`zone` field on `NodeInfo` (`model.py:294-338`); reachability is strictly per-edge (knows + firewall + remote vuln), never set-based | — |
| firewall rules, incoming & outgoing | **YES** | `FirewallConfiguration.{incoming,outgoing}` (`model.py:277-284`); flip via `override_firewall_rule/block_traffic/allow_traffic` (`static_defender_actions.py:97-144`); enforced at `exploit_remote_vulnerability:170-188` | **BOTH** — changes what the agent can DO (firewall gate) and what the encoder SEES (`firewall_config_array`, visible+service-mapped) |
| explicit edges in the node/graph data structure | ground-truth: **frozen** (`knows_/access_/dos_graph` built once, `generate_network.py:97-312`, consulted for `reachable_count` only); encoder: `evolving_visible_graph` edges = **agent traversal history** (`:284-324`) | — | editing encoder edges changes SEES but **not DO** → fails 0.4 (not behaviourally real) |

**How `static_defender` exercises this path & whether it reports affected nodes.** When a static
defender is attached, `step()` re-encodes **unconditionally** (`:568` `... or self.static_defender_agent`)
because "the defender may have acted with modifying actions" — but it does **NOT** populate
`self._last_dynamic_events`, so its firewall edits are **not** tagged as dynamic-change events and are
**not** drift-logged as connectivity. The mechanism (firewall flip) is exercised and observable; the
**event-reporting/affected-node plumbing is not** wired for it. A connectivity change type would reuse
the firewall mechanism but must add the `_last_dynamic_events` reporting itself (0.2, 1.4).

**Plainly:** Component 1 (credentials) **cannot be reproduced** (disclosed limitation). Component 2
(connection set) **can only be approximated** — by flipping firewall rules per `(node, port)`, which is
a port-level, not node-set-level, connectivity primitive; there is no subnet to resample. Neither
component is reproduced faithfully; both approximations must be disclosed as such, not passed off as the
DynPen operation.

## 0.4 Is a connectivity change behaviourally real, in both directions? [ARTIFACT — decisive]

**(a) Sever prevents an action — YES, concrete.** `block_traffic(target, port, incoming=True)` sets a
BLOCK rule; on the agent's next remote exploit of that target on that port,
`exploit_remote_vulnerability` returns `FirewallBlock` at `:180-188` (or `:170-178` for source
outgoing) instead of owning the node. Real, deterministic (firewall check precedes the success-rate
roll).

**(b) Add makes a node reachable/discoverable that was not — PARTIAL.** `allow_traffic` can **restore**
reachability to an **already-discovered** node whose port was blocked (re-opens an exploit path). It
**cannot make an undiscovered node discoverable**: discovery requires entry into `_discovered_nodes`,
which only grows via a `Reconnaissance` outcome (`attacker_actions.py:286-303`) or the join mechanism's
synthesized recon — **not** via any firewall change. So the "reachable" clause is satisfiable; the
DynPen "addition discovers new nodes" clause is **not reproducible via connectivity** (it is the
membership-join change type).

**(c) Severing connectivity to an OWNED node — DOES NOT revoke ownership.** Ownership is the
`agent_installed` flag (`owned_nodes`/`root_owned_nodes` = nodes with `agent_installed[==ROOT]`,
`cyberbattle_env.py:1046-1048`). `agent_installed` is set False in **exactly one place**: defender
`reimage_node` (`static_defender_actions.py:49`). No firewall/connectivity path touches it; there is no
runtime re-check that an owned node is still reachable. **So severing a connection keeps the node owned
(and, if unreachable, simply owned-and-unreachable).** `reachable_count` itself derives from the
**frozen** initial `access_shortest_paths` (`cyberbattle_env.py:1088-1100`), so it does not respond to a
runtime firewall flip either.

**Consequence for STEP 3.6 (per the spec's own 0.4c branch):** the mechanical (arithmetic-loss) term of
the decomposition **does not exist** for connectivity — there is no root-owned departure analogue.
Cost, if any, is **entirely forward-looking behavioural**: fewer *new* nodes get owned in the remaining
episode because a pivot/exploit path was cut. Per the spec, report **total cost only**, labelled, with
this reason — do **not** manufacture a mechanical term. (This is the anticipated branch, not a failure:
the task explicitly says "If 0.4(c) shows ownership is unaffected by connectivity, say so and report
total cost only.")

**Does 0.4 trip the STOP?** No. There **is** a behaviourally-real sever (a) and a partial add (b), so a
connectivity change is not "perceived but irrelevant." It is real but **narrower** than DynPen: sever +
restore, no ownership loss, no node discovery. That narrowing is the finding to carry, not a workaround
to hide.

## 0.5 Which checkpoints are the right ones? [ARTIFACT]

The frozen static agents (dynamic_mode=none, patch off — verified by config diff; the `F1_adapted`
folders are dynamic_mode=both and are **NOT** these) that F1-R (30-40) and F2 (80-100) evaluated:

| band | run folder (under `.../jobs/0dfa230d/tmp/taskF1/`) | seed → topology |
|---|---|---|
| 30-40 | `runs/trpo_250k_F1_static_seed{42,100,123,200,300}` | **all 5 seeds → `scalability_30_40/44`** (single shared topology, 34 nodes) |
| 80-100 | `f2_runs/trpo_250k_F2_static_band80-100_seed{42,100,123,200,300}` | 42→`/5` (85n), 100→`/100` (88n), 123→`/18` (95n), 200→`/2` (84n), 300→`/67` (80n) — **5 distinct topologies** |

**Structural asymmetry to preserve (identical to the F-series):** 30-40 is 5 seeds on ONE topology;
80-100 is 5 (seed,topology) pairs. Each 250k static checkpoint has `checkpoint_250000_steps.zip` +
`checkpoint_vecnormalize_250000_steps.pkl` present. Every STEP 2 cell must reuse exactly these agents on
exactly these topologies, or the cost figures are not comparable to published F1-R/F2. (Note: F4 may
promote these to 500k checkpoints; if the connectivity eval is to sit beside the *converged* F-series it
should read `CKPT_STEP=500000` once F4 settles — flag at STEP 1, do not mix budgets silently.)

## 0.6 Rate feasibility, both conditions [ARTIFACT — reasoned from source + topology data; a measured pass belongs in STEP 1.7 once the type exists]

A connectivity type slots into the same `maybe_apply_dynamic_step` cadence as the legacy property change
(`num_iterations % change_interval == 0`, `cyberbattle_env.py:449`), i.e. one deterministic event per
`change_interval` steps.

- **Matched event count.** Membership fires ≈11/ep (30-40) and ≈15.7/ep (80-100) over ~300-step
  episodes → to match, `change_interval ≈ 300/11 ≈ 27` (30-40) and `≈ 300/15.7 ≈ 19` (80-100). Directly
  settable; same machinery, so cross-type rate matching is by event count exactly as the spec requires.
- **Eligible pool & exhaustion.** Firewall-rule-bearing `(node,port)` pairs among discovered visible
  nodes: 30-40 topo44 = **45 incoming + 45 outgoing rules** across 34 nodes; 80-100 = **93–122 each
  direction** across 80–95 nodes. Crucially, **firewall flips are reversible** (unlike node departure,
  which permanently shrinks the pool): a "resample" re-draws which rules are BLOCK vs ALLOW, so the pool
  **does not exhaust mid-episode** at either band for the localised magnitude.
- **Localised magnitude:** sustainable at both bands (few flips/event, reversible).
- **DynPen-faithful magnitude (r∈[1,n], ~n/2):** **NOT faithfully reproducible per node** — a node has
  ~1.3 firewall-controlled ports, not n potential connections, so its "connection set" cannot be
  resampled to size r∈[1,n]. It **is** reproducible **network-wide**: flip r firewall rules across the
  whole network with r ∝ n (≈1.3n rules available each direction), which **does** make the perturbation
  scale with n — preserving STEP 2's magnitude-scaling logic (perturbation ∝ n) even though the per-node
  subnet-set does not exist. This reinterpretation must be disclosed: "magnitude scales with n" is at
  the network level (count of flipped rules), not DynPen's per-node connection-set level.

## Pre-existing scaffold (context, not a result)

`..._compressed.py` already carries `self._last_connectivity_event` and a `change_type="connectivity"`
tag (`:186, :303-323`), but the comment there states it fires as a **consequence of the agent's own
successful exploit** (h1→h2 window), and "There is currently **no independent, agent-action-decoupled
connectivity-change mechanism** in this codebase; this is the only existing 'connectivity' mutation
site." So the drift **schema** already admits a connectivity tag, but no independent connectivity
**change** exists — STEP 1 would add the independent (h2→h3) mechanism and must not collide with this
existing agent-coupled tag.

## Avenue not taken (recorded for completeness, deliberately NOT built)

A firewall-based **"reachability change"** IS implementable on this platform: traversal to an
**already-discovered** node blocked (`block_traffic` → `FirewallBlock` at
`attacker_actions.py:180-188`) and later restored (`allow_traffic`), reported as an endpoint
node-feature perturbation (`firewall_config_array`, `..._compressed.py:462-476`). It is a real,
sever-and-restore perturbation. It was **not** built here, by decision above.

**Naming rule (binding, if it is ever built by a later task):** such a condition must be named
**reachability** and **never connectivity** — it does not reproduce DynPen's connectivity category
(no credential set `Cr`, no subnet/connection-set, no ownership loss on sever, no node discovery on
add), and it shares the property condition's observation channel (endpoint node features). Calling it
"connectivity" would claim a category coverage this platform does not have.

## GATE — reported 0.1–0.6 (this section preserved as the original decision record)

Summary for the decision:
- **0.1:** encoder sees real edges + edge features (NNConv, 2-hop). STOP condition not met. But the
  implementable connectivity primitive is **node-feature-mediated, not an edge change**.
- **0.2:** re-encode fires if the type returns endpoint IDs; non-zero drift additionally requires
  refreshing `x` AND flipping a visible, service-mapped port (double structural-zero hazard).
- **0.3:** credentials — not reproducible; subnet/connection-set — not reproducible (no subnet);
  firewall — reproducible, both DO and SEES.
- **0.4:** sever real (a); add only restores, cannot discover (b); ownership not revoked (c) → **3.6
  mechanical term absent, total cost only.**
- **0.5:** checkpoints/pairing confirmed (30-40 one shared topo; 80-100 five distinct).
- **0.6:** matched rate settable; localised sustainable; DynPen-faithful magnitude only at network
  level (∝n), not per-node subnet.

**Net:** the task is **runnable as a disclosed firewall-based sever+restore approximation**, but it
reproduces neither DynPen component faithfully and lacks two of DynPen's behavioural consequences
(ownership loss, node discovery on add).

**Decision taken (2026-07-29): do NOT proceed to STEP 1. Connectivity is closed.** STEP 0 falsified the
premise the task was built on; the category is reported as a not-fully-reproducible platform limitation
(this card), and the firewall primitive is preserved only as the "reachability" avenue-not-taken above.
