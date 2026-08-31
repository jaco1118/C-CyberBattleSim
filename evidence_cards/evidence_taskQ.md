# Task Q — two scope audits on the central claims (+ interval check)

Read-only: source and existing logs only. Nothing trained/evaluated/run. Q1.4 reads the existing gate drift
logs (`attenuation_drift_logs/drift_{band}.csv`) column-subset only (low footprint); verified non-contending
with the Task Z eval (its completed-CSV count advanced normally throughout).

---

## Q1 — DOES THE 100% FIGURE COVER JOIN EVENTS? Verdict: (i) — joins are FILTERED OUT; the claim is real but NARROWER.

**Q1.1 — the exact filter behind the 100% figure.** Response = `change_drift_slice > 0`, over the event set
selected by (identical across the RQ1 scripts):
`d[(d.change_type ∈ {"property","membership_leave"}) & (d.relevant==True) & (d.touched_node_visible==True) &
(d.event_phase ∈ {"immediate","attributed"})]` — quoted at `a2.py:16`, `x_stepA.py:15`, `x_stepB.py:12`
(e.g. `x_stepA.py:15`: `return d[(d.change_type == ct) & (d.relevant == True) & (d.touched_node_visible ==
True) & ...]`). **Enters:** `property` and `membership_leave` events on already-visible nodes. **Excluded:**
`membership_join` — excluded at the **change_type** stage (it is never in the `{property, membership_leave}`
set), and it would also fail `touched_node_visible` for undiscovered nodes. Also excluded: `no_change`
sanity rows, visibility-lag artefacts, `fired`-phase (not-yet-attributed) rows. [ARTIFACT]

**Q1.2 — where the joins go (filter cascade).** Task T gate counts (`evidence_taskT.md:179-182`), 5-seed grid:
raw drift rows 187,647 / 640,230 / 775,122 (=1,602,999) → drop `touched_node_visible=False` −24,377 → drop
`event_phase ∉ {immediate,attributed}` −1,287,823 (the null control, max drift exactly 0) → **290,799
retained for attenuation**. The ~43,000 headline set is the **response-rate subset** = `{property,
membership_leave}` × relevant × visible × immediate/attributed; the degenerate perception table
(`evidence_taskX.md:132`) counts it at the two analysed bands: 30-40 property 14,849 + membership 7,749;
80-100 property 14,739 + membership 5,520 = **42,857**. **Join rows survive into the 290,799 "retained" pool
but are removed from the 42,857 response set by the change_type filter** — they are routed to the *separate*
join analysis (coverage + attenuation_ratio slope, `evidence_taskT.md:283-289`), not the 100% figure. [ARTIFACT]

**Q1.3 — joins ARE in the logs.** `membership_join` rows per band: **4,294 / 10,984 / 11,455**
(10-15/30-40/80-100); of these, undiscovered-at-fire (`touched_node_visible=False`): **3,256 / 10,062 /
11,059**. (`membership_leave` for comparison: 5,321 / 21,445 / 28,088.) [ARTIFACT]

**Q1.4 — full-vector drift for joins, split by discovery. The STOP condition is NOT triggered.** [FINDING]

Prediction (pre-stated): undiscovered joins → exactly zero; discovered joins → non-zero. **Confirmed, once
the right column is used.** Two source facts settle it:
- The join's **own** contribution to the pooled vector is `delta_h_v` (`cyberbattle_env_compressed.py:829`
  → `_node_delta_vector:735-745`): for `membership_join`, `after = h3.node_embeddings.get(node_id, zero)`.
  An undiscovered joined node is never added to `evolving_visible_graph`, so it is absent from
  `h3.node_embeddings` → `after = zero` → **`delta_h_v = 0` exactly.** Measured: `delta_h_v_norm` for
  undiscovered joins has **max = 0.00e+00 at every band — never non-zero** (values are either exactly 0 or
  NaN on episode-end flush rows; zero nonzeros).
- `change_drift_full` is **NOT per-event**: it is the **global step-level** h2→h3 pooled drift
  (`:816` `change_drift_full = self._rel_drift(h2.combined, h3.combined)`) **stamped identically onto every
  event row that fired that step** (`:868`, inside the `for event in events` loop). So an undiscovered-join
  row co-firing with a `membership_leave`/`property` inherits *that* event's drift.

Measured contamination, undiscovered joins: rows with non-zero global `change_drift_full` = 96 / 231 / 275;
**of those, 96/96, 231/231, 275/275 (100.0%, zero residual) shared their step with another event whose own
`delta_h_v_norm > 0`.** So every non-zero is a co-firing artefact of the shared column — **not** the encoder
responding to an undiscovered node. **The encoder does not see undiscovered nodes; the premise holds.**

**Q1.5 — correct wording.** The claim *"the full pooled observation vector responds to 100.0% of change
events, at every band, with zero exceptions"* is **true only for changes to discovered nodes.** Corrected:

> "The full pooled observation vector responds to 100.0% of changes **to nodes the agent has discovered**
> (property changes and departures of discovered nodes), at every band."

with the required companion disclosure: **joins of nodes never discovered within the episode (≈52%/83%/92%
of joins, rising with scale — `evidence_taskT.md:236-238`) move the pooled vector by exactly zero**, and are
handled separately as a coverage/exploration limit, not counted in the 43,000 nor treated as
perceived/unperceived in the attenuation sense. **Consequence for the cancelled research question:** the
load-bearing sentence *"any change to a discovered node necessarily moves the vector, so 'not perceived'
categories are structurally empty"* stays valid **only with the "to a discovered node" qualifier** — there
*is* a class of change events that move the vector by exactly zero (undiscovered joins), but it is a
**discovery** failure, not a **perception** failure, and must be labelled as such wherever the cancellation
is justified. Do not state the claim without the qualifier.

---

## Q2 — WHAT GRAPH DOES THE ENCODER RUN ON? Verdict: report Task P as a stated PROXY.

**Q2.1 — runtime construction.** The encoder runs on `self.evolving_visible_graph` (an `nx.DiGraph`), via
`encode(): data = from_networkx(self.evolving_visible_graph)` (`cyberbattle_env_compressed.py:382`). Its
edges are added **only** by `add_edge_evolving_visible_graph(source, target, vuln_key)` (`:284-324`), called
on a **successful agent traversal** (reward>0 guard). Node features `x` are the per-node vectors (`:264-269`);
the ground-truth graph is edge-cleared at construction (`generate_network.py:309`). [ARTIFACT]

**Q2.2 — is it a tree? NO.** From source, not from what discovery "usually" produces:
- **>1 parent per node — YES possible.** `add_edge_evolving_visible_graph` adds a directed `source→target`
  edge for *each* successful exploit (`:313` `self.evolving_visible_graph.add_edge(source_node,
  target_node)`; `:294` merges edge features if the pair already exists). A node exploited from two different
  sources gets **two incoming edges**. Nothing dedups to a single parent.
- **Cycles — YES possible.** Exploiting `A→B` and later `B→A` (both successful) creates both directed edges;
  no acyclicity check exists. So the structure is a general DiGraph, **not necessarily a tree.** [FINDING]

**Q2.3 — DFS, BFS, or neither? NEITHER.** It is the agent's **exploit/traversal history**, ordered by the
policy's action sequence — not a spanning-tree construction. Its depth/branching is **policy-determined**:
it can be star-like (everything exploited from the starter) or deep (node-to-node pivoting) or contain
multi-parent/cyclic structure, and this is not fixed by any tree algorithm. So "two-hop reach" on the
runtime graph is not a fixed structural property the way it is on a chosen tree. [FINDING]

**Q2.4 — the probe is a PROXY, not a reproduction.** Task P's propagation ran on a **DFS spanning tree**
(`probe_p.py:60` `nx.dfs_tree(A, nodes[0]).edges()`, `A = net.access_graph.subgraph(nodes)`, node set
BFS-collected). This **resembles** the runtime graph (both sparse, tree-ish for single-parent discovery) but
does **not reproduce** it. Divergence points: (1) the DFS tree forces **exactly one parent** and **no
cycles**, while the runtime graph allows both (Q2.2); (2) DFS-tree **depth is deterministic and deep**, while
runtime depth is policy-dependent and could be shallow/star-like; (3) DFS-tree edge features pick **one**
vuln (`probe_p.py:edge_vuln`), while runtime **aggregates all** exploited vulns per pair (`:300-302`). The
two-hop-reach, propagation-dominance, 1/N, and degree-correlation findings describe **this proxy**. [FINDING]

**Q2.5 — what a real comparison needs.** The runtime `evolving_visible_graph` **structure is not logged** —
the drift log stores only norms/counts/n_discovered, never the edge list or per-node degrees. To compare the
proxy against the real graph one would need to **log the `evolving_visible_graph` edge set / degree
distribution / depth at change events**. This is the **same per-event structural-logging addition Task N/P
identified as unblocking four gates** (degree per departure, per-event propagation, action counterfactual,
coverage↔events linkage) — so a real-vs-proxy comparison would ride on that one addition, **raising that
task's value**. (Not added here, per the DO-NOT list.) [FINDING]

**Q2.6 — wording decision.** Task P's findings must be reported as **properties of a stated proxy** (a deep
DFS spanning tree over the access graph), **not** as properties of the environment, until the runtime graph
is logged and the proxy validated against it. The robust, structure-independent parts (2-hop message-passing
depth = the encoder's 2 layers; 1/N dilution) can be stated as encoder/architecture facts; the
structure-dependent parts (direct-vs-propagation share, degree correlations, extremal-slice dominance) are
proxy-conditional and must carry that label. [FINDING]

---

## Q3 — INTERVALS ON THE THREE CORRELATIONS [FINDING]

corr(propagation, degree) on the sparse proxy, Fisher-z 95% CIs:

| band | r | n | 95% CI |
|---|---|---|---|
| 10-15 | +0.657 | 33 | [+0.406, +0.816] |
| 30-40 | +0.796 | 100 | [+0.710, +0.858] |
| 80-100 | +0.383 | 120 | [+0.218, +0.526] |

**Q3.2 — pairwise separation (two-sided z on Fisher z):**
- 10-15 (+0.66, n=33) vs 30-40 (+0.80, n=100): z=−1.43, **p=0.153 — NOT separated.**
- 30-40 (+0.80) vs 80-100 (+0.38): z=+4.98, **p<0.001 — SEPARATED.**
- 10-15 (+0.66) vs 80-100 (+0.38): z=+1.88, p=0.060 — not separated (borderline).

**The "rise then fall" shape is NOT supported: +0.66 and +0.80 are statistically indistinguishable.** The
real shape is a **flat (high) start followed by a fall** — only the drop at 80-100 is a real feature. This is
the simpler observation, and it is what should be reported as the unexplained (Task M) phenomenon: **not a
peak at 30-40, but a decline at the largest band.** [FINDING]

---

## GATE — reported, stop

**Q1:** verdict **(i)** — joins are filtered out at `change_type`; the 100% figure is real but must be
reworded to "100% of changes **to discovered nodes**," with the undiscovered-join zero-response disclosed as
a coverage (not perception) limit. STOP condition not triggered (undiscovered joins move the vector by
exactly zero; the non-zero `change_drift_full` is a global-column co-firing artefact, 100% explained, zero
residual). **Q2:** the runtime graph is a policy-dependent DiGraph that can have multiple parents and cycles;
Task P's DFS-tree is a **proxy**, so Task P's structure-dependent findings must be labelled as proxy
properties (real comparison needs the Task-N logging addition). **Q3:** the non-monotone shape is a
flat-then-fall, not a rise-then-fall (+0.66 vs +0.80 indistinguishable). Nothing modified except this card;
Task Z eval untouched.
