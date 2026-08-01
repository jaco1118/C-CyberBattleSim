# Task X — the two analyses RQ1 is missing

Analysis only, existing data. STEP A reported here; **STOPPING at the gate — STEP B not begun.**

## STEP A — perception axis for node-PROPERTY change

**PROVENANCE:** node-property (all **patch** = vulnerability removal; env default `change_type="patch"`,
never overridden — no service events) eval on the **F1 static (30-40, `scalability_30_40/44`) and F2
static (80-100, one topology/seed) agents at 250k**. Data: `eval_out/drift_static_seed*_evalproperty.csv`
(30-40) and `f2eval_out/…` (80-100), 5 seeds each. No 10-15 property data exists (property was only run
at 30-40 and 80-100). Filter (same as the existing figures): `change_type=="property" & relevant==True &
touched_node_visible==True & event_phase∈{immediate,attributed}`; property events are all `immediate`.
Response rate at τ=0 per slice = fraction with `change_drift_slice > 0`, per seed, then mean(sd) across
5 seeds. **Note:** these agents were later retrained by F4 to 500k; the perception figures here are on the
**250k** checkpoints (existing data), matching the 250k basis of the 0.72/0.71/0.42/0.41 figures being
completed.

### A.1 + A.2 — response rate at τ=0, ALL FOUR slices, property vs membership on the same F-series basis [FINDING]

| band | change type | full | mean | max | min | n_events |
|---|---|---|---|---|---|---|
| 30-40 | **property (patch)** | **1.000 (sd .000)** | **1.000 (sd .000)** | 0.718 (sd .020) | 0.713 (sd .021) | 14,849 |
| 30-40 | membership_leave | 1.000 (sd .000) | 1.000 (sd .000) | 0.813 (sd .021) | 0.827 (sd .018) | 7,749 |
| 80-100 | **property (patch)** | **1.000 (sd .000)** | **1.000 (sd .000)** | 0.420 (sd .047) | 0.410 (sd .050) | 14,739 |
| 80-100 | membership_leave | 1.000 (sd .000) | 1.000 (sd .000) | 0.642 (sd .048) | 0.656 (sd .042) | 5,520 |

The max/min columns **reproduce the previously-quoted property figures** (0.72/0.71 at 30-40; 0.42/0.41
at 80-100), validating the filter. **The two missing slices are now MEASURED, not argued: the mean slice
and the full vector respond to 100.0% of property events at BOTH bands, every seed (sd 0.000).** So the
"~58% of property events unperceived at 80-100" reading was a mis-attribution of the *extremal-slice*
insensitivity: a property change to a non-extremal node leaves the max/min pool untouched, but the mean
and full vector always move. **Property change is perceived, via the mean/full slices, at 100% — the
argument the methodology rested on is confirmed by measurement.**

*(Basis note: the membership max/min here are the F-series values (0.81/0.83 at 30-40; 0.64/0.66 at
80-100), computed on the SAME agents as the property data for a like-for-like comparison. They differ
from the multi-topology GATE membership figures the task cited (84.0/82.8; 43.0/36.1) — a dataset
difference, largest at 80-100 — so the same-basis comparison is the one in the table.)*

### A.3 — property events with EXACTLY zero full-vector change drift [FINDING]

| band | n property events | exact zero (==0.0) | \|Δ\|≤1e-12 |
|---|---|---|---|
| 30-40 | 14,849 | **0 (0.0000)** | 0 (0.0000) |
| 80-100 | 14,739 | **0 (0.0000)** | 0 (0.0000) |

**Zero exact-zero full-vector drifts, at either band.** Zero dropped. The unperceived branch of the
classification has **no candidate members** from property data. The historical "structural zero" defect
(property events re-encoding to exactly zero because the change reported no affected node) is **fixed and
does not recur** — every one of ~29,600 property events moved the full vector.

### A.4 — characterisation of the exact zeros [FINDING]

**No exact zeros exist (A.3), so there is nothing to characterise** — the residual-defect-vs-genuine-
non-response question does not arise for property change. (Had any existed: they could have been split by
seed, but NOT by node or node-kind — node identity is not logged, only `n_touched_nodes` — and not by
operation, since all property events are patch. This is noted so a future zero, if one ever appears,
is known to be only seed-characterisable from current logging.)

### A.5 — exact-zero full-vector drift among MEMBERSHIP_LEAVE (expected 0) [FINDING]

| band | n leave events | exact zero full |
|---|---|---|
| 30-40 | 7,749 | **0 (0.0000)** |
| 80-100 | 5,520 | **0 (0.0000)** |

Zero, as expected (the full vector responds to 100% of leave events). **A.3 and A.5 are consistent** — no
exact-zero full-vector drift for either change type; no reconciliation needed.

---

## GATE (STEP A) — the decision this gate exists for

**Property change produces NO genuine exact zeros (A.3) and is perceived at 100% on the mean/full slices
(A.1).** Therefore the **unperceived / BLIND branch of the classification is NOT populated by property
data** — property change does not supply the missing "change reached the encoder but the output did not
move" cases. Consequence for the planned larger work: it is **NOT rendered unnecessary** — since neither
membership nor property yields unperceived events (both move the full vector on 100% of events, A.1/A.5),
the strict unperceived branch can still only come from a condition that changes an *undiscovered* node
(the very work the gate was meant to possibly obviate). The extremal-slice failures (max/min) are real
but are pooling insensitivity to *non-extremal* change, not non-perception.

**Reported A.1–A.5. STOPPING. STEP B not begun; awaiting acceptance.**

---

# STEP B — the relevance axis (+ B.3a graded perception) [RAN after STEP A gate]

**PROVENANCE:** F1 static (30-40/topo44) + F2 static (80-100) @250k; same drift data as STEP A. Zero dropped beyond the standard filter (relevant & visible & event_phase∈{immediate,attributed}).

## B.1 The flag, quoted [ARTIFACT]

`cyberbattle_env_compressed.py:722-726`:
```
722  def _is_event_relevant(self, node_ids):
723      return any(
724          node_id in self.owned_nodes or node_id in self.discovered_nodes or node_id in self._drift_acted_on_nodes
725          for node_id in node_ids
726      )
```
An event is relevant iff **ANY** touched node is **owned OR discovered OR acted-on** (source/target earlier this episode). Disjunction over three conditions; `any()` over the touched nodes (for batch events).

## B.2 The degeneracy — CONFIRMED [FINDING]

`discovered` **is** one of the disjuncts, and both change types' eligibility requires the node to be **discovered** (`_get_removal_eligible_nodes` = discovered∩running∩unprotected; property patches discovered running nodes). So **every leave and every property event is relevant by construction — the raw flag is CONSTANT True** (verified: property = {True: 2881/2881}; membership_leave = all True). It cannot carry the axis. Components, separately:
- **discovered** — constant True (the degenerate disjunct).
- **owned** — varies, but logged only as `was_owned` in the separate `leaveown_*` CSV, and **only for membership at 80-100** (30-40 membership eval wrote no leaveown; property is not a departure so has none).
- **acted-on** (`_drift_acted_on_nodes`) — **not logged separately**, folded into the combined flag.

The only usable, varying component is `was_owned`, and only for 80-100 membership.

## B.3 Relevance rate [FINDING]

- Raw flag `relevant`: **constant 1.000** (property & membership_leave, both bands) — reported as constant.
- `was_owned` (owned component, 80-100 membership_leave): **0.232 (sd 0.050)** across 5 seeds — i.e. ~23% of departing nodes were owned; the rest were discovered-but-unowned. **This is the one relevance signal that varies.** (30-40: no leaveown logged; property: no departure, no was_owned.)

## B.3a GRADED perception — channel count (the discriminating axis) [FINDING]

Count of {mean, max, min} slices responding at τ=0, per event (full excluded — it always moves). Distribution (mean of per-seed proportion (sd) across 5 seeds):

| band | change | cc=0 | cc=1 | cc=2 | cc=3 | n |
|---|---|---|---|---|---|---|
| 30-40 | property | 0.000 (.000) | 0.227 (.016) | 0.115 (.009) | **0.659** (.023) | 14,849 |
| 30-40 | membership_leave | 0.000 (.000) | 0.142 (.019) | 0.076 (.008) | **0.782** (.020) | 7,749 |
| 80-100 | property | 0.000 (.000) | **0.524** (.064) | 0.123 (.037) | 0.353 (.034) | 14,739 |
| 80-100 | membership_leave | 0.000 (.000) | **0.299** (.045) | 0.104 (.020) | 0.597 (.047) | 5,520 |

**This is the channel-collapse, graded.** `cc=0` is **empty everywhere** (the mean slice always fires, so ≥1 channel always responds). `cc=3` (all three channels) **falls with scale** (property 0.659→0.353; membership 0.782→0.597) while `cc=1` (mean only; both extremal slices silent) **rises with scale** (property 0.227→0.524; membership 0.142→0.299). **At 80-100, a majority of property events (52%) are perceived on the mean channel ALONE** — property collapses to single-channel more than membership (52% vs 30% at cc=1), consistent with property being the smaller perturbation. Unlike the binary full-vector axis (saturated at 100%), this axis discriminates by band and by change type.

## B.4 Cross-tab: relevance × BOTH perception measures [FINDING]

**Table 1 — binary relevance × binary (full-vector) perception: fully DEGENERATE.** relevant = 1.000 and full-responds = 1.000 for every cell, so all events fall in the single `[relevant × perceived]` cell (30-40 property 14,849; 30-40 membership 7,749; 80-100 property 14,739; 80-100 membership 5,520). The table cannot discriminate — exactly as anticipated.

**Table 2 — binary relevance × channel count: discrimination is entirely on the perception axis.** Relevance = 1 (a single column), and the rows are the B.3a channel-count distribution. So Table 2 = the B.3a table, in the `relevant` column only.

**The was_owned split that would make relevance non-degenerate is NOT computable here.** was_owned lives in `leaveown_*` (keyed by seed, episode, node_id; no step) and perception in the drift CSV (keyed by seed, episode, step; **no node_id**). Only (seed, episode) is shared, and batch leave events break any rank-alignment, so the joint `was_owned × channel-count` table cannot be built from existing logs — it needs per-event node identity in the drift log (a re-run). This is the same logging gap that blocked Task W STEP 2.

## B.5 Empty cells, and WHY [FINDING]

- **The entire `not relevant` column is empty** — because of the **experiment setup**, not the agent: both change types only ever touch *discovered* nodes, so the flag is True by construction. To populate it needs a condition that changes an undiscovered node (the very work Task X's gate was weighing).
- **The `cc=0` row is empty** — because of the **agent/representation**: mean pooling moves whenever any discovered node's embedding changes, so at least one channel always responds. This is a genuine property of the encoder+mean-pool, not a setup artefact.
- **The was_owned-split cells are not "empty" but "not computable"** — a **logging limitation of the setup** (no node_id in the drift log), not a statement about the agent.

## GATE (STEP B) — summary

The relevance axis **cannot be populated from existing data**: the raw flag is constant (setup-degenerate), its only varying component (`was_owned`, 80-100 membership) can't be joined to perception, and the "not relevant" cell requires an undiscovered-target condition. **The graded perception axis (B.3a) is the one usable, discriminating result** — it shows channel-collapse growing with scale and worse for property than membership, with no noise baseline or agent-dependent quantity. **Reported B.1–B.5. STOPPING; awaiting acceptance.**
