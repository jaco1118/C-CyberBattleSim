# Task AN — the arithmetic null for the extremal response rates

Analysis on existing gate drift logs (`cyberbattle/agents/attenuation_drift_logs/drift_{band}.csv`, 5 seeds)
+ source checks. Nothing trained/evaluated/run; only this card modified. Did not touch the running 750k eval.

## STEP 0 (gate)

**0.1 — p = 64 per channel [ARTIFACT].** `node_embeddings_dimensions=64` (`cyberbattle_env_compressed.py:88`
default, `:123` assigned); encoder `out_channels: 64` (`gae/logs/default/SecureBERT/train_config_encoder.yaml:13,16`).
Each pooled slice is `observation_embedding[i*dim:(i+1)*dim]` with `dim = self.node_embeddings_dimensions`
(`:740,743`) → the max/min slice is a coordinate-wise extreme over **p = 64** dimensions.

**0.2 — n_discovered per event [ARTIFACT].** Recorded per row (`n_discovered`, col 7). At membership-leave
events (relevant + visible + immediate/attributed), per band: **mean 6.0 / 22.1 / 69.9**, median 6 / 23 / 71,
between-seed sd 0.2 / 0.7 / 1.1. **These are far below the band midpoints (12 / 35 / 90)** — using midpoints
would understate the expected rate, exactly as the task warns.

**0.3 — departing-node-held-extreme flag: UNAVAILABLE, not recoverable [FINDING].** The drift logs record
`change_drift_max/min` (whether the slice moved) but **no flag for whether the departing node itself held a
coordinate-wise max/min**. Recovering it needs the per-node embeddings at the event (to test `h_v ==` the
pool's coord-max in any of 64 dims); the logs save only norms, never per-node vectors. So it **cannot be
recovered from existing logs**. (Task P STEP 2's `v_held_max` was computed on the offline probe, not the
runtime graph.) **→ STEP 2 (empirical null) is blocked;** STEP 1 (needs only 0.1/0.2) and STEPs 3–5 proceed.

## STEP 1 — the closed-form null on the correct denominator [FINDING]

Per-event expected response under exchangeability `1−(1−1/n)^64` using **that event's own n_discovered**
(averaged over events, not the band midpoint, not averaging n first).

**Reconciliation first (this determines the sign):** the manuscript's reported rates **98.5 / 84.0 / 43.0**
(max) are `change_drift_max>0` over **ALL** membership-leave events (28,088 at 80-100) — reproduced exactly
(43.0% at 80-100). But the exchangeable null assumes **v is a member of the pool of n**, which holds only on
the **perceived subset** (relevant + visible + attributed, where v is actually in the discovered pool). Over
that subset the observed rate is higher, and it is the correct basis for the null:

| band | slice | OBSERVED (perceived subset) | EXPECTED null `1−(1−1/n)^64` | OBS − EXP | (manuscript all-leave) |
|---|---|---|---|---|---|
| 10-15 | max | 99.4% ± 0.2 | 100.0% ± 0.0 | **−0.6** | 98.5% |
| 30-40 | max | 90.4% ± 1.0 | 94.4% ± 0.4 | **−4.0** | 84.0% |
| 80-100 | max | 63.1% ± 1.9 | 60.7% ± 0.6 | **+2.5** | 43.0% |
| 10-15 | min | 99.4% ± 0.2 | 100.0% ± 0.0 | −0.6 | 98.5% |
| 30-40 | min | 90.5% ± 1.0 | 94.4% ± 0.4 | −3.9 | 82.8% |
| 80-100 | min | 62.5% ± 1.7 | 60.7% ± 0.6 | **+1.8** | 36.1% |

**1.3 sign [FINDING]:** the residual (observed − expected) is **small and near zero at every band** — slightly
**below** the null at 10-15/30-40 (where arithmetic already saturates near 100%) and slightly **above** at
80-100 (**+2.5 max, +1.8 min**). **So the headline decline in the extremal response rate (99% → 63%) is
DOMINATED by the closed-form arithmetic null (100% → 61%): removing one of n exchangeable nodes stops holding
a coordinate extreme at rate `1−(1−1/n)^64`, which falls from ~100% to ~61% as n_discovered grows 6→70.** The
part arithmetic does NOT explain is a few percent, and it is marginally **positive at the largest band** — a
small excess consistent with encoder propagation adding to what the departing node alone accounts for.

**What the manuscript should say:** report **observed − expected**, not the raw decline. The raw decline is
mostly the operation, not a finding; the finding is the small residual. And the reported 98.5/84.0/**43.0**
are the **all-leave** rates — comparing 43.0 to the null (60.7) would wrongly read "far below," but that is a
denominator mismatch (non-perceived leaves whose max cannot move by construction); the clean, like-for-like
comparison is the perceived subset (63.1 vs 60.7).

## STEP 2 — the empirical null [BLOCKED by 0.3]

Requires whether the departing node held a coordinate extreme (0.3), which is not recorded and not recoverable
from existing logs. Cannot be computed here. It becomes available only if a re-run logs the per-node
embeddings at each event (the option-(c) logging built and then reverted in Task L, or a real-graph probe
re-run) — recorded as depending on that data.

## STEP 3 — the tension with Task P [FINDING]

Task P: on removals where the departing node held **no** coordinate max, the max slice still moved in
**87–100%** of cases (propagation). If that held on the runtime graph the max would respond to ~100% of
events; it responds to 43.0% (all-leave) / 63.1% (perceived) at 80-100.

**3.2 — same threshold / filter / definition? NO — three differences, and they are the cheapest explanation:**
- **Graph:** Task P ran on the **offline DFS spanning-tree PROXY**; the response rates come from the
  **runtime `evolving_visible_graph`**. Task Q/M established the proxy overstates propagation (and its
  degree/propagation findings are proxy properties — `open_items.md` OI-1).
- **"Moved" definition:** Task P used `norm(newmax − Hmax) > 1e-6` on the probe (STEP 2); the response rate
  uses `change_drift_max > 0` (a *relative* drift) on the runtime slices — different quantities.
- **Event set:** Task P used degree-spaced removal *trials*; the response rate uses *leave events*
  (all-leave 43% vs perceived 63%).

**3.3 — what the evidence supports:** the tension is **substantially a definition/graph artefact**, not a
contradiction in the environment. On the runtime perceived subset the max response is **63.1%, close to the
arithmetic null (60.7%)** — i.e. on the real graph the max slice moves at roughly the exchangeable rate, NOT
the ~100% the proxy implied, so **runtime propagation does NOT dominate the max slice the way Task P's proxy
suggested.** This is consistent with the OI-1 demotion. **But it cannot be fully settled without the
real-graph probe re-run (Task L STEP 3 enables it):** the proxy figure and the runtime figure are not
like-for-like, and only re-running Task P's `v-held-no-max → max-still-moved` decomposition on the real graph
would give the matching quantity. Not resolved by preferring either figure.

## STEP 4 — the null control is NOT a tautology [FINDING]

**4.1:** On a no-change step (`nodes_changed` falsy), `drift_h3 = self._drift_snapshot_fresh()`
(`cyberbattle_env_compressed.py:698`), and `_drift_snapshot_fresh` calls `self.encode(self.evolving_visible_graph)`
**afresh** (`:714`) — an **independent re-encode**, not the pre-change array. (The `nodes_changed`-true branch
reuses the cache at `:696`; the no-change branch deliberately does NOT.) So `change_drift = rel_drift(h2, h3)`
compares two independent `encode()` calls on the same graph. **This is case (a): independent re-encode.**

**4.2 — the control STANDS.** Suggested sentence for the manuscript: *"The 1,287,823 exact-zero drifts are not
a control-flow artefact: on a no-change step the post-change snapshot is produced by an independent re-encode
of the graph (`_drift_snapshot_fresh` calls `encode()` afresh), so the exact zero reflects encoder determinism
rather than a reused array."* (4.3 not applicable; 4.4: a stronger control would perturb the input by an
epsilon and confirm non-zero drift — feasible offline, not run here.)

## STEP 5 — property change: which runs produced usable events [FINDING]

**5.1:** **Usable property-change events were produced in the F1/F2 EVAL sweeps**, not the gate/attenuation
runs. Verified: the gate drift logs (`attenuation_drift_logs/drift_{30-40,80-100}.csv`) contain **only
`membership_leave` and `membership_join` — ZERO property events**. The property cost figures come from the
**F1** eval (30-40, `runs/trpo_250k_F1_static_seed*`, `eval_out/score_*_evalproperty.csv`, cost +0.0174
[+0.0083,+0.0269], `evidence_taskF1.md:289`) and **F2** eval (80-100, `f2_runs/trpo_250k_F2_static_band80-100_seed*`,
`f2eval_out/score_*_evalproperty.csv`, `evidence_taskF2.md:158-159`), where the eval config set
`patch_service_dynamic_enabled=True`.

**5.2 — both the run-status table and the results text need correcting:**
- The results §4.2 ("property produced no usable events because it was **disabled in the frozen
  checkpoints**") is **imprecise on both counts**: property was **not** disabled in the checkpoints (it is an
  **eval-time config flag** `patch_service_dynamic_enabled`, not baked into a frozen policy), and it **did**
  produce usable events — in the F1/F2 evals. §4.2 is only true of the **gate/attenuation** runs, whose eval
  config left property off (hence zero property events there).
- **Corrected statement:** *"Property change was run in the F1/F2 evaluation sweeps and produced usable events
  at both bands (cost figures reported there). It was not enabled in the gate/attenuation drift runs, which
  measured membership change only — so the attenuation analysis contains no property events. This is a
  configuration difference between the two eval sweeps, not a property of the frozen checkpoints."* The
  run-status table must distinguish the F1/F2 eval (property run) from the gate/attenuation run (property not
  run), rather than marking property "run at both bands" without qualification.

## GATE — reported, stop

STEP 0: p=64, n_discovered 6/22/70 at leave events, **0.3 unavailable → STEP 2 blocked**. **STEP 1 (the
high-value result): the extremal response decline is DOMINATED by the closed-form arithmetic null
(1−(1−1/n)^64); the residual is small (−0.6/−4.0/+2.5 max) and marginally positive at scale — the manuscript
should report observed − expected, and use the perceived subset (63.1) not the all-leave 43.0 for the null.**
STEP 3: the Task P vs response-rate tension is largely a proxy/definition/event-set artefact; full resolution
needs the real-graph probe re-run (OI-1 / Task L STEP 3). STEP 4: null control **stands** (independent
re-encode). STEP 5: property events came from F1/F2 evals, not the gate runs; §4.2 and the run-status table
both need the correction above. No remedy chosen; manuscript not edited. Nothing run or modified except this
card.

## STEP 1 DENOMINATOR RECONCILIATION (per follow-up request) — and a CORRECTION to STEP 1 [FINDING]

Filter-stage counts, per band (membership_leave), response rates, and exact-zero full-vector counts. Zero
NaN dropped anywhere; "zero dropped" stated where true.

| stage | predicate | 10-15 | 30-40 | 80-100 | full>0 | max>0 | full==0 exactly |
|---|---|---|---|---|---|---|---|
| 1 ALL leave | `change_type=="membership_leave"` | 5,321 | 21,445 | 28,088 | **100.0%** | 98.6 / 84.0 / **43.0**% | **0** |
| 2 drop NaN full | (0 dropped — none NaN) | 5,321 | 21,445 | 28,088 | 100.0% | 98.6/84.0/43.0% | 0 |
| 3 PERCEIVED subset | `& relevant & touched_node_visible & event_phase∈{immediate,attributed}` | 4,251 | 14,935 | 11,523 | 100.0% | 99.4/90.4/**63.2**% | 0 |

**(a) The "perceived subset" predicate** (`x_stepA.py:15`, `a2.py:16`, quoted):
`d[(d.change_type == ct) & (d.relevant == True) & (d.touched_node_visible == True) &
(d.event_phase.isin(["immediate","attributed"]))]`.

**(b) Denominator of 98.5 / 84.0 / 43.0 (max):** **ALL membership_leave rows** (5,321 / 21,445 / **28,088**),
response = `change_drift_max > 0`. Nothing excluded; zero NaN dropped.

**(c) Denominator of the 100.0% full-vector figure:** the **SAME — all membership_leave rows** (28,088 at
80-100); `change_drift_full > 0` = **100.0%**. So the 100%-full and the 43.0%-max are on the **same
denominator**, and they are consistent (removing an in-pool node always moves the full vector; it moves the
coordinate-max only 43% of the time).

**(d) Reconciliation of the three:** all-leave = 28,088; perceived = 11,523 (drops **16,565** from all-leave).
Those 16,565 dropped events **provably DID move the full vector** (full>0 is 100% over all-leave, exact-zero
count 0), so they were perceived in the only sense that matters here — they are dropped by the **relevance**
filter, which for a leave is a **known undercount**: the departing node is removed from `discovered/owned`
BEFORE `_is_event_relevant` evaluates, so `relevant` collapses to the acted-on fraction (~40%, Task A2 STEP 3.2),
not "was in the pool". So the perceived subset (63.2%) is a **buggy-narrow** denominator, not a perception
boundary.

**(e) Any leave event with full-vector drift exactly zero? NO** — `change_drift_full == 0` count is **0** at
every stage, every band. The manuscript's "100.0%, zero exceptions" holds. No more-serious finding.

**Which denominator the manuscript should use, in one sentence:** use the **ALL membership-leave denominator
(28,088 at 80-100)** for the observed-versus-expected comparison — it is the single denominator on which the
100%-full and the max/min figures are jointly and consistently defined, and every leave event provably moved
the full vector (so every departing node was in the pool, satisfying the exchangeability precondition of the
null) — whereas the 63.2% "perceived subset" is **not comparable** because its relevance filter spuriously
drops ~16,565 in-pool leave events via the leave-relevance undercount (A2), not a perception boundary.

**CORRECTION to STEP 1 (consequence):** STEP 1 above chose the perceived subset (63.1) as the null basis; that
was the wrong denominator. The correct observed max at 80-100 is **43.0%** (all-leave), not 63.1%. The
closed-form null must therefore be recomputed over the **all-leave** set (not done here per the "no new
computation" instruction). Because 43.0% is well below the perceived-subset null (60.7%), the residual sign
will most likely **flip to negative** (observed BELOW the null) — meaning the extremal decline is **more** than
arithmetic predicts, which points at the **inverse-degree selection rule** (low-degree departures move the
extreme less than a random draw), not at encoder propagation. This supersedes STEP 1's tentative "+2.5, propagation
adds" reading. The follow-up computation of the all-leave null is the one remaining step, and it is flagged, not run.

## STEP 1/2 SUPERSEDED — all-leave null at nominal p, with the INDEPENDENCE caveat [FINDING; sign UNDETERMINED]

Per the follow-up: the null `1−(1−1/n)^p` assumes the p=64 coordinates are **independent**. Message passing
correlates GNN embeddings across coordinates, so the effective number of independent chances is almost
certainly **far below 64**, and the null is extremely sensitive to it (at n≈70: p=64→null 60.2, p=32→36.9,
p=20→25.0 — the sign of observed−null flips between 64 and 32). **Nominal p=64 therefore gives an
independence-assuming UPPER BOUND on the null** (independence maximises P(a node holds ≥1 coord max)), which
biases observed−expected **negative by construction.**

**Denominator: ALL-LEAVE (28,088 at 80-100), as accepted.** Confirmed 0.2's n_discovered (6.0/22.1/69.9) was
on the *perceived* subset; recomputed over **all-leave: 6.3 / 22.6 / 70.5** (nearly identical — the
denominator barely changes n; it changes the *observed* rate). Per-event null averaged over all-leave:

| band | n_disc (all-leave) | slice | OBSERVED (all-leave) | NULL @ p=64 (UPPER BOUND) | OBS − NULL@64 |
|---|---|---|---|---|---|
| 10-15 | 6.3 | max / min | 98.5% / 98.5% | 100.0% | −1.4 / −1.5 |
| 30-40 | 22.6 | max / min | 84.0% / 82.8% | 94.1% | −10.1 / −11.2 |
| 80-100 | 70.5 | max / min | 43.0% / 36.1% | 60.5% | **−17.5 / −24.3** |

**CONCLUSION WITHHELD (per instruction).** At nominal p=64 the residual is negative and grows with n, but
**p=64 is an upper bound on the null**; the true effective p is lower, which raises observed−expected and can
flip the sign (e.g. p_eff≈32 → null≈37 < observed 43 → residual becomes **positive**). **So it is NOT yet
determined whether the extremal decline exceeds arithmetic** — i.e. whether the residual points at the
inverse-degree selection rule (if negative) or at encoder propagation (if positive). This supersedes STEP 1's
"+2.5 propagation adds" AND the earlier "flips negative → selection rule" flag: **the number is −17.5 at p=64,
the assumption is independence, and the sign is open.** Report the number and the assumption together.

**Estimating the EFFECTIVE p empirically (what it takes; NOT run):** the assumption-free estimator is, over
real pooled embeddings at a leave event, **the number of DISTINCT nodes that hold the maximum in ≥1 of the 64
coordinates**, k, divided by pool size n. Under independence k/n is large (≈ the p=64 null); under strong
cross-coordinate correlation a few nodes monopolise the extremes and k/n is small. `k/n` **is** the empirical
null (STEP 2's quantity), and `p_eff = log(1−k/n)/log(1−1/n)` recovers the effective width. It requires the
**per-node embeddings of the pool at each leave event on the real graph** — which **Task L STEP 3's current
logger does NOT record** (it logs edges + pooled obs + node identity/degree; the per-node-embedding logging
was the option-(c) addition that was reverted). It is, however, a **free byproduct of the OI-1 real-graph
probe re-run**: that re-encodes the logged `evolving_visible_graph`, yielding per-node embeddings, from which
k (distinct max-holders) is a numpy `argmax` + `unique` — cheap once the re-encode exists. **So the effective
p, the empirical null (STEP 2), and the sign of the residual are all decided by the same OI-1 probe re-run;
until then the sign is open.** Not run here.

## ROBUST-ACROSS-p STATEMENT (defensible NOW, independent of the open sign) [FINDING]

The null was evaluated at p = 64, 48, 32, 24. Both the residual's sign AND which curve falls faster change
with p — but at **every** value the null itself falls steeply with n: from **100.0% to 59.9%** (10-15 → 80-100)
even at the most generous p = 64, and further at lower p. **So a substantial part of the extremal
response-rate decline is a pure counting effect (the departing node stops holding a coordinate extreme as the
pool grows) at ANY plausible p; only the SIZE of that arithmetic part — and hence the sign of the residual —
is open.** This is the statement the manuscript can make now: the decline is *not* a clean encoder finding;
it is arithmetic plus an undetermined residual.
