# Task P — offline encoder probe: is the attenuation in the pooling or the encoder?

> **[2026-07-31 DEMOTION — the degree correlations do NOT support any scenario-degree claim]** The
> propagation–degree correlations reported below (**+0.66 / +0.80 / +0.38**, STEP 3.2) were computed against
> the **DFS spanning-tree PROXY's undirected degree — mean ≈ 2, a narrow range** (a spanning tree has n−1
> edges over n nodes, so degree ≈ 2 by construction). They were **NOT** computed against the scenario degree
> (knows out-degree 7.9/22.3/54.9). This is stronger than "proxy property" (Task Q Q2.6): **those correlations
> currently support no claim about scenario degree, which is exactly the quantity RQ3 names as its structural
> predictor. RQ3 therefore has NO measured link between its predictor (scenario knows out-degree) and
> propagation at present.** Consequence (recorded in `open_items.md`): the probe must be **re-run against
> knows out-degree on the REAL `evolving_visible_graph`** once Task L STEP 3 logs it; **until then no
> cross-band structural claim in RQ3 has measured support.**


Pure frozen-encoder forward passes on existing graphs. No agent, no training, no episodes.

## STEP 0 — setup [ARTIFACT]

- **0.3 Encoder:** `gae/logs/default/SecureBERT/encoder.pth` + `model_spec.yaml` (the `graph_encoder_path`
  in every run's train_config). 2 layers: `NNConv`(768-d edge feats) → `GCNConv`, `BatchNorm` after each,
  `activation: null`. Loaded frozen, `eval()`. Only one encoder on disk under that path.
- **0.1 Which graphs.** Mean n_discovered at leave events (existing logs): **24.8 (30-40), 69.4 (80-100)**.
  **Crucial:** the encoder runs on `from_networkx(evolving_visible_graph)` — the agent's **traversal**
  graph (edges added only on a successful exploit, `add_edge_evolving_visible_graph`), which is **sparse
  and tree-like (edges ≤ owned count)**, NOT the dense scenario graph. **[CORRECTION — see the header
  demotion below]** the degrees the task's premise cites (7.9/22.3/54.9) are **knows_graph OUT-degrees**
  (Task H, Task Y 0.1 reconciliation), NOT access-graph degrees as earlier written here. A BFS tree over the ~1-hop-complete dense access graph collapses to a **star**
  (depth 1) — unrepresentative of incremental pivoting — so the sparse proxy is a **DFS spanning tree**
  (deep, mean degree ~2, diameter >2), size = mean n_discovered, drawn over each scenario's access graph
  with real node features and per-edge vulnerability embeddings. A **dense** access-subgraph on the same
  nodes is also probed as an upper bound. **Caveat carried throughout:** the real discovered graph's
  depth is between a star and a deep tree and is not logged, so the direct/propagation SHARE is
  bracketed, not pinned (see verdict).
- **0.2 Removals:** sampled across the full degree range (ordered by degree, evenly spaced), degree
  recorded. **30 removal trials per graph, 4 scenarios per band** → 33/100/120 trials at 10-15/30-40/80-100
  (sparse). Zero dropped beyond graphs too small to build.

## STEP 1 — the decomposition (SPARSE = the encoder's real input) [FINDING]

`ΔpooledMean = DIRECT + PROPAGATION`, DIRECT = `(h_bar − h_v)/(N−1)` (removed node leaves the pool),
PROPAGATION = `mean_{survivors}(h'_i − h_i)` (encoder moving survivors' embeddings). Magnitudes (median):

| band | N | direct | propagation | prop/direct (median / mean) | frac prop>direct |
|---|---|---|---|---|---|
| 10-15 | 11 | 0.068 | 0.127 | 2.38 / 2.46 | 0.88 |
| 30-40 | 25 | 0.024 | 0.065 | 2.64 / 2.76 | 0.81 |
| 80-100 | 69 | 0.009 | 0.022 | 2.53 / 2.87 | 0.95 |

- **1.2 slopes vs N:** direct **−1.10** (≈ the pure-1/N-dilution −1.0), propagation **−0.97** (also ≈ −1).
  **Propagation dilutes at the same 1/N rate as the direct term.**
- **1.3 prop/direct vs N:** ≈ **constant at ~2.5** (2.38 → 2.64 → 2.53), a hair steeper if anything
  (direct −1.10 vs prop −0.97). **Propagation does NOT grow as a share with N** — it is a scale-invariant
  ~2.5× the direct term, and it DOMINATES the pooled change at every band (81–95% of removals).

(Dense/upper-bound: propagation ≈ 1.0 vs direct ≈ 0 — on a dense graph embeddings collapse toward the
mean so the direct term vanishes and propagation is everything; prop/direct ~10⁵–10⁷. Unrealistic input,
reported only as the bracket.)

## STEP 2 — extremal slices are propagation-DOMINATED [FINDING]

| band | v held a coord-max (frac) | among v-held-NO-max: max slice still changed (prop only) | max-slice change: full / direct-only (median) |
|---|---|---|---|
| 10-15 | 0.85 | 1.00 | 0.253 / 0.057 |
| 30-40 | 0.61 | 0.87 | 0.142 / 0.008 |
| 80-100 | **0.32** | **0.93** | 0.079 / **0.000** |

**2.3:** the max slice moves on 87–100% of removals where the departing node held **no** extreme — i.e. by
propagation alone — and the propagation-inclusive change (0.079 at 80-100) dwarfs the direct-only change
(0.000, since the departing node rarely holds a max at scale). **The thesis account — extremal response
explained purely by whether the departing node was extremal — is INCOMPLETE: the majority of
extremal-slice movement is the encoder redistributing survivors' embeddings, not the removed node.**

## STEP 3 — reach and drivers [FINDING]

- **3.1 hop distance (layer count CONFIRMED):** mean survivor |Δh_i| by hops from the removed node —
  1hop 0.52/0.45/0.50, **2hop 0.32/0.26/0.31, 3+hop EXACTLY 0.0 (max 0.00e+00)** at 10-15/30-40/80-100.
  Propagation reaches exactly two hops and is exactly zero beyond — the encoder is 2-layer as believed
  (the earlier apparent "2-hop = 0" was a star-graph artefact from BFS over the dense access graph, now
  excluded). **Nothing believed-frozen is unfrozen.**
- **3.2 propagation vs degree:** corr(prop, degree) = **+0.66 / +0.79 / +0.39** (positive at every band);
  high-degree removals propagate more than low-degree (median 0.43 vs 0.12 at 10-15, etc.). **Consequence
  for the selection rule (characterised, not altered):** the inverse-degree weighting removes preferentially
  LOW-degree nodes — exactly the removals that propagate LEAST — so it has been systematically sampling the
  least-disturbing departures, which lowers every measured response rate. This is a genuine downward bias
  on the reported attenuation response rates.
- **3.3 norm / atypicality:** corr(prop, ||h_v||) weak (+0.15–0.25); corr(prop, distance-from-mean) weak
  (−0.01 to +0.08). **Propagation is driven by DEGREE, not by the removed node's embedding norm or
  atypicality** — it is a structural (connectivity) effect, cleanly separable from atypicality.

## STEP 4 — verdict

### 4.1 [FINDING]
On the encoder's realistic (sparse, multi-hop) input the pooled change is **dominated by PROPAGATION**
(≈2.5× the direct term, 81–95% of removals), that dominance is **scale-invariant** (not growing with N),
and — the one thing invariant across graph structures — **both terms dilute at ~1/N**, so the attenuation
TREND is 1/N whether it is pooling or encoder; the direct-vs-propagation SHARE itself depends on the
discovered graph's depth (star ⇒ direct-dominated; deep tree ⇒ propagation-dominated) which is not logged.

### 4.2 Recommendation: **(c)** — both contribute and their shares must be reported separately [FINDING]
- (a) "attenuation is a pooling property" is **not supported**: on realistic input the pooled movement is
  propagation-dominated and the extremal-slice response is majority-propagation.
- (b) "purely an encoder property" **overstates**: the direct (pooling) term is real and both terms scale
  identically at 1/N, so the *scaling* claim (attenuation ∝ 1/N) is unaffected.
- (c) is correct: **the 1/N SCALING is genuine and shared, but the pooled MAGNITUDE is
  propagation-dominated, and the extremal-slice response is majority-propagation** — the encoder spreads a
  removal across its 2-hop neighbourhood before pooling sees it.

### 4.3 Claims in the draft that must change [FINDING]
1. **"A localised change is progressively lost in the pooling step."** The change is **not localised in the
   representation** — 2-hop message passing spreads it (2-hop |Δh_i| is ~60% of 1-hop), and the pooled
   movement is propagation-dominated (~2.5× direct) on realistic input. Restate: the change is redistributed
   by the encoder across its 2-hop neighbourhood, then both the redistributed and the removed-node terms
   dilute at 1/N.
2. **The extremal-slice account** (response explained by whether the departing node was extremal) is
   incomplete — 87–100% of max-slice movement on non-extremal-node removals is propagation.
3. **Any claim that the mean-slice slope −1.09 ≈ −1 evidences pooling-dilution specifically** must be
   dropped: propagation dilutes at −0.97 too, so the slope does not distinguish the two — it was the
   inference this task was asked to replace, and it is not diagnostic.
4. **The reported response rates carry a downward bias** from the inverse-degree selection rule preferring
   low-propagation (low-degree) removals — this must be disclosed wherever response rates are quoted.

**Open item (needs a re-run, out of scope here):** the direct-vs-propagation share can only be pinned by
logging the real `evolving_visible_graph` structure (its degree distribution / depth) at leave events; the
offline bracket is star ⇒ direct-dominated, deep-tree ⇒ propagation-dominated (~2.5×).
