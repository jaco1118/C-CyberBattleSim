# Task H — extend scale to a 200-250 band (optional 130-170 interpolation)

STEP 0 verification. Report 0.1–0.6 and STOP; do not train until 0.4 is reported and agreed. Generation
of the STEP-1 sets is gated on 0.1/0.2/0.5 acceptance. Source-quoted.

> Band placement (accepted from the task): 200-250, midpoint 225, relative width 22% (= 80-100), ratio
> 2.5× to 80-100, upper bound = the 250 RQ2 names. One band. 130-170 optional, second priority, an
> INTERPOLATION point (1/N: 1/90=.0111, 1/150=.0067, 1/225=.0044 — closer to the new band than the old).

## 0.1 Generator and parameters [ARTIFACT]

Existing bands were produced by `env_generation/generate_graphs.py` with per-band configs identical
except `num_nodes_range` (`config/generation_config_{30-40,80-100}nodes.yaml` are byte-identical apart
from that line). Full parameter set (both existing configs, carried verbatim into the new
`generation_config_200-250nodes.yaml`):

- `num_services_range: [1,2]` (per-node count), `homogeneity_range: [0.1,0.1]`,
  `firewall_rule_{incoming,outgoing}_probability_range: [0,0]` (⇒ **no firewall BLOCK rules** in any
  band), `knows_neighbor_probability_range: [0.2,0.8]`, `data_presence/partial_visibility/
  need_to_escalate: [0.2,0.8]`, `service_shutdown/probing/exploit_detection: [0,0]`,
  `success_rate: [0.6,1]`, `value_range: [1,100]`, `min_presence_each_category: 0.05`,
  `minimum_{access,knows,dos}_connectivity_threshold: 0.5` (rejection-acceptance thresholds).
- Category mix: `--percentage_type random` (a convex combination per graph); nlp extractor
  `SecureBERT` (existing sets have only `network_SecureBERT.pkl`).

**The one parameter that changes network CHARACTER with n — reported and MEASURED (Task-H amendment 1),
not inferred:** `knows_neighbor_probability` is applied **per node-pair** — `for node_id in
nodes_graph.nodes: if node_id != node: if random.random() < knows_neighbor_probability:
discovery_list.append(node_id)` (`generate_network.py:191-195`), scaled per-vuln by confidentiality
impact.

MEASURED knows/access degree over 25 scenarios/band (existing bands' actual files):

| band | n | knows mean-deg | knows med-deg | knows density | access mean-deg | access density |
|---|---|---|---|---|---|---|
| 10-15 | 12.5 | 7.9 | 9.3 | 0.693 | 8.9 | 0.774 |
| 30-40 | 34.7 | 22.3 | 26.4 | 0.661 | 24.2 | 0.720 |
| 80-100 | 89.3 | 54.9 | 64.6 | 0.620 | 61.8 | 0.698 |
| **200-250** | **243** | **161.4** | **189.0** | **0.667** | **195.5** | **0.808** |

**Correction to the naive arithmetic:** a flat `p=0.5·(n−1)` predicts 5.75/17/44.5 — the MEASURED
knows-degrees are **~1.3× higher** (7.9/22.3/54.9), because effective per-pair p ≈ 0.62–0.69 (not 0.5:
confidentiality-impact scaling + multiple recon vulns per node). **Precise statement of what grows:**
mean knows-degree grows **~linearly with n** (degree/n ≈ 0.61–0.64), while edge density (connected
fraction of ordered pairs) is **roughly constant but weakly DECLINING** — knows 0.693→0.661→0.620,
i.e. ~0.65, **not 0.5**. So the qualitative reading (degree ∝ n, density ≈ const) holds; the constants
were off. A 225-node graph is proportionally denser than an 85-node one (degree ~2.5× larger), not
merely larger. **Choice: keep `[0.2,0.8]` fixed, identical to every existing band.**
Reason: the existing 10-15/30-40/80-100 curve ALREADY has degree ∝ n (they all share `[0.2,0.8]`), so
comparability REQUIRES reproducing it; scaling `p ∝ 1/n` to hold degree constant would make the new
band non-comparable to the existing three. **Disclosed confound (inherited, not introduced): the whole
scaling curve confounds n with mean degree — "robustness vs N" is really "robustness vs (N and
density)."** This is a property of the existing curve; the new band continues it. (This is separate from,
and compounds, the attribution caveat the task already states.)

**Seed handling (Task-H amendment 5):** `generate_graphs.py` has **no `--seed` and no `random.seed()`**.
**Two kinds of reproducibility, stated explicitly for the thesis:** the existing three bands are
reproducible **only from their archived scenario files** (`data/env_samples/scalability_*`), NOT
re-derivable from the generator (an unseeded rerun yields different graphs). For the new band I will set
and record an explicit global seed before generation, so the 200-250 set is reproducible **from the
generator as well** — a stronger guarantee than the existing bands have, disclosed as such.

## 0.2 Frozen encoder at 2.5× its range [ARTIFACT — architecture done; EMPIRICAL PENDING]

- **Architecture is node-count-AGNOSTIC.** The frozen encoder is 2 layers, `NNConv`(768-d edge feats)
  → `GCNConv` (`gae/model.py`, confirmed in Task E). Both are message-passing convs whose weights act on
  FEATURE dimensions and each node's neighbourhood, **not** on the total node count; forward takes
  `(x, edge_index, edge_attr)` (sparse), so it accepts any graph size. The only N²-scaling object is the
  DECODER's dense adjacency `torch.zeros((num_nodes,num_nodes))` (`gae_utils.py:51`), used at TRAINING
  time only — it is not in the frozen encoder's inference path. So nothing in the encoder breaks at 225.
- **Training size:** GAE trained via `train_gae.py` on `num_environments: 500` graphs; the available
  candidate sets are ~100-node (`syntethic_deployment_20_graphs_100_nodes`, `emulated_network_100_nodes`)
  and the ≤100-node scalability sets. So 225 is ~2.25× the training node range — hence the empirical
  check below is required, not optional.
- **EMPIRICAL [FINDING] — encoder behaves sensibly at 243 nodes, band is USABLE, one caveat.** Full-graph
  encode (every running node + every access-graph edge), per-node embedding-norm distribution:

  | | 80-100/topo5 (n=85) | 225-test (n=243) |
  |---|---|---|
  | per-node norm mean / median | 46.4 / 49.9 | 159.1 / 174.1 |
  | CoV (sd/mean) | 0.27 | 0.31 |
  | frac near-zero | 0.00 | 0.00 |
  | cosine-to-mean-direction (→1 = collapsed) | 0.905 | 0.890 |

  **No degeneration / saturation / collapse:** no zero-collapse, node-level diversity preserved (CoV
  comparable), cosine-to-mean-direction *lower* at 243 (no directional collapse toward a constant).
  **BUT per-node magnitude inflates ~3.4×** (median 174 vs 50): the first layer (`NNConv`) sum-aggregates
  over a neighbourhood whose degree ∝ n (access-degree ≈195 at 243 vs ≈62 at 85), and the frozen
  `BatchNorm` runs in eval mode with running stats calibrated on ≤100-node training graphs, so it
  under-normalizes the larger activations. **Why the band is still usable:** the attenuation metric is
  RELATIVE (`‖h3−h2‖/‖h2‖`), so a uniform magnitude scaling **cancels exactly** — cross-band
  relative-drift comparisons stay valid, and the 1/n dilution the study measures is independent of this
  scaling. **Disclosed caveat: absolute embedding magnitudes are NOT comparable across bands; only
  relative drift is** (already how the analysis is done). Encoder NOT retrained. Test scenario:
  `data/env_samples/graphs_scalability_200_250_test_2026-07-29_23-14-19/1/`.

## 0.3 Hard limits in the code [ARTIFACT]

- **Observation shape is n-INDEPENDENT.** The pooled graph embedding is
  `node_embeddings_dimensions(64) × |aggregations|` plus fixed tails; `create_discrete_features` appends
  only `[len(discovered), len(owned)]` scalars. No fixed max-nodes array or padding anywhere
  (`grep` for max_nodes/node-count arrays: none). So a 225- or 150-node graph produces the same-shape
  observation — nothing truncates.
- **`max_services_per_node`** caps the per-node firewall/service arrays (`:462-485`) — a PER-NODE cap
  (nodes have ~1.2 services), unaffected by n.
- **Action-space cap `sample_subset_samples: 100`** — `__balance_action_space_by_outcome` keeps ≤100
  actions **per outcome class** (`:1054-1055`). This cap ALREADY binds at 30-40 and 80-100 (owned×
  discovered pairs ≫ 100 there), so it is not a new binding at 250 — BUT the **fraction of the true
  action space the 100 sampled actions represent shrinks monotonically with n**, so the 200-250 agent
  sees a proportionally sparser slice of its action space. Behaviour-affecting, but a CONTINUATION of the
  mechanism the 80-100 band already experiences, not a new-at-250 artifact. **Disclosed.**
- **Nearest-neighbour action search** (`find_closest_action_embedding`, cdist over
  `action_embeddings`) is bounded by that cap (≤~100×#outcomes actions), so the SEARCH cost does not
  blow up with n. However **`create_continuous_action_space` iterates owned×discovered pairs = O(n²)**
  every graph-changing step (`:981-982`) — ~62.5k pairs at 250 vs ~6.4k at 85 — which is the dominant
  throughput cost and exactly why 0.4 must be MEASURED.
- **Episode cap** `min(ownable×K(25), episode_iterations=300)` → 300 binds at 200-250 (ownable large),
  same as 80-100; K is inert at this band (as Task R found for 80-100).

## 0.5 Donor pools [ARTIFACT]

Existing join pool `join_donor_pool_20_topologies` = **20 topologies of 10-15 nodes** (measured: sizes
10–15). It is a FIXED small pool used by every band's join eval. **Confound:** a joined node at 80-100
comes from a sparse 10-15-node donor, so it is structurally mismatched to the large training topology —
the source of the standing "~2.2× weaker-pool" caveat. Band-matched pools at 200-250 would AVOID the
size mismatch but at the cost of cross-band join comparability.

**DECISION (Task-H amendment 3): KEEP the existing shared 20-topology (10-15-node) pool for the new
band's join arm.** A band-matched pool is the correct fix but belongs to **Task G**, which regenerates
pools for ALL bands at once; introducing it for one band only would create a cross-band asymmetry
*inside a single experimental arm* — worse than a confound that is consistent across all bands and
already disclosed (same principle as giving both F4 bands the same budget). The ~2.2× weaker-pool caveat
applies uniformly to the new band's join numbers, as to the others. STEP 1.2 is amended accordingly (no
band-matched pool generated). Membership-LEAVE needs no donor pool; only JOIN does.

## 0.6 Convergence at these sizes [ARTIFACT]

A 225-node scenario is a strictly larger RL task than 80-100 (O(n²) action-space construction, ~2.5×
more nodes to own, sparser action-space coverage per 0.3). Task F4 found 80-100 not converged at 250k
and targeted 500k by criterion. **Implication: the 200-250 budget is very likely ≥500k and must be set
by the SAME F4 convergence criterion applied per band, not a fixed budget copied across.** Note: the F4
resume runs emit no tensorboard, so convergence here (as in F4) is measured by evaluating checkpoints at
window boundaries — the reconstruction must be used for this band too.

## 0.4 Throughput — THE DECIDING NUMBER [PENDING clean-box measurement]

Requires (a) the 225-node test scenario (generating) and (b) a free box — a training-fps number
measured while F4 + the F4 convergence eval are running would be meaningless. **Deferred until the box
is idle (after the F4 convergence eval completes).** Will report measured steps/sec at 225 (and 150 if
the optional band stays in play) beside the known 249 (30-40) / 89 (80-100).

**MEASURED fps (this box, amortized over 6 rollouts of n_steps=4096; fresh TRPO, static, device=cuda
but CPU-bound on env encode + O(n²) action-space, GPU idle):**

| band | n | fps (this box) | historical (evidence_taskR) |
|---|---|---|---|
| 30-40 | 34 | **75.8** | 249 |
| 80-100 | 85 | **47.9** | 89 |
| **225-test** | 243 | **21.3** | — |

fps is stable across window length (75.8 vs 75.3 at 1 vs 6 rollouts), so not a warmup artifact — **this
box is genuinely ~2–3× slower than the historical 249/89** (different hardware; the CPU-bound env work
dominates, GPU idle). The current-box fps is the operative reference: the actual STEP-2 runs execute
here, at these rates.

**Projection for five 200-250 runs** (fps 21.3):

| budget | per-run | 5 CONCURRENT wall-clock | 5 sequential (total compute) |
|---|---|---|---|
| 500k (F4 target) | 6.5 h | ~6.5 h (penalty-adj 9.3 h) | 32.6 h |
| **750k (F4 large-band actual — 80-100 needed 750k)** | 9.8 h | **~9.8 h (penalty-adj 14.0 h)** | 48.9 h |
| 1.125M (1.5× of 750k) | 14.7 h | ~14.7 h (penalty-adj 21.0 h) | 73.4 h |

**Fallback decision (amendment 4): NOT triggered.** Under **concurrent** execution (5 parallel, exactly
how F4 ran — 5×OMP4 = 20 threads < 32 cores, GPU idle, so ~solo fps sustained; penalty-adj assumes a
conservative 0.7× slowdown), five 200-250 runs finish in **~10–14 h at the realistic 750k budget and
~15–21 h at 1.5× (1.125M)** — both comfortably under 30 h. (Sequential execution would exceed 30 h even
at 500k, but the experiment runs concurrently.) **→ 200-250 is VIABLE; proceed with the ratio-placed
band, no cost-fallback midpoint needed.** The optional 130-170 band would be cheaper still (n≈150 →
fps between 47.9 and 21.3, ~30 fps → ~7 h/run to 750k) and remains droppable on its own cost.

## GATE STATUS — COMPLETE

All of 0.1–0.6 reported (no STEP-2 training started; the 0.4 fps job is the short measurement STEP 0
itself authorises):
- **0.1** params comparable; density confound inherited + MEASURED (degree ∝ n, density ~0.65); keep
  [0.2,0.8]; seed the new band.
- **0.2** encoder node-count-agnostic; empirically sensible at 243 (no collapse), ~3.4× magnitude
  inflation that RELATIVE drift cancels → band usable.
- **0.3** obs n-independent; action cap already binds (coverage shrinks with n, disclosed); O(n²)
  action-space build is the throughput cost; K inert.
- **0.4** measured fps 75.8/47.9/21.3; **5 concurrent 200-250 runs fit in ~10–21 h ≤ 30 h → VIABLE,
  fallback not triggered.**
- **0.5** keep shared donor pool (amendment 3).
- **0.6** budget ≥500k (likely ≥750k like 80-100), set by the F4 criterion per band.

**Awaiting user acceptance of the STEP 0 gate before STEP 1 generation / STEP 2 training.** Per amendment
4 the throughput fallback is pre-decided (not triggered), so STEP 2 is unblocked on cost; STEP 1
generation still awaits explicit 0.1/0.2/0.5 acceptance.
