# Evidence card — Task C (recon-synthesis crash fix) + Task H0 (150/250-node generation feasibility)

Branch: `attenuation-pooling-scale`. Numbers, provenance, and ARTIFACT/FINDING labels only.
No thesis wording, no conclusions.

---

# TASK C

## STEP 0 answers

### 0.1 — Crash reproduction

**Location**: `_synthesize_recon_vulnerability`, `cyberbattle/_env/cyberbattle_env.py:793-809` (pre-fix line numbers). Failing call: `cyberbattle_env_compressed.py:1161`, `np.concatenate((embedding, outcome_embedding))`.

Reproduced directly (script strips a real node to zero vulnerabilities, calls the exact production methods in the exact order the join path calls them):
```
parent node: Node_4, vulnerabilities before strip: 5, after strip: 0
Traceback (most recent call last):
  File ".../cyberbattle_env.py", line 809, in _synthesize_recon_vulnerability
    self.refresh_vulnerabilities_embeddings_for_node(parent_node_id)
  File ".../cyberbattle_env_compressed.py", line 1161, in refresh_vulnerabilities_embeddings_for_node
    "embedding": np.concatenate((embedding, outcome_embedding))
ValueError: zero-dimensional arrays cannot be concatenated
```
**Trigger condition**: `existing_embeddings = [v.embedding for v in parent_info.vulnerabilities.values() if v.embedding]` empty — for real data, this means specifically zero vulnerabilities on the parent (not an empty-vs-None distinction; `.vulnerabilities` is always a dict).

### 0.2 — Empty-dict fallback downstream

Crashes immediately (ValueError above) — does not propagate as NaN, is not silently skipped. `map_outcome_to_onehot` doesn't intercept it: the synthesized vulnerability's only outcome is `Reconnaissance`, valid for both local/remote, so `np.concatenate` always executes.

### 0.3 — Design question: TWO findings

**Finding A** (node-level, pre-existing, independent of any fix): `convert_node_info_to_observation`, `cyberbattle_env_compressed.py:489-497` — `mean_vulnerabilities_embedding` (768-dim node feature, `vulnerability_embeddings_dimensions=768`, confirmed via `906 = 64*2+768+10`) already zero-fills whenever `len(node_info.vulnerabilities)==0`, NOT gated by `node_info.visible`. Structural, unrelated to Task C's fix.

**Finding B** (action-space level, NOT anticipated by the original framing, more severe, confirmed empirically): a naive `np.zeros(768)` fallback for the *synthesized vulnerability's own embedding* would be zero-norm. `find_closest_action_embedding` uses `scipy.spatial.distance.cdist(x, y, 'cosine')`. Tested at real scale (906-dim, 51 candidates, one zero-vector planted):
```
distance at zero-candidate index (25): nan
np.argmin selected index: 25 (zero-candidate is at index 25)
SELECTED THE ZERO/NaN CANDIDATE: True
```
`np.argmin` does not skip/deprioritize NaN — it can select it. A naive zero fallback would make the synthetic action get *selected*, not merely unreachable.

**Decision (from user)**: neither Option 1 (floor) nor Option 2 (zeros) as originally recorded. Implement a widened-donor search instead (see STEP 1).

### 0.4 — Membership-change crash risk: CORRECTED, not structurally impossible

`_get_join_eligible_parents` (`cyberbattle_env.py:729-733`) filters only on `discovered_nodes` + `status==Running` — **no vulnerability-count filter**. Checked the actual 8 topologies used by the completed Task T gate (via `grid_topology_id_map.json`):

| band | zero-vuln nodes in the 8 used topologies |
|---|---|
| 10-15 | 0 |
| 30-40 | topology 3: 1 |
| 80-100 | topology 10: 2, topology 67: 1, topology 90: 1, topology 17: 7 |

So the crash was theoretically reachable during membership-only runs. Checked all 15 `skip_info_*.json` (which independently catch and count this exact exception): `skip_reason_counts: {}`, `n_episodes_skipped: 0` everywhere — **never fired.** The completed gate ran close to a latent failure, not on a structural guarantee.

### 0.5 — Vulnerability distribution / what property change strips

`_patch_random_vulnerability` (`cyberbattle_env.py:811-828`) removes exactly one vulnerability per invocation, default `change_type="patch"` (never overridden in `train_config.yaml`), `change_interval=20` → up to 15 events/300-step episode. Per-node vulnerability counts in the 8 actually-used topologies: 10-15 band min 1-5 (no zeros); 30-40 min 0-2 (topology 3 has 1 zero-vuln node); 80-100 min 0-2 (topology 100 has 53/88 nodes at exactly 1 vulnerability; topology 17 already has 7 zero-vuln nodes). **Stripping to zero is on the main path, not a rare edge case**, once property change is enabled at any real scale.

---

## STEP 1 implementation

**Fix**: `cyberbattle/_env/cyberbattle_env.py`, commit `0b06f61`. Widened donor search, replacing the single-parent-reuse-or-crash fallback:
1. Parent's own vulnerabilities — **exact original insertion-order logic, unchanged**. Checked directly: sorting this tier by vulnerability_ID instead would have picked a *different* embedding on 3 of 11 real nodes on the regression topology — confirmed and avoided; this tier stays byte-for-byte identical to pre-fix behaviour.
2. Any other discovered node's vulnerabilities — new, deterministic (sorted by node_id then vulnerability_ID, first truthy embedding).
3. Any node anywhere in the topology — new, same deterministic selection.
4. `NoVulnerabilityEmbeddingInTopology` (named exception) raised if the whole topology has no embedding — never fabricated.

Selection is sort-based, not RNG-based: chosen because it never consumes episode RNG state (so it can never perturb any other draw's sequence in an episode where it fires), and which real embedding gets reused doesn't matter substantively (only that it's real and well-formed).

Join-sponsor eligibility is deliberately **not** gated on vulnerability count anywhere (unchanged) — coupling it to property change would shrink the sponsor pool as patching proceeds, entangling two change types meant to be independent.

### Tier/exception/determinism verification (direct method calls, not a full rollout)

```
TIER 2 (parent empty, discovered non-parent has vulns): embedding found, len=768
TIER 3 (parent + all discovered empty, only an undiscovered node has vulns): embedding found, len=768
TIER 4 (whole topology empty): CORRECTLY RAISED NoVulnerabilityEmbeddingInTopology
DETERMINISM (same stripped config, 2 independent envs): trial 0 == trial 1 embedding: True
TIER 1 unchanged check (3 nodes previously found to mismatch insertion-vs-sorted order):
  Node_3: matches original insertion-order selection = True
  Node_6: matches original insertion-order selection = True
  Node_9: matches original insertion-order selection = True
```

### Post-fix cdist/argmin re-verification (empirical, real scale, real continuous action space)

```
action_embeddings size after rebuild: 12
NaN count in full distance matrix: 0
synthetic candidate: distance=0.969458, percentile among all candidates=41.67%
argmin selects a different (real) candidate, not the synthetic one
```
41.67th percentile — solidly mid-range, not an extreme. The widened-donor fix did not reintroduce the outlier problem in a different form.

---

## Byte-identical regression — MANDATORY, corrected after a self-caught methodology error

**First attempt failed** (1691/2000 obs mismatches, seed 12345) — traced to reusing the wrong baseline (`golden_seed*_2000.pkl`, captured earlier under `patch_service_dynamic_enabled=True`) against a comparison run configured with `patch_service_dynamic_enabled=False` per Task C's mandate. That mismatch alone explains the divergence; not a real regression. Corrected by re-deriving the baseline from the actual pre-Task-C commit (`b008aef`, via `git archive`) with the mandated config (`patch_service_dynamic_enabled=False`, `drift_logging=False`), 2000 steps, both required seeds:

```
=== compare seed 12345 ===
PASS: all 2000 steps byte-identical (seed=12345, drift_logging=False, patch_service_dynamic_enabled=False)
  _synthesize_recon_vulnerability calls (this process): 0, needed new tier: 0
=== compare seed 54321 ===
PASS: all 2000 steps byte-identical (seed=54321, drift_logging=False, patch_service_dynamic_enabled=False)
  _synthesize_recon_vulnerability calls (this process): 0, needed new tier: 0
```
Zero-execution counter confirmed by instrumentation (wrapping the real bound method, not re-derived): the new code path executed zero times, consistent with `local_baseline_single_topology` having no zero-vulnerability nodes and property change disabled.

---

## Post-fix smoke test (300-500 steps, NOT an experiment, NOT a result)

`patch_service_dynamic_enabled=True`, 400 steps, `scalability_10_15/1`, `drift_logging=True` for inspection only:
```
completed 400 steps, crashes: 0
change_type value_counts: NaN=389, property=20, membership_leave=9
env.patch_service_dynamic_enabled reset to: False
```
No crash (1.5 criterion met); property events fire (20/400, exactly the deterministic `change_interval=20` period; 1.5 criterion met). Third criterion ("well-formed, not NaN or zero everywhere") — **investigated rather than assumed**, since `change_drift_full` was exactly 0.0 for all 20 property rows:

**FINDING (new, not previously known, out of scope to fix): property-change events currently have no mechanism connecting them to the observation at all.** `maybe_apply_dynamic_step` (`cyberbattle_env.py:433-459`) calls `_apply_legacy_dynamic_change()` (property) but its effect is never included in the function's return value (`removed + joined` only) — so `step()`'s `if nodes_changed:` re-encode gate (`cyberbattle_env_compressed.py:625`) never fires for a property-only event. Verified directly (not from the aggregate drift number): patched a node already in `evolving_visible_graph`, its feature vector `x` was **bit-for-bit unchanged** after the vulnerability deletion; calling `update_node_evolving_visible_graph` explicitly **does** produce a real, non-trivial change (max abs diff 0.00345) — confirming the mechanism exists but is never invoked as a consequence of property change. `update_node_evolving_visible_graph` (`cyberbattle_env_compressed.py:280`) has exactly one call site (`update_evolving_visible_graph_after_step:957`), gated on the *agent's own* action outcome type, unrelated to property change. So `change_drift_full=0.0` for property events is exact and structural, not a numerics artifact and not attenuation — the observation is never refreshed at all unless the agent happens to act on that same node for an unrelated reason. Not fixed (out of scope); flagged here as a new, more fundamental blocker for Task D than the crash this task fixed.

`agent_drift_full`, `norm_h1`, `norm_h2`, `norm_h3` are non-degenerate (real, varying, non-null values) — only `change_drift_full` (the h2→h3 property-specific delta) is affected by this gap.

---

## Membership vs. property event rate (for Task D planning, not acted on)

Recomputed correctly after catching a double-counting error (episode-end flush rows share `event_phase="fired"` with the original firing — first pass wrongly counted both as separate events, inflating join counts past the `dynamic_max_joins_per_episode=3` cap; corrected by deduplicating on `event_id`):

| band | membership_leave (events/episode) | membership_join (events/episode) | combined | property (deterministic, if enabled) |
|---|---|---|---|---|
| 10-15 | 7.09 | 2.83 | 9.92 | ~15 (up to `episode_iterations/change_interval`=300/20, less if episode ends early) |
| 30-40 | 11.23 | 2.81 | 14.04 | ~15 |
| 80-100 | 14.04 | 2.79 | 16.83 | ~15 |

`membership_join` correctly bounded just under the 3-per-episode cap across all bands (confirmed, not assumed, after the dedup fix). `membership_leave` scales up with band size; `membership_join` stays roughly flat (capped).

**The two rates differ substantially, and the direction of the mismatch changes across bands**: property's fixed ~15/episode is ~51% higher than combined membership at 10-15 (9.92), roughly comparable at 30-40 (14.04), and ~11% lower than combined membership at 80-100 (16.83). Comparing BLIND-vs-attenuated across change types at a single band would conflate disturbance intensity with change type; the mismatch is band-dependent, not a constant offset.

---

## Files changed

- `cyberbattle/_env/cyberbattle_env.py` — commit `0b06f61`

## Dropped/skipped counts

Zero skipped anywhere in this task's verification runs (all scripted checks completed; the one FAIL was a self-caught test-harness baseline mismatch, corrected and re-run, not a dropped/skipped case).

## Confirmed at end of task

`patch_service_dynamic_enabled` is `False` in the repo's live `config/train_config.yaml` (unchanged this task) and was explicitly reset to `False` on the smoke-test env instance after the smoke test completed.

---

# TASK H0 (parallel, read-only — no code changed, no topology generated)

### H0.1 — Generator location
`cyberbattle/env_generation/generate_graphs.py`, entry point `generate_network_graphs()`. Node count controlled by `config['num_nodes_range']` (line 178, `random.randint(range[0], range[1])`), set via `--generation_config`.

### H0.2 — Never used above 100 nodes
Confirmed two ways: (a) every generation-config template on disk (`generation_config_{s,m,l}.yaml` and default) tops out at `generation_config_l.yaml`'s `num_nodes_range: [80, 100]`; (b) every saved `generation_config.yaml` alongside an existing sample set matches — `scalability_80_100`: `[80,100]`, `num_graphs: 100`; `syntethic_deployment_20_graphs_100_nodes`: `[100,100]`, `num_graphs: 20`. No config anywhere requests more than 100.

### H0.3 — Comparability requirements
Generator explicitly controls: `num_nodes_range`, `num_services_range`, category percentages (`ics/iot/routers/unix/windows`), connectivity thresholds (`minimum_{access,dos,knows}_connectivity_threshold`), `homogeneity_range`. Vulnerability-count-per-node and outcome-type mix **emerge** from the vulnerability classifier's output given the sampled services — not set directly, confirmed by reading `generate_network_graphs`: the classifier runs once per node's assigned services, producing whatever vulnerability/outcome distribution the classifier predicts, with no explicit target distribution enforced (the `privesc_coverage_ok` rejection check exists in the code but is commented out, line 220-223). Donor-pool compatibility for membership join is **not addressed by the generator at all** — the existing `join_donor_pool_20_topologies` is a separately hand-curated folder, unrelated to this generator's own selection process.

### H0.4 — Feasibility estimate (reasoned, not measured — no timing log found, nothing generated)
`save_model_nlp_extractors_versions` (`cyberbattle/utils/envs_utils.py:83-88`) re-extracts features once per requested NLP model (default list of 8: bert, roberta, distilbert, gpt2, SecBERT, SecRoBERTa, SecureBERT, CySecBERT) — this project only uses SecureBERT downstream, so requesting just that one model cuts this specific cost ~8x. 8 topologies at 150-250 nodes is a substantially smaller total workload (≈1600 total nodes) than the existing 80-100 band's already-successful 100-graph batch (≈9000 total nodes). **Estimate: minutes to a small number of hours, not impractical** — explicitly an estimate, not a measurement; no generation was run, per instruction.

### Two items added to the record before any generation (per explicit instruction — not started)
1. **Distribution mismatch risk**: vulnerability-count-per-node emerges from the classifier rather than being set directly (confirmed above), so a generated 150/250-node graph's distribution may not resemble the existing bands' distributions (documented per-band min/max/mean in Task C's 0.5 table above, for comparison once/if generation happens). A full 100-graph set should not be committed to before checking this on a small trial batch.
2. **Donor pool gap**: the join donor pool is hand-curated (`join_donor_pool_20_topologies`) and does not exist for nodes above 100. Task H (the RQ2 scale grid) therefore needs donor-pool construction as a *separate* piece of unscoped work, in addition to topology generation itself — flagged, not started.
