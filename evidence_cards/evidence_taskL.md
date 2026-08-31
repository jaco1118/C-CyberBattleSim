# Evidence card — Task L (Local multi-topology action-space collapse)

Branch: `attenuation-pooling-scale`. Report only, no thesis wording, no conclusions.
Numbers and provenance only. Each number labelled ARTIFACT (a measurement/output of the
tooling, not itself the phenomenon under study) or FINDING (a direct empirical result about
the systems under study).

Revised scope note (per user instruction): the two planned 250k matched-pair training runs
(STEP 1 of the original Task L) were cancelled before launch. Nothing was trained for this
task. All evidence below is read-only inspection plus zero-training-cost checks (space
construction, checkpoint loading, direct method calls).

---

## L.1 — Raw (unpadded) Local action-space size and observation shape, band 10-15

FINDING. Measured directly by instantiating `CyberBattleLocalEnv` (via
`wrap_graphs_to_local_envs`, `cyberbattle/utils/envs_utils.py:47-56`) on each of the 8
topologies of the 10-15 band individually, with no padding applied (single-topology
construction, matching each env's own `__init__`-time catalogue at
`cyberbattle_env_local.py:74-83`).

| topology_id (original) | raw action_space | catalogue size (action_space − 2) | observation shape |
|---|---|---|---|
| 61 | Discrete(480) | 478 | (3152,) |
| 54 | Discrete(578) | 576 | (3152,) |
| 62 | Discrete(212) | 210 | (3152,) |
| 29 | Discrete(565) | 563 | (3152,) |
| 4  | Discrete(227) | 225 | (3152,) |
| 76 | Discrete(224) | 222 | (3152,) |
| 1  | Discrete(130) | 128 | (3152,) |
| 45 | Discrete(333) | 331 | (3152,) |

Range: 130–578 (4.45x spread) across 8 topologies of the same nominal size band.

Observation shape is constant (3152) across all 8 — not part of the collapse mechanism.
Traced to source: `get_node_feature_vector` (`cyberbattle_env_local.py:256-261`) encodes a
single node's own fixed-schema features (firewall slots capped at `max_services_per_node`,
fixed-width vulnerability embeddings) — independent of the topology's total vulnerability
catalogue size. FINDING: action space varies per topology; observation space does not, for
a structural reason unrelated to the padding question.

Dropped counts: 0 of 8 topologies excluded; zero skipped.

---

## L.2 — Cross-topology policy transfer failure

FINDING. Checkpoint used: `cyberbattle/agents/logs/local_dynamic_5seed_2026-07-18_01-15-51/TRPO_x_control_SecureBERT/checkpoints/1/checkpoint_990000_steps.zip`
— a pre-existing (pre-pivot) TRPO checkpoint trained on `local_baseline_single_topology`
(1 topology, 11 nodes, within the 10-15 band's size range), single-topology training (no
padding applied at training time, since `train_agent.py`'s cross-topology padding pass at
lines 558-589 only grows a catalogue when there is more than one topology in the loaded
batch to pad against; a join-headroom allocation is still added even for a single topology,
since headroom sizing depends on the donor pool, not the training topology count).

ARTIFACT (tooling output, not itself the phenomenon): loading this checkpoint reports
`model.action_space = Discrete(1597)`, `model.policy.action_net = Linear(in_features=64,
out_features=1597, bias=True)` — i.e. the policy's actual output layer is hard-sized to
1597 categories (304 raw catalogue + 2 switch actions + 1293 join headroom slots specific
to that training run's donor-pool sizing), not the 304 a naive unpadded reading would
suggest.

Target: a different topology of the same band — `grid_topologies_10-15` sequential slot 1
(original topology id 61, from `grid_topology_id_map.json`), raw `action_space =
Discrete(480)`.

**Attempt 1 — feed an observation from the new topology into the loaded policy, no env swap:**
`model.predict(obs)` does **not** raise an exception. It silently returns `action=[1515]`.
FINDING: 1515 ≥ 480 (the target topology's real `action_space.n`) — a structurally
meaningless, out-of-range index for this topology. Quoted directly, no paraphrase.

**Attempt 2 — `TRPO.load(checkpoint, env=<new topology's vec env>)`, SB3's own
space-compatibility check:**
```
ValueError: Action spaces do not match: Discrete(1597) != Discrete(480)
```
Quoted verbatim from the raised exception. SB3 itself refuses to bind this checkpoint to
the new topology's environment.

**Attempt 3 — actually apply the mismatched action (1515) to the new topology's env directly:**
```
other_env.calculate_discrete_action(1515)
  -> IndexError: list index out of range
```
(`flattened_action_space` on this topology has only 480 entries; direct call, quoted
verbatim.)
```
other_env.step(1515)
  -> AttributeError: 'CyberBattleLocalEnv' object has no attribute 'outcome'
```
(quoted verbatim — the full `step()` path fails differently and later than the direct
`calculate_discrete_action` call, downstream code assumes `self.outcome` was set by a
branch the `IndexError` prevented from ever running.)

No padding, reshaping, or workaround was applied at any point in this test, per instruction.
The three failures above are the result, not an obstacle that was worked around.

Dropped counts: 1 checkpoint tested, 1 target topology tested; 0 excluded; zero skipped.

---

## L.3 — Compressed env action-space dimension across the same 8 topologies

FINDING. Measured directly, same 8 topologies of band 10-15, via
`wrap_graphs_to_compressed_envs` + `set_pca_components(768)`:

All 8 topologies (original ids 61, 54, 62, 29, 4, 76, 1, 45) report **identical** action
space shape: `(906,)` = 64 (node embedding dim) × 2 + 768 (vulnerability embedding dim) +
10 (outcome one-hot dim), per `cyberbattle_env_compressed.py`'s
`spaces.Box(..., shape=(node_embeddings_dimensions*2 + vulnerability_embeddings_dimensions
+ outcome_dimensions,))`.

Confirms: the Compressed env's action space dimension is identical across all 8 topologies
of the band. Dropped counts: 0 of 8 excluded; zero skipped.

---

## Provenance summary

- Files read, not modified: `cyberbattle/_env/cyberbattle_env_local.py`,
  `cyberbattle/_env/cyberbattle_env_compressed.py`, `cyberbattle/utils/envs_utils.py`,
  `cyberbattle/agents/train_agent.py`, the checkpoint listed above, and the 8 topology
  pickles under `cyberbattle/data/env_samples/scalability_10_15/` and
  `cyberbattle/data/env_samples/grid_topologies_10-15/`.
- No file was edited. No training was run. No padding, alignment, or reshaping was added
  anywhere in the course of producing this evidence.
- No thesis wording or conclusion is drafted here; this card is numbers and provenance only.
