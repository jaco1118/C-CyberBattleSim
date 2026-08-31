# Task V — two mechanism checks (reading only; source-quoted)

No runs, no training, no code changes. All file:line refer to the current tree.

---

## CHECK 1 — why does the leave rate grow with band?

### Verdict (one sentence)
The mechanism is **identical at every band** — `change_interval` is a fixed constant (20) at all three
bands and enters leave only as a per-step *rate* anchor, not a firing schedule — so the sub-linear rate
growth is **emergent from network size** (a floor ∝ 0.5·n, a discovery-limited eligible pool, and
shorter episodes at 10-15), **not a band-dependent design choice**; but the task's "attempt every
change_interval, varying success rate" framing is mechanically imprecise (see 1.4).

### 1.1 When is a leave attempted? [source]
Leave is **NOT** gated by `change_interval`. In `maybe_apply_dynamic_step` (`cyberbattle_env.py`),
the property/patch change *is* interval-gated but leave is called **every step**:
```
449  if self.patch_service_dynamic_enabled and self.num_iterations % self.change_interval == 0:
450      touched_node = self._apply_legacy_dynamic_change()          # PROPERTY: interval-gated
...
455  if self.dynamic_mode in ("leave", "both"):
456      removed = self._apply_dynamic_leave()                        # LEAVE: called EVERY step
```
Inside `_apply_dynamic_leave`, `change_interval` sets the **expected per-step removal rate**, not an
interval (`cyberbattle_env.py:578-581`):
```
578  # calibration anchor: at full ramp, expected removals per step equal 1/change_interval,
...
581  target_rate = ramp * (1.0 / self.change_interval)
```
with a per-node Bernoulli draw (`:601-605`) and a soft ramp toward a floor (`:570-576`):
```
570  floor = self._get_dynamic_floor()
571  room  = max(0, alive - floor)
576  ramp  = min(1.0, room / max(1, self.num_nodes - floor))
605  hits  = [n for n in eligible if random.random() < probabilities[n]]
```
`change_interval` is a **fixed constant read from configuration** (`__init__` default + train_config),
**not derived from network size**. The only n-dependent quantities are the floor
(`_get_dynamic_floor` = `max(dynamic_min_alive_nodes=5, ceil(0.5·num_nodes))`, `:524-526`) and the
ramp's `num_nodes - floor` normaliser — both properties of the network, not the schedule.

### 1.2 Configured `change_interval` per band [source]
Identical at all three bands in the runs that produced the figures (natural-membership condition inherits
the training `change_interval`):
- 30-40 (`runs/trpo_250k_F1_static_seed42/train_config.yaml`): `change_interval: 20`
- 80-100 (`f2_runs/trpo_250k_F2_static_band80-100_seed42/train_config.yaml`): `change_interval: 20`
- 10-15 (`f4_runs/f4_static_10-15_seed42/train_config.yaml`): `change_interval: 20`
Floor params identical too: `dynamic_min_alive_nodes: 5`, `dynamic_min_alive_fraction: 0.5`,
`dynamic_batch_interval: 150`, `dynamic_batch_size_mean: 1.0`. **They do not differ — the interval does
not answer the question; the network-size-dependent floor/pool/episode-length do.**

### 1.3 What happens when a leave is attempted but no eligible node exists? [source]
**Silent skip, no retry, no exception, not recorded** (`cyberbattle_env.py:565-573`):
```
565  eligible = self._get_removal_eligible_nodes()
566  if not eligible:
567      return []                      # silent: empty list, nothing logged
...
572  if room == 0:                      # floor reached
573      return []                      # silent
```
Eligible = discovered ∩ running ∩ not-{starter,source,target,interest} (`_get_removal_eligible_nodes`,
`:510-518`). A step with no eligible node (or room 0) returns `[]`; `maybe_apply_dynamic_step` then
appends **no** `membership_leave` event (`:457 if removed:`), so the "failed attempt" leaves **no trace**.

### 1.4 Attempts vs successes vs eligible-pool size, per band [what the data supports]
**Attempts are NOT logged separately from successes, and the eligible-pool size is not logged at all.**
The drift CSV header has no `attempt`, `eligible`, `pool`, or selection-`distance` column (only a
`pooling_mode` matched "dist"); the only leave record is a `change_type="membership_leave"` drift event,
which is a **success** (a node was actually removed). So from existing logs I can report the SUCCESS rate
(the given 7.09 / 11.23 / 14.04) but **cannot** report attempts or pool size — stated plainly rather than
inferred.

**Moreover the "attempt" concept does not map onto the code** (a finding about what the data can
support): there is no discrete "one leave attempted, succeeded/failed" per interval. Every step runs a
per-*node* Bernoulli draw over the whole eligible set (`:601-605`) plus an independent Poisson batch
trigger (`:607-617`); a step removes 0, 1, or several nodes. "Number of attempts" is therefore ill-defined
— it is neither 1/step nor 1/interval — which is itself why it is not (and cannot cleanly be) logged.

### 1.5 Which explanation the evidence supports [verdict]
The **defensible "identical mechanism, emergent rate" explanation holds** (change_interval fixed at 20
across bands; rate difference is a property of the network), **with one correction to its mechanism**:
the rate is not "fixed-interval attempts with a varying success rate" but a **per-step probabilistic
ramp**. The sub-linear growth 7→11→14 is driven by three network-size effects, all emergent: (a) the
floor ∝ 0.5·n caps cumulative removals low at small n; (b) the eligible pool is discovery-limited, and
fewer nodes are ever discovered at 10-15; (c) 10-15 episodes are shorter (K binds: ownable×25 < 300).
Consistency check: the full-ramp ceiling is 300/`change_interval` = 15 removals/episode, and 80-100
(14.04) sits right at it while smaller bands fall below — exactly the emergent-ceiling signature, not a
band-dependent design decision.

---

## CHECK 2 — is action selection a pure minimum?

### Verdict (one sentence)
**Yes — a pure `np.argmin` over cosine distance**, single candidate, so the resolved distance is an order
statistic (the minimum) and is usable only as a **relevance** signal, not a perception measure.

### 2.1 `find_closest_action_embedding` in full [source, `cyberbattle_env_compressed.py:1064-1099`]
```
1064  def find_closest_action_embedding(self, action_vector, no_output=False):
1065      metric_mapping = {
1066          'l1':   lambda x, y: np.linalg.norm(x - y, ord=1, axis=1),
1067          'l2':   lambda x, y: np.linalg.norm(x - y, ord=2, axis=1),
1068          'inf':  lambda x, y: np.linalg.norm(x - y, ord=np.inf, axis=1),
1069          'cosine': lambda x, y: distance_cosine.cdist(x, y, 'cosine').flatten()
1070      }
1075      if self.distance_metric not in metric_mapping: raise ValueError(...)
      # [1085-1091: empty-action guard — ADDED BY TASK D3, see note below; never fires for
      #  static/membership/property, so it does not affect this analysis]
1091      if not self.action_embeddings:
1092          return self.starter_node, self.starter_node, "__no_valid_action__", model.LateralMove(), 0.0
1092      embeddings_array = np.array(list(self.action_embeddings.values()))
1093      vector_segment   = np.atleast_2d(np.array(action_vector, dtype=np.float32))
1094      distances = metric_mapping[self.distance_metric](vector_segment, embeddings_array)
1095      min_index = np.argmin(distances)
1096      action, distance = list(self.action_embeddings.keys())[min_index], distances[min_index]
1097      closest_source_node_index, closest_target_node_index, vulnerability_index, outcome_type = action
1099      return closest_source_node_index, closest_target_node_index, vulnerability_index, outcome_type, distance
```
**Selection is `np.argmin` over a distance vector** (`:1095`) — a pure nearest-neighbour argmin. Metric is
**cosine** (`distance_metric: cosine` in every train_config; `scipy.spatial.distance.cdist(..., 'cosine')`,
`:1069`). Not a softmax, not top-k, not temperature-weighted. **(The `if not self.action_embeddings`
guard at :1091-1092 was added by Task D3 for a degenerate empty-action-space case; it is inert for the
static/membership/property conditions and does not alter the argmin.)**

### 2.2 How many candidates influence the outcome
**Exactly one** — the single argmin. No combination of multiple candidates (`:1095-1096`).

### 2.3 Is the resolved distance available / logged?
**Available: yes.** `distance = distances[min_index]` (`:1096`) and returned as the 5th value (`:1099`);
the caller captures it (`:534 ... outcome, distance = self.find_closest_action_embedding(...)`). It is
**used** for the distance penalty (`:620 self.reward += self.penalties_dict['distance_penalty'] * distance`)
and placed in `StepInfo` (`:657 min_distance_action=distance`; schema `:57`). **Logged: NO — not in the
drift CSV** (no distance column in the drift header; `StepInfo.min_distance_action` is returned to the RL
loop each step but not persisted by the drift logger).

### 2.4 What logging it per step would take (do NOT implement)
The distance is **already computed and returned**, so no new computation. Minimal change: thread the
returned `distance` into the drift path (e.g. capture it in `step()` where the drift row is built and add
one column to `_build_drift_row`), OR persist `StepInfo.min_distance_action` per step. **Behaviour-neutral
by construction**: it reads a value already produced for the reward and returned in `StepInfo`; it touches
neither the argmin selection nor the reward. (Reported only — not implemented.)

### 2.5 Why the invalid-action count is always zero (rebuild trigger) [source]
A departed node's actions are **purged from the candidate set** in `remove_node_dynamic`
(`cyberbattle_env_compressed.py:335-339`):
```
335  self.action_embeddings = {
336      k: v for k, v in self.action_embeddings.items()
337      if k[0] != node_id and k[1] != node_id
338  }
339  self.processed_pairs = {p for p in self.processed_pairs if node_id not in p}
```
and the candidate set is **rebuilt whenever the graph changes** — on the agent's own graph-changing
action (`:567-589 if action_changed_graph ... create_continuous_action_space(...)`) and on a dynamic
membership/property change (`:633-641 if nodes_changed: ... self.create_continuous_action_space()`).
Because the argmin (`:1095`) runs over `action_embeddings`, which never contains a removed node's keys,
the agent **cannot select an action referencing a departed node** — hence the invalid-action count is
structurally zero. This is a property of the continuous action space, now sourced.

### 2.6 Tie-breaking
`np.argmin` returns the **first** index of the minimum (NumPy default), so equidistant candidates are
broken by **insertion order of `self.action_embeddings.keys()`** (`:1095-1096`). No explicit tie-break
rule beyond that.

---

## Things noticed while reading (unsolicited, per the OUTPUT clause)

1. **The CHECK-1 inference's mechanism is wrong even though its conclusion is right.** The design doc's
   "a leave is attempted every change_interval steps; what varies is the success rate" does not match the
   code: leave runs a per-*node* Bernoulli draw **every** step (`:456, :601-605`), and `change_interval`
   is a rate anchor (`:581`), not a firing period. The *conclusion* (band-independent mechanism, emergent
   rate) survives, but the thesis should describe it as a per-step probabilistic ramp toward a floor, not
   an interval-attempt/success process — otherwise 1.4's "attempts vs successes" implies a discrete
   attempt that does not exist.

2. **80-100's rate sits at the mechanism's ceiling.** 300/`change_interval` = 15 removals/episode at full
   ramp; 80-100 measures 14.04 (essentially the ceiling), so at the large band the rate is
   *change_interval-limited*, while at 10-15 it is *floor/discovery/episode-limited*. The sub-linearity is
   this crossover, not a single cause — worth stating precisely.

3. **The empty-action guard I added in Task D3 lives inside `find_closest_action_embedding`** (`:1091-1092`).
   It never fires for static/membership/property (their action space is never empty), so it does not
   affect the membership analysis or the invalid-action-zero property — but it is now the one place the
   argmin can be bypassed, and any future reader of this function should know it is a D3 artifact.

4. **Cross-reference (join cap):** a separate reading established the join arm is hard-capped at
   `dynamic_max_joins_per_episode=3` (~2.8 realised/episode) with only ~0.12/episode ever discovered —
   so the leave-vs-join asymmetry (~14 vs ~3) is a *design cap on join*, not a symmetric mechanism. The
   fixed absolute join cap (3, not ∝ n) is a second scale-dilution effect on the join perturbation,
   independent of the donor-pool confound (recorded in the membership/join discussion, not this card).
