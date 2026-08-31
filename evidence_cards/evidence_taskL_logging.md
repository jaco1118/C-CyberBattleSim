# Task L — the one logging addition, and the re-run  (logging task)

> **Filename note (Amendment 4):** the name `evidence_taskL.md` was already taken by an unrelated
> earlier "Local multi-topology action-space collapse" task. This logging + re-run task uses its own
> file: **`evidence_cards/evidence_taskL_logging.md`** (this file). The old card is left untouched.

Branch `attenuation-pooling-scale`. STEP 0 read-only (accepted); STEP 1 writes code; STEP 2 is the
regression gate; STEP 3 the re-run (waits for the Task Z eval). The overriding constraint: the addition
must not change behaviour or RNG consumption — met by an **append-only, read-only side logger** that never
touches the drift CSV, env state, or any generator.

## STEP 0 (accepted) — summary

- **0.1** At the drift-snapshot point (`_log_drift_rows`, `cyberbattle_env_compressed.py:812`, called after
  `maybe_apply_dynamic_step`): (a) node identity, (b) `evolving_visible_graph` + edges, (c) degree
  (pre-change for leave), (d) discovered/owned, (e) obs vectors, (f) action set — all available;
  **(g) policy output distribution is NOT env-visible** (external SB3 model — needs an eval hook).
- **0.2** Per-event drift feasible, RNG-free (frozen encoder, `eval()`, no Dropout/sampling) — **offline
  route chosen** (log structure, compute drift offline; zero live code).
- **0.3** Disk ~250–400 MB, obs vectors the bulk; float side-store + don't log the full action set.
- **0.4** Byte-identical regression exists (Task D) but covers only `drift_logging=False`; must be extended
  to `drift_logging=True` (STEP 2).
- **0.5** STEP 3 waits for the Z eval.

## AMENDMENT 3 (claims-audit, reported before STEP 1) — the regression description is correctly scoped [FINDING]

Checked every "byte-identical / no-effect" description against what was actually run:
- `evidence_taskD.md:112,118,121` and `evidence_taskC.md:107,110`: all quote the byte-identical PASS **with
  the explicit qualifier** `drift_logging=False, patch_service_dynamic_enabled=False`. Not overbroad.
- `dissertation_log_v2.md:46`: *"a golden-baseline regression check … confirmed observations and rewards are
  byte-identical between the pre-instrumentation code and the post-instrumentation code **with
  `drift_logging=False`**. With `drift_logging=True`, targeted checks confirmed [functional: no-change ~0
  drift, per-slice differences, delta conventions, relevance tagging, …]."* — the byte-identical claim is
  scoped to logging-OFF; the logging-ON path is described only via **functional** checks, **not** a
  trajectory-identity claim. Also correctly scoped.

**No card overclaims.** The one honest residual to record: **every reported RQ1/RQ3 result was produced with
`drift_logging=True`**, yet the byte-identical proof only ever covered `drift_logging=False`. The assumption
that the logging-ON path yields the same trajectory as logging-OFF has been supported functionally but never
proven byte-identically. The word "diagnostic-only" (`dissertation_log_v2.md:18`) could invite the stronger
reading; the code's own comment is careful ("no effect … **when off**", `cyberbattle_env_compressed.py`).
**STEP 2's `drift_logging=True` regression closes this gap** — it will, for the first time, prove the
logging-ON path (under which all results were generated) is trajectory-identical to logging-OFF. No manuscript
edit made; reported only.

## STEP 1 — what was added [ARTIFACT]

**Design:** a standalone side logger (`cyberbattle/utils/event_graph_logger.py`, `EventGraphLogger`) invoked
only when **both** `drift_logging` and a new **`event_graph_logging`** flag are on. It **reads** state at the
existing drift points and **writes its own append-only files**. It does **not** touch `drift_logger.py`,
`_build_drift_row`, any drift column, env state, or any RNG. Join key to the drift CSV: `(run_id, seed,
episode, step)`.

**Diff summary (purely additive):** `cyberbattle_env_compressed.py` **+74 lines, 0 deletions** (import;
2 `__init__` kwargs `event_graph_logging` / `event_graph_log_dir` + state init; a read-only pre-change capture
block before `maybe_apply_dynamic_step`; a read-only `log_step(...)` call after `_log_drift_rows`). One new
file `cyberbattle/utils/event_graph_logger.py`. **No existing file column removed/renamed/changed (1.6).**
Confirmed: with the flag off (default), `env.event_graph_logging=False`, `_event_graph_logger=None`, no side
files created, drift CSV keeps its **51 columns** unchanged.

**Schema of the side files** (per logged change-step; a step is logged only if it fired a dynamic event):
- `event_graph.jsonl` — one JSON object per change-step:
  `run_id, seed, scenario_id, episode, step`; `obs` = byte offsets/lengths into the float store for
  `pre_graph`/`post_graph` (256 floats) and `pre_discrete`/`post_discrete` (2 floats); `pre_edges`,
  `post_edges` (lists of `[u,v]` node-id strings — the **real runtime graph**, 1.2); `action_keys_pre_count`,
  `action_keys_post_count`, `action_keys_added`, `action_keys_removed` (candidate-action-set change, 1.4 —
  counts + delta, full set rebuildable offline from the graph per 0.3); `events` = one entry per fired event
  with `event_index, change_type, node_ids, changed_node_id, changed_node_degree` (pre-change degree in the
  visible graph; **`None` for an undiscovered join** — honest, not 0), `changed_node_discovered`,
  `changed_node_owned` (1.1).
- `event_obs.f32` — raw little-endian **float32** bytes for the observation vectors (1.3 + **Amendment 2:
  no float16** — exact zeros must survive the null controls).

**1.5 (per-event drift):** taken via the **offline route** (STEP 0.2) — the logged pre/post edge lists +
observations are the inputs a downstream task computes per-event drift from. No live per-event encode was
added (zero RNG exposure); no partial live version was implemented (per the instruction).

## AMENDMENT 1 — full policy input logged + reproduced offline [FINDING]

- **What is logged:** the **full** observation, both pre- and post-change: `graph_embeddings` (256-d) **and**
  `discrete_features` (2-d), not just the 192-d pooled vector.
- **Where VecNormalize sits:** **outside** the env — the env returns the **raw (pre-normalisation)**
  observation; the eval loop applies `VecNormalize.normalize_obs` (`taskZ_eval.py` casts to float32 first).
  The logged vector is therefore **pre-normalisation**, and VecNormalize's running statistics **are saved
  with the checkpoints** (`checkpoint_vecnormalize_*.pkl`). To reproduce the policy input offline:
  `vecn.normalize_obs(logged_raw_obs)`.
- **Acceptance demonstration (one-event, verbatim):** ran a 3-episode eval (F1 static seed42, topo44,
  membership change, `drift_logging=True` + `event_graph_logging=True`), capturing the exact normalised obs
  the eval loop fed to `model.predict` at each step, then reproduced it **from the log alone** (`event_obs.f32`
  post-change vector → `normalize_obs`). Over **47 change-steps cross-checked**:
  `max |reproduced policy input − value eval loop used| = 0.000e+00 → PASS (==0)` (and logged raw obs vs
  env-returned raw obs also `0.000e+00`). The log is sufficient to reconstruct the exact policy input.

## DISK (measured vs 0.3 estimate) [ARTIFACT]

Demo: 47 change-steps → `event_obs.f32` 97,008 B (2,064 B/step = (256+2)×2×4, exact) + `event_graph.jsonl`
319,706 B. Extrapolated to a full sweep (~50,000 change-steps): **obs ≈ 103 MB, jsonl ≈ 340 MB, total ≈
443 MB** — close to the 0.3 estimate (250–400 MB), now dominated by the **edge-list node-id strings** in the
jsonl (switching action keys from full-set to counts+delta cut the jsonl 24×, from an 8.4 GB trajectory).
Gzipping the jsonl would roughly halve it if the sweep total needs trimming; not applied yet. Actual sweep
disk will be reported at STEP 3.

## GATE (STEP 1)

Added: an append-only, read-only side logger (`event_graph_logging` flag) capturing per change-step the real
`evolving_visible_graph` edges (pre/post), the full pre/post observation (float32), the changed node's
identity/degree/discovered/owned/type, and the candidate-action-set counts+delta. **+74 additive lines + one
new module; no existing drift column touched; default path provably inert (flag off → nothing runs).**
Amendment 1 acceptance **PASS (max diff 0)**; Amendment 2 honoured (float32); Amendment 3 reported (regression
descriptions correctly scoped; the logging-ON assumption is closed by STEP 2). **STEP 2 not started** (it is
the next gate); **STEP 3 waits for the Z eval.** Awaiting acceptance before STEP 2.

## STEP 2 — prove it changed nothing [FINDING]

Harness `run_L_regression.py`: runs the **`drift_logging=True`** path with a **fixed action sequence**
(`RandomState(12345)`) + seeded env RNG, comparing OLD (HEAD via `git stash`) vs **NEW(flag off)** vs
**NEW(flag on)** — the drift CSV cell-by-cell (NaN==NaN equal) over every pre-existing column, plus the
trajectory (returned obs/reward/done). 800 steps, both bands, `dynamic_mode=both`+`patch_service` (mixed).

**2.1 / 2.3 — PASS (byte-identical), with `PYTHONHASHSEED=0`:**

| band | old vs new_off | old vs new_on | new_off vs new_on | trajectory |
|---|---|---|---|---|
| 30-40 | **0** cells | **0** cells | **0** cells | identical |
| 80-100 | **0** cells | **0** cells | **0** cells | identical |

`old == new_off == new_on`, zero differing cells, identical trajectories. **The Task-L addition — including the
flag-ON side-logging path STEP 3 uses — is byte-identical to pre-Task-L on the `drift_logging=True` path**
(this upgrades the Amendment-3 logging-ON assumption from a functional check to a byte-identical one).

**A PRE-EXISTING non-determinism, surfaced here, NOT caused by Task L [FINDING].** The regression first showed
hundreds of differing cells; root cause: **two IDENTICAL runs of the *unmodified* code differ** — 24
deterministic cells (4 `membership_join` deferred-attribution rows: `change_type, change_fired, event_id,
step_fired, visibility_lag_steps, node_origin_is_join`) + ~470 cells in `delta_h_v_norm` and
`attenuation_ratio_{mean,max,min,full}`. Cause: **`PYTHONHASHSEED` randomises string-keyed set/dict iteration
order across processes, reordering the join donor pool so `random.choice(available)` picks a different donor**
(`cyberbattle_env.py:781`; donor set at `:189`). Not threads (persists at `set_num_threads(1)`);
**`PYTHONHASHSEED=0` collapses it to 0.** It is confined to **join-related** columns (donor attribution +
per-node `delta_h_v_norm`, and the `attenuation_ratio` dividing by it — already an ARTIFACT per Task T); the
**trajectory, `change_drift_*`, `norm_*`, counts, and every headline figure are deterministic.** But the
**originally-reported gate drift logs were generated without pinning `PYTHONHASHSEED`**, so their
`delta_h_v`/`attenuation_ratio`/join-attribution columns carry this variability — **STEP 3 must set
`PYTHONHASHSEED=0`**, and it will reproduce the headline figures (deterministic columns) but **not** bit-match
those join columns; disclose rather than treat as a regression miss.

**2.2 — RNG unchanged.** The side logger only reads state and writes its own files; it draws from no generator
and mutates no env state. **Proven, not asserted:** `new_off`==`new_on` byte-identical — any RNG draw on the
flag-on path would shift the whole join/leave stream; it does not. (The offline per-event-drift route keeps
zero live code.)

**2.4 — no failure** (2.1–2.3 pass at `PYTHONHASHSEED=0`).

### ITEM A — candidate action set rebuildable offline? NO — STOP and report [FINDING]
Not possible from the current log: (1) only action-key **counts+delta** are logged, **not the baseline set**,
so the pre/post sets can't be reconstructed; (2) only the **pooled** obs is logged, **not per-node
embeddings**, so the 906-d action embeddings needed to **snap** the policy output can't be formed. Therefore
the **observation-channel** term (both outputs snapped to the same pre-change set — the weight-bearing one)
and the **total-effect** term are **not computable**; only the counts-based representation-level term is.
**Disk cost of the fix (measured):** (a) full 906-d embeddings ~**173 GB** (infeasible); (b) full key set +
per-node emb ~**6.6 GB**; (c, minimal) per-node emb + per-node vuln-id lists → derive keys+embeddings offline
~**2.2 GB**. Decision reserved for you: add option (c) (~2.2 GB) or accept those two behavioural terms are out
of scope. No code added for this.

### ITEM B — per-event identity, not per-step? CONFIRMED [FINDING]
`events` is a **list, one entry per fired event**, each with its own `event_index/change_type/node_ids/
changed_node_id/degree/discovered/owned`. Real co-firing step (demo ep1 step109): `event_index=0`
`membership_leave` Node_33 (deg 1, disc 1, owned 1) **vs** `event_index=1` `membership_join` (undiscovered,
deg `None`) — fully distinguishable. Exactly what offline per-event drift needs; the step-level stamping
limitation (Task Q's contaminated join rows) is retired for the newly-logged quantities.

## GATE (STEP 2)

Regression **PASS** (old==new_off==new_on, 0 cells, identical trajectory, both bands, `PYTHONHASHSEED=0`); RNG
unchanged (proven). **ITEM B PASS.** **ITEM A: rebuild NOT possible → reported and STOPPED** (fix = ~2.2 GB
option (c), your call). Pre-existing non-determinism (join donor selection under `PYTHONHASHSEED`) surfaced —
**STEP 3 must pin `PYTHONHASHSEED=0`**; it touches only join `delta_h_v`/`attenuation_ratio`/attribution, not
the headline figures. **STEP 3 still waits for the Z eval and the ITEM A decision.** Awaiting direction.

## STEP 3 PREP — the action-embedding STALENESS finding (architecture, goes in the thesis) [FINDING]

While validating ITEM A option (c), the offline reconstruction of the candidate action embeddings from the
current per-node embeddings **failed: max |action_emb[:64] − node_emb[src]| = 0.360** (over 200 sampled keys,
30-40). Source-verified cause — this is an **architecture finding, not just an obstacle**:

- `create_continuous_action_space` builds each pair's 906-d embedding from `self.node_embeddings` at the time
  the pair is processed (`cyberbattle_env_compressed.py:1032-1038`: `running_owned_nodes = {node:
  self.node_embeddings[node] ...}` → `__add_vulnerabilities_to_action_space(..., source_node_embedding, ...)`),
  and **caches the pair in `processed_pairs`, SKIPPING re-bake on every later rebuild** (`:1045-1049`
  `if (source_node, target_node) in self.processed_pairs: continue`). A pair is first processed at a
  **discovery step**, so its action embedding is **frozen at discovery-time node embeddings** while the
  observation vector stays **live** (re-encoded every graph change). The 0.360 is the measured size of that
  mismatch.

**Consequence for the whole thesis (recorded, manuscript NOT edited):** the encoder **propagation** term
(≈2.5× the direct term, Task P) has **NO route to behaviour through the action embeddings** — they are stale
by design. It can reach behaviour **only through the observation vector.** So the behavioural question has
**TWO live channels, and the third is zero by construction, not by measurement:**
1. **observation channel** — the policy's output changes because its input (the pooled observation) changed. **LIVE.**
2. **candidate-membership channel** — actions involving a departed node are removed from the set. **LIVE.**
3. **action-embedding channel** — the 906-d embeddings would carry the change, but they are frozen at
   discovery time. **DEAD by construction** (`processed_pairs`, `:1045-1049`).

**Exact-reconstruction cost (measured, for the record):** the action space re-bakes **~9,700 distinct 906-d
embeddings per episode** (~35 MB/ep) as node embeddings drift → **~700–1000 GB for the sweep** → logging the
action embeddings is infeasible. The observation-channel and candidate-membership terms are instead obtained
by **deterministic replay** (Decision 1), which needs no extra disk.

**Disclosed observation, NOT to be chased (per the 15-Aug hard stop):** because action embeddings are frozen
at discovery and the graph keeps growing, **staleness accumulates over an episode, and would accumulate faster
on larger networks** — a possible contributor to scale effects **entirely separate from pooling or
propagation**. Reported with the 0.360 attached; no experiment is designed for it (a new direction, not a gap
in an existing one).

## DECISION 1 — replay VERIFIED (policy in the loop); DECISION 2 — reverted to STEP-1 logger [FINDING]

**Action selection is STOCHASTIC** — `replay_verify.py`: `a, _ = mdl.predict(norm(obs), deterministic=False)`
(quoted). So seed+PYTHONHASHSEED alone is not obviously sufficient; it was verified, not assumed.

**(a) Byte-identical replay of ONE FULL EVALUATION EPISODE, policy in the loop** (PYTHONHASHSEED=0,
`set_num_threads(1)`, seed 42, `dynamic_mode=both`):

| band | (a) policy run1 vs run2 | (b) forced-action replay vs policy |
|---|---|---|
| 30-40 | **0 cells** (334 rows) | **0 cells** |
| 80-100 | **0 cells** | **0 cells** |

**(a)** two identical policy-in-loop runs are byte-identical → determinism-based replay works even with
stochastic action selection, once PYTHONHASHSEED is pinned. **(b)** feeding the recorded raw action sequence
(`actions.npy`) **instead of** calling the policy also reproduces the episode byte-identically → **replay does
NOT depend on policy determinism** (the insurance holds). The STEP-3 eval logs `actions.npy` per episode
(the action sequence) so replay is guaranteed regardless of any future non-determinism.

**DECISION 2 — reverted.** With (a) and (b) verified, the option-(c) per-node-embedding + vuln-id additions
were removed (env back to **+74 lines**, 0 deletions; logger back to its STEP-1 schema). The observation-channel
and total-effect behavioural terms come from **deterministic replay**, not from disk. **Logger back to the
~440 MB STEP-1 footprint** (edges + full obs + per-event identity/degree/discovered/owned + action-key
counts/delta). Byte-identical regression on the reverted logger: re-confirmed (see below).

## STEP 3 — the instrumented re-run: COMPLETE, headline reproduced [FINDING, 2026-08-01]

Re-ran the attenuation eval sweep (`compute_attenuation_analysis.py --collect --analyze`, gate manifest:
3 bands × 5 seeds × 8 topologies × ≤400 ep, target 200 events/change-type) with **`PYTHONHASHSEED=0`**, the
reverted STEP-1 event-graph logger enabled, and per-seed `actions.npy` saved. Three edits to the harness are
**gated on `YEG=1`** (inert by default): redirect the drift dir (originals untouched), enable
`event_graph_logging` with a **unique per-env dir** (avoids float-store offset collisions across the 8
topologies), and capture the action sequence. Outputs in `cyberbattle/agents/attenuation_step3_logs/`.

- **3.1 completion.** All 15 run-configs executed, no crashes. Episodes completed (0 skipped): 730/1900/2000 at
  10-15/30-40/80-100. `membership_leave` events **5,121 / 21,388 / 27,998**. `membership_join` budget-exhausted
  shortfall (317/1000 at 30-40, 397/1000 at 80-100) — under-sampling, not structural. `property` 0 events —
  **structurally impossible** under the frozen config (`patch_service_dynamic_enabled=False`), as expected.
- **3.2 reproduction (checked BEFORE interpretation).** Max-slice `membership_leave` response rate reproduced
  vs the reported headline:

  | band | STEP-3 re-run | reported | Δ |
  |---|---|---|---|
  | 10-15 | 98.4 | 98.5 | 0.1 |
  | 30-40 | 84.3 | 84.0 | 0.3 |
  | 80-100 | 42.5 | 43.0 | 0.5 |

  All within **<1 pp** — far inside the pre-registered ~3–4 pp hash-seed spread (CA-2). full-slice 100% every
  band; min-slice 97.8/83.1/35.5. **The hash-pinned re-run reproduces the headline; downstream interpretation
  is unblocked.**
- **3.3 inventory.** 66,853 change-step records across 119 `event_graph.jsonl` (39/40/40 per band) = 377 MB;
  `event_obs.f32` 138 MB; `actions.npy` 4.88 GB (906-dim float32/step × 5 seeds × 3 bands). Each record:
  run_id/seed/scenario/episode/step, `pre_edges`/`post_edges` (evolving_visible_graph), action-key counts +
  delta, per-event `changed_node_{id,degree,discovered,owned}`, obs float-store offset.

**Per-node embeddings confirmed obtainable by re-encoding the logged `evolving_visible_graph`** (`pre_edges`/
`post_edges` = its edge list; node features in the obs store) → the reverted ~440 MB logger needs **no
revisiting** for the effective-p estimate. **Unblocks:** OI-1 real-graph probe re-run (knows out-degree ↔
propagation, the missing RQ3 link), RQ2(c) counterfactual, Task AN effective-p.

**PROVISIONAL caveat (carried, Task Q):** the analysis banner flags the shared cross-band
`join_donor_pool_20_topologies` donor pool (431 vs 192 novel donor/topology pairs) — a directional gate read
only, affecting the **join** columns, **not** the leave response rates reproduced above; join figures must be
regenerated on same-band donor pools before the thesis.
