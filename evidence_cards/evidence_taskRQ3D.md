# Task RQ3D — renormalized top-decile-loss vs departures comparison

Goal: check whether RQ3(d)'s "biggest-loss episodes have fewer root departures" finding
(top-decile-loss 0.70 vs rest 1.23, rescued at commit `1d6aaab`, never merged into the live
`evidence_taskCX.md`) survives once departures are normalized by how many roots the episode's
paired undisturbed baseline actually held — the raw comparison is confounded because an agent
already doing badly owns fewer roots to begin with.

## STEP 0 — source verification [FINDING, corrects the task's own premise]

The figures do **not** come from Task F3, contrary to the task's stated expectation. Traced to
`evidence_taskCX.md` PART 3 §3.9 (rescued text, commit `1d6aaab`) — confirmed via the diff's own
file header and cross-checked against §3.5's adjacent "mechanical vs behavioural (from was_root /
`root_owned_departures`)" section, which names the same quantity. F3's own on-disk-ness is
independently contradicted by the rescued text itself (line 83: *"the original F2/F3 cost-eval
harness is unrecoverable... no F2/F3 output CSVs found"*).

The per-episode data needed to recompute this (`event_episode.jsonl`, carrying
`final_root_owned_count` + `root_owned_departures`) has a correct, committed producing mechanism
in `cyberbattle_env_compressed.py` (Task CX B), but the actual output file is **missing** from all
9 checked locations (`cx_step2_static/registration/replay` × 3 bands) — `event_graph_logging` was
never turned on for those runs. Confirmed a **find-lost-raw-data problem, not a
rebuild-the-aggregation-script problem** (the task's stated belief was inverted).

## Checkpoint population identity [FINDING, three rounds of correction — see below]

Settled: this task uses **the dissertation's standard 5-seed × 3-band checkpoint grid**
(`trpo_250k_tuned_compressed_band{10-15,30-40,80-100}_seed{42,100,123,200,300}_2026-07-26_*`),
verified **dynamically trained** (not static — `app.log` shows `[DynamicEnv]` events firing from
step 2 of training). This **is** Task CX PART 3's own "adapted gate checkpoints":
`evidence_taskCX.md:272` names them "the... 'adapted' GATE checkpoints"; `evidence_taskF1.md:14`'s
"gate grid slot 3" matches `grid_topology_id_map.json`'s `"30-40": {"3": "44"}` byte-for-byte; the
archive folder is literally named `2026-07-26_trpo_5seed_gate`. Same population `evidence_taskRQ2C.md`
used. Three prior descriptions of this population in this task's own addenda were each wrong in a
different way (assumed "adapted" checkpoints were missing/needed fresh training; assumed
"static-trained... used this session for Y-N30-N60/RQ2B/O-8" was the same population — it wasn't,
it doesn't span all 3 bands) before landing here. No new training was needed at any point.

## Rollout [ARTIFACT]

`cyberbattle/agents/rq3d_rollout.py` (committed `1d07e75`, manifest `rq3d_manifest.yaml` committed
`d22dd3e` — see `standing_rules.md` SR-1, the manifest was previously only in the untracked
`attenuation_gate_archive/`). Inference-only (`TRPO.load` + `VecNormalize.load(training=False)` +
stochastic `model.predict`), confirmed from source (0.5.2/0.5.4): no `.learn()`, no optimizer, and
`event_graph_logging`/`drift_logging` proven read-only + I/O-only by tracing every gated code path.

Change arm mirrors Task CX's own condition (`dynamic_mode='both'`, `change_interval=20`,
`allow_undiscovered_removal=True`, `uncapped_join=True`; property stays discovered-only/unset,
matching CX's actual condition where it never fired). Static arm: `dynamic_mode='none'`, no
relaxation — the undisturbed pairing baseline. 8 topologies/band (standard grid), 15 change + 10
static episodes per (seed × topology) = 600/400 per band, 3000 total (Addendum 1/2's deliberately
smaller-than-original sample — the original effect is large (~43%), so this remains well-powered).

**Result: 1800 change + 1200 static episodes, zero errors, exact target counts on all 3 bands.**

## Static-pairing convention [ARTIFACT]

`static_root_owned_count(seed, topology, band) = mean(final_root_owned_count)` across static-arm
episodes sharing that seed and topology — exogenous, mirrors RQ1(a)/RQ1(b)'s convention. Zero
episodes excluded (every change episode had a valid non-zero paired denominator, by design since
both arms ran across the identical (seed, topology) set).

## Ranking metric [FINDING — STEP 0.3, revised after the full-scale data]

Original ranking metric unconfirmable from the rescued text (STEP 0.3). Proposed closest match:
`loss(episode) = static_root_owned_count - final_root_owned_count`, pooled across bands, top 10%
of positive-loss episodes. **Scale confound discovered on the full dataset:** static root-owned
count scales sharply with band (mean 7.83/23.08/29.62, max 12/31/48 for 10-15/30-40/80-100), so an
absolute-loss pooled ranking structurally excludes the smallest band — confirmed empirically: zero
10-15 episodes in the pooled top-decile out of 600. This also revises the STEP 0.3 read: since the
original reports non-zero loss-share for all three bands (18/32/34%), its ranking was almost
certainly not pooled-absolute-loss the way this analysis's primary metric is. A **within-band**
ranking variant was added as a supplementary check (not a silent redefinition — the authorized
primary/pooled metric is reported in full).

## Result [FINDING]

**Direction reverses from the original, under every variant tried** (pooled and within-band, raw
and renormalized, all 3 bands): top-decile-loss episodes have **more** departures than rest, not
fewer (original: 0.70 < 1.23; this rollout, pooled raw: 4.16 > 2.58; within-band raw:
10-15 2.45>1.46, 30-40 4.83>4.15, 80-100 2.73>2.30). Statistically resolved (95% CI excludes 0) for:
pooled (raw+renorm), within-band 10-15 (raw+renorm), within-band 30-40 (raw only). Not resolved at
this sample size for 80-100 under either ranking, or within-band 30-40's renormalized comparison.

**Caveat, load-bearing:** this analysis's `loss` metric is built directly from
`final_root_owned_count`, which `root_owned_departures` mechanically depletes — a positive
correlation between this `loss` and departure count is expected to some degree by construction,
independent of any genuine mechanical-vs-behavioural story. If the original's own ranking metric
was residual/score-based (netting out the mechanical channel, per CX's own §3.5 "behavioural
residual" framing) rather than raw-count-based, that fully explains the reversal without implying
the original finding was wrong. **Read as evidence of two different, unreconciled loss
definitions, not as a refutation of the original claim.**

Full per-band tables, bootstrap CIs, and both caveats (gross-count, donor-pool) preserved in
`cyberbattle/agents/rq3d_renormalized_results.md` (committed `3d8c9aa`, alongside the 6 small
per-episode `event_episode.jsonl` files themselves and `compute_rq3d_renormalize.py`).

## Addendum 7 — does a behavioural-residual ranking resolve the reversal? [FINDING — no, does not cleanly resolve]

Pure analysis on already-collected data, no new episodes. Re-checked the rescued text (commit
`1d6aaab` §3.8): the exact formula is `residual = loss - gross mechanical` (mechanical read
literally as `root_owned_departures`, same units, per §3.5's identical usage) — but **§3.9
explicitly ranks on "positive-loss episodes", not residual**, so the rescued text does not
evidence a residualized original ranking; no surviving code implements one either
(`compute_attenuation_analysis.py` has no decile/residual logic). This exercise is therefore
exploratory on this rollout's own data, not a reproduction of a residualized original method.

Three rankings compared (pooled departures, top-decile vs rest): original (0.70 < 1.23, fewer in
top group) vs this rollout's raw-loss ranking (4.16 > 2.58, more in top group, **resolved**) vs
this rollout's behavioural-residual ranking (2.19 < 2.45, fewer in top group — **matches the
original's direction** — but 95% CI [-0.549, +0.041] brackets 0, **not resolved**). The
residualized ranking moves the direction back toward the original, consistent with the
metric-definition explanation mattering, but the result isn't statistically resolved and the
residualization is circular in the opposite direction from the raw-loss ranking (subtracting
departures out of the ranking variable mechanically biases toward fewer departures in the top
group, symmetric to how raw loss mechanically biased toward more). **Does not cleanly separate
"different checkpoint population" from "different loss definition" as the cause of the reversal —
per Addendum 7's explicit stop condition, both raw findings are reported honestly side by side
with this caveat, no further rollout launched to chase it.**

Also found while re-running the pooled comparison: the pooled RENORMALIZED figure under the
raw-loss ranking is a near-exact tie (0.1637 vs 0.1637) — confirmed a genuine pooling/composition
artifact (verified by reproducing it from the already-reported per-band figures), not a bug. The
per-band breakdown remains the trustworthy view; the pooled renormalized number should not be
read as "no effect."

Committed: `compute_rq3d_behavioural_residual.py`, results appended to
`rq3d_renormalized_results.md`.

## Standing rules adopted from this task [record, project-wide — see `standing_rules.md`]

- **SR-1:** commit manifests/configs alongside scripts, not just scripts.
- **SR-2:** default `event_graph_logging=True` (+ `drift_logging=True`) for any real, non-smoke
  run — confirmed read-only/I-O-only from source; its absence on the original CX run is exactly
  what forced this task into a fresh rollout instead of a pure recompute.
