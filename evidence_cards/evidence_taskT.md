# Evidence card — Task T STEP 5 (revised): five-seed TRPO attenuation gate

Branch: `attenuation-pooling-scale`. Numbers, provenance, and ARTIFACT/FINDING labels only.
No thesis wording, no conclusions.

**Path note**: the task specified `/Users/jaco_chan/Documents/Cowork Dissertation Playground/
Remote_mount/evidence_taskT.md`. That path does not exist on this machine (`/Users` doesn't
exist at all — this session runs on a Linux host, not the local Mac the path implies). Written
here instead, alongside `evidence_taskL.md`, in the repo's `evidence_cards/` folder.

"PROVISIONAL: shared confounded donor pool, approx 2.2x weaker at the large band; membership
change type only; property change disabled."

---

## STEP 0 answers

### 0.1 — Full aggregation-site audit

Every `groupby`, `pivot`, `pivot_table`, `merge`, `join`, `concat`, `set_index`, `reset_index`,
`drop_duplicates`, `nunique`, `idxmax`, `idxmin`, `rolling`, and dict keyed on episode/event id
in `compute_attenuation_analysis.py`:

| line(s) | operation | keys | seed included? | justification |
|---|---|---|---|---|
| 152, 156, 178 (`Counter`, `setdefault`) | per-seed summary accumulation in `collect_band_data` | change_type | N/A | Structural: inputs are the return value of one seed's own `_collect_one_seed` call, already filtered to that seed before being summed. |
| 259 | `df = df[df['seed'] == seed]` | seed | ✓ (the fix itself) | — |
| 385/389 (pre-fix) → now `.groupby(['seed','scenario_id','event_id']).ngroups` | visibility_stats distinct-event counting | seed, scenario_id, event_id | ✗ then ✓ | **Confirmed broken pre-fix** (see ACTION 1). `event_id` is generated per env instance (`CyberBattleCompressedEnv._drift_event_counter`), restarting at 0 for every one of 8 topologies × 5 seeds = 40 instances writing one band's CSV. Confirmed on real data: 119 raw-unique `event_id` values vs 1957 true distinct `(seed, scenario_id, event_id)` tuples on band 10-15 — a >16× undercount. **Fixed** (commit `b008aef`). |
| 453, 455 | `attenuation_df.groupby(group_cols)` | seed, scenario_id, episode, change_type | ✓ | Episode identity requires all four; episode numbering restarts independently per seed's own run. |
| 547–548 | `slice_df.groupby([...])` | seed, scenario_id, episode, n_discovered_bin | ✓ | Same reasoning. |
| 993–994 | `rr_df.set_index(['slice','n_discovered_bin'])` | slice, n_discovered_bin | not present, correctly so | Structural: `rr_df` is already the output of `compute_response_rates`, one row per (change_type, slice, tau, bin) — seed already collapsed correctly upstream at 547. |
| 1107–1108 | `valid.groupby('seed')` | seed only | ✓ (intentionally sole key) | This *is* the between-seed variance computation. |
| 1137, 1212, 1213 | `pd.concat(...)` | — | N/A | Vertical stacking only; every row keeps its own seed/band columns. |

**Verdict**: one previously-unaudited site found broken, now fixed (ACTION 1). Every other site
either includes seed or has a structural reason not to.

### 0.2 — Regression against old single-seed PPO data

Superseded by a deterministic version at the user's request (ACTION 2, below) after the first,
stochastic 0.2 check (re-collecting episodes from the old PPO checkpoints) showed small
(~3–4 point) shifts on max/min response rate attributable to unseeded environment stochasticity
during collection (confirmed via grep: zero `np.random.seed`/`random.seed`/`torch.manual_seed`
calls anywhere in `compute_attenuation_analysis.py` — episode collection has never been
deterministic run-to-run, only the checkpoint's trained weights are fixed).

### ACTION 2 (replaces stochastic 0.2) — deterministic regression, analyze-only, fixed CSV

Method: checked out `compute_attenuation_analysis.py` at `429d966~1` (commit `2ce6c83`, the
"pre-429d966" state) into a scratch dir; ran both that version and the current version with
`--analyze` only (no re-collection) against the SAME already-collected single-seed CSV
(`regression_check_old_ppo/attenuation_drift_logs/drift_10-15.csv`, 32,186 rows).

- `attenuation_episode_aggregates.csv`: identical row count (242 both), new version adds 16
  columns (`seed`, `agent_drift_{mean,max,min}`, `change_drift_{mean,max,min}`,
  `norm_h{1,2,3}_{mean,max,min}` — all intentional additions), **zero columns removed, zero
  differing values in all 18 pre-existing columns** (row-for-row comparison after identical
  sort).
- `response_rate_by_bin.csv`: only 4 of many rows differ, only in `ci_lo`/`ci_hi` (by
  ~0.01–0.02 percentage points); every `response_rate` point estimate is byte-identical.
- Traced the CI jitter to `cyberbattle/utils/math_utils.py:49`,
  `bootstrap_ci`'s `rng = np.random.default_rng()` — no fixed seed, identical in both code
  versions, pre-existing and unchanged by the refactor.

**Verdict: PASS. Zero definitional changes.**

### 0.3 — Seed identification

Recorded per row, not inferred: `_build_drift_row` writes `"seed": self.drift_seed` directly
from the env's own attribute; `collect_band_data` sets it explicitly from
`train_config['seeds_runs'][0]` (the checkpoint's own saved training seed) — never inferred
from a directory name or path. A seed producing zero rows for a band would not crash (guarded
by `len(rates) < 2` / `len(per_seed_rel) >= 2` checks) but would silently disappear from the
`seeds_present` list in the between-seed section rather than being called out explicitly —
minor completeness gap, not a correctness bug (`per_seed_skip_info`/`per_seed_shortfall` track
every attempted seed independently of row count).

### 0.4 — Stopping target and resulting episode counts

Target is **per seed** (`target_relevant_events_per_change_type: 200`, checked inside
`_collect_one_seed`'s own loop against `relevant_counts()` filtered to that seed). Resulting
counts (final, from the completed run):

| band | attempted | completed | skipped |
|---|---|---|---|
| 10-15 | 750 | 750 | 0 |
| 30-40 | 1910 | 1910 | 0 |
| 80-100 | 2000 | 2000 | 0 |
| **total** | **4660** | **4660** | **0** |

vs. old single-seed gate: 130 / 300 / 400 (830 total). Roughly 5× at 10-15 (750≈5×150), less
than 5× at 30-40 and 80-100 because several seeds there stopped on the 400-episode budget cap
for `membership_join` before reaching 5×200 (see per-seed shortfall table below) — i.e. the
band-level total is not simply the old number times five, and stating only the total would
hide that.

**Does the stopping condition interact with n_discovered?** All completed episodes up to the
stop point are retained in full — no episode is selectively dropped based on its own
characteristics. But the STOPPING POINT (how many total episodes get collected) differs
systematically by how fast relevant events accumulate, which differs by band/seed. This is a
real, reportable limitation: bands/seeds where events accumulate slowly get a smaller
`n_discovered` sample overall, not a differently-selected one.

### 0.5 — Per-run completion table

All 15/15 reached checkpoint ≥250000 and logged "Training finished" — no partial or missing
runs.

| band | seed | wall clock | reached 250000 |
|---|---|---|---|
| 10-15 | 42/100/123/200/300 | 583s/589s/559s/587s/620s (sequential) | Y/Y/Y/Y/Y |
| 30-40 | 42/100/123/200/300 | 1056s/997s/1017s/1041s/1077s (sequential) | Y/Y/Y/Y/Y |
| 80-100 | 42/100/123/200/300 | ~48–49 min each (5-way parallel, not comparable to solo cost) | Y/Y/Y/Y/Y |

### 0.6 — Per-slice norm columns

All 9 present (`norm_h{1,2,3}_{mean,max,min}`). Null pattern (band 10-15, smaller smoke check
generalizes to all bands): h1/h2 nulls come from episode-end **flush** rows only (a pending
event never resolved, flushed with no live snapshot at all); h3 nulls = flush rows + attributed
rows (attribution only has h1/h2 by construction) — refining the assumption that h3 nulls are
attributed-only.

### 0.7 — Training-time dynamic config

Quoted from `logs/trpo_250k_tuned_compressed_band10-15_seed42_.../train_config.yaml`:
`dynamic_mode: both` (line 41), `patch_service_dynamic_enabled: false` (line 71). These
checkpoints were trained WITH membership leave/join active — not zero-shot naive to membership
changes. Only property/connectivity changes would be genuinely novel to them.

### 0.8 — Change types fired, final counts

| band | membership_leave rows | membership_join rows | property |
|---|---|---|---|
| 10-15 | (see filtering table below) | — | 0 (confirmed, `patch_service_dynamic_enabled=False`) |
| 30-40 | — | — | 0 |
| 80-100 | — | — | 0 |

(Raw change_type row counts subsumed by the filtering table in "Gate row counts" below —
repeating both would double-count; property confirmed absent in all three bands' pre-flight
check, `gate_summary.txt` lines 21-29.)

---

## ACTION 1 — event_id collision fix, before/after

Fixed in `compute_attenuation_analysis.py:389-406` (commit `b008aef`): disambiguated by
`(seed, scenario_id, event_id)` instead of raw `event_id`. Recomputed from already-collected
CSVs, no re-collection:

| dataset | band | OLD n_fired/n_never/frac | NEW n_fired/n_never/frac | BLIND flag (>30%) |
|---|---|---|---|---|
| live 5-seed grid | 10-15 | 132 / 118 / 89.4% | 2124 / 1096 / 51.6% | True → True (unchanged) |
| live 5-seed grid | 30-40 | 335 / 320 / 95.5% | 5360 / 4445 / 82.9% | True → True (unchanged) |
| live 5-seed grid | 80-100 | 268 / 265 / 98.9% | 5577 / 5131 / 92.0% | True → True (unchanged) |
| single-seed stand-in* | 10-15 | 84 / 68 / 81.0% | 375 / 167 / 44.5% | True → True (unchanged) |
| single-seed stand-in* | 30-40 | 170 / 160 / 94.1% | 731 / 533 / 72.9% | True → True (unchanged) |
| single-seed stand-in* | 80-100 | 201 / 193 / 96.0% | 1117 / 987 / 88.4% | True → True (unchanged) |

*The literal original 830-episode gate's CSV bytes no longer exist (overwritten by later runs,
confirmed via `find`/`git log` — never git-tracked). Stand-in = freshly re-collected
single-seed data from the same PPO checkpoints (the ACTION 2 regression-check run).
`dissertation_log_v2.md` was checked and never quoted a specific number for this metric, only
qualitative framing — no published number is being corrected.

**The categorical BLIND-candidate flag never changes state** in any of the 6 checks (always
True). The fix substantially changes the *magnitude* (roughly halves the fraction at the two
smaller bands, smaller relative change at 80-100) without overturning the categorical verdict.

`membership_leave` shows 0/0/undefined everywhere — expected (only `membership_join` produces
`fired`-phase rows; leave/property events can only touch an already-visible node, per the
structural guarantee established in prior work).

---

## Gate row counts (every filter stage, live 5-seed grid)

| stage | 10-15 | 30-40 | 80-100 | total |
|---|---|---|---|---|
| raw drift CSV rows | 187,647 | 640,230 | 775,122 | 1,602,999 |
| dropped: `touched_node_visible=False` (visibility-lag artefacts) | 3,256 | 10,062 | 11,059 | 24,377 |
| further dropped: `event_phase` not in {immediate, attributed} | 170,877 | 548,336 | 568,610 | 1,287,823 |
| retained for attenuation analysis | 13,514 | 81,832 | 195,453 | 290,799 |
| encoder determinism check (no_change rows, max drift) | 0.00e+00 (170,877 rows) | 0.00e+00 (548,336 rows) | 0.00e+00 (568,610 rows) | — |

`attenuation_episode_aggregates.csv`: 6,305 rows (`membership_leave` 750+1901+1990=4641,
`membership_join` 606+714+344=1664). `response_rate_by_bin.csv`: 81 data rows.

Zero rows skipped for any other reason anywhere in the pipeline — stated explicitly, matching
`gate_summary.txt`'s own B1 accounting (4660 attempted, 4660 completed, 0 skipped).

---

## Per-seed shortfall, band 80-100 (FINDING — statistical-power limitation, not just a total)

| seed | membership_leave (target 200) | membership_join (target 200) | stopped on |
|---|---|---|---|
| 42 | 2158 (exceeded) | 89 | budget cap (400 episodes) |
| 100 | 2296 (exceeded) | 81 | budget cap |
| 123 | (exceeded) | 88 | budget cap |
| 200 | (exceeded) | 72 | budget cap |
| 300 | (exceeded) | 66 | budget cap |

All 5 seeds hit the 400-episode budget cap on `membership_join` without reaching target
(pooled: 396/1000). Same pattern at 30-40, less severe (pooled: 517/1000, individual seeds
150/187/180/187/... — three of five short, two reaching target). **Consequence, stated as a
limitation, not folded into the total**: the large band has systematically less statistical
power for `membership_join` than the small bands — every seed at 80-100 stopped on the budget
cap, not the target, so its `membership_join` sample is thinner and this must be read as a
data-volume caveat on that band/change-type cell specifically, not as evidence the effect itself
is weaker there.

---

## Memory / file-size measurements

| band | file size | rows | peak RSS (pandas read_csv) |
|---|---|---|---|
| 10-15 | 70MB | 187,647 | 330MB |
| 30-40 | 245MB | 640,230 | 927MB |
| 80-100 | 304MB | 775,122 | 1,078MB |

Scaling is sub-linear to linear with row count. All three load comfortably given ~60GB total
system RAM (peak usage <2% of total even at the largest band).

---

## Two separate mechanisms (per explicit instruction — do not merge)

### (a) COVERAGE — joined nodes never discovered, therefore never attributed

A discovery/exploration limit, not a pooling/attenuation result. Leave-vs-join event volume
asymmetry and corrected never-attributed fraction, per band (FINDING):

| band | membership_leave events (2158-type scale) | membership_join events (fired, corrected) | never-attributed fraction (corrected) |
|---|---|---|---|
| 10-15 | (thousands, comfortably exceeds target every seed) | 2124 | 51.6% |
| 30-40 | (thousands, comfortably exceeds target every seed) | 5360 | 82.9% |
| 80-100 | ~2158–2296 per seed alone, comfortably exceeds target | 5577 | 92.0% |

At 80-100 specifically, per-seed: ~2158 and 2296 leave-event counts against 89 and 81
join-events-relevant-toward-target (a different, smaller denominator than the "fired" count
above — the two numbers measure different things: relevant-toward-stopping-target vs.
total-fired-and-tracked). Plainly: **the agent goes on to discover the vast majority of leaving
nodes (it already owns/has seen them), but a large and growing-with-scale fraction of *joined*
nodes are never discovered at all within an episode — this is a coverage/exploration limit,
not a pooling result.**

### (b) ATTENUATION — among changes that WERE attributed, the per-slice response rate

The pooling result, computed only on the subset of events that DID get attributed (i.e., were
actually perceived). `membership_leave`, pooled across all 3 bands, 5 seeds (FINDING):

| slice | overall response rate (tau=0.0) | slope vs n_discovered |
|---|---|---|
| mean | 100.0% | 1.078 [FINDING] |
| full | 100.0% | 1.292 [FINDING] |
| max | 64.4% | 8.850 [ARTIFACT — divides by norm(delta_h_G_slice), exactly 0 for 39-43% of events] |
| min | 60.4% | 10.315 [ARTIFACT, same reason] |

Between-seed spread (mean ± std across 5 seeds), by band:

| band | mean | max | min | full |
|---|---|---|---|---|
| 10-15 | 100.0% ± 0.0% | 98.5% ± 0.5% | 98.5% ± 0.3% | 100.0% ± 0.0% |
| 30-40 | 100.0% ± 0.0% | 84.0% ± 2.0% | 82.8% ± 1.8% | 100.0% ± 0.0% |
| 80-100 | 100.0% ± 0.0% | 43.0% ± 1.5% | 36.1% ± 0.8% | 100.0% ± 0.0% |

Per-slice absolute and relative drift, episode-level median with bootstrap 0.95 CI,
`membership_leave` (FINDING; absolute drift = rel_drift × norm_h1_s, never a norm difference):

| band | slice | relative drift median [CI] | absolute drift median [CI] | between-seed (mean of medians ± std) |
|---|---|---|---|---|
| 10-15 | mean | 0.2157 [0.2106, 0.2210] | 0.1750 [0.1706, 0.1796] | rel 0.2050±0.0074, abs 0.1639±0.0096 |
| 10-15 | full | 0.1674 [0.1619, 0.1730] | 0.3673 [0.3553, 0.3795] | rel 0.1557±0.0170, abs 0.3403±0.0218 |
| 30-40 | mean | 0.0853 [0.0837, 0.0871] | 0.0461 [0.0450, 0.0474] | rel 0.0772±0.0030, abs 0.0416±0.0032 |
| 30-40 | full | 0.0472 [0.0454, 0.0491] | 0.1080 [0.1046, 0.1115] | rel 0.0397±0.0042, abs 0.0924±0.0118 |
| 80-100 | mean | 0.0237 [0.0232, 0.0244] | 0.0098 [0.0094, 0.0103] | rel 0.0223±0.0012, abs 0.0088±0.0002 |
| 80-100 | full | 0.0050 [0.0046, 0.0057] | 0.0135 [0.0127, 0.0146] | rel 0.0037±0.0001, abs 0.0102±0.0002 |

`n_events_pre_at_floor` (norm_h1_s ≤ 1e-9): **0 in every slice/band/change_type cell** —
reported alongside every relative-drift figure per convention, confirmed clean throughout.

`membership_join` mechanism (attenuation_ratio slope, since SNR is structurally undefined for
join — every retained row is `attributed`-phase, an h1→h2 pair with no h3, so
`change_drift_full`/`agent_drift_full` doesn't apply; not a data-volume shortfall):

| slice | slope vs n_discovered |
|---|---|
| full | 0.600 [0.542, 0.658] [FINDING] |
| mean | 0.794 [0.735, 0.851] [FINDING] |
| max | 1.163 [0.920, 1.416] [ARTIFACT] |
| min | 1.162 [0.894, 1.448] [ARTIFACT] |

SNR (membership_leave, two-sided with noise-floor caveat) [FINDING]: 66.4% of steps have a
zero agent-driven noise floor (trivially detectable on those steps — argues against
attenuation, not for thin data); on the remaining steps, SNR at ~100 discovered nodes = 0.492
[0.385, 0.632], log-log slope vs n_discovered = -0.804 [-0.842, -0.765] — SNR near/below 1,
declining with scale.

---

## Comparison to old PPO single-seed gate (orientation only, not a target)

| metric | old (PPO, 1 seed, 830 ep) | new (TRPO, 5 seed, 4660 ep) |
|---|---|---|
| mean/full response rate | ~100% | 100.0% / 100.0% |
| mean/full slope | 1.101 / 1.325 | 1.078 / 1.292 |
| max/min response rate (pooled) | 61.0% / 56.8% | 64.4% / 60.4% |
| max/min ARTIFACT slope | ~9 / ~10 | 8.850 / 10.315 |

Differences are expected (different algorithm, 5× the seeds, corrected event_id methodology
feeding a different visibility picture) and are not tuned toward.

---

## Files written or changed

- `cyberbattle/agents/compute_attenuation_analysis.py` — ACTION 1 fix + option B backup
  (commit `b008aef`)
- `cyberbattle/agents/attenuation_manifest.yaml` — points at the 15-run TRPO grid (unchanged
  in this session, from commit `429d966`)
- `cyberbattle/agents/attenuation_drift_logs/drift_{10-15,30-40,80-100}.csv` — regenerated
  (live 5-seed grid)
- `cyberbattle/agents/attenuation_drift_logs/{shortfall,skip_info}_{band}.json` — regenerated
- `cyberbattle/agents/attenuation_analysis_output/*` — regenerated (gate_summary.txt,
  response_rate_by_bin.csv, attenuation_episode_aggregates.csv, 3 PNGs)
- `attenuation_gate_archive/2026-07-26_trpo_5seed_gate/` — dated archive copy of all of the
  above plus the manifest and `grid_topology_id_map.json` (621MB, untracked, not in git)
- `evidence_cards/evidence_taskT.md` — this file

## Outstanding / not done in this task

- Property and connectivity change types remain untested (structurally impossible /
  mechanism doesn't exist yet — unchanged from prior tasks).
- The 0.3 completeness gap (a zero-row seed silently disappearing from the between-seed
  section rather than being flagged) was noted but not fixed — out of this task's scope.
- Same-band donor pools (the PROVISIONAL banner's caveat) remain a separate, not-yet-started
  follow-up.
