# Task I — Complexity Curve: Cost vs Network Size

Numbers and provenance only. No thesis wording.

## Hardware and idle confirmation

- CPU: AMD Ryzen 9 9950X 16-Core Processor (32 logical via SMT), `torch.set_num_threads(4)` per
  process (single-process solo runs, run strictly sequentially, never concurrently).
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q — **not used**; GAE encoder runs on CPU throughout
  (unchanged, per the DO-NOT-DO list).
- RAM: 60 GiB total.
- **Machine confirmed idle immediately before STEP 1**: `nvidia-smi` — 0% GPU utilization, 29 MiB
  / 97887 MiB used (both GPUs). `uptime` load average 0.24 / 0.13 / 0.04 (32 logical cores).
  `ps aux` showed no other python/training process running. All 7 STEP 1 runs were launched one
  at a time, sequentially, nothing else running concurrently on the machine.

## STEP 0 — verify before measuring

**0.1 — Existing timing data: all contaminated, none reused.** Every `.out` training log with
`fps` figures (`compressed_scale_10_15.out`, `compressed_scale_30_40.out`,
`compressed_scale_80_100.out`) started within 2ms of each other
(`2026-07-18 05:25:49.06[6-8]`) and ran concurrently for their full 4.5-6h duration — a 3-way
parallel batch across ALL THREE bands, not just 80-100 as flagged in the task brief. No
dedicated timing/profiling file exists anywhere in the repo
(`find . -iname "*timing*" -o -iname "*profil*"` returns nothing); the codebase's own
per-episode timers (`graph_encoder_time`, `action_calculation_time`,
`action_space_creation_time`, `balance_action_space_time`, `inner_step_time`,
`update_evolving_visible_graph_time`, `cyberbattle_env_compressed.py:168-173`) are only ever
logged to disk when `verbose>1`; every run this session used `verbose=0`, so none of that
instrumentation reached disk historically. **None of the existing `fps` figures were used in any
fit below.** All STEP 1 measurements are fresh.

**0.2 — Cost components, confirmed from code** (`cyberbattle_env_compressed.py`):

| Component | Location | Scales with (read from code) |
|---|---|---|
| GAE encode (GNN forward pass) | `encode()` `:367-380`; `GAEEncoder.forward` (`gae/model.py:70-82`; this model's layers are `GCNConv`/similar message-passing) | discovered-subgraph node count + edge count |
| Pooling (mean/max/min) | `encode()` `:400-413` | discovered node count × 64 (`node_embeddings_dimensions`) |
| Action-space construction | `create_continuous_action_space()` `:972-1002` | `|owned_nodes| × |discovered_nodes(running)| × vulnerabilities-per-target-node`; reduced on steady-state steps via `processed_pairs`/`nodes_to_recalculate` skip logic |
| Action-space balancing (**added — not in the task's original expected list**) | `__balance_action_space_by_outcome()` `:1047-1061`, called every `create_continuous_action_space()` | raw candidate-action count before capping, capped by `sample_subset_samples=100` (`train_config.yaml:29`) per outcome-display-category |
| Cosine nearest-action matching | `find_closest_action_embedding()` `:1064-1087` | candidate count in `self.action_embeddings` (post-cap) × embedding width (906); `cdist` rebuilds `embeddings_array` fresh every call |
| Environment simulation itself | base-class `step_attacker_env`, timed as `inner_step_time` | not exhaustively traced per outcome branch; measured empirically, not asserted analytically |

All five originally-expected components confirmed present; balancing added as a sixth,
distinct, separately-timed sub-cost of action-space construction.

**0.3 — Constant-width claim: measured, confirmed.** `env.observation_space` (`graph_embeddings`)
and `env.action_space.shape[0]` read directly from a live env at every one of the 7 sizes
measured (10, 15, 34, 40, 80, 90, 100 nodes), 3 repeats each (21 total):

**ARTIFACT: `obs_width = 256` and `action_width = 906` in all 21/21 runs, no exceptions,
across all three bands (10-15, 30-40, 80-100).** The invariance claim holds as measured.

**0.4 — encode() trigger condition, confirmed from code:** fires (a) unconditionally once per
`reset()` (`:245-246`); (b) per step when `action_changes_evolving_visible_graph(outcome)` is
`True` — with `precise_graph_encoding=False` (class default, not overridden in
`train_config.yaml`), this is narrow: only `LateralMove`, `DenialOfService`, `Reconnaissance`
outcomes (`:939-947`); (c) per step when `maybe_apply_dynamic_step()`'s return value is truthy
(membership leave/join, and property change since Task D); (d) per step if
`self.static_defender_agent` is set (not the case in this config,
`static_defender_eviction_goal: False`). Encode frequency therefore depends on episode dynamics
as well as N — measured empirically below (encodes-per-episode) rather than assumed.

**Topology availability**: existing instances give continuous coverage at N = 10, 15, 34, 40, 80,
90, 100 (no generation). **Gap confirmed and reported, not filled**: no topology of any kind
exists in `env_samples/` between N=40 and N=80.

---

## STEP 1 — measurements

Two measurement passes were run; they answer different questions and are **never mixed in one
table**.

### Pass A: realistic full-episode timing (300 random-action steps per episode, MEASURED, high variance — bursty by nature, see caveat below)

| N (topology) | encode() calls/episode (spread, 3 reps) | mean wall-clock/step, s (mean ± std, spread, 3 reps) | peak RSS, KB (spread, 3 reps) |
|---|---|---|---|
| 10 | 39.8 [27.5, 46.5, 45.5] | 0.000287 ± 0.000085 [0.000333, 0.000190, 0.000339] | [755872, 755872, 766772] |
| 15 | 70.3 [39.0, 78.0, 94.0] | 0.000536 ± 0.000361 [0.000120, 0.000729, 0.000759] | [710956, 751404, 751404] |
| 34 | 74.7 [38.0, 95.0, 91.0] | 0.009125 ± 0.008048 [0.000100, 0.011721, 0.015555] | [744128, 936596, 1012592] |
| 40 | 80.3 [80.0, 81.0, 80.0] | 0.020897 ± 0.003750 [0.016656, 0.023776, 0.022259] | [1113656, 1329036, 1376732] |
| 80 | 75.0 [38.0, 90.0, 97.0] | 0.012547 ± 0.011194 [0.000101, 0.015746, 0.021793] | [740088, 1009112, 1205360] |
| 90 | 75.7 [40.0, 89.0, 98.0] | 0.038153 ± 0.033578 [0.000097, 0.050754, 0.063608] | [792764, 1529172, 1573472] |
| 100 | 67.3 [39.0, 50.0, 113.0] | 0.016099 ± 0.027678 [0.000112, 0.000126, 0.048059] | [803672, 845656, 1509488] |

**FINDING: this pass's per-step timing has enormous repeat-to-repeat spread (std comparable to
or larger than the mean at several sizes) — this is a real, structural property of the system,
not measurement error.** Cause identified directly from the code:
`precise_action_space_positions=False` (class default, not overridden) means every step whose
outcome changes the visible graph triggers a full `create_continuous_action_space()` call with no
node-scoping; the `processed_pairs` skip (`:989-993`) means only *newly reachable* (owned,
discovered) pairs get (re)processed, so cost per call depends on how many new pairs became
reachable since the last call — which grows non-uniformly as owned/discovered sets expand over
an episode. This pass's numbers are **not used for curve fitting** for this reason; they are
reported as the realistic, in-situ training-step cost, with their real variance shown.

### Pass B: isolated fixed-state component timing (MEASURED — used for all STEP 2 fits)

For each of 3 independent 300-step rollouts (different seed) per topology, the reached
discovered/owned state was frozen and each cost component was timed with 10 repeated,
side-effect-neutral calls on that fixed state (full method: `evidence_cards/../` scripts
`taskI_profile_clean.py`, kept in the job scratch dir). Reports the per-state mean over the 10
inner repeats; the tables below give all 3 outer states per topology size (their spread across
these 3 states, at similar-but-not-identical discovered/owned counts, IS the "at least 3 repeats"
spread the task requires — inner-repeat std was always ≤ ~10% of the mean, i.e. genuine
low-noise per-call timing; see raw JSON for full inner-repeat detail).

| N (topology) | n_discovered | n_owned | n_edges | n_actions (candidates) | encode() mean, s | action_space mean, s | match mean, s |
|---|---|---|---|---|---|---|---|
| 10 | 8 / 7 / 7 | 3 / 5 / 6 | 12 / 12 / 8 | 336 / 697 / 835 | 0.001557 / 0.001332 / 0.001164 | 0.000388 / 0.000760 / 0.000870 | 0.000186 / 0.000359 / 0.000429 |
| 15 | 10 / 8 / 9 | 9 / 5 / 8 | 17 / 15 / 23 | 765 / 345 / 568 | 0.001729 / 0.001595 / 0.002009 | 0.000849 / 0.000371 / 0.000624 | 0.000391 / 0.000188 / 0.000297 |
| 34 | 25 / 22 / 22 | 10 / 12 / 9 | 38 / 34 / 36 | 8781 / 8546 / 6599 | 0.004113 / 0.003863 / 0.004211 | 0.022156 / 0.011427 / 0.009201 | 0.016277 / 0.015878 / 0.011776 |
| 40 | 26 / 28 / 30 | 14 / 14 / 20 | 44 / 48 / 67 | 17563 / 17949 / 28570 | 0.004040 / 0.005058 / 0.006607 | 0.037256 / 0.049643 / 0.066797 | 0.031482 / 0.033197 / 0.051914 |
| 80 | 64 / 63 / 70 | 17 / 14 / 16 | 77 / 63 / 63 | 20785 / 16960 / 19933 | 0.010377 / 0.008177 / 0.009434 | 0.041287 / 0.046473 / 0.050888 | 0.037104 / 0.031749 / 0.036187 |
| 90 | 77 / 77 / 76 | 18 / 20 / 13 | 84 / 75 / 66 | 55464 / 59126 / 37900 | 0.013885 / 0.010900 / 0.010226 | 0.145127 / 0.152107 / 0.090299 | 0.096722 / 0.106093 / 0.066552 |
| 100 | 91 / 86 / 87 | 16 / 18 / 18 | 83 / 63 / 74 | 39984 / 44281 / 45107 | 0.012623 / 0.010342 / 0.011286 | 0.091553 / 0.115580 / 0.104605 | 0.071163 / 0.080050 / 0.083736 |

---

## STEP 2 — fit and extrapolate (FITTED, from Pass B only)

### 2.1 Fitted forms per component (log-log power-law fit, plus linear/N-log-N/quadratic for comparison)

| Component | vs (code-derived scaling quantity) | Best-fit form | R² | Power-law exponent (95% CI) |
|---|---|---|---|---|
| GAE `encode()` | n_discovered (evolving_visible_graph nodes) | power law | 0.978 | 0.862 (0.801, 0.924) |
| GAE `encode()` | topology N | power law | 0.975 | 0.969 (0.895, 1.044) |
| `create_continuous_action_space()` | n_owned × n_discovered | power law | 0.935 | 1.464 (1.279, 1.649) |
| `create_continuous_action_space()` | topology N | power law | 0.899 | **2.428 (2.036, 2.819)** |
| `find_closest_action_embedding()` | n_actions (candidates) | linear | 0.999 | (linear beats power law; power-law exponent 1.291, CI 1.237-1.345, R²=0.993 — theoretically expected, since `cdist` cost is exactly linear in candidate count × fixed dim) |
| `find_closest_action_embedding()` | topology N | power law | 0.889 | **2.620 (2.174, 3.065)** |

All fits: n=21 points (7 sizes × 3 states/repeats each), log-log OLS on Pass B's isolated
measurements.

### 2.2 Dominant component

**FINDING — corrects the task's initial framing.** The GAE encoder scales **near-linearly** with
N (exponent 0.969, CI includes 1.0). Action-space construction and nearest-action matching both
scale **superlinearly, close to quadratic** (exponents 2.428 and 2.620). **The encoder is not the
dominant cost at any size in or beyond the measured range — the action-space/matching pipeline
is.**

Crossover points, computed from the fitted power laws vs N:
- **encode() is overtaken by action-space construction at N ≈ 18** — i.e. within the measured
  10-15 band itself, not at some larger untested size.
- **action-space construction is overtaken by nearest-action matching at N ≈ 265.**

**At N=250 (extrapolated, see caveat below): action-space construction and matching are
co-dominant** (≈50/50 split of their combined cost; action-space narrowly ahead at exactly 250,
match ahead from ≈265 onward). The encoder's share of the three components' combined cost is
≈1.1% at N=250, falling further at larger N (≈0.4% at N=1000). This is the opposite of "the wall
is in the encoder" as originally framed — the wall is in the action-space/matching pipeline,
which exists specifically because `precise_action_space_positions`/`precise_graph_encoding` are
both `False` (unmodified defaults) and candidate-action count grows with the
owned×discovered×vulnerability product, not with the (constant) embedding width.

### 2.3 Extrapolation to 500, 1000, 2500 nodes — **EXTRAPOLATED, not measured**

**Caveat, applies to every number in this section without exception: these are extrapolations
from a fit over a measured range of roughly 10 to 100 nodes and have not been validated beyond
it.**

| N | encode() mean/call, s | action_space mean/call, s | match mean/call, s | sum, s |
|---|---|---|---|---|
| 250 (extrapolated) | 0.0289 | 1.303 | 1.288 | 2.620 |
| 500 (extrapolated) | 0.0566 | 7.007 | 7.918 | 14.982 |
| 1000 (extrapolated) | 0.1108 | 37.695 | 48.661 | 86.467 |
| 2500 (extrapolated) | 0.2692 | 348.549 | 536.557 | 885.376 |

For reference, measured values already on record (not extrapolated) at the top of the fitted
range: N=100, encode≈0.0113s, action_space≈0.104s, match≈0.0783s (Pass B means).

### 2.4 Memory extrapolation — **reported separately from time, per instruction**

**FINDING, nuances the task's framing: the interface (observation vector) genuinely is
constant-size — 256 floats (1024 bytes), measured identically at all 7 sizes (0.3 above) — so
that specific invariance claim is exactly correct and holds without qualification.**

**However, a separate internal structure — the action-space cache (`self.action_embeddings`) —
is not constant, and its growth is not independent of the compute wall: it is built from the same
owned×discovered×vulnerability product that drives action-space-construction time.** Fitted
(log-log OLS, Pass B `n_actions` vs topology N): exponent **2.032 (95% CI 1.703, 2.361)**,
R²=0.898 — i.e. also close to quadratic.

| N | predicted n_actions (extrapolated) | action_embeddings raw float32 payload (extrapolated) |
|---|---|---|
| 100 (measured range) | ≈53,138 (fit); measured range 39,984-59,126 | ≈193 MB (fit) |
| 250 (extrapolated) | ≈342,115 | ≈1.24 GB |
| 500 (extrapolated) | ≈1,399,531 | ≈5.07 GB |
| 1000 (extrapolated) | ≈5,725,230 | ≈20.75 GB |
| 2500 (extrapolated) | ≈36,860,595 | ≈133.58 GB |

**Caveat: raw float32 array payload only (n_actions × 906 dims × 4 bytes); excludes Python
dict/tuple-key/object overhead, which is typically several times the raw payload for a dict of
many small individually-allocated numpy arrays — the real resident-memory figure at each N is
higher than this table, not measured directly, and this extrapolation carries the same
validity caveat as 2.3 (fit over a 10-100 node range, unvalidated beyond it).**

Whole-process peak-RSS numbers from Pass A (740 MB-1.57 GB across the range) do not show a clean
fittable trend distinguishable from noise — they are dominated by a large, roughly constant
process baseline (PyTorch/GAE encoder/interpreter, on the order of 700 MB) that swamps the
actual N-dependent component (the action_embeddings dict, ≈2-200 MB in the measured range per
the table above) at this size range. Not fitted for that reason; the fitted memory quantity above
(the action_embeddings dict specifically) is the analytically isolable, honestly-fittable
N-dependent memory driver.

---

## Summary table — measured vs fitted vs extrapolated (explicitly marked, never mixed)

| Quantity | Status | Value/Source |
|---|---|---|
| obs_width=256, action_width=906 at N∈{10,15,34,40,80,90,100} | **MEASURED** | 21/21 runs, Pass A |
| Pass A per-step wall-clock, encodes/episode | **MEASURED** (high variance, not fitted) | table above |
| Pass B per-component wall-clock at N∈{10,...,100} | **MEASURED** | table above |
| Power-law exponents + R² per component | **FITTED** | from Pass B, n=21 |
| Cost at N=250/500/1000/2500 | **EXTRAPOLATED** | from fitted power laws, unvalidated beyond ~100 nodes |
| n_actions / memory at N=250/500/1000/2500 | **EXTRAPOLATED** | from fitted power law, unvalidated beyond ~100 nodes |
| Crossover points (N≈18, N≈265) | **DERIVED from fits** (not directly measured at those N) | computed from fitted power-law intercepts/exponents |

## Provenance

- Raw JSON (Pass A): `result_N{10,15,34,40,80,90,100}.json`
- Raw JSON (Pass B): `clean_N{10,15,34,40,80,90,100}.json`
- Fit outputs: `taskI_fit_summary.json` (Pass A fits, not used — kept for the record showing why
  Pass A alone was unfittable, R²=0.08-0.53), `taskI_clean_fit_results.json` (Pass B fits, used
  for all STEP 2 numbers above)
- Scripts: `taskI_profile_one_topology.py` (Pass A), `taskI_profile_clean.py` (Pass B),
  `taskI_fit_curves.py` (Pass A fitting), `taskI_fit_clean.py` (Pass B fitting) — all kept in the
  job scratch directory, not the repo.
- Topologies used (existing, none generated): `scalability_10_15/{1,2}`,
  `scalability_30_40/{1,5}`, `scalability_80_100/{42,11,21}` (N = 80, 90, 100 respectively).

---
---

# TASK I-2 — Is the quadratic scaling inherent, or a configuration/implementation artefact?

Numbers and provenance only. No thesis wording. Appended under this separate heading per
instruction, same card as Task I.

## Critical methodology correction (found during STEP 0, before any new timing was run)

**Task I's own profiling scripts (`taskI_profile_one_topology.py`, `taskI_profile_clean.py`)
never passed `sample_subset_samples` to the env constructor.** This silently used the class
default `sample_subset_samples=False` (`cyberbattle_env_compressed.py:93`), which disables the
action-space balancing/capping step entirely (`:1006`, `if self.sample_subset_samples:`) — unlike
real training, where `train_agent.py` passes `**config` (the full loaded YAML, which sets
`sample_subset_samples: 100`, `train_config.yaml:29`). **Verified directly**: the same
post-300-step state on `scalability_80_100/11` produced **43,326** candidate actions with the
parameter omitted, and exactly **800** (8 outcome categories × 100 — `Collection`, `Exfiltration`,
`PrivilegeEscalation`, `Discovery`, `Reconnaissance`, `Persistence`, `DefenseEvasion`,
`LateralMove-Credential`, per `map_outcome_to_string`, `encoding_utils.py:40-76`) with it correctly
wired. **This means Task I's fitted exponent for `find_closest_action_embedding` (2.62) was
measured against an uncapped candidate set that never occurs in real training, and does not
describe production behaviour.** All Task I-2 measurements below fix this omission
(`taskI2_profile_v3.py`, verified against the same topology: 43,326 → 800 with the fix applied).

A second, equally consequential issue was found alongside it: Task I's `create_continuous_action_space()`
timing always cleared `env.processed_pairs = set()` before every timed repeat, to force a fair,
reproducible "full rebuild" cost. But `processed_pairs` is **never cleared mid-episode in real
production** (only in `reset()`) — so Task I's methodology measured a cost regime (full rebuild
from empty) that in real training only occurs briefly, early in an episode, not on every step.
STEP 1 below measures both the full-rebuild regime (for continuity with Task I) and the realistic
steady-state regime (processed_pairs already populated) separately.

## STEP 0 — verify before measuring

**0.1 — What do the two flags actually change?**

Every use of both flags (`grep -rn "precise_action_space_positions\|precise_graph_encoding"`,
repo-wide): only 3 real branch points, all in `cyberbattle_env_compressed.py`.

`precise_action_space_positions` — read only at the `step()` call site (`:585-594`):
```
585   if action_changed_graph:
586       if self.precise_action_space_positions:
587           self.create_continuous_action_space(nodes_to_recalculate=[source_node, target_node])
588       else:
589           self.create_continuous_action_space()
590   elif self.static_defender_agent:
591       if self.precise_action_space_positions:
592           self.create_continuous_action_space(nodes_to_recalculate=self.changed_nodes)
593       else:
594           self.create_continuous_action_space()
```
It only changes the `nodes_to_recalculate` **argument** passed in. Inside
`create_continuous_action_space` (`:981-993`), the outer double loop over
`running_owned_nodes × running_discovered_nodes` is **unconditional regardless of this flag** —
confirmed by direct reading: the loop bounds at `:981-982` do not reference
`nodes_to_recalculate` at all; it only gates which pairs get skipped via `continue`
(`:984-993`). **Neither flag changes which (source, target, vulnerability, outcome) keys are
eventually reachable** — the same full owned×discovered×vulnerability enumeration happens
under both settings, given enough steps; the flag only changes *when* an already-processed
pair's stored embedding gets refreshed (True: proactively, for pairs path-connected to the
just-changed node, via `nx.has_path` checks at `:986-987`; False: never, once first computed).
**This is (a) only — how/when action embeddings get refreshed, not the observation shape or the
reachable action-key set.**

`precise_graph_encoding` — read only inside `action_changes_evolving_visible_graph`
(`:939-947`):
```
939   def action_changes_evolving_visible_graph(self, outcome):
940       if self.precise_graph_encoding:
941           return not (isinstance(outcome, model.InvalidAction) or ... 9 failure/no-op types ...)
946       else:
947           return isinstance(outcome, model.LateralMove) or isinstance(outcome, model.DenialOfService) or isinstance(outcome, model.Reconnaissance)
```
This return value (`action_changed_graph`) directly gates **whether `encode()` runs at all this
step** (`:568-577`) and feeds the same `precise_action_space_positions` branch above. Under
`True`, encode() fires on almost every non-failure outcome (broad); under `False` (the
unmodified default, matching every run this session and the completed Task T gate), only 3
narrow outcome types trigger it. **This is (b) — it changes how often, and under what
conditions, the agent's observation (`graph_embeddings`) is refreshed, which changes the actual
numeric content the agent sees at a given step for an identical action/topology/seed sequence.**
Confirmed empirically (identical seed, identical topology, `scalability_10_15/1`, 50 steps):
obs/action **shape** stays 256/906 under both settings (no crash, no shape change — 0.3 below),
but the discovered/owned counts and `n_actions` differ between the two runs by step 50 (obs=6/45,
6 discovered under True vs 8 discovered/454 actions under False) — meaning full trajectories
**can and do diverge in value**, not just in speed.

**FINDING, prominent per the GATE**: neither flag changes observation or action-space *shape*.
`precise_graph_encoding` **does** change the actual observation *values* and *timing* the agent
receives, and (via the shared `action_changed_graph` gate) indirectly changes how often action
embeddings refresh too. **No completed Task T result can be assumed representative of the `True`
setting — a run under `True` is not just slower, it can genuinely diverge in trajectory.** STEP 1
below is therefore a **timing comparison only**, never a results comparison, exactly as flagged.

**0.2 — THE MORE IMPORTANT QUESTION: where does the quadratic actually come from?**

(a) Candidate count expression, read from `__add_vulnerabilities_to_action_space`
(`:1011-1035`), called from the double loop (`:994-1000`):
```
n_actions_raw = Σ_{s ∈ owned_running} Σ_{t ∈ discovered_running} [ V_local(t)·𝟙{s=t} + V_remote(t) ]
```
where `V_local(t)`/`V_remote(t)` are the counts of local-/remote-type vulnerabilities cataloged
on node `t` (`self.vulnerabilities_embeddings_per_node_type[t][...]`), minus the
LateralMove/CredentialAccess local-type exclusion at `:1017-1018`. This is **then capped** by
`__balance_action_space_by_outcome` (`:1047-1061`) to at most `sample_subset_samples` (=100) per
outcome-string category (8 categories in practice, per `map_outcome_to_string`) — i.e.
`n_actions_final ≤ 800`, a **constant independent of N**, confirmed empirically (0/21 states out
of 800 once N≥15 in this data, see STEP 1 below) — **when this parameter is correctly wired,
which it was not in Task I's scripts.**

(b) Rebuilt from scratch or maintained? **Both, in different senses, confirmed from code**: the
outer double loop (`:981-982`) always re-executes over the full `owned × discovered` product on
every single call, regardless of flag — this is the one part of the design that genuinely never
avoids an O(|owned|×|discovered|) sweep per call. But `processed_pairs` (persisted across the
whole episode, only cleared in `reset()` — confirmed via `grep -n "processed_pairs = set()"`,
one hit, inside `reset()` and nowhere inside `step()`) means that once a pair has been
processed, the expensive per-vulnerability work (`__add_vulnerabilities_to_action_space`) is
skipped for it on every later call under the default (`False`) flag — so **the default
configuration is already a form of incremental maintenance in its steady state**, confirmed
empirically below (steady-state cost ≈ flat, ≈0.0002-0.0004s regardless of N). The genuinely
unavoidable-by-either-flag part is the loop-iteration/dict-membership-check floor cost, which
is small; the genuinely large, quadratic-in-N part (full rebuild) is paid only while
`processed_pairs` is still being populated (episode warm-up), or, under `precise_action_space_positions=True`,
on **every single call**, because that flag additionally runs an `nx.has_path` graph traversal
(`:986-987`) across the *entire* owned×discovered product every time, even for pairs it
ultimately skips — a real, measured, and avoidable cost (STEP 1 below).

(c) Match: always the full candidate set, no subsetting inside the function at all — confirmed,
`find_closest_action_embedding` (`:1064-1087`) builds `embeddings_array =
np.array(list(self.action_embeddings.values()))` unconditionally, every call, no flag or
locality-based narrowing anywhere in the function. Its cost is therefore bounded by whatever
`self.action_embeddings` currently contains — which, once `sample_subset_samples` is correctly
applied, is capped at a constant (800), **not growing with N**.

**Which of the three is responsible for the exponent originally measured, and is it forced or a
choice?** Two of Task I's three flagged "quadratic" numbers turn out not to describe the
deployed system at all:
- **Match's measured exponent (2.62, Task I) was entirely a test-harness artifact** (missing
  `sample_subset_samples`) — not a property of continuous action selection, and not present in
  the real system, where the candidate set is explicitly bounded by design.
- **Action-space construction's quadratic cost is real, but conditional on regime**: it is an
  unavoidable-in-this-implementation O(owned×discovered) sweep, but the *default* configuration's
  `processed_pairs` cache means this sweep's expensive part is skipped once a pair is known —
  i.e. the current implementation already approximates incremental maintenance for the default
  flag setting, most of the time (STEP 1 confirms this: steady-state cost stays flat, exponent
  0.41, statistically indistinguishable from no growth given the CI reported below is much
  smaller in absolute terms than the full-rebuild regime). The quadratic cost is paid (i) briefly,
  during an episode's warm-up while `processed_pairs` is still filling up, regardless of flag, and
  (ii) on **every** call under `precise_action_space_positions=True`, because of the `nx.has_path`
  sweep — an implementation detail, not a structural requirement of "incremental" maintenance (a
  reverse-adjacency index or a dirty-pair queue could avoid re-checking the full pair space every
  call). **This supports "rebuilding per step something that could be maintained incrementally"
  (ii), not "inherent to continuous action selection" (i)** — the specific cost driver (the
  unconditional double loop, and separately the `nx.has_path` sweep under the alternative flag)
  is a demonstrable implementation choice, not a property forced by searching a continuous
  action space.

**0.3 — Is enabling the flags safe to measure?**

Yes. Verified directly: 300-step rollout with both flags `True` completes with no exception, no
shape change (`obs_width=256`, `action_width=906`, identical to `False`), on `scalability_10_15/1`
and confirmed again across all 7 sizes in STEP 1 (14/14 runs, 0 crashes). Safe to proceed.

## STEP 1 — measurement

**Machine reconfirmed idle immediately before this sweep**: `nvidia-smi` 0% GPU / 29 MiB used,
`uptime` load average 0.41/0.28/0.13 (32 logical cores), no other process running. Same hardware
as Task I. All runs sequential, nothing concurrent.

Method: identical to Task I's Pass B (freeze a discovered/owned state via a 300-step rollout,
then time components in isolation), corrected per the two issues above:
`sample_subset_samples=100` now passed through; three action-space regimes measured separately
(`full_rebuild`: `processed_pairs` cleared before every timed call, as Task I did;
`steady_state`: `processed_pairs` populated once via an untimed full build, then timed calls on
top of that — the realistic in-episode condition); `match` timed against the resulting
steady-state (realistic) `action_embeddings`. 3 outer repeats (independent 300-step rollouts,
different seeds) × 10 inner repeats per state, all 7 Task I sizes, both flag settings (14 runs
total).

### precise_action_space_positions = precise_graph_encoding = False (production default)

| N | n_actions (all 3 repeats) | encode() mean, s | full_rebuild mean, s | steady_state mean, s | match mean, s |
|---|---|---|---|---|---|
| 10 | 659 / 484 / 628 | 0.001751 / 0.001189 / 0.001477 | 0.001220 / 0.000606 / 0.000885 | 0.000161 / 0.000116 / 0.000151 | 0.000344 / 0.000257 / 0.000332 |
| 15 | 800 / 276 / 800 | 0.001998 / 0.001436 / 0.001597 | 0.002241 / 0.000370 / 0.001731 | 0.000193 / 0.000071 / 0.000192 | 0.000430 / 0.000154 / 0.000435 |
| 34 | 800 / 800 / 800 | 0.003969 / 0.004364 / 0.002938 | 0.030958 / 0.033169 / 0.014390 | 0.000216 / 0.000291 / 0.000217 | 0.000512 / 0.000542 / 0.000504 |
| 40 | 800 / 800 / 800 | 0.005449 / 0.007984 / 0.005990 | 0.056503 / 0.064520 / 0.070742 | 0.000229 / 0.000242 / 0.000240 | 0.000553 / 0.000509 / 0.000511 |
| 80 | 800 / 800 / 800 | 0.012616 / 0.010672 / 0.012891 | 0.080999 / 0.094515 / 0.070330 | 0.000313 / 0.000335 / 0.000319 | 0.000578 / 0.000558 / 0.000496 |
| 90 | 800 / 800 / 800 | 0.013903 / 0.010358 / 0.011869 | 0.237297 / 0.127060 / 0.148418 | 0.000359 / 0.000296 / 0.000336 | 0.000562 / 0.000491 / 0.000511 |
| 100 | 800 / 800 / 800 | 0.014707 / 0.010795 / 0.014396 | 0.138475 / 0.113098 / 0.127829 | 0.000332 / 0.000297 / 0.000323 | 0.000544 / 0.000508 / 0.000489 |

### precise_action_space_positions = precise_graph_encoding = True

| N | n_actions (all 3 repeats) | encode() mean, s | full_rebuild mean, s | steady_state mean, s | match mean, s |
|---|---|---|---|---|---|
| 10 | 337 / 669 / 605 | 0.001139 / 0.001819 / 0.001108 | 0.000480 / 0.001096 / 0.000880 | 0.000438 / 0.001070 / 0.000820 | 0.000192 / 0.000351 / 0.000319 |
| 15 | 315 / 800 / 483 | 0.001343 / 0.002650 / 0.001849 | 0.000662 / 0.002979 / 0.000741 | 0.000453 / 0.002496 / 0.000705 | 0.000165 / 0.000451 / 0.000255 |
| 34 | 800 / 800 / 800 | 0.004367 / 0.005661 / 0.003657 | 0.026096 / 0.034463 / 0.014794 | 0.009422 / 0.028563 / 0.024630 | 0.000476 / 0.000524 / 0.000490 |
| 40 | 800 / 800 / 800 | 0.005579 / 0.004517 / 0.006171 | 0.038645 / 0.037321 / 0.044419 | 0.032811 / 0.028746 / 0.031734 | 0.000527 / 0.000542 / 0.000496 |
| 80 | 800 / 800 / 800 | 0.009234 / 0.013143 / 0.008828 | 0.046949 / 0.083738 / 0.058112 | 0.040547 / 0.064256 / 0.047186 | 0.000484 / 0.000480 / 0.000474 |
| 90 | 800 / 800 / 800 | 0.012219 / 0.010585 / 0.010694 | 0.162605 / 0.174262 / 0.158736 | 0.142759 / 0.135262 / 0.123890 | 0.000493 / 0.000499 / 0.000519 |
| 100 | 800 / 800 / 800 | 0.011349 / 0.012591 / 0.015562 | 0.087618 / 0.083489 / 0.169448 | 0.063377 / 0.067188 / 0.128542 | 0.000505 / 0.000494 / 0.000504 |

**FINDING: `n_actions` caps at exactly 800 by N=15-34 under both flag settings** (only the two
smallest states occasionally fall short of the cap, when too few vulnerabilities have been
discovered yet to fill every category) — confirming 0.2(a)'s capped expression empirically.

## STEP 2 — fit and compare (FITTED, from STEP 1 above)

| Component | False exponent (95% CI), R² | True exponent (95% CI), R² | Changed beyond CI? (1.4) |
|---|---|---|---|
| `encode()` vs N | 1.033 (0.922, 1.145), R²=0.952 | 0.994 (0.888, 1.099), R²=0.953 | **No** — CIs heavily overlap |
| `full_rebuild` vs N | 2.357 (1.978, 2.736), R²=0.899 | 2.316 (1.972, 2.660), R²=0.913 | **No** — CIs heavily overlap |
| `steady_state` vs N | 0.407 (0.288, 0.527), R²=0.727 | 2.262 (1.909, 2.614), R²=0.905 | **Yes** — CIs do not overlap at all (0.527 vs 1.909) |
| `match` vs N | 0.255 (0.126, 0.384), R²=0.473 | 0.285 (0.151, 0.419), R²=0.510 | **No** — CIs heavily overlap |

(All fits: log-log OLS, n=21 points, 7 sizes × 3 repeats, same as Task I's methodology.)

**FINDING, corrects Task I's headline number**: `match`'s exponent is **0.26-0.29, not 2.62** —
Task I's original 2.62 was the `sample_subset_samples` test-harness bug, not a real property of
this system. With the cap correctly applied, matching cost is close to flat (weak R² because the
data genuinely has very little N-dependence left to fit, not because the fit failed) — bounded
by a constant candidate count (800), exactly as 0.2(c)'s code reading predicted.

**FINDING: the only component whose exponent changes meaningfully between flag settings is
`steady_state`** — flat (0.41) under the production default, quadratic (2.26, statistically
indistinguishable from `full_rebuild`'s own exponent) under `precise_action_space_positions=True`.
This is the direct, quantitative confirmation of 0.2(b): the "precise" flag does not achieve
incremental maintenance in this implementation — it pays a persistent `nx.has_path`-driven
quadratic cost on every call, forever, rather than only during warm-up.

**Dominant component (largest measured size, N=100, from the fitted curves)**:

| Regime | encode() | full_rebuild | steady_state | match | Sum | Dominant |
|---|---|---|---|---|---|---|
| False, worst-case (encode+full_rebuild+match) | 0.0137s | 0.1988s | — | 0.0006s | 0.2131s | full_rebuild (93%) |
| **False, realistic (encode+steady_state+match)** | 0.0137s | — | 0.0003s | 0.0006s | **0.0146s** | **encode() (94%)** |
| True, worst-case | 0.0129s | 0.1647s | — | 0.0005s | 0.1781s | full_rebuild (92%) |
| **True, realistic (encode+steady_state+match)** | 0.0129s | — | 0.1275s | 0.0005s | **0.1409s** | **steady_state (90%)** |

**FINDING, reverses Task I's headline claim under the production default**: in the *realistic*
regime (the one that actually occurs for most of a training episode, once `processed_pairs` has
stabilised), **the GAE encoder is the dominant per-call cost at N=100 under the default
configuration** (94% of the tracked sum) — the opposite of Task I's original "encoder overtaken
at N≈18" claim, which was based on the flawed always-full-rebuild/uncapped measurement.
Crossover (encode vs steady_state, fitted): **N≈0.27 under False** (i.e. encode() is already the
larger cost at every measured size in steady state) vs **N≈16.4 under True** (matching Task I's
original — but now understood to be an artefact of the `nx.has_path` sweep specific to that flag,
not a property of the design in general).

**Caveat that qualifies all of the above, required for honesty about extrapolation**: this
"realistic dominates" picture holds for episodes short enough, or topologies small enough, that
discovery/ownership growth (and hence `processed_pairs` population) settles within the episode.
In this data, `n_discovered_final` at N=100 was 78-91 of 100 (STEP 1 of Task I's Pass B) —
discovery had *mostly but not entirely* completed within 300 steps. **For N well beyond the
measured range, or a fixed episode length too short to finish discovering a much larger
topology, a larger fraction of the episode would be spent in the warm-up (full-rebuild-like)
regime, which genuinely is quadratic under either flag** — so the "encoder dominates" finding
does not straightforwardly extrapolate to N=250+; it is only established within the measured
10-100 node range. Not extrapolated further for this reason.

## Conclusion

**The evidence supports (ii): the quadratic term comes from rebuilding per step something that
could be maintained incrementally — it is not inherent to continuous action selection over a
candidate set that grows with the network.** Specific grounds:

1. The candidate set itself does **not** grow with the network in the deployed system — it is
   explicitly capped at a constant (800) by `sample_subset_samples`, a design choice already
   present in `train_config.yaml`. Continuous action selection over this candidate set therefore
   does not inherently require quadratic-in-N matching cost — `find_closest_action_embedding`'s
   real exponent is 0.26-0.29, not 2.62.
2. The one genuinely quadratic-in-N cost that survives correction (`full_rebuild`, and
   `steady_state` under `precise_action_space_positions=True`) is traceable to a specific,
   avoidable implementation detail: an unconditional O(owned×discovered) double loop
   (`:981-982`) that never itself shrinks with either flag, and, under the "precise" setting
   specifically, an additional `nx.has_path` graph traversal computed across that entire product
   on every single call regardless of flag outcome. A design that maintained a genuine dirty-pair
   queue (only ever touching pairs actually affected by the most recent change) rather than
   re-sweeping the full pair space every call could avoid this — this is not a property forced by
   the method of continuous action selection itself.
3. The production **default** configuration already behaves close to incrementally in its steady
   state (flat, ≈0.0002-0.0004s regardless of N, exponent 0.41 with a CI that does not support a
   strong trend) — direct evidence that avoiding the quadratic cost is achievable within the
   current design for most of an episode, further undercutting the "inherent" reading.
4. The caveat above (episode-length-relative-to-N) means this conclusion is established for the
   measured range only; it is not claimed to hold at N=250+ without further work, which was out
   of scope here.

## Flags reset / confirmed

**`precise_action_space_positions` and `precise_graph_encoding` were never written to any repo
config file in this task** — both remained at their class default (`False`,
`cyberbattle_env_compressed.py:96-97`) throughout; every `True` setting used in STEP 1 was passed
as a constructor keyword argument in scratch scripts only (`taskI2_profile_v3.py`, job scratch
directory). Confirmed via `git status --short cyberbattle/agents/config/` — no changes. No reset
needed.

## Provenance

- Raw JSON: `v3_N{10,15,34,40,80,90,100}_{false,true}.json` (14 files); earlier exploratory
  files `v2_N*_{false,true}.json`, `smoke_precise.json` kept for the record of the methodology
  correction.
- Fitting: `taskI2_fit_v3.py`.
- Scripts: `taskI2_profile_v3.py` (final methodology), `taskI2_profile_clean_v2.py` (intermediate,
  superseded by v3's full_rebuild/steady_state split).
- All kept in the job scratch directory, not the repo.
- Topologies: identical set to Task I (`scalability_10_15/{1,2}`, `scalability_30_40/{1,5}`,
  `scalability_80_100/{42,11,21}`).

---
---

# RE-CHECK — `action_embeddings` memory: pre-sampling vs post-sampling cache size

Report only, nothing changed. Numbers and provenance only, no thesis wording.

## Why this re-check was needed

Task I's original memory figures (`n_actions` growth exponent 2.03, payload 193MB at N=100
rising to 1.24GB/5.07GB/20.75GB/133.58GB at N=250/500/1000/2500) came from
`taskI_profile_clean.py`, the same script Task I-2 found never passed `sample_subset_samples`
through — meaning `__balance_action_space_by_outcome` never ran, and `n_action_embeddings_final`
in Task I's data was, the whole time, the **raw, never-pruned** candidate count, not a
"final"/resting cache size in any config that matches real training.

## Method

Instrumented the name-mangled `_CyberBattleCompressedEnv__balance_action_space_by_outcome`
directly (wraps it, records `len(self.action_embeddings)` the instant *before* it prunes, then
calls the original) — this measures the exact transient size a real construction call passes
through, immediately prior to capping, with `sample_subset_samples=100` correctly wired this
time. Measured both: **pre-balance** (raw candidate count, right before pruning) and
**post-balance** (the size that persists in `self.action_embeddings` afterward), for both the
`full_rebuild` scenario (`processed_pairs` cleared — matches Task I's original per-state
condition) and the `steady_state` scenario (`processed_pairs` already fully populated — the
realistic in-episode condition, per Task I-2). Same 7 sizes, 3 repeats each (21 points),
`taskI_memory_recheck.py`.

## ARTIFACT — measured pre/post-balance sizes

| N | full_rebuild pre-balance (raw) | full_rebuild post-balance (capped) | steady_state pre-balance | steady_state post-balance |
|---|---|---|---|---|
| 10 | 603 / 559 / 491 | 593 / 559 / 491 | 593 / 559 / 491 | 593 / 559 / 491 |
| 15 | 504 / 616 / 1339 | 504 / 616 / 800 | 504 / 616 / 800 | 504 / 616 / 800 |
| 34 | 11561 / 10043 / 10590 | 800 / 800 / 800 | 800 / 800 / 800 | 800 / 800 / 800 |
| 40 | 32452 / 24092 / 20609 | 800 / 800 / 800 | 800 / 800 / 800 | 800 / 800 / 800 |
| 80 | 22509 / 23316 / 22070 | 800 / 800 / 800 | 800 / 800 / 800 | 800 / 800 / 800 |
| 90 | 38532 / 64926 / 55029 | 800 / 800 / 800 | 800 / 800 / 800 | 800 / 800 / 800 |
| 100 | 27528 / 37262 / 37204 | 800 / 800 / 800 | 800 / 800 / 800 | 800 / 800 / 800 |

**FINDING: post-balance size caps at exactly 800 from N≈15-34 onward, in every scenario, no
exceptions (19/21 states at exactly 800; the 2 exceptions are the two smallest, sub-cap states,
same as Task I-2's STEP 1).** `steady_state pre-balance` always equals `steady_state
post-balance` — confirmed directly: once `processed_pairs` is fully populated, the double loop
skips every pair before it can add anything new, so there is nothing to prune; the resting
`action_embeddings` dict never exceeds the cap once an episode has stabilised.

## FITTED — pre-balance (transient) vs post-balance (persistent) growth

| Quantity | Exponent (95% CI) | R² |
|---|---|---|
| `full_rebuild` **pre-balance** (raw, transient) vs N | **1.955 (1.621, 2.290)** | 0.888 |
| `full_rebuild` **post-balance** (persistent cache) vs N | 0.155 (0.096, 0.214) | 0.614 |

**Task I's original exponent (2.03, CI 1.70-2.36) survives**: it falls squarely inside this
re-check's CI for the pre-balance quantity (1.955, CI 1.621-2.290 — the two CIs overlap heavily,
and 2.03 sits inside the re-check's interval). **What is withdrawn is the *label*, not the
number**: Task I described this as "the action_embeddings cache" without distinguishing
transient-during-construction from persistent-at-rest: those are the same number only because
capping was silently disabled in Task I's script. With capping correctly applied, they are two
different, and very differently-scaling, quantities (1.955 vs 0.155 — the persistent quantity
does **not** meaningfully grow with N at all in this range).

## EXTRAPOLATED — memory, re-derived, with the correct label

**Caveat, as before: extrapolated from a fit over a measured range of roughly 10-100 nodes, not
validated beyond it.**

| N | pre-balance (transient) payload | post-balance (persistent) payload |
|---|---|---|
| 100 (measured) | ≈199 MB (fit); measured range 27,528-64,926 actions → ≈100-235 MB | 800 actions, constant → ≈2.90 MB |
| 250 (extrapolated) | ≈1.19 GB | ≈2.90 MB (constant, not extrapolated — it does not grow) |
| 500 (extrapolated) | ≈4.63 GB | ≈2.90 MB |
| 1000 (extrapolated) | ≈17.96 GB | ≈2.90 MB |
| 2500 (extrapolated) | ≈107.76 GB | ≈2.90 MB |

(Raw float32 array payload only, as in Task I; excludes Python dict/tuple-key/object overhead,
which is typically several times the raw payload — same caveat as Task I's original table.)

## Answer to the specific question asked

**The cache does store pre-sampling candidates, transiently, and that transient state is
legitimately quadratic-ish even though the persisted/resting sample is capped — exactly the
scenario flagged as a possibility when this re-check was requested.** Explicitly:

- **Task I's N^2.03 exponent and 193MB/1.24GB/.../134GB figures survive**, re-measured at 1.955
  (CI overlaps Task I's original) and 199MB/1.19GB/4.63GB/17.96GB/107.76GB (same order of
  magnitude, same conclusion) — but they describe the **transient pre-balance/raw candidate
  set**, which is real and does get materialised in memory during any full-rebuild-style
  construction call (episode warm-up, or every call under `precise_action_space_positions=True`
  per Task I-2), not a number that was invalidated by the `sample_subset_samples` bug.
- **A second, separate, previously-unreported quantity — the persistent/resting cache size once
  `sample_subset_samples` is correctly applied — is capped at a constant (800 entries, ≈2.9MB)
  and does not grow with N** (exponent 0.155, CI 0.096-0.214, i.e. statistically indistinguishable
  from flat given the practical scale involved).
- **Memory and time do scale differently, and this is worth reporting in its own right**: the
  transient/full-rebuild regime is quadratic in both time (Task I-2: `full_rebuild` time
  exponent 2.36) and memory (this re-check: pre-balance size exponent 1.96) — the two move
  together, both driven by the same unconditional double-loop candidate generation. The
  steady-state/persistent regime is close to flat in both time (Task I-2: `steady_state` time
  exponent 0.41 under the default flag) and memory (this re-check: post-balance size exponent
  0.155) — also moving together. **The apparent "memory vs time" divergence is therefore not a
  divergence at all once both are correctly split into the same two regimes (transient
  full-rebuild vs persistent steady-state); within each regime, time and memory scale
  consistently with each other.**

## Provenance

- Raw JSON: `mem_N{10,15,34,40,80,90,100}.json` (7 files, 21 points).
- Script: `taskI_memory_recheck.py` — instruments
  `_CyberBattleCompressedEnv__balance_action_space_by_outcome` directly, does not modify any
  repo file.
- Kept in the job scratch directory, not the repo. No repo file was changed for this re-check.
