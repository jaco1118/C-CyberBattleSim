# Run record — Task REPLAY-FLAG-CLOSEOUT

Date: 2026-08-12.

## Commands
```
cd analysis/replay_flag_closeout_2026-08-12
/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/bin/python compute_replay_flag_signature.py
/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/bin/python compute_graphdepth_sweep_narrow_signature.py
```

## Input paths (read-only)
`cyberbattle/agents/{cx_step2_replay,graphdepth_sweep}/drift_<band>.csv`. Source: `compute_attenuation_analysis.py:391-401` (CX_REPLAY mechanism), `cyberbattle/_env/cyberbattle_env_compressed.py` (CX_REPLAY_PROBE, lines 709/892-896). Raw session transcripts for launch-command recovery.

## Precondition check (before interpreting any number)
`grep -rn "CX_REPLAY" cyberbattle/` (no pathspec restriction) → 5 hits total: two are `CX_REPLAY_PROBE`
(a read-only diagnostic logger, never touches dynamic-event application), three are in
`compute_attenuation_analysis.py`/`compute_rq2c_action_divergence.py` — the action-substitution
(`action = _replay[_ri:_ri+1]` replacing `model.predict(state)`) and its comment. No code path
reads or replays a *recorded event log* — the environment's own `maybe_apply_dynamic_step` →
`_apply_dynamic_leave` → `_get_removal_eligible_nodes` runs exactly as in a live rollout, gated by
this run's own `self.allow_undiscovered_removal`. **Conclusion: the unknown_fraction statistic is
a valid measurement of cx_step2_replay's own flag, not an echo of a recorded sequence.**

## Row counts, cx_step2_replay (membership_leave only)

| band | n_rows_read | n_dropped_missing_col | n_rows_used | n_visible | n_not_visible | unknown_fraction |
|---|---|---|---|---|---|---|
| 10-15 | 12,012 | 0 | 12,012 | 2,131 | 9,881 | 0.8226 |
| 30-40 | 49,295 | 0 | 49,295 | 6,479 | 42,816 | 0.8686 |
| 80-100 | 56,035 | 0 | 56,035 | 10,174 | 45,861 | 0.8184 |
| POOLED | 117,342 | — | 117,342 | 18,784 | 98,558 | 0.8399 |

Zero rows dropped for missing column at every band — stated explicitly. Escape route: median
discovered/topology fraction 0.80/0.833/0.909 (10-15/30-40/80-100) — never near 100%, so ample
room existed for the flag to matter.

**These numbers are row-for-row IDENTICAL to `cx_step2_registration`'s own numbers** (same
12,012/49,295/56,035 totals, same 2,131/6,479/10,174 visible counts, same 0.8399 pooled fraction —
verified directly against `analysis/flag_ground_truth_2026-08-12/flag_ground_truth_signature.csv`).
This is expected, not suspicious, and is itself corroborating evidence: `cx_step2_replay` forces the
same action sequence as `cx_step2_registration` under the same per-seed RNG seeding and the same
`CX_DIAG=1` environment configuration, so the dynamic-leave RNG draws reproduce byte-identically —
exactly the "Task L verified byte-identical" replay-fidelity property referenced in the code's own
comment (`compute_attenuation_analysis.py:392`). Had `cx_step2_replay`'s own flag actually been off
(contradicting its `CX_DIAG=1` launch), the eligible pool size would differ from the original run's,
the RNG draw outcomes would diverge from the very first differing Bernoulli hit, and the two
datasets' row counts would not match exactly — they do, which independently rules out a silent
flag mismatch.

## Launch record, cx_step2_replay (re-verified from raw transcript)
```
REPO=/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim
MAN=$REPO/attenuation_gate_archive/2026-07-26_trpo_5seed_gate/attenuation_manifest.yaml
RDIR=$REPO/cyberbattle/agents/cx_step2_replay
rm -rf $RDIR; mkdir -p $RDIR/probe
export USER=slchan PYTHONHASHSEED=0 CX_DIAG=1 CX_REPLAY=1
export CX_REPLAY_ACTIONS_DIR=$REPO/cyberbattle/agents/cx_step2_registration
export CX_REPLAY_PROBE=1 CX_REPLAY_PROBE_DIR=$RDIR/probe
nohup env YEG_DRIFT_DIR=$RDIR python compute_attenuation_analysis.py --collect --manifest $MAN > $RDIR/replay.log 2>&1 &
```
`CX_DIAG=1` is set. Searched for a resume/relaunch (`find .../cx_step2_replay -iname "*resume*"`,
no pathspec restriction) — none found; only one launch command exists in any transcript.

## Row counts, graphdepth_sweep addendum (membership_leave only)

| band | n_rows_read | n_dropped_missing_col | n_rows_used | n_visible | n_not_visible | unknown_fraction |
|---|---|---|---|---|---|---|
| 10-15 | 1,399 | 0 | 1,399 | 1,399 | 0 | 0.0000 |
| 30-40 | 1,672 | 0 | 1,672 | 1,672 | 0 | 0.0000 |
| 80-100 | 3,127 | 0 | 3,127 | 3,127 | 0 | 0.0000 |
| POOLED | 6,198 | — | 6,198 | 6,198 | 0 | 0.0000 |

Identical to `graphdepth_sweep_wide`'s own numbers exactly — same underlying event stream under
narrower embedding logging. Launch command (raw transcript): `RQ2C=1 LEG=1 YEG_DRIFT_DIR=graphdepth_sweep`,
no `CX_DIAG` — same pattern as `graphdepth_sweep_wide`.

## STEP 2 — Inventory of run_metadata files whose flag record contradicts real behaviour

`grep -rl '"allow_undiscovered_removal": true' cyberbattle/agents/*/eventgraph_*/run_metadata_*.json`
(no pathspec restriction beyond the fixed `eventgraph_*/run_metadata_*.json` layout, which is the
only place this file is ever written) → exactly 6 directories, 15 `run_metadata_*.json` files each
(3 bands × 5 seeds), **every one of which records `allow_undiscovered_removal: true`** — `false` is
never recorded anywhere in this project's history (`grep -rl '"allow_undiscovered_removal": false'`
→ zero hits).

| directory | n_metadata_files | real behaviour | verdict |
|---|---|---|---|
| `graphdepth_sweep` | 15 | OFF (this task) | **FALSE record** |
| `graphdepth_sweep_wide` | 15 | OFF (FLAG-GROUND-TRUTH) | **FALSE record** |
| `rq2c_replay` | 15 | OFF (FLAG-GROUND-TRUTH) | **FALSE record** |
| `cx_step2_registration` | 15 | ON (FLAG-GROUND-TRUTH) | true record |
| `cx_step2_replay` | 15 | ON (this task) | true record |
| `cx_step2_static` | 15 | **moot** — 0 membership_leave events at any band (`dynamic_mode=none`, verified by direct row count) | not a contradiction; flag never exercised |

**45 run_metadata files (3 directories × 15) carry a false `allow_undiscovered_removal` record.**

## Exact condition that produces a false record
A run launched with `RQ2C=1` (or `CX_STATIC=1`) and `CX_DIAG` unset or not equal to `"1"` gets a
`run_metadata_s<seed>.json` that unconditionally reports `allow_undiscovered_removal` (and
`uncapped_join`, and `patch_service_dynamic_enabled`) from `CX_REMOVAL`/`CX_JOIN`/`CX_PATCH`
(default `"1"`⇒True), while the environment actually constructed for that run only receives those
three kwargs inside `build_band_envs`'s `if os.environ.get("CX_DIAG") == "1":` guard — so if
`CX_DIAG` was not set, none of the three were ever applied, and the environment fell back to the
constructor defaults / the checkpoint's own frozen `train_config.yaml` values.

## Do the other two flags suffer the same bug?
Confirmed from source, not assumed: `uncapped_join` and `patch_service_dynamic_enabled` are built
by the exact same pattern, in both the metadata writer (`compute_attenuation_analysis.py:122-123`,
unconditional) and the environment builder (`:171,174`, inside the same `CX_DIAG=="1"` guard) —
**yes, both are equally unreliable for the same 3 directories.** Consequence check: `patch_service_
dynamic_enabled`'s real value for all 3 false-record directories is `False` (the checkpoints' own
frozen `train_config.yaml`, confirmed directly in `analysis/visibility_puzzle_2026-08-11/`), and
property events never appear in any reported figure drawn from these 3 directories (0 throughout,
confirmed). `uncapped_join`'s real value is likewise `False`, but no reported figure separately
analyses join-event volume or capacity from `graphdepth_sweep`, `graphdepth_sweep_wide`, or
`rq2c_replay` specifically — the join-related published figures (never-attributed fraction,
RQ3(a) gate counts) trace to `attenuation_drift_logs` and `cx_step2_registration` respectively,
neither of which is in the false-record set.

## Does any reported figure change?
No. Checked, not assumed: every number this project has published from `graphdepth_sweep`,
`graphdepth_sweep_wide`, or `rq2c_replay` was computed from that dataset's own data throughout —
each dataset is internally self-consistent (one real configuration governed the whole run), and
that real configuration (`allow_undiscovered_removal=False`) is, per `compute_attenuation_
analysis.py:252-253`'s own comment, the *intended* standard/non-diagnostic attenuation config for
these three, not an accidental deviation. The zero-degree finding, propagation-to-direct medians,
per-hop magnitudes (graphdepth_sweep_wide/graphdepth_sweep), and the RQ2(c) target-survival/
choice-unchanged figures (rq2c_replay) all stand as published. Only the run_metadata *receipts*
for these three directories are wrong; the thesis text describing them (checked: the 2026-08-02
RQ2(c) dissertation-log entry describes the mechanism as "gated RQ2C=1", never claims a relaxed
condition) was never wrong about this.

## Bottom line for cx_step2_replay
**(b) FLAG WAS ON**, established by (i) the launch command (`CX_DIAG=1`, re-verified from the raw
transcript), (ii) the behavioural signature (0.8399 pooled unknown fraction, matching a genuinely
relaxed run, with the escape route checked and ruled out), and (iii) the row-for-row exact match to
`cx_step2_registration`'s own numbers, which is only possible if the same eligibility pool governed
both runs. Metadata is right. **The Section 4.4 sentence stands as written.**

## Outputs (committed alongside this record)
`compute_replay_flag_signature.py`, `cx_step2_replay_signature.csv`, `replay_flag_run_output.log`,
`compute_graphdepth_sweep_narrow_signature.py`, `graphdepth_sweep_narrow_signature.csv`,
`graphdepth_sweep_narrow_run_output.log`, this record.

## Wipe test
Reproducible from the committed scripts and the already-on-disk `cx_step2_replay/`,
`cx_step2_registration/`, and `graphdepth_sweep/` data (raw data itself not committed, per this
project's standing convention; scripts and output are).
