# Run record — Task VISIBILITY-PUZZLE

Date: 2026-08-11.

## Command
```
cd analysis/visibility_puzzle_2026-08-11
/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/bin/python compute_visibility_puzzle.py
```

## Input paths (read-only, no re-run of anything)
- `cyberbattle/agents/graphdepth_sweep_wide/drift_<band>.csv`
- `cyberbattle/agents/cx_step2_registration/drift_<band>.csv`
- `cyberbattle/agents/logs/trpo_250k_tuned_compressed_band10-15_seed42_2026-07-26_11-56-51/TRPO_x_control_SecureBERT/train_config.yaml`
- `cyberbattle/_env/cyberbattle_env.py` (constructor defaults, `_get_removal_eligible_nodes`, `_apply_dynamic_leave`) and `compute_attenuation_analysis.py` (`build_band_envs`, `_cx_write_run_metadata`) — read, not modified.
- Raw session transcripts `~/.claude/projects/*/*.jsonl` — searched for the actual launch commands (see below).

## RNG
None — deterministic read/aggregate over already-existing files.

## Row counts

| band | graphdepth_sweep_wide n_leave | =visible | cx_step2_registration n_leave | =visible | volume ratio |
|---|---|---|---|---|---|
| 10-15 | 1,399 | 1,399 (100.0%) | 12,012 | 2,131 (17.7%) | 8.59x |
| 30-40 | 1,672 | 1,672 (100.0%) | 49,295 | 6,479 (13.1%) | 29.48x |
| 80-100 | 3,127 | 3,127 (100.0%) | 56,035 | 10,174 (18.2%) | 17.92x |

## Headline result — root cause found, not just correlated

**This is a real behavioural difference between two deliberately different, correctly-separated
runs, not a logging artefact.** The candidate hypothesis given at the start of this task (drift
file downstream of / derived from the leave-embedding logger's skip guard) is refuted: `_log_drift_rows`
(`cyberbattle_env_compressed.py:1118`, called `:776`) and `_log_leave_embeddings` (`:1260`, called
`:812`) are separate, sequentially-called functions with independently computed visibility logic;
drift-row writing happens first and is never filtered by the leave-embedding guard.

The real cause is a launch-configuration difference, confirmed at three independent levels:

1. **Launch commands** (recovered verbatim from the raw session transcripts):
   - `graphdepth_sweep_wide`: `RQ2C=1 LEG=1 YEG_DRIFT_DIR=graphdepth_sweep_wide ... --collect` — no `CX_DIAG`.
   - `cx_step2_registration`: `CX_DIAG=1 YEG_DRIFT_DIR=$CXDIR ... --collect --analyze` — the command's own
     inline comment reads "CX registration arm (removal+join+patch, property discovered-only)".

2. **Code**: `compute_attenuation_analysis.py:168-174` only copies `CX_REMOVAL`/`CX_JOIN`/`CX_PATCH`
   into `train_config_for_env` (the kwargs actually passed to `CyberBattleCompressedEnv(...)`)
   inside `if os.environ.get("CX_DIAG") == "1":`. Without `CX_DIAG=1`, none of the three flags are
   applied to the constructed environment. `_get_removal_eligible_nodes` (`cyberbattle_env.py:520-524`)
   then bases the candidate pool on `self.discovered_nodes` (constructor default
   `allow_undiscovered_removal=False`, `:83`) instead of the whole topology — every leave event
   necessarily targets an already-discovered, hence visible, node: exactly the observed 100%.

3. **Data**: the representative checkpoint's own frozen `train_config.yaml` confirms
   `patch_service_dynamic_enabled: False` is baked in (line 71) and `allow_undiscovered_removal` is
   **absent as a key entirely** — both checkpoints predate the flag's introduction (established in
   the earlier FLAG-PROVENANCE task). So even the fallback-to-`train_config` path (had `CX_DIAG` been
   set differently) could not have supplied `True` for `graphdepth_sweep_wide`.

**Second, independently confirmed finding: a metadata bug, not just a behavioural difference.**
`_cx_write_run_metadata` (`compute_attenuation_analysis.py:118-126`) computes its `"flags"` dict
directly from `os.environ.get("CX_REMOVAL"/"CX_JOIN"/"CX_PATCH", "1") == "1"` **unconditionally** —
it does not check `CX_DIAG` — and is called whenever `CX_DIAG==1 OR CX_STATIC==1 OR RQ2C==1`
(line 248). So for any `RQ2C=1`-only run, `run_metadata_s<seed>.json` **falsely** records
`allow_undiscovered_removal: true` / `uncapped_join: true` / `patch_service_dynamic_enabled: true`
even though none of those three kwargs were ever passed to the constructed environment. This
affects `graphdepth_sweep_wide` and `rq2c_replay` (both launched `RQ2C=1` only, confirmed from
transcripts) — their metadata is not a reliable record of what was actually run. `cx_step2_registration`
and `cx_step2_replay` (both launched `CX_DIAG=1`) are unaffected — their metadata happens to match
reality because the relaxation really was applied.

**Volume gap, same root cause.** `_apply_dynamic_leave`'s per-node draw probability is
`min(_DYNAMIC_P_MAX=0.25, target_rate * weights[n] / weight_sum)` (`cyberbattle_env.py:618-621`),
and `sum(p) == target_rate` only holds before the 0.25 clip. With a small `discovered_nodes`-only
pool, more individual nodes hit the per-node cap, pulling the realised total leave rate below
`target_rate`; a full-topology pool dilutes the same `target_rate` over far more nodes so the clip
binds less and total volume is higher. This matches the observed 8.6x–29.5x gap, which *widens*
rather than narrows with topology size (larger topology ⇒ `discovered_nodes` is a smaller fraction
of it, especially early in an episode) — consistent with the mechanism, not merely correlated with it.

## Does this touch anything already reported? Confirmed, not assumed.

- **Zero-degree finding, propagation-to-direct medians, per-hop magnitudes**: sourced *exclusively*
  from `graphdepth_sweep_wide` (GRAPH-DEPTH-WIDE STEP 3/4, FLAG-PROVENANCE Part B, PROVENANCE-COMMIT
  Item 3 — all read `graphdepth_sweep_wide/` only). This is one single, internally self-consistent
  dataset: every event in it was generated under the *same* actual configuration
  (`allow_undiscovered_removal=False`, `patch_service_dynamic_enabled=False`), whatever its own
  metadata claims. Per the code comment at `compute_attenuation_analysis.py:252-253` — "Task RQ2C
  reuses the same per-seed seeding + provenance metadata (but NOT the CX_DIAG constraint relaxation
  in build_band_envs -- RQ2C runs the standard attenuation config)" — this is the *intended*,
  standard (non-diagnostic) configuration, matching the original published RQ2/RQ3 runs, not an
  accidental deviation. **Not affected.**
- **RQ3(a) gate counts**: `analysis/rq3a_gate_recompute_2026-08-09/compute_rq3a_gate_pooled.py:36`
  and `compute_rq3a_gate_unfiltered.py:41` both hard-code `DATA_DIR = os.path.join(AG, "cx_step2_registration")`.
  `cx_step2_registration` was launched with `CX_DIAG=1` — the relaxation was genuinely applied, and
  its metadata is accurate. **Not affected.**
- **RQ2(c) findings** (the 76%/94%/99% target-survival and 72%/90%/95% choice-unchanged figures,
  `dissertation_log_v2.md` 2026-08-02 entry) draw on `rq2c_replay/`, which *is* one of the two
  `RQ2C=1`-only, metadata-mismatched datasets identified here. The finding's substance is unaffected
  regardless (it studies the agent's behavioural response to whatever leave events occurred, which
  does not depend on which eligibility regime produced them), but `rq2c_replay`'s own
  `run_metadata_*.json` "flags" dict is subject to the same false-`true` bug documented above — flagged
  for accuracy, not because it changes the finding.

## Anything else noticed that looks wrong
The metadata bug above (`_cx_write_run_metadata` never checking `CX_DIAG`) is itself the main new
finding of this task, beyond the puzzle's own two symptoms (visibility rate, event volume) — it
means **no `run_metadata_s<seed>.json` file for any `RQ2C=1`-launched run in this project can be
trusted for its `allow_undiscovered_removal`/`uncapped_join`/`patch_service_dynamic_enabled` values**;
the launch command (not the metadata) is the only reliable record for those three fields on any
`RQ2C`-only run. Confirmed present on `graphdepth_sweep_wide` and `rq2c_replay`; `cx_step2_registration`
and `cx_step2_replay` are unaffected because they were launched with `CX_DIAG=1`, under which the
metadata and the actually-applied kwargs are computed from the same env vars and agree.

## Outputs (committed alongside this record)
`compute_visibility_puzzle.py`, `puzzle_volume_and_visibility.csv`, `puzzle_run_output.log`.

## Wipe test
Reproducible from the committed script and the already-on-disk `graphdepth_sweep_wide/` and
`cx_step2_registration/` data (raw sweep output itself not committed, per this project's standing
convention; script and output are). The launch-command evidence is quoted verbatim in the script's
own docstring since it is not re-derivable from disk alone.
