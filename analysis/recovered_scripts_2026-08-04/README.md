# Recovered scripts — project-wide manifest/script audit, 2026-08-04

Preserved from `.claude/jobs/0dfa230d/tmp/{taskF1,taskI}/` (ephemeral job scratch, not part of the
git repo, not guaranteed to persist) as part of a project-wide audit for the same failure mode that
lost Task F3's and the original Task CX pipeline's manifests (see `standing_rules.md` SR-1). These
scripts were found still present on disk — genuinely at risk, not yet actually lost — so this is a
preservation copy, not a reconstruction.

**Not reviewed, re-run, or validated as part of this audit** (per the audit's own scope: "find and
commit only," no re-running analyses). Copied as-is. If a script here is later needed, verify it
against its citing evidence card before trusting its output.

## Scope

`taskF1/` (76 files) is a single shared scratch workspace that turned out to cover far more than
Task F1 — it contains the producing scripts for several tasks' evidence cards:

| filename prefix | supports | evidence card |
|---|---|---|
| `taskF1_*`, `check1_*`, `check2_*`, `a2.py` | Task F1 / A2 (eval-mode decision, cost probe, headroom diag) | `evidence_taskF1.md`, `evidence_taskA2.md` |
| `taskF1R_reanalyze.py` | Task W / A2's reanalysis pass | `evidence_taskW.md`, `evidence_taskA2.md` |
| `taskF2_*` | Task F2 (train/eval/calibrate/step3 analysis) | `evidence_taskF2.md` |
| `taskF3_*`, `run_f3_*.sh` | Task F3 (characterize, mech/ownership eval+analyze) | `evidence_taskF3.md`, `evidence_taskZ.md` (cites F3's mech scripts) |
| `taskF4_train.py`, `conv_*`, `launch_f4_*.sh` | Task F4 (convergence check, 750k extension) | `evidence_taskF4.md` |
| `taskZ_*`, `z_*`, `launch_z*.sh` | Task Z (three-arm ablation train/eval, preflight, step2/repro analysis) | `evidence_taskZ.md` |
| `condB_multitopo_*` | Condition B multi-topology gate-checkpoint job | `evidence_conditionB_multitopology.md` |
| `d3_*`, `repro_d3.py`, `verify_d3.py`, `launch_d3_3040.sh` | Task D3 | `evidence_taskD3.md` |
| `demo_L.py`, `run_L_regression.py`, `compare_L_regression.py`, `replay_verify.py`, `drive_L_regression*.sh`, `rv_*.sh` | Task L (regression/replay verification) | `evidence_taskL.md`, `evidence_taskL_logging.md` |
| `w_step4.py`, `w_step5.py` | Task W | `evidence_taskW.md` |
| `x_stepA.py`, `x_stepB.py` | Task X | `evidence_taskX.md` |
| `probe_m.py`, `m_step2.py` | Task M | `evidence_taskM.md` |
| `probe_p.py` | Task P | `evidence_taskP.md` |
| `q1_joins*.py` | (query/probe, task unclear from filename alone — not traced further, per the audit's time-box) | -- |
| `encoder_test.py`, `throughput_test.py`, `count_strand.py` | diagnostic/utility scripts, no single card traced | -- |

`taskI/` (8 files): `taskI_*`, `taskI2_*` — Task I's profiling/curve-fitting scripts.
Card: `evidence_taskI.md`.

## What this does NOT cover

Bulk output data (result CSVs/JSONs, per-run logs, checkpoint directories) that lived alongside
these scripts in the same job-tmp folders was **not** copied here — consistent with this project's
existing convention of keeping bulk artifacts off git (see `evidence_cards/artifact_manifest.tsv`
for the tamper-evident checksum record of what bulk data has been separately preserved). Only the
small, text-format scripts are preserved in this directory.
