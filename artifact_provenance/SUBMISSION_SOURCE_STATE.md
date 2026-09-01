# Proposed submission source state

Audit date: 2026-09-01. No research artifact was copied, moved, deleted, reset, checked out, committed, or modified during this reconciliation.

## Repository state

- Current branch: `snr-abs-sensitivity`
- Current HEAD: `8c2862ba2ba7424bb27e3ca67c5a9630a486bbde`
- Local `main`: `7cdfb2b7bcfc9bd4b0f104819d3eedbd3cce3508`
- Cached `myrepo/main`: `7cdfb2b7bcfc9bd4b0f104819d3eedbd3cce3508`
- Cached and remotely verified upstream `origin/main`: `f5ab18c1a01c97de2781c4731c9a4a172716c7c0`
- `d0153c6e0b21f96b6ddf5eaccfdc6f92b0f6ab82`: **not present in the local Git object database**. The personal `myrepo` remote could not be queried without credentials, so its presence, tree and ancestry cannot be verified in this audit.
- `8c2862ba` is current HEAD. Both local `main` and `origin/main` are ancestors of it; it is not an ancestor of either `main` tip in the reverse direction. Current HEAD is therefore a later dissertation branch state, not the local main tip.
- Whether `d0153c6e` descends from `8c2862ba`: **UNKNOWN**, because `d0153c6e` is not locally available. Git cannot perform an ancestry test without the object.

Current `git status --short` before creating this report and the two requested manifests was:

```text
AM analysis/README.md
AM evidence_cards/README.md
?? ARTIFACT_AUDIT.md
?? ARTIFACT_MISSING_OR_RISKY.md
?? ARTIFACT_SUBMISSION_PLAN.md
?? analysis/uncertain_folders.zip
?? get_sizes.py
```

`AM` means each README was staged as a new file and then modified again in the working tree, so neither the staged nor working version is in current HEAD.

## Origin of current uncommitted files

Present before the artifact audit began:

- `analysis/README.md`
- `evidence_cards/README.md`
- `analysis/uncertain_folders.zip`
- `get_sizes.py`

Created during the artifact audit/reconciliation:

- `ARTIFACT_AUDIT.md`
- `ARTIFACT_SUBMISSION_PLAN.md`
- `ARTIFACT_MISSING_OR_RISKY.md`
- `SUBMISSION_SOURCE_STATE.md` (this report)
- `RQ1C_FINAL_CHECKPOINT_ALLOWLIST.tsv`
- `RQ3D_FINAL_INPUTS.tsv`

## Which state contains the final dissertation analyses?

Local `main` does not contain the final dissertation analysis set. The following paths cited by the audit reports are committed at current HEAD and absent from local `main`:

| Final evidence path | Current HEAD | Local main | `d0153c6e` |
| --- | --- | --- | --- |
| `analysis/rq1a_regression_recovered_2026-08-07/` | Present | Absent | Cannot inspect |
| `analysis/rq1b_mech_split_scale_2026-08-08/` | Present | Absent | Cannot inspect |
| `analysis/nodecount_ci_2026-08-09/` | Present | Absent | Cannot inspect |
| `analysis/nodecount_ci_n60_2026-08-16/` | Present | Absent | Cannot inspect |
| `analysis/rq1c_mde_2026-08-10/` | Present | Absent | Cannot inspect |
| `analysis/graph_depth_2026-08-10/decomposition_wide/` | Present | Absent | Cannot inspect |
| `analysis/three_lookups_2026-08-15/` | Present | Absent | Cannot inspect |
| `analysis/provenance_commit_2026-08-11/` | Present | Absent | Cannot inspect |
| `analysis/rq3a_gate_recompute_2026-08-09/` | Present | Absent | Cannot inspect |
| `analysis/rq3b_slice_recompute_2026-08-09/` | Present | Absent | Cannot inspect |
| `analysis/rq3c_rebuild_2026-08-10/` | Present | Absent | Cannot inspect |
| `analysis/rq3d_ranking_overlap_2026-08-22/` | Present | Absent | Cannot inspect |
| `analysis/reward_convergence_2026-08-15/` | Present | Absent | Cannot inspect |
| `analysis/y_reward_stage_2026-08-19/` | Present | Absent | Cannot inspect |
| `analysis/snr_abs_2026-08-22/` | Present | Absent | Cannot inspect |
| `analysis/snr_trace_2026-08-22/` | Present | Absent | Cannot inspect |
| `evidence_cards/claims_audit.md` | Present | Absent | Cannot inspect |
| `evidence_cards/artifact_manifest.tsv` | Present | Absent | Cannot inspect |
| `cyberbattle/agents/attenuation_manifest.yaml` | Present | Absent | Cannot inspect |
| `cyberbattle/agents/y_robustness/scripts/run_stage_earlyckpt.sh` | Present | Absent | Cannot inspect |

These checks test committed tree entries, not merely similarly named working files. The listed paths are therefore reproducibly present in commit `8c2862ba`. No claim that they are unchanged in `d0153c6e` can be made until that commit is made locally readable.

## Recommended base commit

**Provisional recommendation: `8c2862ba2ba7424bb27e3ca67c5a9630a486bbde`.** It is the only available commit demonstrated to contain the final dissertation code and analyses, and the final thesis results postdate local `main`.

This recommendation must be revisited before committing or packaging if `d0153c6e0b21f96b6ddf5eaccfdc6f92b0f6ab82` is intended as a later final-submission commit. Safest resolution: make that commit available read-only (for example by an authenticated fetch or a supplied Git bundle), then compare `8c2862ba..d0153c6e`, verify ancestry, and repeat the final-path table. Do not select `d0153c6e` merely from its hash without inspecting its tree.

## Files proposed for a deliberate final submission commit

Subject to review, add only documentation/provenance files:

- `analysis/README.md` — use the reviewed working-tree version, not silently the currently staged earlier version.
- `evidence_cards/README.md` — same caveat.
- `ARTIFACT_AUDIT.md`
- `ARTIFACT_SUBMISSION_PLAN.md`
- `ARTIFACT_MISSING_OR_RISKY.md`
- `SUBMISSION_SOURCE_STATE.md`
- `RQ1C_FINAL_CHECKPOINT_ALLOWLIST.tsv`
- `RQ3D_FINAL_INPUTS.tsv`

Creating that commit on top of the verified base would change **documentation and provenance only**, not simulator code, analysis algorithms, processed results, checkpoints, topologies, or raw experimental data.

## Files proposed to remain uncommitted/excluded

- `analysis/uncertain_folders.zip` — unreviewed archive; do not include automatically.
- `get_sizes.py` — local audit helper, not dissertation research code unless separately justified.
- All ignored research artifacts remain in place and uncommitted until the packaging plan is approved.
- No checkpoint, topology, raw log, cache, credential, virtual environment, or `.git` directory should be added to this documentation-only source commit.
