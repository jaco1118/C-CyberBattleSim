# Task O8 — drift/action-success coupling + the zero-graph-info floor (Arm 4)

Two independent parts. Part 1 is read-only (existing drift logs, no new training). Part 2 extends the
Task Z three-arm pooling ablation with a fourth arm: all graph_embeddings dims zeroed (not just the
extremal channels), to test whether the policy uses graph_embeddings AT ALL. Both parts complete.

## Part 1 — per-step drift vs action-success (binary proxy) [FINDING]

Point-biserial correlation between per-step observation drift caused by the agent's OWN action
(`agent_drift_full`) and `agent_action_succeeded` (binary: reward>0), computed from the already-logged
`attenuation_step3_logs/drift_*.csv` (no new training).

**LABELLING (load-bearing):** this is "drift vs per-step action-success (binary proxy)". It is NOT the
same claim as "drift vs magnitude of score change" — no continuous per-step reward is logged anywhere in
the drift CSVs, only the binary success flag. A continuous-reward version would need a fresh run and is a
separate future item, not done here.

**Result** (all p ~ 0 at n=1.8e5–7.7e5 steps):

| band | n_steps | point_biserial_r | p | mean_drift\|succ=1 | mean_drift\|succ=0 |
|---|---|---|---|---|---|
| 10-15  | 180,972 | 0.538 | ~0 | 0.524 | 0.030 |
| 30-40  | 631,918 | 0.520 | ~0 | 0.514 | 0.015 |
| 80-100 | 769,514 | 0.530 | ~0 | 0.634 | 0.011 |

Moderate positive coupling, stable across scale (r essentially flat 0.52–0.54 across a 5.3x node-count
range): successful actions move the observation substantially; failed actions barely move it.

Script: `cyberbattle/agents/compute_drift_score_corr.py`. Committed on `attenuation-pooling-scale`
(1eaffe5), results captured to `attenuation_step3_logs/drift_actionsuccess_corr_results.txt`.

## Part 2 — Arm 4: does graph_embeddings carry ANY usable signal? [FINDING]

Extends Task Z's three-arm ablation (`evidence_taskZ.md`; Arm 1 full-256 [mean,max,min], Arm 2
mean-only-128, Arm 3 256-d extremal-zeroed) with **Arm 4: all 256 graph_embeddings dims held at 0.0**
(post-VecNormalize, same placement/rationale as Arm 3 — VecNormalize's running stats still see the raw
obs). The policy retains only `discrete_features` (owned_nodes, discovered_nodes) as observation signal.
This is the most basic ablation: does the graph encoder's output matter at all, vs the two-scalar
discrete summary alone?

### 2.1 Build

`ExtremalMask` (in `cyberbattle/agents/extremal_mask.py`) parameterized: `mask_slice` now defaults to
`EXTREMAL_SLICE=slice(64,192)` (Arm 3, unchanged behaviour) or can be set to `FULL_SLICE=slice(0,256)`
(Arm 4). New gated CLI flag `--zero_graph_embeddings` in `train_agent.py` (mutually exclusive with
`--extremal_mask`). Pre-flight generalized from hardcoded mean/tail index ranges to a boolean
complement-of-slice check, correct for both a partial mask (Arm 3) and a full mask (Arm 4, which has no
"outside" region). Committed `b72e3c5`.

**Bit-exact pre-flight** (Z_PREFLIGHT=1, real training obs stream, 5120 steps):
`masked_width=256, distinct_in_masked=[0.0], n_distinct=1, max_abs_outside_diff=0.0` — all 256 dims
exactly zero, no distortion. Committed `87c183d` (`o8_arm4/preflight/arm4_preflight.json`).

**Config**: `o8_arm4/arm4_base.yaml`, cloned from the committed Arm-1 training config (same 256-d
[mean,max,min] aggregations, `dynamic_mode=none`, 250k iterations, `sample_subset_samples=100`,
5 seeds via `num_runs=5`/`load_seeds`) with `zero_graph_embeddings=True`.

**CAUGHT DURING LAUNCH — CLI-override bug (same pattern as the earlier `--finetune_model` issue):** the
first launch passed `--train_config arm4_base.yaml` without also passing `--num_runs`/`--load_seeds` on
the CLI. `train_agent.py`'s `config.update(vars(args))` always applies argparse's *defaults* for every
unset flag, silently overwriting the yaml's `num_runs=5`/`load_seeds=.../seeds_all` with the CLI defaults
(`num_runs=1`, `load_seeds="config"`). Only seed 42 trained (mechanically fine, wrong scope). Caught by
checking `Training finished` / checkpoint-subdir counts before committing results; the incomplete run was
deleted and relaunched with `--num_runs 5 --load_seeds <path>` explicit on the CLI. Verified this time:
`seeds.yaml` lists all 5 seeds, 5 checkpoint subdirs per topology, no errors. **Any future
`train_agent.py` invocation with a yaml-only config MUST pass every CLI-overridable field explicitly on
the command line** — the yaml value is silently discarded otherwise.

### 2.2 Training

5 seeds (42/100/123/200/300) x {topo #44, topo #34}, 250k steps each, thread-capped
(`OMP/MKL/OPENBLAS/NUMEXPR/VECLIB_NUM_THREADS=1`), ~71 min wall time for both topologies concurrently.
No errors. All 10 final checkpoints (`checkpoint_250000_steps.zip`) confirmed present.

### 2.3 Eval + FLOOR contrast

`taskZ_eval.py` extended: `arm==4` branch zeroes all 256 `graph_embeddings` dims at eval time (mirroring
training); `zero_graph_embeddings` added to the env-constructor `_SKIP` set. 200 stochastic episodes x 5
seeds x {static, change} x {#44, #34}, no errors.

**Per-arm mean terminal root-owned COUNT:**

| topo | arm1 (full) | arm2 (mean-only) | arm3 (extremal-zero) | arm4 (ALL-zero, O8) |
|---|---|---|---|---|
| #44 static | 9.774 | 9.730 | 9.706 | 9.573 |
| #44 change | 6.327 | 6.343 | 6.251 | 5.799 |
| #34 static | 7.535 | 7.634 | 7.495 | 7.395 |
| #34 change | 5.206 | 5.252 | 5.264 | 5.052 |

**FLOOR contrast (arm1 − arm4, under CHANGE, paired by seed, 10k paired-bootstrap 95% CI), MDE from the
existing 3-arm static pool (0.085 nodes / 0.9% on #44, 0.162 nodes / 2.1% on #34):**

| topo | mean diff | 95% CI | verdict |
|---|---|---|---|
| #44 (primary)   | +0.528 | [+0.438, +0.631] | **EFFECT PRESENT** (CI excludes 0) |
| #34 (secondary) | +0.154 | [-0.012, +0.321]  | EFFECT ABSENT (well-powered) |

**Reading (contrast with Task Z's INFO/CAPACITY/RAW, all EFFECT ABSENT or UNDERPOWERED on both topos):**
graph_embeddings DOES carry usable signal on the primary topology — removing it entirely costs ~0.53
root-owned, ~6.2x the MDE, clearly resolved. But specifically the extremal-pooling channels don't matter
(Arm 1 vs Arm 3 was null on both topos) — the signal that matters lives in the MEAN channel (or the
next-escalation channel), not the max/min pooling. On the secondary topology (#34) the FLOOR effect is
not resolvable at this sample size (CI upper bound +0.321 is close to but does not clear the 0.162 MDE),
so this reading is topology-dependent, not a clean universal "graph info always helps" result.

Committed `7879a42` (`taskZ_eval.py`, `compute_z_mde.py` FLOOR-contrast extension, 4 preserved eval
CSVs in `rq2b_10-15_eval/arm4_*.csv`). Config/seeds/preflight at `87c183d`, wrapper/CLI flag at `b72e3c5`.
All on branch `rq2b-10-15`. Training run folders (`logs/o8_arm4_*`) gitignored per repo convention.
