# Run record — Task RECOMPUTE CONVERGENCE ON EPISODE REWARD (STEP 0 only, gated)

Date: 2026-08-15.

## Command
```
cd analysis/reward_convergence_2026-08-15
/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/bin/python step0_reward_signcheck.py
```

## Input paths (read-only)
`cyberbattle/agents/compute_convergence_check.py` (imported, unmodified). tfevents files for 35
(population, cell, seed) combinations across 4 populations — exact paths in the script itself and
in the reply this record accompanies. `evidence_cards/evidence_taskY3.md` (read via `git show
5193e32:...`, only present on branch `taskY2-pilot-n30`) used to disambiguate the two
differently-dated `yN30_s<seed>_stg1_*` folder sets.

## RNG
None — deterministic read of already-logged tensorboard scalars.

## Row counts
35/35 (population, cell, seed) combinations readable. 0 unreadable. 0 with `pre` negative. 0 with
`pre` within 1.0 of zero. Every window (`npre`, `nfin`) fully populated at 12 points, for every row
— no partial/degenerate windows, including the N=90 seeds whose final-stage folders span ~500k
local steps rather than ~250k.

## Headline result
`rollout/ep_rew_mean`'s earlier-window mean ranges from ~2,150 to ~20,900 across all 35 rows,
comfortably positive and far from zero everywhere. A relative rule is safe on this signal for
every seed checked — the opposite finding from the attenuation-ratio precedent the task's framing
raised as a live concern. Full per-row table in `step0_reward_output.log`.

## Outputs (committed alongside this record)
`step0_reward_signcheck.py`, `step0_reward_output.log`, this record.

## Wipe test
Reproducible from the committed script and the already-on-disk tfevents files (not committed, per
convention; the script and its output are).

## Note on this task's gate
Per the task's explicit instruction, STEP 1 (applying the rule and computing verdicts) has not
been run. This commit covers STEP 0 only. Waiting for confirmation to proceed.
