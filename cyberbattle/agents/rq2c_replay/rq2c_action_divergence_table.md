# Task RQ2C-1 -- action-divergence (does the choice follow the view or only the action set)

> RQ2C: single-node membership-leave events only; batch events excluded; 80-100 band agent is not confirmed converged (see Task F4); group (i) events have no choice-changed metric by construction.

> SCOPE: measured on FRESH episodes from a per-seed-seeded STOCHASTIC rollout of the trained policy -- the exploratory regime that actually produces membership_leave events (the DETERMINISTIC policy discovers ~nothing and fires 0 leaves, so it cannot measure RQ2(c)). Same checkpoints / seeds / bands as the headline attenuation sweep, and the same standard attenuation config (patch_service OFF, no CX_DIAG constraint relaxation). These are NOT the literal reported headline episodes: the stored attenuation_step3_logs actions are not faithfully replayable (that sweep lacked per-seed seeding; only a distributional <1pp reproduction was ever checked, never a trajectory-identical one -- see the replay-fidelity scope correction). Nor is this the stochastic-action-selection headline NUMBERS themselves. The COUNTERFACTUAL pre/post policy predicts are deterministic=True (noise-free before/after comparison -- that is where 'no stochastic sampling' matters). Reproducible: torch/np/random seeded per seed + set_num_threads(1) + PYTHONHASHSEED=0.

Source files (87) under `rq2c_replay/rq2c`; 9156 total single-node membership_leave event records. Bootstrap: 10000 iters, episode-clustered, 0.95.

## Per band (ARTIFACT counts; FINDING = group-(ii) divergence rate)

| band | total records (ARTIFACT) | clean-step records (ARTIFACT) | excl no_cand_pre (ARTIFACT) | excl no_cand_post (ARTIFACT) | excl no_obs_pre (ARTIFACT) | n_group_i (ARTIFACT) | n_group_ii (ARTIFACT) | n_changed (ARTIFACT) | rate group-ii (FINDING) | boot 0.95 CI | wilson 0.95 CI | emb_dist mean/median (ARTIFACT) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10-15 | 4965 | 4965 | 0 | 0 | 0 | 1176 | 3789 | 1062 | **0.280** | [0.264, 0.297] | [0.266, 0.295] | 2.810 / 1.192 |
| 30-40 | 1483 | 1483 | 0 | 0 | 0 | 101 | 1382 | 167 | **0.121** | [0.098, 0.144] | [0.105, 0.139] | 1.117 / 0.507 |
| 80-100 | 2708 | 2708 | 0 | 0 | 0 | 27 | 2681 | 121 | **0.045** | [0.037, 0.053] | [0.038, 0.054] | 0.454 / 0.123 |

## Per seed within band (the across-condition significance unit)

| band | seed | clean-step records | n_group_i | n_group_ii | n_changed | rate group-ii (FINDING) | boot 0.95 CI |
|---|---|---|---|---|---|---|---|
| 10-15 | 42 | 1018 | 251 | 767 | 180 | 0.235 | [0.206, 0.263] |
| 10-15 | 100 | 987 | 219 | 768 | 210 | 0.273 | [0.239, 0.308] |
| 10-15 | 123 | 1080 | 292 | 788 | 219 | 0.278 | [0.241, 0.319] |
| 10-15 | 200 | 979 | 197 | 782 | 259 | 0.331 | [0.293, 0.366] |
| 10-15 | 300 | 901 | 217 | 684 | 194 | 0.284 | [0.249, 0.318] |
| 30-40 | 42 | 296 | 14 | 282 | 28 | 0.099 | [0.057, 0.142] |
| 30-40 | 100 | 312 | 23 | 289 | 45 | 0.156 | [0.111, 0.199] |
| 30-40 | 123 | 262 | 17 | 245 | 13 | 0.053 | [0.027, 0.085] |
| 30-40 | 200 | 315 | 23 | 292 | 47 | 0.161 | [0.114, 0.220] |
| 30-40 | 300 | 298 | 24 | 274 | 34 | 0.124 | [0.078, 0.168] |
| 80-100 | 42 | 557 | 4 | 553 | 21 | 0.038 | [0.023, 0.058] |
| 80-100 | 100 | 508 | 8 | 500 | 30 | 0.060 | [0.043, 0.075] |
| 80-100 | 123 | 480 | 3 | 477 | 14 | 0.029 | [0.017, 0.045] |
| 80-100 | 200 | 505 | 6 | 499 | 30 | 0.060 | [0.040, 0.079] |
| 80-100 | 300 | 658 | 6 | 652 | 26 | 0.040 | [0.026, 0.053] |

## Implausible-value check (mandatory, OUTPUT AND REPORTING)

- Pooled per-band rates { 10-15: 0.280, 30-40: 0.121, 80-100: 0.045 } are neither all-0 nor all-1 -> not in the degenerate regime the spec flags. Still cross-check: group-i non-empty (membership test lives) and some emb_dist>0.
  - 10-15: n_group_i=1176, n_group_ii=3789, emb_dist>0 in 3789/3789 group-ii events.
  - 30-40: n_group_i=101, n_group_ii=1382, emb_dist>0 in 1382/1382 group-ii events.
  - 80-100: n_group_i=27, n_group_ii=2681, emb_dist>0 in 2681/2681 group-ii events.

## Interpretation key
- rate group-ii near 0 => FINDING: behaviour follows the ACTION SET, not the view (the preferred action, when it survives, is reselected).
- rate group-ii materially > 0 => FINDING: the VIEW itself changes the choice even when the preferred action is still available.
- group (i) (preferred action removed) carries no changed metric by construction; its size is the share of leaves that act purely through candidate-set membership.
