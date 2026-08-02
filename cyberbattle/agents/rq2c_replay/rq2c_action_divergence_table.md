# Task RQ2C-1 -- action-divergence (does the choice follow the view or only the action set)

> RQ2C: single-node membership-leave events only; batch events excluded; 80-100 band agent is not confirmed converged (see Task F4); group (i) events have no choice-changed metric by construction.

> SCOPE: measured on FRESH episodes from a per-seed-seeded STOCHASTIC rollout of the trained policy -- the exploratory regime that actually produces membership_leave events (the DETERMINISTIC policy discovers ~nothing and fires 0 leaves, so it cannot measure RQ2(c)). Same checkpoints / seeds / bands as the headline attenuation sweep, and the same standard attenuation config (patch_service OFF, no CX_DIAG constraint relaxation). These are NOT the literal reported headline episodes: the stored attenuation_step3_logs actions are not faithfully replayable (that sweep lacked per-seed seeding; only a distributional <1pp reproduction was ever checked, never a trajectory-identical one -- see the replay-fidelity scope correction). Nor is this the stochastic-action-selection headline NUMBERS themselves. The COUNTERFACTUAL pre/post policy predicts are deterministic=True (noise-free before/after comparison -- that is where 'no stochastic sampling' matters). Reproducible: torch/np/random seeded per seed + set_num_threads(1) + PYTHONHASHSEED=0.

Source files (54) under `rq2c_replay/rq2c`; 10169 total single-node membership_leave event records. Bootstrap: 10000 iters, episode-clustered, 0.95.

## Per band (ARTIFACT counts; FINDING = group-(ii) divergence rate)

| band | total records (ARTIFACT) | clean-step records (ARTIFACT) | excl no_cand_pre (ARTIFACT) | excl no_cand_post (ARTIFACT) | excl no_obs_pre (ARTIFACT) | n_group_i (ARTIFACT) | n_group_ii (ARTIFACT) | n_changed (ARTIFACT) | rate group-ii (FINDING) | boot 0.95 CI | wilson 0.95 CI | emb_dist mean/median (ARTIFACT) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10-15 | 4965 | 4965 | 0 | 0 | 0 | 1176 | 3789 | 1062 | **0.280** | [0.264, 0.296] | [0.266, 0.295] | 2.810 / 1.192 |
| 30-40 | 3146 | 3146 | 0 | 0 | 0 | 197 | 2949 | 307 | **0.104** | [0.092, 0.117] | [0.094, 0.116] | 0.910 / 0.442 |
| 80-100 | 2058 | 2058 | 0 | 0 | 0 | 18 | 2040 | 101 | **0.050** | [0.038, 0.060] | [0.041, 0.060] | 0.451 / 0.129 |

## Per seed within band (the across-condition significance unit)

| band | seed | clean-step records | n_group_i | n_group_ii | n_changed | rate group-ii (FINDING) | boot 0.95 CI |
|---|---|---|---|---|---|---|---|
| 10-15 | 42 | 1018 | 251 | 767 | 180 | 0.235 | [0.206, 0.262] |
| 10-15 | 100 | 987 | 219 | 768 | 210 | 0.273 | [0.239, 0.307] |
| 10-15 | 123 | 1080 | 292 | 788 | 219 | 0.278 | [0.242, 0.319] |
| 10-15 | 200 | 979 | 197 | 782 | 259 | 0.331 | [0.293, 0.366] |
| 10-15 | 300 | 901 | 217 | 684 | 194 | 0.284 | [0.250, 0.317] |
| 30-40 | 42 | 3146 | 197 | 2949 | 307 | 0.104 | [0.092, 0.117] |
| 80-100 | 42 | 2058 | 18 | 2040 | 101 | 0.050 | [0.039, 0.060] |

## Implausible-value check (mandatory, OUTPUT AND REPORTING)

- Pooled per-band rates { 10-15: 0.280, 30-40: 0.104, 80-100: 0.050 } are neither all-0 nor all-1 -> not in the degenerate regime the spec flags. Still cross-check: group-i non-empty (membership test lives) and some emb_dist>0.
  - 10-15: n_group_i=1176, n_group_ii=3789, emb_dist>0 in 3789/3789 group-ii events.
  - 30-40: n_group_i=197, n_group_ii=2949, emb_dist>0 in 2949/2949 group-ii events.
  - 80-100: n_group_i=18, n_group_ii=2040, emb_dist>0 in 2040/2040 group-ii events.

## Interpretation key
- rate group-ii near 0 => FINDING: behaviour follows the ACTION SET, not the view (the preferred action, when it survives, is reselected).
- rate group-ii materially > 0 => FINDING: the VIEW itself changes the choice even when the preferred action is still available.
- group (i) (preferred action removed) carries no changed metric by construction; its size is the share of leaves that act purely through candidate-set membership.
