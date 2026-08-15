# Run record — Task CONVERGENCE DENOMINATOR CHECK (STEP 0 only, per the task's own gate)

Date: 2026-08-15.

## Command
```
cd analysis/convergence_denominator_2026-08-15
/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/bin/python compute_denominator_check.py
```

## Input paths (read-only)
`cyberbattle/agents/compute_convergence_check.py` (imported directly, unmodified — not
reimplemented). `cyberbattle/agents/logs/trpo_250k_tuned_compressed_band<band>_seed<seed>_2026-07-26_*/
TRPO_x_control_SecureBERT/TRPO_1/events.out.tfevents.*` for all 15 manifest checkpoints (the same
15 the accepted Table IV.1 run used, commits `2108f6d`/`780b65a`).

## RNG
None — deterministic read of already-logged tensorboard scalars.

## Row counts
15/15 checkpoints readable (both windows non-empty, `npre`/`nfin` both 12-13 points per seed). 0
unreadable. 0 seeds with `pre==0`. Stated explicitly, per the task's degenerate-case convention.

## Headline result
Band 10-15's earlier-window mean of `train/Root owned nodes` is 2.33–2.67 nodes across its 5 seeds
(smallest 2.3333, median 2.5833). At this denominator, **a change of only 0.117–0.133 whole nodes
already exceeds the 5% tolerance** — an order of magnitude below one whole node. Band 30-40's
denominator is 15.08–16.83 (1 node = 6.35–6.63%, close to but still over the 5% line). Band
80-100's denominator is 18.25–28.83 (1 node = 4.58–5.48%, straddling the 5% line). Full per-seed
table and all aggregates in `denominator_check_output.log`.

## Outputs (committed alongside this record)
`compute_denominator_check.py`, `denominator_check_output.log`, this record.

## Wipe test
Reproducible from the committed script and the already-on-disk tfevents files (not committed, per
convention; the driver script and its output are).

## Note on this task's gate
Per the task's explicit instruction ("Do not proceed past STEP 0. There is no STEP 1 in this
task... Report the answers to Q1 through Q6 and wait."), no further action is taken beyond this
commit and the reply. The dissertation log and evidence cards were not touched in this task.
