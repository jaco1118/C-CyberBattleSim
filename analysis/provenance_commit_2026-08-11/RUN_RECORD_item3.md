# Run record — Item 3: per-hop mean shift-norm on the wide population

Date: 2026-08-11.

## Command
```
cd analysis/provenance_commit_2026-08-11
/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/bin/python compute_hop_shares_wide.py
```

## Input paths (read-only)
`cyberbattle/agents/graphdepth_sweep_wide/leaveembed_<band>/*/*.jsonl` (the GRAPH-DEPTH-WIDE STEP 3 sweep, logger commit `1522b71`), filtered to `n_touched_nodes==1`.

## RNG
None.

## Row counts

| band | single-node events | events with zero reachable survivors (excluded from every hop mean) |
|---|---|---|
| 10-15 | 1,361 | 623 (45.78%) |
| 30-40 | 1,508 | 764 (50.66%) |
| 80-100 | 2,819 | 2,197 (77.94%) |

Cross-check note: these counts are close to but not bit-identical to the earlier `departing_node_degree==0` counts reported in TASK LOGBOOK-CLOSE's STEP 0 (617/762/2,196) — a small, real difference (6/2/1 events) between two related but distinct predicates ("no survivor has any finite hop distance" vs "the departing node's own undirected degree is exactly 0"), not reconciled further here since it does not change any reported mean.

Per-band, per-hop-bin event and survivor counts are in `item3_hop_shares_wide.csv` in full (hops 1–10 individually, plus the `unreachable` bin).

## Outputs (committed alongside this record)
`compute_hop_shares_wide.py`, `item3_hop_shares_wide.csv`, `item3_zero_reachable_counts.csv`, `item3_run_output.log`.

## Headline result
Hop-1 two-level mean shift-norm: 0.311 / 0.319 / 0.324 (10-15/30-40/80-100) — **lower** than `probe_p.py`'s synthetic-proxy figures (0.52/0.45/0.50), confirming the predicted direction, not a rise. Hop-2: 0.100/0.099/0.100, also lower than the proxy (0.32/0.26/0.31). Hop bins 3 through 10 are exactly 0.0000 at every band with the number of contributing events shown even where the mean is zero. The one-level pooled alternative diverges from the two-level figure most at hop 1 (by 0.07–0.09), confirming the two-level mean is measurably pulled down by small-survivor-count events relative to naive pooling — reported, not resolved, per the task's instruction to show both.

## Wipe test
Reproducible from the committed script and the already-on-disk `graphdepth_sweep_wide/` data (raw sweep output itself not committed, per this project's standing convention against committing bulk run data; script and output are).
