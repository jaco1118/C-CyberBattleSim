# Run record — N=30 vs N=60 and N=60 vs N=90 robustness-difference CI + MDE

Date: 2026-08-16.

## Command
```
cd cyberbattle/agents
python analysis/nodecount_ci_n60_2026-08-16/compute_nodecount_ci_n60.py
```

## Method, confirmed identical to the existing comparison
- Resample count: **NB=10,000**, matching `compute_nodecount_ci.py` (N=30 vs N=90) exactly.
- Resampling unit: **seed** -- 5 seeds per cell resampled with replacement, independently per
  cell, each bootstrap draw's pooled robustness computed by concatenating all episodes across
  the resampled seeds and taking the ratio of pooled means. Same two-level construction as
  `compute_nodecount_ci.py` and `compute_neighbour_comparison.py`.
- RNG seed: 11, matching both prior scripts' own SEED constant.
- MDE: pooled between-seed SD (ddof=1) of the raw per-seed robustness values from both sides of
  each comparison, equal per-seed weight, no z-multiplier -- matching `compute_rq1c_mde.py`'s
  construction exactly (itself an adaptation of `compute_z_mde.py`'s ablation-MDE convention).

## N=60 data source
`y_robustness/out/n60_ci7/` -- the SELECTED run from Task N=60-ROBUSTNESS Amendment 2 (ci=7,
42.23% achieved churn, in the declared 40.5-43.5% band). NOT `y_robustness/out/n60/` (ci=5,
47.18%, rejected) and NOT `y_robustness/out/n60_ci8/` (ci=8, 40.12%, rejected).

## Per-seed robustness (all three cells, for reference)
| seed | N=30 | N=60 | N=90 |
|---|---|---|---|
| 42  | 0.6802 | 0.7029 | 0.6805 |
| 100 | 0.7686 | 0.6570 | 0.6486 |
| 123 | 0.6740 | 0.6281 | 0.6755 |
| 200 | 0.6983 | 0.6781 | 0.6924 |
| 300 | 0.7066 | 0.7342 | 0.7194 |
| mean | 0.7056 | 0.6801 | 0.6833 |
| sd | 0.0376 | 0.0409 | 0.0258 |

## Results

| comparison | point diff (larger - smaller) | 95% CI | includes 0? | MDE (robustness units) | \|diff\|/MDE |
|---|---|---|---|---|---|
| N=30 vs N=90 (existing, `nodecount_ci_2026-08-09`) | -0.0223 (N90-N30) | [-0.0574, +0.0117] | yes | 0.0326 | 0.675 |
| N=30 vs N=60 | -0.0255 (N60-N30) | [-0.0679, +0.0176] | yes | 0.0394 | 0.647 |
| N=60 vs N=90 | +0.0032 (N90-N60) | [-0.0355, +0.0424] | yes | 0.0323 | 0.099 |

All three comparisons' 95% CIs include zero. Sign convention: larger cell minus smaller cell
throughout (N=30 vs N=60 -> N60-N30; N=60 vs N=90 -> N90-N60), matching the existing text's
"-0.022, the 90-node cell the lower" framing for N=30 vs N=90.

## Outputs (committed alongside this record)
`compute_nodecount_ci_n60.py` (committed pre-run, `8b9ddcc`), `run_output.log`,
`nodecount_ci_n60_result.csv`, this record.

## Wipe test
Reproducible from the committed script and the already-on-disk score CSVs in
`y_robustness/out/{n30,n60_ci7,n90}/` (not committed here, per convention; those three
directories were committed in their own respective prior tasks).
