# Run record — STEP 1 (full) + Additions 1 and 2

Date: 2026-08-15.

## Command
```
cd analysis/reward_convergence_2026-08-15
/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/bin/python step1_full_comparison.py
```

## Addition 1 — population (a), both conventions
Largest convention-induced difference: **band 10-15, seed 123: −4.81% (stop=250000) → +10.00%
(default), diff +14.81pp.** Band-level: 10-15 mean|Δ%| 14.55%→12.58% (within 2/5→2/5, verdict
unchanged NOT CONVERGED); 30-40 mean|Δ%| 5.29%→7.16% (within 4/5→3/5, verdict unchanged NOT
CONVERGED under the mean criterion at stop=250000 already, more clearly so at default); 80-100
mean|Δ%| 19.91%→19.93% (within 1/5→1/5, essentially unchanged). **All three band-level verdicts
for population (a) are NOT CONVERGED under both conventions** — individual seed figures swing
substantially, band verdicts do not.

## Addition 2 — reward's cross-convention swing vs root-owned-count's, all 35 rows
- root-owned-count |diff|: mean 3.285pp, median 2.310pp, max 14.808pp.
- reward |diff|: mean 1.880pp, median 0.603pp, max 14.876pp.

**One sentence, as asked, no recommendation**: reward's cross-convention swing is smaller than
root-owned-count's typically (median ~0.6pp vs ~2.3pp, roughly a quarter; mean ~1.9pp vs ~3.3pp),
but not uniformly — reward's own worst case (14.88pp, Task Y N=90 seed200) is essentially as large
as root-owned-count's worst case (14.81pp, band 10-15 seed123) — so the evidence is partial rather
than clean: most of the instability found so far is attributable to the small-count signal, but a
comparably large instability shows up in reward too on at least one cell, meaning the window-anchor
choice contributes something independent of which signal it's applied to.

## STEP 1 — default convention (designated the project standard from here on)
Full 35-row table (both metrics, both within/no) in `step1_full_comparison_output.log`.
Population/cell-level summary:

| population/cell | verdict (root-owned) | verdict (reward) | AGREE? |
|---|---|---|---|
| a-manifest/10-15 | NOT CONVERGED (12.58%, 2/5) | NOT CONVERGED (13.50%, 1/5) | AGREE |
| a-manifest/30-40 | NOT CONVERGED (7.16%, 3/5) | NOT CONVERGED (12.25%, 1/5) | AGREE |
| a-manifest/80-100 | NOT CONVERGED (19.93%, 1/5) | NOT CONVERGED (13.14%, 1/5) | AGREE |
| b-N30/N30 | CONVERGED (4.18%, 4/5) | CONVERGED (1.71%, 5/5) | AGREE |
| **b-N60/N60** | **NOT CONVERGED (5.70%, 3/5)** | **CONVERGED (1.08%, 5/5)** | **DISAGREE** |
| b-N90/N90 | CONVERGED (2.18%, 5/5) | CONVERGED (1.50%, 5/5) | AGREE |
| c-lowdeg-N30/N30(lowdeg) | CONVERGED (2.67%, 4/5) | CONVERGED (3.62%, 4/5) | AGREE |

**Of the three previously-flagged outcomes to watch: exactly one materialized.**
- Task Y's N=60 cell now meets the rule under reward (5.70%/3-5 NOT CONVERGED → 1.08%/5-5
  CONVERGED). **This is the one DISAGREE row in the whole table.**
- Task Y's N=30 and N=90 cells (Table IV.4's only columns) do NOT fail under reward — both remain
  CONVERGED under both metrics.
- The lower-neighbour cell's seed 42 does NOT change status — NOT within tolerance under both
  metrics (root-owned +5.88%, reward +7.06%).

Counts: 35/35 rows computed under both metrics and both conventions, 0 unreadable, 0 undefined
(consistent with Q2's earlier finding that reward has no sign/zero problem anywhere in this set).

## Outputs (committed alongside this record)
`step1_full_comparison.py`, `step1_full_comparison_output.log`, this record.

## Wipe test
Reproducible from the committed script and the already-on-disk tfevents files (not committed, per
convention).

## Note
Per the task's explicit instruction, no recorded value is corrected, no convention is chosen on
the basis of which gives a better-looking answer, and no action is recommended from the AGREE/
DISAGREE result. Reported and stopped.
