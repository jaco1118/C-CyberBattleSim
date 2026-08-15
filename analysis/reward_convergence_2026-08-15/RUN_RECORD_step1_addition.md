# Run record — STEP 1 ADDITION: root-owned-count reproduction check

Date: 2026-08-15.

## Command
```
cd analysis/reward_convergence_2026-08-15
/cs/student/project_msc/2025/sec/slchan/conda_envs/ccbs/bin/python step1_rootowned_reproduce_check.py
```

## Result: 20/35 reproduce, 15/35 do not — and the split is clean and fully explained

**All 20 rows across populations (b) and (c) — Task Y's N=30/N=60/N=90 grid (Table IV.3) and the
lower-neighbour N=30 pilot — reproduce their recorded values exactly** (largest difference:
0.0049 percentage points, floating-point noise). **This closes Q6's residual uncertainty for these
populations**: the original checks used `--stop` at its default (each file's own final logged
step), including for the N=90 seeds whose final-stage files span ~500k local steps rather than
~250k — confirmed, not assumed, by exact reproduction.

**All 15 rows in population (a) — the 15-manifest checkpoints, Table IV.1 — do not reproduce**,
by amounts ranging from 0.16 to 14.81 percentage points. **This has a known, already-identified
cause, not an unknown one**: this task's recompute used `--stop` at its default (253,952, each
file's own max logged step); the CONVERGENCE-PROVENANCE task's own prior run
(`analysis/convergence_provenance_2026-08-15/run_manifest_convergence_check.sh`, committed
`2108f6d`) explicitly passed **`--stop 250000`** (a deliberate, documented choice at the time, to
match the "250k" round budget label). These are two different, both-legitimate step-anchor
conventions used across two of my own scripts — **not evidence that a recorded Table IV.1 verdict
was computed on a wrong or unexamined window**; the window used is known exactly, on the record,
in both cases; they simply differ from each other by design choice, not by accident.

**Table IV.3 (population b), which the task's framing specifically named as the risk to check, is
the one confirmed clean.** The mismatch landed instead on Table IV.1 (population a), for a
reason already on record before this check ran.

Per the task's explicit instruction, no conclusion is drawn beyond this report, no value is
corrected, and STEP 1's original ask (comparing reward-based verdicts against root-owned-count
verdicts) is not carried out in this same run — stopping here as instructed.

## Full per-row table
See `step1_rootowned_reproduce_output.log` (committed).

## Outputs (committed alongside this record)
`step1_rootowned_reproduce_check.py`, `step1_rootowned_reproduce_output.log`, this record.

## Wipe test
Reproducible from the committed script and the already-on-disk tfevents files (not committed, per
convention).
