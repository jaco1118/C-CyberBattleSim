# Run record — N=60 robustness column, Amendment 2 STAGE A/B (ci=7, ci=8 churn-only)

PROVENANCE BANNER: commissioned 2026-08-15, after the convergence criterion changed
from root-owned-node-count to episode reward (independent decision), which made the
N=60 cell newly eligible for Table IV.5.

Date run: 2026-08-16.

## Statement of procedure
Three full 200-episode/5-seed evaluations of this cell were run, at
change_interval 5, 7, and 8. ci=7 was selected on achieved churn alone, against a
tolerance (42.0% +/-1.5pp) declared before any of the three ran. Robustness was not
computed or inspected for any candidate until after ci=7 was selected.

| change_interval | pooled achieved churn | outcome |
|---|---|---|
| 5 | 47.18% | rejected, out of band |
| 7 | 42.23% | selected |
| 8 | 40.12% | rejected, out of band |

None of the three runs is deleted; all three remain on disk and in this record.

## Why this run exists
STEP 2 at ci=5 (`y_robustness/out/n60/`, committed `9a49be0`) achieved 47.18% pooled
churn against the 42.0% reference -- outside the +/-1.5pp band. That run is REJECTED
as a table column per Amendment 2, but is NOT deleted and stays on disk/in the
record as a measured, disclosed miss. Amendment 2 abandons 30-episode single-seed
probe calibration for this cell (it could not resolve ci=5 from ci=6: probe read
42.17%, the 200-episode main run on the same seed read 45.97%) and instead runs the
full 200-episode/5-seed design directly at two candidate change_interval values,
selecting on ACHIEVED CHURN ALONE, with robustness deliberately not computed at
this stage.

## Script change disclosed
`run_stage_n60.sh` was modified beyond the CI_N60 value: added a second required
env-var guard, `N60_TAG`, used only to route OUTDIR (`y_robustness/out/n60_${TAG}`)
and the log filenames (`logs_run/n60_<seed>_${TAG}_<cond>.log`) to a tag-specific
path. This was necessary because the script's OUT_DIR and log-file naming were
hardcoded to the exact paths ci=5's committed run already occupies, and
taskF2_eval.py names its CSVs by (seed, eval_cond) only, not by change_interval --
running ci=7 or ci=8 through the unmodified script would have silently overwritten
the committed ci=5 CSVs/logs (score_static_seed<seed>_eval<cond>.csv collision,
the exact mechanism already identified in STEP 1's own incident and, independently,
in the JOIN CAP PROVENANCE task's finding about the CX run_metadata writer). No
other line of the script changed: same seeds, same checkpoints/topologies, same
episode count, same thread caps, same nice level, same failure accounting.
One fix needed after the first edit: an apostrophe inside the N60_TAG guard's
error message broke bash's `${VAR:?message}` parsing (`unexpected EOF`) on the
first launch attempt; both ci=7 and ci=8 failed immediately with a syntax error
and produced no output. Fixed by rewording the message to avoid the apostrophe,
verified with `bash -n`, then both re-launched from a clean state (confirmed no
partial output existed from the failed attempts before re-running).

## STAGE A: design
5 seeds x {static, membership_matched} x 200 episodes, ci=7 and ci=8, each as its
own 10-process concurrent batch (`N60_TAG=ci7`/`ci8`), run concurrently with each
other. Same checkpoints/topologies as ci=5 (`logs/yN60_s<seed>_stg7_2026-08-05_23-47-20`,
CKPT_STEP=250000, `graphs_yN60_s<seed>_2026-08-03_17-15-34/1`).

## STAGE B: churn only -- no robustness figure computed, read, or reported

### ci=7
| seed | leave/ep | churn (% of 60) |
|---|---|---|
| 42  | 24.87 | 41.45% |
| 100 | 24.91 | 41.52% |
| 123 | 25.92 | 43.20% |
| 200 | 25.34 | 42.23% |
| 300 | 25.65 | 42.75% |

Pooled mean leave/ep = 25.338 -> **churn = 42.23%**. Distance from 42.0% = **+0.23pp**.
**In band [40.5, 43.5]%: YES.**

### ci=8
| seed | leave/ep | churn (% of 60) |
|---|---|---|
| 42  | 23.70 | 39.50% |
| 100 | 23.43 | 39.05% |
| 123 | 24.32 | 40.53% |
| 200 | 24.55 | 40.92% |
| 300 | 24.37 | 40.62% |

Pooled mean leave/ep = 24.074 -> **churn = 40.12%**. Distance from 42.0% = **-1.88pp**.
**In band [40.5, 43.5]%: NO.**

### Episode counts
ci=7: 2000 launched (5 seeds x 2 conditions x 200), 2000 completed, 0 dropped.
ci=8: 2000 launched, 2000 completed, 0 dropped. Confirmed from every one of the 20
`[F2 ...] episodes=200 ...` summary lines; 0 Tracebacks in any of the 20 logs;
orchestrator `failed=0` for both batches (all 20 PIDs individually exit=0).

## Selection, per the rule fixed in advance (Amendment 2)
Exactly one of the two lands in band: ci=7 does, ci=8 does not. Per the pre-declared
rule ("If exactly one of ci=7 and ci=8 lands in band, it is selected"), this is a
mechanical outcome of the rule applied to the numbers above, not a judgement call.
No robustness figure for either candidate was computed, opened, or looked at before
selection.

## STAGE C: ci=7 robustness (raw root_owned count, matched-undisturbed-paired)
| seed | static root_owned mean | matched root_owned mean | robustness |
|---|---|---|---|
| 42  | 26.9300 | 18.9300 | 0.7029 |
| 100 | 28.3250 | 18.6100 | 0.6570 |
| 123 | 30.9350 | 19.4300 | 0.6281 |
| 200 | 29.8100 | 20.2150 | 0.6781 |
| 300 | 28.0650 | 20.6050 | 0.7342 |

Undefined (zero static mean) seeds: 0.

Cell mean robustness = 0.6801. Cell SD = 0.0409.

## Outputs (committed alongside this record)
`n60_ci7/score_static_seed<seed>_eval{static,membership_matched}.csv`,
`n60_ci7/leaveown_static_seed<seed>_evalmembership_matched.csv` (5 seeds each);
same for `n60_ci8/`. Plus the 20 `logs_run/n60_*_ci{7,8}_*.log` files and both
orchestrator logs (`run_orchestrator_n60_ci7.log`, `run_orchestrator_n60_ci8.log`).
NOT committed: `drift_*.csv` (bulk, consistent with this directory's convention).
`run_stage_n60.sh`'s N60_TAG addition committed separately.

## Wipe test
Reproducible from `run_stage_n60.sh` with `CI_N60`/`N60_TAG` set per run, and the
already-on-disk checkpoints/topologies (not committed, per convention).
