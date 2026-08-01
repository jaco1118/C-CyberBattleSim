# Claims audit — numbers written into the draft, checked against the code/logs

Each entry: the claim as written, what the code actually produces, and the disposition. A CLOSED item means
the check is finished; any residual action is stated (usually a writing/relabel task, not a recompute).

---

## CA-1 — "arithmetic share of robustness cost is 69/72/76% across three bands"  [CLOSED — figures correct, LABEL wrong]

**As written in the draft:** the arithmetic (mechanical) share of robustness cost is 69, 72 and 76 per cent
**"across three bands"** (implying the 10-15 / 30-40 / 80-100 node bands).

**What the three figures actually are** (Task Z STEP 0.1; source `evidence_taskF3.md:237-239`, computed by
`taskF3_mech_analyze.py` from the per-departure `was_root` records in `f3_mech_out/` and `f3_rel_out/`):
they are **two bands under three CHURN CONDITIONS**, not three bands —

| figure | condition | mechanical (root-owned dep/ep = Σ was_root) | count-cost | share |
|---|---|---|---|---|
| 69% | **30-40**, fixed-**absolute** churn (~33%) | 5.39 [5.20, 5.58] | 7.79 | 5.39/7.79 = 0.692 |
| 72% | **80-100**, fixed-**absolute** churn (~18%) | 2.69 [2.26, 3.07] | 3.74 | 2.69/3.74 = 0.719 |
| 76% | **80-100**, fixed-**relative** churn (32.3%) | 5.06 [4.31, 5.71] | 6.66 | 5.06/6.66 = 0.760 |

There is **no 10-15 figure** in this series, and 80-100 appears **twice** (once per churn definition).

**The numbers are correct** — each is an exact event-by-event count of `was_root` divided by the count-metric
cost; verified arithmetic above. **Do not change any figure.**

**Residual action (writing only, not this project's job):** relabel the series wherever it is presented as
"three bands." The correct caption is **"30-40 fixed-absolute / 80-100 fixed-absolute / 80-100 fixed-relative
(two node bands × two churn definitions)."**

**Reconciliation with the Task-N N2 gate** (which said band 30-40 "cannot count the arithmetic component"):
no contradiction — N2 was about the `leaveown_*` file (`was_owned`, exists only at 80-100); this series uses
the F3 **mech CSVs** (`was_root`, exist at both bands). Different files. For the root-owned COUNT cost,
`was_root` is the correct arithmetic component and it covers both bands. See Task Z addendum for the full
re-examination of N2 against the mech CSVs.

---

## CA-2 — join-donor selection is non-deterministic under PYTHONHASHSEED (pre-existing defect)  [OPEN — disclose; not a Task-L regression]

**What:** the dynamic membership-**join** donor is chosen by `random.choice(available)`
(`cyberbattle/_env/cyberbattle_env.py:781`), where `available` derives from a **string-keyed set** of donor
indices (`self._used_donor_pool_indices_this_episode`, `:189`). Python randomises string hashing per process
(`PYTHONHASHSEED`), so **the set/list iteration order — and hence which donor `random.choice` picks — differs
across processes** even at a fixed RNG seed.

**Evidence:** two byte-for-byte identical runs (same code, same seed, `set_num_threads(1)`) differ; pinning
`PYTHONHASHSEED=0` collapses the difference to **0**. Discovered during Task L STEP 2's byte-identical
regression (`evidence_taskL_logging.md`, STEP 2). **Not caused by Task L** — reproduced on the unmodified code.

**Affected columns** (join events only): `delta_h_v_norm`, `attenuation_ratio_{mean,max,min,full}`, and the
deferred-attribution columns `change_type/change_fired/event_id/step_fired/visibility_lag_steps/
node_origin_is_join`. **Deterministic and unaffected:** the trajectory, `change_drift_*`, `norm_*`, counts,
and every headline figure (response rates, robustness, arithmetic share).

**Consequence, stated plainly:** the **originally-reported gate drift logs were NOT hash-pinned**, so their
join `delta_h_v_norm` and `attenuation_ratio` columns are **not bit-reproducible** run-to-run. **Bounded
impact** (stated, not used to avoid the disclosure): join events are **excluded from the ~43,000 headline
response set** (CA-1 / `evidence_taskQ.md` Q1 — the set is property + membership_leave), so no headline number
depends on these columns; and `attenuation_ratio` was already an ARTIFACT (Task T). **Action:** any re-run
(Task L STEP 3 included) must set `PYTHONHASHSEED=0`; even then it reproduces the headline figures but will
**not** bit-match those join columns from the un-pinned originals — disclose, do not treat as a regression
miss. Fixing the defect itself (seed the donor pick deterministically) is out of scope here; recorded for the
methodology.
