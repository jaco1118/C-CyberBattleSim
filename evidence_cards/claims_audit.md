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

---

## CA-3 — "forced-action replay from actions.npy, Task L verified byte-identical" over-scopes what was validated  [CLOSED — scope correction; no headline number affected]

**As written / as relied on:** `evidence_taskCX.md:413` ("Forced-action replay from actions.npy (Task L
verified byte-identical): 4,410 episodes across 3 bands × 5 seeds"), and the RQ2C-1 task premise ("actions.npy
per band/seed — Task L verified byte-identical"), both read as: the full `attenuation_step3_logs` sweep can be
faithfully **replayed** from its stored per-seed `actions_s*.npy` to recreate the recorded trajectory.

**What was actually validated as byte-identical** — two SMALL controlled tests, neither the sweep itself:
1. Task L STEP 2 `drift_regression_check` (`evidence_taskL_logging.md` STEP 2): **2 topologies/bands, 800
   steps, a FIXED `RandomState(12345)` action sequence** (a synthetic fixed-action stream, *not* the stored
   policy actions), `old==new_off==new_on`, 0 cells — proves the **instrumentation** is inert, not that a
   stored-policy trajectory replays.
2. Task L Amendment 1 (`evidence_taskL_logging.md:219`): **ONE full evaluation episode**, policy in the loop,
   PYTHONHASHSEED=0 — a single-episode reproduction, not the 4,410-episode sweep.

**What was NOT validated, and is in fact false for the sweep:** the `attenuation_step3_logs` recording ran
under `YEG=1` **only** — it never went through the CX_DIAG per-seed seeding block (`torch/np/random.seed(seed)
+ set_num_threads(1)`), so its trajectory was not seed-pinned and its stored actions are **not** faithfully
replayable. **Empirical proof (RQ2C-1, 2026-08-02):** replaying `actions_s42.npy` for band 10-15 produced a
**completely different** single-node-leave sequence — steps `30,72,86,92,97,126,199,…` vs the recorded
`2,19,26,44,74,98,231,248,…` — with different episode lengths. Forcing the stored action *vectors* onto a
differently-seeded churn stream makes `find_closest_action_embedding` snap each vector to whatever is nearest
in a **mismatched** graph: a self-consistent-looking but **invalid "Frankenstein" trajectory**, not the
agent's real behaviour. (The CX replay `cx_step2_replay` DID reproduce only because both the CX recording and
its replay used **CX_DIAG** per-seed seeding — a different, seed-pinned run; the attenuation sweep was not.)

**Do not conflate:** "byte-identical replay" (validated only on the fixed-action regression + one episode) is
a **stronger, different** property than the sweep's headline reproduction. The sweep's reproduction that WAS
checked (`evidence_taskL_logging.md` STEP 3: max-slice `membership_leave` response rate 98.4/84.3/42.5 vs
98.5/84.0/43.0, Δ<1pp) is **distributional**, inside the ~3–4pp hash-seed spread — NOT trajectory-identical.
The <1pp property is the correct, weaker claim to cite for the sweep.

**Consequence / bounded impact:** no headline number depends on replaying the stored sweep actions — the
headline reproduction is the distributional <1pp check, which stands as such. This corrects a **methodology**
statement only. **Action taken (RQ2C-1):** because the stored actions are not faithfully replayable, RQ2(c)
was measured NOT by replay but by a **fresh, per-seed-seeded stochastic live rollout** of the same
checkpoints/seeds/bands under the same attenuation config, computing the pre/post counterfactual inline (see
`evidence_taskRQ2C.md`). Recorded for the methodology; no recompute of any existing figure.
