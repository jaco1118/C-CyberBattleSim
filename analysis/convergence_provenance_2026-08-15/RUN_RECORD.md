# Run record — Task CONVERGENCE-PROVENANCE

Date: 2026-08-15.

## Question 1 — can the reported checkpoints be convergence-checked now?

### 1.1 — what survives, named

Every one of the 15 manifest checkpoint run folders
(`cyberbattle/agents/logs/trpo_250k_tuned_compressed_band<band>_seed<seed>_2026-07-26_*/`,
confirmed present for all 3 bands × 5 seeds) contains:
- `TRPO_x_control_SecureBERT/TRPO_1/events.out.tfevents.<...>` — a real TensorBoard event log,
  containing the scalar tag **`train/Root owned nodes`** (confirmed present; 62 logged points for
  the band-10-15/seed42 file, steps 4,096 → 253,952) among 60 total scalar tags.
- `TRPO_x_control_SecureBERT/checkpoints/1/checkpoint_<N>_steps.zip` at every 5,000-step interval
  from 5,000 to 250,000 (50 checkpoints per seed, dense, single continuous run — no gap, no
  separate resumed stage).
- `train_config.yaml`, `seeds.yaml`, `split.yaml`, `app.log`.

**No per-seed "stage boundary" records exist because there are no stages** — each of the 15 is one
uninterrupted training run to 250k, unlike the F1_static/F2_static population's explicit
250k→500k→750k resume structure. Total steps per seed: 250,000 (target); the tfevents file's own
final logged step is a few thousand steps past this (e.g. 253,952), a normal rollout-batch overshoot
matching `compute_convergence_check.py`'s own documented behaviour for exactly this situation.

### 1.2/1.3 — computed, all three bands (script: `run_manifest_convergence_check.sh`, committed
`2108f6d`, applying the already-committed, unmodified `compute_convergence_check.py`, commit
`4173c53`, confirmed an ancestor of current HEAD — the exact tool Table IV.2's own population
(Task Y) uses, confirmed by that tool's own docstring: "Same method used for the real 80-100 band
and every N=90 cell check")

Method: `train/Root owned nodes`, 50,000-step windows, `--stop 250000`, threshold 5%, ≥4/5 seeds —
identical invocation style to Task Y's own 250k-stage checks.

| band | seed | Δ% | within band (<5%)? |
|---|---|---|---|
| 10-15 | 42 | +7.20% | no |
| 10-15 | 100 | +21.98% | no |
| 10-15 | 123 | −4.81% | YES |
| 10-15 | 200 | +36.87% | no |
| 10-15 | 300 | −1.92% | YES |
| 30-40 | 42 | −0.34% | YES |
| 30-40 | 100 | −18.20% | no |
| 30-40 | 123 | +4.55% | YES |
| 30-40 | 200 | +3.05% | YES |
| 30-40 | 300 | −0.33% | YES |
| 80-100 | 42 | +23.87% | no |
| 80-100 | 100 | +1.68% | YES |
| 80-100 | 123 | +15.56% | no |
| 80-100 | 200 | −10.09% | no |
| 80-100 | 300 | +48.37% | no |

| band | mean\|Δ%\| | seeds within tolerance | training used | verdict |
|---|---|---|---|---|
| **10-15** | **14.55%** | **2/5** | 250k (single stage, no extension attempted) | **NOT CONVERGED** |
| **30-40** | **5.29%** | **4/5** | 250k (single stage, no extension attempted) | **NOT CONVERGED** (mean criterion fails narrowly; seed-count criterion alone would pass) |
| **80-100** | **19.91%** | **1/5** | 250k (single stage, no extension attempted) | **NOT CONVERGED** |

**FINDING, and it reverses the thesis's implicit framing:** under Table IV.2's own exact method,
applied to the checkpoints that actually back this project's reported results, **all three bands
fail to converge at 250k — not only band 80-100.** Band 10-15 (mean 14.55%, 2/5) and band 80-100
(mean 19.91%, 1/5) are comparably far from the criterion; band 30-40 (mean 5.29%, 4/5) narrowly
misses on the mean threshold alone. This is a materially different picture from the F1_static/
F2_static and Task-Y populations, where 10-15 converged cleanly (1.0–4.18% depending on population)
at the same 250k point. A plausible, disclosed-not-suppressed reason: the 15-manifest checkpoints
were trained with 8-topology switching (`RandomSwitchEnv`, `switch_interval_episodes`, established
earlier this project), unlike F1_static (one shared topology) or F2_static (topology-per-seed but
still single-topology per run) — episode-to-episode topology switching is a real, additional source
of training-curve variance that this metric cannot distinguish from genuine policy instability. This
is offered as a candidate explanation, not a computed fact, and does not change the measured result
above.

### 1.4 — N/A, given 1.2/1.3 answered directly.

### 1.5 — is the criterion even applicable to a single-stage run?

**Yes, applicable, and this needed checking rather than assuming.** Read
`compute_convergence_check.py` in full: the "final training stage" the criterion measures is not a
separately-resumed training leg — it is the run's own final 50,000-step window compared against the
immediately preceding 50,000-step window of the *same continuous* training curve (`windows are
anchored at the run's own final logged step`, docstring line 12; `pre, npre = wmean(stop-2*window,
stop-window); fin, nfin = wmean(stop-window, stop)`, lines 60-61). A single continuous 250,000-step
run has two such windows available ([150k,200k) and [200k,250k)) with no resume needed. The phrase
"final training stage" in the task's own framing is a red herring for these runs specifically — the
criterion needs a training curve at least `2×window` long, which 250k comfortably satisfies; it does
not need multiple resumed stages, and none of the 15 manifest checkpoints have any.

### Bottom line, Question 1
**(a) ANSWERED**, for all three bands, using Table IV.2's own exact method on already-logged
training records — no training, resumption, or new evaluation. Artifact:
`analysis/convergence_provenance_2026-08-15/manifest_convergence_output.log` (raw tool output) +
the table above. All three bands are NOT CONVERGED at 250k on the checkpoints that actually back
this project's reported attenuation/pooling/SNR figures.

---

## Question 2 — which population does Table IV.2 report?

### 2.1/2.2 — traced exactly, not by resemblance

**Task Y's N=30/N=60/N=90 degree-controlled grid, POST-EXTENSION final state** (the pre-extension
table in `evidence_cards/evidence_taskY.md:325-336` — 4.18/4.52/5.08%, training totals 1.25M/6.25M/
4.75M — was itself superseded five days later and does NOT match Table IV.2's figures; the TRUE
match requires combining two dated entries):

- **N=30**: `evidence_taskY.md:309-310` — "CONVERGED at stage 1 (250k). mean|Delta%|=4.18%, 4/5
  within band." Unchanged after this (no extension needed) — training used = 1.25M (5 seeds × 250k).
  **Matches Table IV.2's stated 4.18%, 4/5, 1.25M exactly.**
- **N=60**: `evidence_taskY.md:379-384` (STEP 2.1, the 2026-08-06 extension) — stage 7 (1.75M, the
  new cap after a disclosed extension beyond the original 1.25M): "mean 5.70%, 3/5." All 5 seeds
  extended together (group-lockstep) to 1,750,000 each → training used = 5 × 1,750,000 = **8,750,000
  = 8.75M**. **Matches Table IV.2's stated 5.70%, 3/5, 8.75M exactly.**
- **N=90**: `evidence_taskY.md:399-404` (STEP 2.2, same extension) — final per-seed steps: seed42
  1.25M, seed100 1.5M, seed123 500k, seed200 500k, seed300 1.0M (only the two seeds not already
  within band were extended). Sum = 1.25+1.5+0.5+0.5+1.0 = **4.75M exactly**. "mean|Delta%| = 2.18%
  (<5%), within-band = 5/5 → CONVERGED." **Matches Table IV.2's stated 2.18%, 5/5, 4.75M exactly.**

**Important side-note, disclosed rather than silently reconciled**: an *earlier* "4.75M" figure for
N=90 (in the pre-extension table) was independently found WRONG by this project's own check
(`dissertation_log_v2.md:1123-1139`, "Task Y-N90-STEPCHECK": true pre-extension sum was 4,250,000 or
4,288,512 depending on method, not 4,750,000). The POST-extension 4.75M that matches Table IV.2 is a
*different, independently correct* number (4,250,000 + 2×250,000 extension = 4,750,000 exactly, from
the explicit per-seed table above) — not a case of the debunked figure resurfacing unnoticed.

### 2.3 — same population or different?

**Different from both the 15-manifest checkpoints and F1_static/F2_static.** Confirmed:
`compute_convergence_check.py`'s commit message describes it as reused for "the real 80-100 band and
every N=90 cell check" — i.e. the SAME *tool* is shared across all three populations, but the
*checkpoints* are not. Task Y's own topologies are freshly generated, degree-calibrated (mean
knows-out-degree ≈20, explicitly controlled — `evidence_taskY.md`'s own STEP 0), structurally
distinct from both the 15-manifest's natural-degree multi-topology bands and F1_static/F2_static's
single/multi-topology-per-band setup.

### Bottom line, Question 2
**(a) ANSWERED, no hedge**: Table IV.2 reports Task Y's N=30/N=60/N=90 degree-controlled grid, in
its POST-EXTENSION final state (`evidence_cards/evidence_taskY.md` STEP 1 + STEP 2;
`dissertation_log_v2.md`'s 2026-08-03 and 2026-08-06 entries). Every one of the six reported figures
(three mean|Δ%|, three seeds-within-tolerance, implicitly the three training totals) traces to an
exact, quoted, dated source — no figure matched by resemblance.

---

## Question 3 — the training-budget statement

### 3.1 — confirmed, artifact named

`evidence_cards/evidence_taskF4.md:134-142` (re-verified directly here, same as Task THREE-LOOKUPS):
"Per-seed window Δ%: −3.1 / +4.8 / +2.1 / +7.9 / +8.8 → **mean|Δ%| = 5.4%, 3/5 within 5% → NOT
CONVERGED at 750k**... the 80-100 band does NOT reach the convergence criterion even at 750k —
**~3× into the released study's budget region**." Both figures confirmed: 750,000 ÷ 250,000 = 3×
exactly; 3 of 5 seeds within tolerance at 750k, both stated directly, not derived.

### 3.2 — which population?

**F1_static/F2_static only.** This specific statement (750k reached, 3/5 seeds, ~3× budget) is a
fact about *that* population's own training history — it required two disclosed resume stages
(250k→500k→750k) that only F1_static/F2_static's checkpoints ever went through. **It does not, and
cannot, transfer to the 15-manifest checkpoints**: those were trained in a single stage capped at
250k and were never resumed to 500k or 750k — there is no 750k checkpoint for them to check, and
none was requested or run in this task (out of scope, would require training). What Question 1 *did*
establish for the 15-manifest checkpoints is a different, new fact at their own actual budget (250k):
none of the three bands converge there either, not just 80-100 — but whether they would also fail at
750k, like F1_static/F2_static did, is unanswered and cannot be answered without further training.

### Bottom line, Question 3
**(a) ANSWERED**: the 750k/3-of-5/~3× statement is confirmed exactly and is specific to
F1_static/F2_static; it does not hold for, and has never been checked against, the 15-manifest
checkpoints.

## Outputs (committed alongside this record)
`run_manifest_convergence_check.sh`, `manifest_convergence_output.log`, this record.

## Scope note
Per this task's explicit instruction, no evidence card or the dissertation log was edited in this
task, even though the Question 1 finding (all three bands fail Table IV.2's own criterion, not only
80-100) is exactly the kind of result this project's standing practice would normally log
immediately. That update is a deliberate follow-up, not done here.

## Wipe test
Reproducible from the committed script and the already-on-disk tfevents files under
`cyberbattle/agents/logs/trpo_250k_tuned_compressed_band*_2026-07-26_*/` (not committed, per
convention; the driver script and its output are).
