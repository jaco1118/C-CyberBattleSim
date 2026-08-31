# Run record — Task THREE-LOOKUPS

Date: 2026-08-15.

## Question Set 1 — the row-writing rule and the episode split

### 1.1 — the row-writing loop, quoted

`cyberbattle_env_compressed.py:1131-1132,1185` (`_log_drift_rows`):
```python
events = dynamic_events or [None]  # None -> the "no dynamic change" sanity-check row
for event in events:
    ...
    self._drift_logger.log(row)
```
and separately `cyberbattle_env_compressed.py:1211` (`_log_attribution_rows`):
```python
for node_id in newly_discovered_node_ids:
    ...
    self._drift_logger.log(row)
```
**One row = one event** (via `_log_drift_rows`, called once per step at `:776`, gated
`if log_this_step:`) **or one no-change sanity row if no event fired that step** — never one row
per event-node-pair and never one row per discovered node at that layer. A *second*, separate loop
(`_log_attribution_rows`, called at `:660` only `if drift_newly_discovered:`) writes **one
additional row per newly-discovered node** that step. Both loops write to the same
`self._drift_logger`, so a step's total row count is `(rows from _log_drift_rows, always >=1) +
(rows from _log_attribution_rows, 0 or more)`.

### 1.2 — leave-then-rediscover

**Yes, a further discovery/attribution row is written.** `newly_discovered_node_ids =
set(self.discovered_nodes) - drift_discovered_before_step` (`:658`) is a fresh per-step set
difference against the state at the *start of that same step* — it carries no episode-long memory
of node IDs that have previously been attributed. A membership_leave prunes the departing node from
`discovered_nodes` (`remove_node_common`, `cyberbattle_env.py:730-736`,
`if node_id in self.discovered_nodes: self.discovered_nodes.remove(node_id)`), so the ID leaves the
set; if that same ID later re-enters `discovered_nodes` (e.g. a join reusing the ID followed by a
fresh `Reconnaissance`), it registers as newly-discovered again on that later step, with no guard
against it.

### 1.3 — batch events

**One row, not k rows.** `_log_drift_rows`'s loop iterates over `dynamic_events` (one dict per
*event*, not per node); a batch event's `node_ids` field is the full list of touched nodes
(`self._last_dynamic_events.append({"change_type": ..., "node_ids": list(removed)})`,
`cyberbattle_env.py:468`, established in an earlier task this project), and the row's
`n_touched_nodes=len(node_ids)` field records the count — it does not fan the batch out into
multiple rows.

### 1.4 — episode split across bands

From `evidence_cards/evidence_taskT.md:78-89` ("0.4 — Stopping target and resulting episode
counts"), read directly, no computation needed:

| band | episodes |
|---|---|
| 10-15 | 750 |
| 30-40 | 1,910 |
| 80-100 | 2,000 |
| **total** | **4,660** |

Independently re-confirmed from `attenuation_drift_logs/drift_<band>.csv` directly (distinct
`(seed, scenario_id, episode)` triples): 750 / 1,910 / 2,000 — exact match.
**Confirmed, not corrected: band 80-100 = 2,000 = 5 seeds × 400-episode budget, exactly**, and per
`evidence_taskT.md`'s own per-seed shortfall table (already on record from an earlier task this
project) all five 80-100 seeds hit the 400-episode cap on `membership_join` without reaching
target — consistent with "exhausted a 400-episode budget on each of five seeds."

### 1.5 — property change configuration

`evidence_cards/evidence_taskT.md:126`: `dynamic_mode: both` (line 41 of the checkpoint's own
`train_config.yaml`), `patch_service_dynamic_enabled: false` (line 71). **Membership-only** (leave +
join active, property mechanism disabled by config). Independently re-confirmed empirically: zero
`property` rows in `attenuation_drift_logs/drift_<band>.csv` at any of the 3 bands (direct count,
this task).

### 1.6 — reconciliation (script: `compute_q1_reconciliation.py`, committed `7989d9a`)

All four headline numbers reproduce exactly from `attenuation_drift_logs/` (the confirmed
Section IV.1 source dataset):

| quantity | thesis | this task, independently computed |
|---|---|---|
| episodes | 4,660 | 4,660 |
| rows | 1,602,999 | 1,602,999 |
| steps | 1,355,136 | **1,355,136** (not found recorded anywhere as a standalone figure — searched `grep -rn "1355136\|1,355,136"` across evidence cards/log/analysis output and `git log --all -S"1355136"`, all empty — computed here as `sum over episodes of nunique(step)`, the one natural reading of "total environment steps" from this data, and it reproduces the thesis figure exactly) |
| retained (attributable-change) rows | 290,799 | 290,799 (290,799/4,660 = 62.4032, matches "62.4") |

**The excess (1,602,999 − 1,355,136 = 247,863) reconciles exactly, but not by the single mechanism
the thesis text names.** Decomposition (this task, same script + one follow-up query):
- Attribution rows alone (`event_phase == "attributed"`): **235,945**.
- Excess from *multi-event steps* — steps where `_log_drift_rows`'s own loop wrote more than one
  non-attributed row because more than one dynamic event fired in the same step (e.g. a leave and a
  join together; observed up to 5 rows in one step at band 80-100): **11,918**.
- `235,945 + 11,918 = 247,863` — exact.

**But the thesis's own narrative framing does not hold up under direct measurement.** The text
attributes the excess to "the 2.1 per cent of steps that reveal new nodes... about 8.7 extra rows
on each such step." Measured directly: **1.360% of all steps have at least one attribution row**
(18,429 / 1,355,136), not 2.1%; and the average is **235,945 / 18,429 = 12.80 attributed rows per
such step**, not 8.7. Checked several alternative denominators to see if a different reading of
"2.1%"/"8.7" would reconcile (2.1% of total rows, of retained rows) — none matched either. **Not
reconstructed or guessed at further, per this task's own instruction** — reported as measured.

**Bottom line, Question Set 1: (a) ANSWERED.** The four headline numbers reconcile exactly
(`compute_q1_reconciliation.py` + `evidence_cards/evidence_taskT.md`), and the row-writing
mechanism is now fully quoted from source. But the specific narrative the thesis currently uses to
explain the 247,863-row excess ("2.1% of steps... ~8.7 extra rows each") is not correct as stated —
the real figures are 1.36% and 12.80, and a second, smaller mechanism (multi-event steps, not
attribution) accounts for 11,918 of the 247,863 and should be disclosed alongside it.

---

## Question Set 2 — convergence figures for the three main bands

Researched via a dedicated background search agent (thorough multi-file, cross-referenced
evidence-card and on-disk read), then independently spot-checked here (both quotes below verified
directly by re-reading the cited files myself, not taken from the agent's report alone).

### 2.1/2.2 — do the four quantities exist, on the same basis as Table IV.2?

**Two different populations share the band labels 10-15/30-40/80-100, and they must not be
conflated:**

1. **The 15-manifest checkpoints**
   (`cyberbattle/agents/logs/trpo_250k_tuned_compressed_band<band>_seed<seed>_2026-07-26_*`,
   confirmed on disk, dated 2026-07-26) — these back the attenuation/pooling/drift figures this
   whole project's evidence trail is built on (`attenuation_drift_logs`, `cx_step2_registration`,
   the SNR/never-attributed work, etc.). **`evidence_cards/evidence_taskW.md:65-75`, quoted
   directly, verified myself**: "the **5-seed TRPO multi-topology attenuation-gate agents**... at
   **250k**... **are NOT the F1/F2 single-topology agents that Task F4 retrained**... So **the gate
   figures are unaffected by F4 and remain valid on their 250k gate checkpoints**." **No convergence
   table (or any of the four quantities) was ever computed for this specific checkpoint population.**

2. **A parallel F1_static/F2_static checkpoint population** (job-scratch-trained 2026-07-28→30,
   same band labels, different topology/training setup: 30-40 on one shared topology, 80-100 on 5
   distinct topologies) — **Task F4 computed exactly the four quantities the task describes, for
   all three bands**, using a pre-registered criterion (`evidence_cards/evidence_taskF4.md:16-20`:
   CONVERGED iff mean|Δ%|<5% across seeds AND ≥4/5 seeds individually <5%, on `train/Root owned
   nodes`, 50k windows) applied via a real, script-backed checkpoint evaluation
   (`conv_eval.py`/`conv_analyze.py`, preserved on branch `attenuation-pooling-scale` commit
   `c05a16a` — confirmed NOT an ancestor of current HEAD — with the raw per-seed CSVs
   (`conv_results.csv`, `conv_results_750k.csv`) still present on disk at
   `~/.claude/jobs/0dfa230d/tmp/taskF1/`, confirmed present).

3. **Table IV.2 itself is most likely neither of the above.** Its own described structure ("mean
   relative deviation... seeds within tolerance... training used... converged/not-converged
   verdict," per this task's own framing) matches **Task Y's N=30/N=60/N=90 degree-controlled grid**
   almost verbatim (`evidence_cards/evidence_taskY.md:334-336`, quoted by the research agent: "|
   mean|Delta%| | ... | | within-band | ... | | verdict | CONVERGED (250k) | NOT CONVERGED (1.25M
   cap) | NOT CONVERGED | | training used | 1.25M total... | 6.25M total... | 4.75M total... |") —
   a third, unrelated population (synthetic topologies calibrated to a fixed mean degree ≈22,
   crossed with node count, not the natural band manifold), confirmed a genuinely separate line of
   work from the band-labelled `yprobe_n90_*` folder name alone (`evidence_taskY.md:250-256`).

**So: (2.1) the four quantities do NOT exist for the three main bands on the same basis as Table
IV.2** (i.e. computed for the 15-manifest checkpoints specifically) — **but they DO exist, computed
on the real pre-registered basis Table IV.2's own methodology matches, for a same-band-labelled but
differently-trained parallel checkpoint population.**

### 2.2 — the four quantities, for the F1_static/F2_static population (verified directly, `evidence_cards/evidence_taskF4.md:112-119` for the 500k table, `:134-142` for the 750k re-check)

| band | mean\|Δ%\| | seeds within 5% | training used | verdict |
|---|---|---|---|---|
| 10-15 | 1.0% | 5/5 | 500k (converged, no further training) | **CONVERGED** |
| 30-40 | 1.1% | 5/5 | 500k (converged, no further training) | **CONVERGED** |
| 80-100 (at 500k) | 6.4% | 1/5 | 500k, then a pre-registered second resume | **NOT CONVERGED** |
| 80-100 (at 750k, final) | 5.4% | 3/5 | 750k (training stopped here, per the pre-registered ceiling) | **NOT CONVERGED** |

Both rows for 80-100 are genuine, independent computed checkpoints (not the same number reported
twice) — the 750k re-check used a fresh windows-650/700/750k evaluation on the resumed checkpoints.

### 2.3 — what IS on record for band 80-100's training

Confirmed directly: **total steps per seed = 750,000** (final, after the pre-registered stopping
rule triggered — "if 80-100 still fails at 750k, STOP and report... training STOPS here",
`evidence_taskF4.md:141-142`, quoted). **Two extension stages**: an initial run to 250k (the F2/F3
baseline), a first resume 250k→500k (checked, NOT CONVERGED, 6.4%/1-of-5), a second resume
500k→750k (checked, still NOT CONVERGED, 5.4%/3-of-5, training stopped per the pre-registered
ceiling — not open-ended). The convergence-related statement ("the 80-100 band does NOT reach the
convergence criterion even at 750k... and is still measurably improving") traces to a computed
number (the per-seed window-Δ% table quoted above), not an impression — this is a real finding
about the F1_static/F2_static population specifically.

### 2.4 — "three times its original training budget"

**The exact phrase does not appear anywhere** — searched by the background agent across evidence
cards, dissertation log (both copies), and `git log --all -p -S` for several close variants, all
empty; no thesis source file (`.tex` or otherwise) exists anywhere in this repository to check a
draft directly. **A closely related, fully computed statement does exist and is the likely origin**:
`evidence_taskF4.md:141-142` / `dissertation_log_v2.md:218`, "the 80-100 band does NOT reach the
convergence criterion even at 750k — **~3× into the released study's budget region**" — i.e.
750k ÷ 250k = 3×, tied directly to the NOT-CONVERGED-at-750k verdict above, not a vague estimate.
**A specific seed count passing the stopping rule does exist**: 3/5 seeds within tolerance at 750k
(quoted above) — not all 5, and not zero.

### Bottom line, Question Set 2

**(b) NOT ON RECORD for the four quantities computed specifically on the 15-manifest checkpoints
that back this project's actual reported band-comparison figures — to obtain them would require
running the same pre-registered convergence check (`conv_eval.py`/`conv_analyze.py`, recoverable
from branch `attenuation-pooling-scale` commit `c05a16a`) against those specific checkpoints, which
were never subjected to it.** BUT this is not a bare absence: the same four quantities, computed
correctly and on a real, pre-registered, script-backed basis, exist for a same-band-labelled
parallel checkpoint population (F1_static/F2_static) — reported in full above — and the
"band 80-100 never converged" qualifier used downstream in four other results traces to that real
computation. What the thesis cannot currently say, honestly, is that this specific finding was
established on the checkpoints its own headline attenuation/pooling numbers come from — only on a
parallel population sharing the same band label and seed numbering. Likely candidate for what Table
IV.2 itself actually reports: Task Y's N=30/N=60/N=90 degree-controlled grid, a third, distinct
population again — not the 15-manifest, and not F1_static/F2_static.

---

## Question Set 3 — how the ablation's MDE was computed

### 3.1 — the expression, quoted, both definitions found

**Bands 30-40 / 80-100** (`evidence_cards/evidence_taskZ.md:181,231-233`, single-arm SD; script not
named in the card's own text, but the table's own arm1-SD column equals the reported MDE exactly:
0.267/6.060 both appear identically as "arm1 SD" and as "MDE (single-arm SD)"):
```
MDE = arm1's own between-seed SD (5 static-condition seed values), band 30-40: 0.267 nodes;
                                                                     band 80-100: 6.060 nodes
```

**Band 10-15** (`compute_z_mde.py`, commit `91df45b`, recovered via `git show` — not an ancestor of
current HEAD, extracted and reproduced in `analysis/three_lookups_2026-08-15/rq2b_reproduction/`,
committed `4a4872a`, then run on its own exact original preserved 12 per-seed CSVs):
```python
static_pool = []
for arm in (1, 2, 3):
    sv = seed_vec(df, arm, topo, "static"); cv = seed_vec(df, arm, topo, "change")
    static_pool.append(sv)
    ...
static_pool = np.concatenate(static_pool)
mde = float(np.nanstd(static_pool, ddof=1))
```
Reproduced exactly: topology #44 (primary) → **MDE = 0.085 nodes** (n=15, 0.9% of static mean
9.737); topology #34 (secondary) → **MDE = 0.162 nodes** (n=15, 2.1% of static mean 7.555). Both
match Table IV.4's stated values to three decimal places.

### 3.2/3.3 — (a) or (b)?

**(a), for every row — none use a power calculation.** Neither `compute_z_mde.py` nor the script
behind `evidence_taskZ.md`'s 30-40/80-100 table contains any power/beta/alpha term, effect-size
formula, or `power=` argument anywhere — both are pure between-seed-SD noise floors. Searched:
`compute_z_mde.py`'s full text (quoted above in full) has no such term; `evidence_taskZ.md`'s own
prose (`:231`, "MDE = single-arm static between-seed SD") states the same thing directly. **3.3 does
not apply** (it is not (b)) — no separate 80%-power figure is needed alongside; the caption's core
claim, "the MDE is one between-seed standard deviation," is correct in form for every row.

### 3.4 — same definition for every row?

**No — not the same population, despite both being "a standard deviation."** Bands 30-40/80-100 use
**arm 1's own 5 static-condition seed values alone** (single-arm SD, n=5). Band 10-15 uses **all
three arms' static-condition seed values pooled together** (n=15,
`np.concatenate([arm1, arm2, arm3])` before taking the SD) — exactly the "combine spread across
three arms" the task's own framing named. Both are legitimately "a between-seed standard deviation,"
so the caption's literal claim survives — but it implies one uniform definition when there are in
fact two, computed over different-sized, differently-composed populations (5 single-arm values vs.
15 pooled-across-arms values), and a reader comparing MDE magnitudes across bands in Table IV.4
would be comparing two different statistics without being told so.

**Bottom line, Question Set 3: (a) ANSWERED.** MDE is a between-seed standard deviation (not a
power-based calculation) on every row, established by direct source reading and, for band 10-15,
exact reproduction from the original preserved data. The caption's core claim is correct. It should
be amended to disclose that the pooling population differs by band (single-arm at 30-40/80-100,
pooled-across-3-arms at 10-15), which the current caption does not state and a reader doing a
cross-band comparison would want to know.

## Outputs (committed alongside this record)
`compute_q1_reconciliation.py`, `q1_run_output.log` (Question Set 1); no new script for Question
Set 2 (verify-only, source/evidence-card reading plus one dedicated research pass, all quotes
independently re-verified); `rq2b_reproduction/` (`compute_z_mde.py`, all 12 original preserved
per-seed CSVs, `q3_reproduction_output.log`) (Question Set 3).

## Wipe test
Question Set 1 reproducible from the committed script and the already-on-disk
`attenuation_drift_logs/` (raw data not committed, per convention). Question Set 2's findings trace
to already-committed evidence cards (`evidence_taskW.md`, `evidence_taskF4.md`, `evidence_taskY.md`)
plus one fact that requires a specific commit to recover (`conv_eval.py`/`conv_analyze.py`, branch
`attenuation-pooling-scale` commit `c05a16a`, not on this branch). Question Set 3's band 10-15
reproduction is fully self-contained and committed (script + all 12 source CSVs); bands 30-40/80-100
trace to `evidence_cards/evidence_taskZ.md`'s own already-committed table.
