# Task SNR-TRACE — identifying the exact statistic behind the published 0.492 / -0.804 SNR figures

Branch: `snr-abs-sensitivity` (continuation of the SNR-ABS investigation; no new branch, per the
task's own header — this task reads and computes, it edits no source).

## STEP 0 — resolving the internal inconsistency before proceeding

**0.1 — which population/filter produced 0.3897, and which produced "0.12-0.21"?**

- **0.3897** = `analysis/snr_abs_2026-08-22/run_output.log` AMENDMENT 1: median(snr_abs) over the
  **full pooled acting-steps population**, all bands, all `n_discovered` — 18,658 rows. Not
  restricted to `n_discovered>=80` at all.
- **"0.12-0.21"** = the range spanned by SNR-ABS's several attempts to reproduce the *point
  figure* specifically, all restricted to the **`n_discovered>=80` (band 80-100) population**,
  2,032 rows: row-level median snr_current=0.1543, episode-level "reaches ≥80" median=0.1839,
  fitted-curve-at-n=100 values 0.12-0.19. These are two different populations answering two
  different questions (a pooled-everything summary vs. an attempted reproduction of the
  ~100-discovered-nodes point figure) — not a contradiction, but worth stating plainly rather
  than left ambiguous.

**0.2 — median snr_current vs median snr_abs, side by side, on the `n_discovered>=80` population**
(already on record in `analysis/snr_abs_2026-08-22/run_output.log`):

| statistic | value | 95% CI | n |
|---|---|---|---|
| median snr_current | 0.1543 | [0.1385, 0.1689] | 2032 |
| median snr_abs | 0.1546 | [0.1393, 0.1691] | 2032 |

Both land in the same ~0.15 neighbourhood — consistent with each other, and with the "0.12-0.21"
range above. Neither is close to the reported 0.492. **Gate cleared**: no genuine internal
contradiction, just two different populations under one report; proceeding to STEP 1.

## STEP 1 — locating the actual computation

`evidence_taskT.md:294-298`, quoted verbatim, is the entire source of the published figures:

> SNR (membership_leave, two-sided with noise-floor caveat) [FINDING]: 66.4% of steps have a
> zero agent-driven noise floor (trivially detectable on those steps — argues against
> attenuation, not for thin data); on the remaining steps, SNR at ~100 discovered nodes = 0.492
> [0.385, 0.632], log-log slope vs n_discovered = -0.804 [-0.842, -0.765] — SNR near/below 1,
> declining with scale.

`evidence_taskT.md`'s own "Files written or changed" section confirms the **only** script
involved is `cyberbattle/agents/compute_attenuation_analysis.py` (commit `b008aef`, "ACTION 1 fix
+ option B backup") — no dedicated Task T script exists.

**The prose alone is NOT sufficient to reconstruct the computation.** It states the two output
numbers and the noise-floor caveat, but gives no aggregation method, no population definition, and
no binning scheme. Every SNR-ABS attempt that failed to reproduce 0.492/-0.804 was a reasonable
reading of this prose — the gap was never a misreading of the sentence, it was that the sentence
doesn't contain the method.

**The actual method, found in `compute_attenuation_analysis.py`:**

1. **Per-event SNR** (`compute_episode_aggregates`, lines 567-576): for every `membership_leave`
   row, `snr = change_drift_full/agent_drift_full`, or `NaN` if
   `agent_drift_full < ZERO_NOISE_FLOOR_THRESHOLD` (`=1e-9`, line 564).
2. **Per-episode median** (lines 577-605): grouped by `(seed, scenario_id, episode, change_type)`
   — **within one band's data at a time** (`compute_episode_aggregates` is called once per band
   inside `analyze_band`, line 751) — `agg_dict['snr'] = 'median'`. This produces one row per
   episode: that episode's own **median** SNR across its own leave events. `n_discovered` is
   *also* median-aggregated to the episode level in the same step (line 584).
3. **Pooling across bands** (line ~1364): `all_episode_df = pd.concat(all_episode_dfs)` —
   the three bands' episode-level tables are concatenated (not re-aggregated) into one pooled
   table before the LEVEL/SLOPE figures are computed. `ct_df` is this pooled table filtered to
   `change_type == 'membership_leave'` (line 953).
4. **LEVEL** (lines 1063-1068): `near_100 = ct_df[(n_discovered>=80) & (n_discovered<=120)]` — a
   **range** filter (not `>=80` alone) on the per-episode median `n_discovered`, applied to the
   pooled (all-bands) table. `mean_snr, lo, hi = bootstrap_series_ci(near_100['snr'])`, and
   `bootstrap_series_ci` (line 611-617) calls `bootstrap_ci` (`cyberbattle/utils/math_utils.py:
   48-54`), which reports **`np.mean(data)`** with a percentile bootstrap CI. So LEVEL is a
   **mean of per-episode medians**, in a window, pooled across bands — not a median of anything,
   and not per-event at all.
5. **SLOPE** (lines 1075-1078): `x = ct_df['n_discovered']`, `y = ct_df['snr']`,
   `fit_loglog_slope(x, y)` (line 723) — an **unbinned** `np.polyfit` log-log OLS directly on the
   per-episode median-SNR values against per-episode median `n_discovered`, over every episode
   with a valid SNR, pooled across bands. No integer-bin aggregation at all — neither Task-T's
   own median-per-bin convention nor Task-W's mean-per-bin convention is used here. Both of those
   are different, separately-established conventions from what this code actually does.

This is a **two-level** aggregation (event → episode-median → pooled-mean-in-a-window /
pooled-unbinned-OLS) that does not match any of the four candidates named in the task's own
STEP 2 — all four of which operate at the per-event level or per-integer-bin level, never with a
per-episode median as an intermediate step.

## STEP 2 — the four prescribed candidates, tested and none match (as instructed, exactly four)

Computed on the matching population (`analysis/snr_abs_2026-08-22/snr_abs_rows.csv`, per-event,
pooled across bands, `membership_leave`, `n_discovered` in [80,120] to match the LEVEL window,
n=2,032):

| candidate | value | vs reported 0.492 |
|---|---|---|
| (a) median of per-event ratio | 0.1543 | no match (Δ=0.338) |
| (b) mean of per-event ratio | 1.5027 | no match (Δ=1.011) — dominated by extreme low-`n_discovered` outliers (the code's own comment notes observed per-event SNR up to 1.86e9; this is exactly why the real pipeline takes a per-episode median before ever averaging) |
| (c) ratio of pooled means (mean(change_drift_full)/mean(agent_drift_full)) | 0.2733 | no match (Δ=0.219) |
| (d) Task-W's mean-per-integer-bin scheme, median across bins | 2.0864 (bin [90,100) alone: 2.3516) | no match (Δ=1.594 / 1.860) |

**None of the four reproduce 0.492.** Per the task's explicit instruction, no fifth candidate was
generated in this search — STEP 2 stops here, exactly as specified.

## The actual reproduction (found via STEP 1, not a STEP-2 search-for-fit)

STEP 1's job was to locate the real code, and having located it, the natural verification is to
run *that exact code path* — not to add it as an unauthorized fifth candidate in STEP 2's
enumerated list. Two independent checks, both exact:

**(A) Replay against the archived original run.** The 2026-07-26 5-seed TRPO gate run's own
output survives on disk, untouched, at
`attenuation_gate_archive/2026-07-26_trpo_5seed_gate/attenuation_analysis_output/`:
- `attenuation_episode_aggregates.csv` — the literal per-episode table (`all_episode_df`) the
  real run produced, 4,641 `membership_leave` rows, 4,517 with valid SNR.
- `gate_summary.txt` — the literal printed output of that run, containing (unmodified, not
  reconstructed):
  ```
  LEVEL: SNR at ~100 discovered nodes (n=306 valid episode-rows in [80,120]): 0.492 [0.385, 0.632]
  SLOPE: log-log SNR-vs-n_discovered slope = -0.804 [-0.842, -0.765] (n=4517)
  ```
  This is the primary source — not an inference, the actual archived output of the run that
  produced the thesis's figures.

  Recomputing LEVEL and SLOPE directly from `attenuation_episode_aggregates.csv`, applying
  `fit_loglog_slope` copied verbatim from the production script:

  | figure | recomputed | archived/reported |
  |---|---|---|
  | LEVEL (n=306) | 0.492 [0.383, 0.632] | 0.492 [0.385, 0.632] |
  | SLOPE (n=4517) | -0.804 [-0.842, -0.765] | -0.804 [-0.842, -0.765] |

  Point estimates match **exactly** (LEVEL to 3 d.p., SLOPE to 3 d.p. including the CI bounds,
  since `fit_loglog_slope`'s bootstrap uses a fixed `rng=42`). The tiny LEVEL CI difference
  (0.383 vs 0.385) is expected: `bootstrap_ci` itself is unseeded, so its own CI is not bit-exact
  run to run, only the point estimate (`np.mean`) is.

**(B) Population match.** 4,641 `membership_leave` episode-rows and n=306/n=4,517 for the two
figures match the archived `gate_summary.txt`'s own reported episode/row counts exactly.

## Verdict

```
0.492 AND -0.804 IDENTIFIED AS: the MEAN (bootstrap-CI'd) of per-episode MEDIAN snr values --
snr computed per membership_leave event as change_drift_full/agent_drift_full (NaN below
ZERO_NOISE_FLOOR_THRESHOLD=1e-9), median-aggregated to one value per (seed, scenario_id,
episode) within each band, then pooled (concatenated, not re-aggregated) across all three
bands. LEVEL = mean of these episode-medians restricted to episodes whose own median
n_discovered falls in the RANGE [80,120]. SLOPE = an UNBINNED log-log OLS (np.polyfit on
log(n_discovered), log(episode-median-snr), bootstrap CI with fixed rng seed=42) over every
episode with a valid (non-NaN) snr, pooled across all three bands. This is
compute_attenuation_analysis.py's own compute_episode_aggregates() plus the LEVEL/SLOPE
emission block near line 1063 -- NOT any of the four candidates named in STEP 2, none of which
used per-episode median-first aggregation.
```

This is not a hedge: the recomputation from the archived episode-level data matches the archived
printed output exactly, including the SLOPE's confidence interval (which is only reproducible
bit-for-bit if the exact same code path, including its fixed RNG seed, is being run).

## Why every SNR-ABS variant landed at ~0.12-0.21 instead

SNR-ABS's closest miss — the median of the same `near_100` episode-level SNR values (i.e. the
correct population, wrong outer statistic) — is **0.2110** (`run_output.log`, Part A cross-check
line: `MEDIAN of the same near_100 per-episode-median-snr values, n=306: 0.2110`). That sits
almost exactly inside SNR-ABS's own "0.12-0.21" range. The entire gap between SNR-ABS's
recomputation and the thesis's 0.492 traces to one design choice: the thesis's own LEVEL figure
takes the **mean**, not the median, of per-episode median-SNR values — and a mean of medians in a
right-skewed ratio distribution (SNR can spike arbitrarily high on low-`n_discovered` episodes)
sits well above the median of the same set. SNR-ABS's per-event-level candidates were even further
off because they skip the per-episode median step entirely, which is what suppresses the extreme
per-event outliers (up to 1.86e9, per the code's own comment) before any cross-episode statistic
is taken.

## What this task does not address

This task identifies the *statistic*; it does not re-litigate whether "mean of per-episode
medians, pooled across bands, in an [80,120] node-count window" is the *right* choice of
statistic for the thesis to report, nor whether the underlying donor-pool confound flagged at the
top of the archived `gate_summary.txt` ("PROVISIONAL... these numbers are a directional gate read
only and must be regenerated on same-band donor pools before appearing in the thesis") has since
been resolved. Both are open questions this task was not asked to answer.

## Outputs (committed alongside this record)

`identify_snr_baseline.py` (committed pre-run), `run_output.log`, this summary.

## Not done in this task

No new evaluation or training. No source file touched (`compute_attenuation_analysis.py` was
read only). No metric definition changed. No `.tex` file touched. No epsilon floor substituted
for any zero denominator.
