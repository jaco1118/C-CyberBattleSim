# Task SNR-ABS — absolute-drift SNR sensitivity check

**SENSITIVITY CHECK: recomputation of an existing reported metric on existing logged data. No new
runs. Not a replacement for the reported figures unless the thesis is edited to say so.**

Branch: `snr-abs-sensitivity`. Dataset: `attenuation_gate_archive/2026-07-26_trpo_5seed_gate/
attenuation_drift_logs/drift_{10-15,30-40,80-100}.csv`, `membership_leave` fired events, 5-seed
TRPO attenuation-gate agents (250k) -- traced in STEP 0 as the source of the reported figures.

## Prediction recorded in advance (per the task) vs outcome

| prediction | outcome |
|---|---|
| median snr_abs lands 0.45-0.55 | **Contradicted.** 0.3897 [0.3693,0.4109] pooled, ~0.15 at the point-figure population |
| qualitative claim (median < 1 while acting) survives | **Survives**, but see the baseline-reproduction issue below |
| contamination slope small, slope_abs within ~0.1 of -0.804 | **Cannot be assessed against -0.804** -- neither slope_current nor slope_abs reproduces -0.804 to begin with (see below) |

Per the task's own instruction, the contradicted prediction is reported plainly, not softened.

## AMENDMENT 3 (resolved first, as a gate) — the 66.4% denominator

| candidate denominator | rows | zero-agent-drift | fraction |
|---|---|---|---|
| events (change_fired rows) | 54,854 | 36,196 | 65.99% |
| **leave-steps (co-firing deduped, (band,seed,episode,step))** | **48,697** | **32,231** | **66.19%** |
| all logged steps (any change_type) | 284,454 | 196,244 | 68.99% |

**The leave-steps denominator (co-firing deduped) reproduces the thesis's 66.4% most closely
(0.21pp off), clearly closer than either alternative (0.41pp and 2.59pp off respectively).**
Treated as confirming population identity, with the residual 0.21pp gap disclosed rather than
forced to zero. This is a real methodological finding in its own right: "66.4% of steps" means
*steps*, not raw event rows -- a step with two co-firing leave events counts once, not twice.

## A bug caught by Amendment 3's own cross-check, before any figure was trusted

The first run's pooled dedup/episode-bootstrap keyed identity on `(seed, episode)` only.
Episode/step numbering restarts independently per band's own training run, so rows from
different bands sharing the same `(seed, episode)` value collided. This showed up as an internal
inconsistency: the same leave-steps computation gave 66.19% in an initial hand check and 67.00%
inside the first script run. Fixed by adding `band` to every pooled key (dedup, groupby,
episode-bootstrap resampling). Confirmed fixed: both computations now agree at 66.19%. FINDING,
not glossed over: this is exactly the kind of error the amendment's own cross-check requirement
was designed to catch, and it worked as intended.

## AMENDMENT 4 — reproduce Task W's numerator slopes before trusting anything else

Task W STEP5's own provenance line specifies **mean**-per-integer-`n_discovered`-bin (not
median -- a different, separately-established convention from Task T's own median-per-bin SNR
method), no `agent_drift_full` filter, `n_discovered>=2`.

| quantity | reproduced | Task W reported |
|---|---|---|
| rows (n_discovered>=2) | 54,832 | 54,832 (exact match) |
| relative `change_drift_full` slope | -1.0057 | -0.988 |
| absolute `change_drift_full*norm_h2` slope | -0.9125 | -0.891 |

**REPRODUCED, within ~2%.** Row count matches exactly; slopes match closely enough to treat this
as confirming the dataset and general row-selection are correct, per the amendment's own
decision rule.

## Baseline-reproduction problem, found and investigated, not smoothed over

Despite Amendment 4 passing, **the SNR *ratio*'s own reported baseline figures (0.492 point
value, -0.804 slope) do not reproduce** under any of several reasonable variants tried:

| variant | result |
|---|---|
| row-level `n_discovered>=80` (band 80-100) | median snr_current = 0.1543 |
| episode-level "reaches `n_discovered>=80` at any point" | median snr_current = 0.1839 |
| binned by `n_discovered_h1` instead of `n_discovered` (=h3) | slope = -0.361 |
| binned by `n_discovered_h2` instead of `n_discovered` (=h3) | slope = -0.725 |
| fitted-curve value at n=100 from median-bin OLS (vs raw sub-population median) | 0.12-0.19 |
| POOLED median-bin slope of snr_current (Task-T-style, all bands) | -0.673 |

All land in a consistent ~0.12-0.21 / slope -0.36 to -1.03 range -- never near 0.492/-0.804.
This is **not** explained by the Amendment-3 pooling bug (point slope estimates don't depend on
episode identity, only the bootstrap CIs do) and is **not** resolved by any n_discovered-variant
tried. Since Amendment 4's numerator-only reproduction succeeded closely on the same dataset, the
gap is specific to reproducing the SNR *ratio*'s own reported baseline, not a sign the dataset or
general approach is wrong. **Reported as an unresolved discrepancy.** The comparison this task
was actually asked for -- snr_abs vs snr_current on the *same* population -- remains internally
consistent and meaningful even though neither figure matches the thesis's exact published number.

## AMENDMENT 2 — contamination factor slope (the decisive diagnostic)

| population | slope of log(norm_h2/norm_h1) vs log(n) | 95% CI | includes 0? |
|---|---|---|---|
| band 10-15 | -0.0230 | [-0.0474, -0.0045] | **No** |
| band 30-40 | -0.0074 | [-0.0241, -0.0018] | **No** |
| band 80-100 | -0.0056 | [-0.0223, -0.0051] | **No** |
| POOLED | -0.0069 | [-0.0114, -0.0054] | **No** |

Contamination factor median (pooled) = 1.0032, p5-p95 = [0.9900, 1.1235] -- matches the thesis's
own disclosed "median within 2.4% of one" and "0.36 to 1.74" range order of magnitude.

**The contamination factor has a small but statistically non-zero slope against n at every band
and pooled.** All four intervals exclude zero. The magnitude is tiny (-0.007 to -0.023) compared
to the reported -0.804 slope, so on its own it cannot explain a large fraction of that figure --
but per the task's own framing, a non-zero slope here means the reported -0.804 is not perfectly
scale-invariant with respect to this factor, even though the effect size is small relative to the
whole slope.

## AMENDMENT 1 — median of the product, and the Spearman correlation

- median(snr_abs), pooled, acting-steps population: **0.3897** [0.3693, 0.4109]
- (cross-check) median(snr_current), same rows: 0.3827 [0.3653, 0.4033]
- naive product-of-medians (NOT used as the result, shown only for comparison): 0.3839
- **Spearman(snr_current, contamination) = -0.3460** (p<1e-300, n=18,658)

The median-of-product (0.3897) and the naive product-of-medians (0.3839) differ by only ~1.5%
here -- small in this instance -- but the correlation itself is not negligible (rho=-0.346,
moderate negative), meaning the two quantities are not independent, and the closeness of the two
approximations should not be assumed to hold in general.

## slope_current vs slope_abs, per band and pooled (Task-T-style median-bin OLS)

| population | slope_current | slope_abs | difference |
|---|---|---|---|
| 10-15 | -0.8253 [-1.0501,-0.2448] | -0.8479 [-1.0629,-0.2742] | -0.023 |
| 30-40 | -0.7234 [-1.0618,-0.5258] | -0.7308 [-1.0706,-0.5503] | -0.007 |
| 80-100 | -1.0262 [-1.2504,-0.1175] | -1.0309 [-1.2478,-0.1075] | -0.005 |
| POOLED | -0.6730 [-0.7838,-0.6448] | -0.6864 [-0.7973,-0.6559] | -0.013 |

The absolute-drift slope is consistently and modestly steeper (more negative) than the relative
one at every band and pooled -- matching Amendment 2's own finding directly, since
`slope_abs = slope_current + slope(contamination)` and the contamination slope is small and
negative everywhere.

## Answer to the task's central question

**Switching to absolute drift does not meaningfully change the qualitative picture on this
recomputation: median SNR stays below 1, and the slope stays negative and of similar magnitude
(within ~0.01-0.02 of slope_current at every population).** The contamination factor's slope is
statistically non-zero but small. **However**, neither `snr_current` nor `snr_abs` as recomputed
here reproduces the thesis's own reported 0.492/-0.804 baseline, despite the dataset and general
row-selection being independently validated (Amendment 4). This sensitivity check therefore
answers "does absolute vs relative change the qualitative story, given a specific reproducible
population" with "no, not meaningfully" -- but it does **not** independently confirm the specific
published numbers themselves, and that gap is reported here rather than assumed away.

## What this task does not address

Switching from relative to absolute drift removes the `norm(h1)/norm(h2)` contamination factor.
**It does not address the separate question of whether `agent_drift`'s own composition changes
with band** (a discovery event moves the pooled vector far more than a single node's change, and
larger networks offer more discovery opportunities). This sensitivity check does not resolve
that question, and should not be read as having closed it.

## Row counts and filter stages

- Total `membership_leave` fired events, all bands: 54,854
- `n_missing_field`: 0 (zero skipped, explicitly)
- `n_zero_noise_floor` (agent_drift_full==0): 36,196
- `n_near_zero_noise_floor` (0 < agent_drift_full < 1e-12): 0
- Acting-steps population (usable for snr_abs/snr_current): 18,658
- Point-figure population (band 80-100, n_discovered>=80): 2,032

## Outputs (committed alongside this record)
`compute_snr_abs.py` (committed pre-run and again after the Amendment-3 bug fix),
`run_output.log`, `snr_abs_rows.csv` (18,658 rows, the full per-row acting-steps population),
this summary.

## Not done in this task
No new evaluation or training. No edit to `cyberbattle_env_compressed.py`, the drift logger, or
any metric definition. No `.tex` file touched. No epsilon floor substituted for any zero
denominator. Medians reported throughout, not means (except where explicitly reproducing Task
W's own mean-per-bin convention for the Amendment-4 cross-check, clearly labelled as such).
