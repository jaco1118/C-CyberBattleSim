# RQ3D: renormalized (departures-per-static-root) top-decile-loss comparison

**POPULATION [ARTIFACT]:** this analysis uses the dissertation's standard checkpoint population (dynamically trained, the band grid, 5 seeds, trained 26 July 2026) -- the same one behind RQ1, RQ2, and RQ3(a-c). It is the same population as Task CX PART 3's own "adapted gate checkpoints" (evidence_taskCX.md:272, evidence_taskF1.md:14; Addendum 5 Branch A) -- **not** a different population -- so the raw-figure reproduction check below applies.

**CAVEATS carried into every number below:**

- **GROSS COUNT:** `root_owned_departures` counts a root lost and immediately re-owned within the same episode as ONE departure, not zero.
- **DONOR-POOL [PROVISIONAL]:** donor-pool confound (Task G pending); `membership_join` draws from a shared pool ~2.2x weaker at the large band.

## Ranking metric [ARTIFACT, STEP 0.3 restated]

The original CX PART 3 text (rescued at commit 1d6aaab) never explicitly stated its top-decile ranking metric or pairing convention (STEP 0 finding). This rebuild's PRIMARY metric: `loss(episode) = static_root_owned_count(seed,topology,band) - final_root_owned_count(episode)`, using the SAME static-pairing convention as the renormalization denominator, ranked POOLED across all 3 bands (matching the original's inferred N-pooled framing), filtered to POSITIVE-loss episodes, top 10% of that positive-loss subset by loss descending = top-decile-loss; remaining 90% = rest.

## SCALE CONFOUND in the pooled ranking [FINDING, discovered on the full dataset]

static_root_owned_count scales sharply with band (this rollout's static-arm mean: 10-15 7.83, 30-40 23.08, 80-100 29.62; max 12 / 31 / 48). Since `loss` is an ABSOLUTE difference, 10-15's episodes can never produce a large enough raw loss value to compete in a POOLED cross-band ranking -- confirmed empirically below: the pooled top-decile group contains **zero 10-15 episodes** out of 600. This revises the STEP 0.3 assessment: since the original text reports a non-zero loss-share for ALL THREE bands (18/32/34%), the original ranking was almost certainly NOT simple pooled-absolute-loss the way this analysis's primary metric is -- it more likely ranked within-band, or used a scale-relative loss definition. A WITHIN-BAND ranking variant is reported below as a supplementary robustness check for exactly this reason -- **not** a silent redefinition of the authorized primary metric, which is reported in full first.

## Episode counts [ARTIFACT]

- Change-arm episodes loaded: **1800**; static-arm: **1200**.
- Positive-loss change episodes (pooled): **1618** of 1800 scored.
- Pooled-ranking top-decile group size: **162**; rest: **1456**.
- Within-band-ranking top-decile group size (summed): **163**; rest: **1455**.

## Exclusions (zero/missing static_root_owned_count denominator) [ARTIFACT]

**Zero excluded.** Every change episode had a valid (non-zero) static_root_owned_count from its paired (seed, topology, band).

## STEP 5 -- raw-figure reproduction check (pooled across bands, primary ranking) [FINDING]

This rollout (mean root_owned_departures, pooled): top-decile-loss = **4.160** (n=162), rest = **2.576** (n=1456). Difference (top-rest) = +1.585, 95% CI [+1.202, +1.964] -- CI excludes 0, difference resolved.

Original CX PART 3 (different rollout, SAME checkpoint population, DIFFERENT ranking metric per the scale-confound finding above): top-decile-loss = 0.7, rest = 1.23. This rollout's pooled top-decile mean (4.160) is HIGHER than its rest mean (2.576) -- the OPPOSITE direction from the original (0.70 < 1.23). Given the ranking metric is confirmed different (not a like-for-like reproduction attempt), this is reported as a metric-definition discrepancy, not a reproduction pass or fail.

**DIRECTION NOTE [important caveat, not just for this pooled result -- it holds in every variant reported below, both rankings, all 3 bands]:** this analysis's `loss` metric is built directly from `final_root_owned_count`, which `root_owned_departures` mechanically depletes -- an episode with more departures has fewer roots left almost by construction, all else equal. So a POSITIVE correlation between this `loss` metric and departure count is expected to some degree simply from how `loss` is defined here, independent of any genuine behavioural-vs-mechanical story. The original CX text's own §3.5 discusses `root_owned_departures` split from a `behavioural residual` (loss minus gross mechanical cost) -- if the original's ranking metric for §3.9 was residual/score-based rather than raw-root-owned-count-based, that would net out exactly this mechanical channel, and could fully explain the reversed direction here without implying the original finding was wrong. **This reversal should be read as evidence the two analyses used different, not-yet-reconciled loss definitions, not as a refutation of the original claim.**

## PRIMARY (pooled ranking): per-band RAW vs RENORMALIZED [FINDING]

| band | group | n | raw mean departures | renorm mean dep/static-root | loss-share % (rollout) | loss-share % (original) |
|---|---|---|---|---|---|---|
| 10-15 | top-decile | 0 | nan | nan | 0.0% | 18% |
| 10-15 | rest | 596 | 1.560 | 0.2075 |  |  |
| 10-15 | **diff (top-rest)** | -- | **+nan** [+nan, +nan] | **+nan** [+nan, +nan] |  |  |
| 30-40 | top-decile | 109 | 4.872 | 0.2013 | 31.2% | 32% |
| 30-40 | rest | 478 | 4.069 | 0.1780 |  |  |
| 30-40 | **diff (top-rest)** | -- | **+0.803** [+0.350, +1.250] | **+0.0232** [+0.0043, +0.0415] |  |  |
| 80-100 | top-decile | 53 | 2.698 | 0.0866 | 31.4% | 34% |
| 80-100 | rest | 382 | 2.291 | 0.0774 |  |  |
| 80-100 | **diff (top-rest)** | -- | **+0.408** [-0.130, +0.976] | **+0.0091** [-0.0083, +0.0275] |  |  |

**Verdict per band (pooled ranking):**

- **10-15**: NO DATA: zero episodes in this band's top-decile group under this ranking.
- **30-40**: SURVIVES: both raw and renormalized differences are resolved (CI excludes 0) and point the same direction.
- **80-100**: UNCLEAR: the raw difference itself is not resolved at this sample size (CI brackets 0) -- cannot assess survival/shrinkage/disappearance with confidence.

## SUPPLEMENTARY (within-band ranking): per-band RAW vs RENORMALIZED [FINDING]

Added specifically because the pooled ranking gives 10-15 zero representation (scale confound above). Every band is ranked against its OWN positive-loss episodes here, so every band has a result.

| band | group | n | raw mean departures | renorm mean dep/static-root |
|---|---|---|---|---|
| 10-15 | top-decile | 60 | 2.450 | 0.2413 |
| 10-15 | rest | 536 | 1.461 | 0.2038 |
| 10-15 | **diff (top-rest)** | -- | **+0.989** [+0.699, +1.284] | **+0.0375** [+0.0068, +0.0685] |
| 30-40 | top-decile | 59 | 4.831 | 0.2014 |
| 30-40 | rest | 528 | 4.150 | 0.1802 |
| 30-40 | **diff (top-rest)** | -- | **+0.681** [+0.024, +1.337] | **+0.0211** [-0.0059, +0.0486] |
| 80-100 | top-decile | 44 | 2.727 | 0.0874 |
| 80-100 | rest | 391 | 2.297 | 0.0776 |
| 80-100 | **diff (top-rest)** | -- | **+0.431** [-0.190, +1.073] | **+0.0098** [-0.0098, +0.0308] |

**Verdict per band (within-band ranking):**

- **10-15**: SURVIVES: both raw and renormalized differences are resolved (CI excludes 0) and point the same direction.
- **30-40**: SHRINKS/UNCLEAR: the raw difference is resolved but the renormalized one is not (CI brackets 0) -- consistent with the raw finding being at least partly an ownership-confound artifact, but not confirmed reversed.
- **80-100**: UNCLEAR: the raw difference itself is not resolved at this sample size (CI brackets 0) -- cannot assess survival/shrinkage/disappearance with confidence.

## Reference: original CX PART 3 figures (context only) [ARTIFACT]

- Loss share, top 10% of positive-loss episodes: 18% / 32% / 34% (10-15 / 30-40 / 80-100).
- Departures (pooled): top-decile-loss = 0.7, rest = 1.23.
- Churn (pooled, not recomputed here -- out of this task's scope): top-decile-loss = 0.36, rest = 0.50.
- Source: rescued findings text at commit 1d6aaab (evidence_taskCX.md PART 3 section 3.9, never merged into the live card). Same checkpoint population as this rollout (Addendum 5 Branch A); ranking metric confirmed different (scale confound finding above).

