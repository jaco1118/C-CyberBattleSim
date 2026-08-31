# Task A2 — why do the two pipelines disagree at band 80-100?

Analysis only, existing drift logs. **Resolved at STEP 1.3 / STEP 2 — it is a filter difference, not a pipeline difference.** The encoder-frozen constraint is respected (no explanation invokes different embeddings).

## VERDICT (one paragraph, up front)

**The disagreement is entirely an event-FILTER difference.** The gate's figures (98.5/84.0/**43.0**/36.1) are computed over ALL immediate leave events; the F-series figures (81.3/**64.2**/65.6) add a `relevant & touched_node_visible` filter. For a leave event the `relevant` flag reduces to "was the departing node **acted-on** earlier this episode" — because the node is removed from `discovered_nodes`/`owned_nodes` *before* the flag is evaluated — and acted-on nodes are more extremal, so the relevant-subset response rate is higher. **Under either filter applied to BOTH pipelines, they agree at 80-100** (immediate: gate 0.430 / F-series 0.436; +rel+vis: gate 0.632 / F-series 0.644), the n_discovered distributions are identical (mean 69.9 vs 69.7), and the two lie on **one** response-rate-vs-n_discovered curve. The "1.5×" came from comparing gate-filter-A (0.43) against F-series-filter-B (0.64). **Recommendation: option (a)** — one fit against n_discovered pooling both pipelines under a single stated filter (see 4.2).

## STEP 1 — provenance

### 1.1 Sources [ARTIFACT]

| figure | pipeline | runs | budget | scenarios | seeds | leave events (immediate) |
|---|---|---|---|---|---|---|
| 84.0/82.8 (30-40), 43.0/36.1 (80-100) | **gate** | `trpo_250k_tuned_compressed_band{30-40,80-100}_seed{42,100,123,200,300}` | **250k** | **8/band** | 5 | 21,445 (30-40); 28,088 (80-100) |
| 81.3/82.7 (30-40), 64.2/65.6 (80-100) | **F-series** | F1 static `trpo_250k_F1_static_seed*` (topo44) / F2 static `trpo_250k_F2_static_band80-100_seed*` | **250k** | **1 (30-40) / 5 (80-100)** | 5 | 11,198 (30-40); 13,775 (80-100) |

### 1.2 F-series budget [ARTIFACT]
The F-series figures are from the **original 250k** eval data (`eval_out/`, `f2eval_out/`), produced by the 250k checkpoints **before** Task F4. They are **NOT** F4's 500k/750k agents. **Both pipelines are at 250k — the same budget.** Not a budget difference.

### 1.3 Same filter? NO — this is the answer [FINDING]

The two figures use **different event filters** (both use the same τ=0 threshold and the same `change_drift_slice > 0` responding definition):
- **gate** (`compute_attenuation_analysis.py:523`): `event_phase == 'immediate'` only — **no relevant/visible filter**.
- **F-series** (`taskF1R_reanalyze.py` `filt`): `relevant==True & touched_node_visible==True & event_phase∈{immediate,attributed}`.

`touched_node_visible` is 1.00 for every leave event (the departing node is always in the pool at h2), so the only operative difference is **`relevant`**. Applying each filter to BOTH pipelines:

| filter | pipeline | 30-40 max/min | 80-100 max/min |
|---|---|---|---|
| A: immediate only (gate's) | gate | 0.840 / 0.829 | **0.430 (sd.015) / 0.361 (sd.008)** |
| A | F-series | 0.744 / 0.768 | **0.436 (sd.065) / 0.361 (sd.027)** |
| B: +relevant+visible (F-series') | gate | 0.904 / 0.905 | 0.632 (sd.019) / 0.625 |
| B | F-series | 0.813 / 0.827 | 0.644 (sd.048) / 0.657 |

**At 80-100 the two pipelines AGREE under either filter** (A: 0.430 vs 0.436, 0.361 vs 0.361; B: 0.632 vs 0.644). The raw 1.5× gap (gate 0.430 vs F-series 0.644) is **gate-A vs F-series-B** — the same data under different filters. **The filter difference is the cheapest explanation and it is the explanation; excluded-first, it is confirmed.**

**Why `relevant` is ~0.40 (not 1.0) for leave — and a correction to Task X STEP B [FINDING].** `_is_event_relevant` = `any(owned OR discovered OR acted-on)` (`cyberbattle_env_compressed.py:722-726`), but `remove_node_common` deletes the departing node from `discovered_nodes` (`cyberbattle_env.py:713`) and `owned_nodes` (`:715`) **before** `_log_drift_rows` calls the flag. So for a leave the discovered/owned disjuncts are already False, and `relevant` reduces to the **acted-on** disjunct — measured at 0.70 (30-40) / 0.40 (80-100), i.e. the fraction of departing nodes the agent had acted on. (This corrects the Task X STEP B claim that leave is "always relevant"; that is true only for property, where the node is not removed. The relevance flag DOES vary for leave — it is the acted-on fraction.)

## STEP 2 — one curve or two?

### 2.1 n_discovered at leave events [FINDING]

| pipeline | band | mean | median | p10–p90 |
|---|---|---|---|---|
| gate | 80-100 | 69.9 | 71 | 60–81 |
| F-series | 80-100 | 69.7 | 71 | 58–82 |
| gate | 30-40 | 22.1 | 23 | 16–28 |
| F-series | 30-40 | 24.2 | 24 | 19–29 |

**Identical at 80-100** (69.9 vs 69.7). The two pipelines sample the same part of the n_discovered axis at the headline band — so a difference there cannot be an n_discovered-sampling difference.

### 2.2 / 2.3 Response rate vs n_discovered — ONE curve [FINDING]

Response rate binned by n_discovered (pooled across bands, filter B for both), gate/F-series, max slice:

| n_disc bin | gate | F-series | (n_gate, n_F) |
|---|---|---|---|
| [15,20) | 0.97 | 0.88 | (2586, 687) |
| [20,30) | 0.89 | 0.81 | (10923, 6453) |
| [30,40) | 0.71 | 0.74 | (823, 749) |
| [55,70) | 0.64 | 0.65 | (3798, 1729) |
| [70,85) | 0.62 | 0.62 | (6488, 3096) |
| [85,200) | 0.64 | 0.65 | (489, 235) |

**The two pipelines lie on essentially one curve.** Gate-minus-F-series gap at matched n_discovered: mean **+0.019** (max) / **+0.010** (min); it is **~0 in the high-n_discovered bins (55–85, the 80-100 region)** and small-positive only at n_discovered 15–30 (the 30-40 region). So at the headline band (80-100 ⇒ n_discovered ~70) the curves **coincide**; the only residual is at low n_discovered.

**The low-n_discovered (30-40) residual (~0.08–0.10, gate above F-series) is scenario variation, not a pipeline effect:** the F-series 30-40 is a SINGLE scenario (topo44) while the gate 30-40 pools 8 — so the F-series point is one draw from the scenario distribution and can sit below the 8-scenario average. (Per the task's DO-NOT, STEP 3 is not pursued because STEP 2 resolves the 80-100 question; this residual is at a different band and is named, not investigated further.)

## STEP 4 — verdict and what the thesis should say

### 4.1 [FINDING]
The 80-100 disagreement is **entirely a filter difference** — the gate reports the response rate over all immediate leave events, the F-series over the `relevant`(=acted-on) subset; under a consistent filter the pipelines agree at 80-100 and lie on one n_discovered curve.

### 4.2 Recommendation: **(a)** — one fit against n_discovered, pooling both pipelines, under a single declared filter [FINDING]
Because the pipelines agree once the filter is harmonised, the choice is **not** gate-vs-F-series (options b/c are moot — they are the same curve) and **not** (d) a range (the evidence separates them cleanly: it is the filter). The remaining decision is **which filter**, and it must be stated:
- **Immediate-only (all leave events)** — recommended as PRIMARY: it is the unbiased population and is the gate's method (98.5→84.0→43.0 is this curve).
- **+relevant (acted-on subset)** — inflates the level (0.64 vs 0.43) via an acted-on selection (acted-on nodes are more extremal); report only as a secondary "among nodes the agent had used" cut, with the acted-on selection disclosed.

So: report the attenuation as **one fit of response rate against n_discovered, pooling gate + F-series, on the immediate-only event set**, with n_discovered (not band label) as the regressor — stronger than the three-point band table and what the methodology said would be done.

### 4.3 Numbers that change in Chapters IV / V [FINDING]
- The gate curve 98.5→84.0→**43.0**/36.1 is the **immediate-only** figure — it stays, but must be **labelled with its filter** and presented as a fit against n_discovered, not three band labels.
- The F-series **64.2/65.6** must **not** be presented as a contradicting measurement — it is the *same curve* on the relevant(acted-on) subset. Either drop it from the headline or present it explicitly as the acted-on cut.
- Any text asserting the two pipelines "disagree" at 80-100 is withdrawn: they agree under a common filter (0.430 vs 0.436).
- Any statement that the leave `relevant` flag is constant/degenerate is corrected: for leave it equals the acted-on fraction (0.70 at 30-40, 0.40 at 80-100).

Reported through STEP 2 + verdict. STEP 3 not pursued (STEP 2 resolved the 80-100 question). Stopping.
