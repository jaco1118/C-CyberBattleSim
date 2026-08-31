# RQ3d clean same-episode ranking-overlap analysis

**Population: the existing RQ3D 3000-episode dataset only.** No new rollout, no retraining, no
source file touched. Data retrieved read-only via `git show 3d8c9aa:...` from branch
`attenuation-pooling-scale` (the commit that added `rq3d_data/{change,static}/eventgraph_<band>/
event_episode.jsonl`) — not re-committed here, since it already has a durable home in git history.

## B. Exact analysis population / filtering rules

- **Source**: `cyberbattle/agents/rq3d_data/{change,static}/eventgraph_{10-15,30-40,80-100}/
  event_episode.jsonl`, commit `3d8c9aa`, branch `attenuation-pooling-scale`. 1800 change-arm +
  1200 static-arm episodes (600+400 per band, exactly as RQ3D originally reported).
- **Definitions, unchanged** (verified against `analysis/rq1b_mech_split_scale_2026-08-08/
  compute_mechanical_share_scale.py:19-26`, the project's canonical "FORMULA B", and reused
  unchanged from `compute_rq3d_renormalize.py` / `compute_rq3d_behavioural_residual.py` on this
  exact dataset):
  ```
  static_root_owned_count(band,seed,scenario_id) = mean(final_root_owned_count) over matching static-arm episodes
  gross_root_loss       = static_root_owned_count - final_root_owned_count
  mechanical_root_loss  = root_owned_departures
  behavioural_residual  = gross_root_loss - mechanical_root_loss
  ```
- **Episode identity**: `(band, seed, scenario_id, episode)` — `episode` alone is only a
  per-(seed,topology) sub-run index (a fresh env starts per block in `rq3d_rollout.py`), so the
  full tuple is the unique key used for every overlap/intersection computation below.
- **Usable population**: a change episode is usable iff its `(band,seed,scenario_id)` has ≥1
  matching static-arm episode. **Result: 1800/1800 usable, 0 excluded** — every change episode in
  this rollout has a valid static pairing (by construction: both arms share the identical
  `(seed,scenario_id)` grid).
- **Zero/negative handling, stated exactly (Question 3)**: for EACH metric independently, episodes
  with that metric ≤ 0 are excluded from *that metric's own* worst-decile ranking (cannot be a
  "loss" episode under a metric that isn't positive) but remain part of the usable population and
  of the *other* metric's ranking if eligible there. This is not a new convention — it reproduces
  `compute_rq3d_renormalize.py`'s own positive-loss filter and `compute_rq3d_behavioural_residual.py`'s
  own positive-residual filter exactly. Worst decile = `ceil(0.10 × n_positive)`, minimum 1, same
  formula both scripts already use.
- **"Rest" in the mechanical-loss diagnostic (Question 6)** = the remaining 90% of the *positive*
  population for that metric (not the full dataset including non-positive episodes) — matching
  `compute_rq3d_renormalize.py`'s own stated convention ("top 10% ... remaining 90% = rest").
  Confirmed by exact reproduction: this definition reproduces RQ3D's own previously-reported pooled
  4.16/2.58 mechanical-loss figures to 2 d.p. (4.160/2.576 here).
- **Pooling**: computed for every quantity, but only treated as valid if all 3 bands are
  represented in **both** pooled worst-decile sets (see A/pooled row) — otherwise flagged, not
  silently used, per your "pooled only if statistically appropriate" instruction.

## A. Per-band table

| band | n usable | gross-loss worst decile n (of n_pos) | gross loss-share | historical loss-share | behav. worst decile n (of n_pos) | behav. residual-share | \|∩\| | Jaccard | % gross-worst exits | % behav-worst new | Spearman ρ (gross,behav) | mech_root_loss: gross-worst mean/median | rest mean/median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10-15  | 600 | 60 (of 596) | **18.2%** | 18% (Δ+0.2pp) | 51 (of 509) | 24.2% | 21 | 0.233 | 65.0% | 58.8% | 0.6305 | 2.450 / 2.0 | 1.461 / 1.0 |
| 30-40  | 600 | 59 (of 587) | **18.9%** | 32% (Δ−13.1pp) | 44 (of 432) | 29.5% | 27 | 0.355 | 54.2% | 38.6% | 0.7373 | 4.831 / 5.0 | 4.150 / 4.0 |
| 80-100 | 600 | 44 (of 435) | **27.7%** | 34% (Δ−6.3pp) | 32 (of 316) | 32.2% | 26 | 0.520 | 40.9% | 18.8% | 0.9272 | 2.727 / 3.0 | 2.297 / 2.0 |
| POOLED (flagged, see below) | 1800 | 162 | 24.2% | — | 126 | 33.0% | — | 0.455 | 44.4% | 28.6% | 0.7606 | 4.160/4.0 | 2.576/2.0 |

**Zero/negative counts** (Question 3, per band, gross_root_loss / behavioural_residual):
10-15: 0 zero / 4 negative gross; 8 zero / 83 negative residual. 30-40: 1/12 gross; 6/162 residual.
80-100: 2/163 gross; 7/277 residual. Negative-residual counts rise sharply with band — expected,
since `behavioural_residual = gross_root_loss − mechanical_root_loss` and larger bands have more
episodes where mechanical departures alone account for all or more of the gross loss.

**Pooled band composition** — why pooled is flagged: gross-loss worst decile (n=162) = {10-15: 0,
30-40: 109, 80-100: 53}; behavioural worst decile (n=126) = {10-15: 0, 30-40: 56, 80-100: 70}.
**Band 10-15 has ZERO representation in either pooled worst decile** — reproducing exactly the
scale confound `evidence_taskRQ3D.md` already found for the pooled raw-loss ranking (absolute
`gross_root_loss` scales with band size, so the smallest band structurally cannot compete). Pooled
figures are shown above for transparency only; **per-band is the primary, trustworthy view**
throughout this analysis.

## C. Interpretation — does mechanical loss materially determine worst-decile membership?

**Yes, substantially, and the effect is largest in the smaller bands.** Under the two most direct
measures:
- **Membership churn**: 65.0% / 54.2% / 40.9% (10-15 / 30-40 / 80-100) of the episodes in the
  gross-loss worst decile are **no longer** in the worst decile once mechanical loss is subtracted
  out. Jaccard overlap is correspondingly low-to-moderate: 0.233 / 0.355 / 0.520.
- **Rank correlation is still fairly high** (Spearman ρ = 0.63 / 0.74 / 0.93, rising with band
  size) — the two metrics are far from independent overall. The apparent tension with the high
  churn numbers above is a genuine boundary effect, not a contradiction: near a decile cutoff on a
  continuously-varying, moderately-correlated pair of metrics, a large fraction of borderline
  episodes swap sides even when the *overall* rank agreement is fairly strong. Both facts are true
  at once and both are reported.

**Practical reading**: at 10-15 and 30-40, mechanical loss is a major driver of gross-loss
worst-decile membership — a majority of "worst" episodes by raw loss are not "worst" once the
mechanical channel is removed. At 80-100, the effect is smaller but still material (41% exit) and
the two rankings converge more (ρ=0.93) — consistent with the already-established finding that the
mechanical share of loss falls with band size (`rq1b_mech_split_scale_2026-08-08`: mechanical
share ~ log(size), negative slope) — less mechanical loss to begin with at 80-100, so removing it
moves the ranking less.

## Question 7 — the existing 2.19-vs-2.45 result

**Exact reproduction**: top-decile (behavioural-residual-ranked, pooled) n=126, mean
`root_owned_departures`=2.190; rest n=1131, mean=2.451; diff=−0.260, 95% CI [−0.549, +0.041] —
matches the previously-reported 2.19 / 2.45 / [-0.549,+0.041] to 2-3 decimal places, on the same
data, same script logic.

**What these two numbers represent, exactly**: the mean `mechanical_root_loss` (root-owned
departure count) in the pooled behavioural-residual worst decile (2.19) versus the pooled rest
(2.45). This is **not** a same-episode overlap statistic — it's a group-mean comparison of a THIRD
variable (departures) under a ranking built from the OTHER two (gross loss minus departures). Per
its own documented circularity caveat, subtracting departures out of the ranking variable
mechanically biases this comparison toward *fewer* departures in the top group — so part of the
2.19<2.45 direction is expected by construction, independent of any real behavioural signal.

**Is it still useful now that the direct overlap result exists? Partially, as a secondary
diagnostic, not as primary evidence.** The overlap/Jaccard/exit-rate results above answer your
actual question — "does membership change" — directly and without that circularity (they compare
episode SETS under two independently-defined rankings, not a group mean built from a subtractive
construction). 2.19/2.45 answers a narrower, construction-biased question ("do the residual-ranked
top episodes show fewer departures") and remains a legitimate cross-check that the two ranking
metrics pull in different directions, but it should no longer be read as the headline evidence for
the mechanical-confound question — the overlap results now fill that role more directly and with a
stated (rather than confounding) methodology.

## The historical 18%/32%/34% comparison — NOT assumed correct, checked directly

| band | historical | fresh RQ3D reproduction | difference |
|---|---|---|---|
| 10-15 | 18% | **18.2%** | +0.2pp — close to an exact match |
| 30-40 | 32% | **18.9%** | −13.1pp — does NOT reproduce |
| 80-100 | 34% | **27.7%** | −6.3pp — does NOT reproduce |

**The fresh population reproduces the historical figure closely for 10-15 only. It does not
reproduce 30-40 or 80-100** — both come in well below the historical value, and the gap is not
uniform across bands (13.1pp vs 6.3pp), so this isn't a single constant offset either. Given the
original pipeline is permanently lost (`git show 1d6aaab`'s own message: *"the CX compute pipeline
that produced those figures is NOT in the stash and remains lost"*) and the original's exact
episode set, checkpoint-loading order, and ranking implementation cannot be inspected, this
partial mismatch cannot be diagnosed further — only reported. **Do not treat 18%/32%/34% as
confirmed by this check**; 10-15's near-exact match is consistent with (but does not prove) the
same underlying process, while the other two bands' mismatches are a real, unresolved discrepancy.

## D. Which existing RQ3d thesis numbers should be retained, replaced, or removed

| existing number | recommendation | why |
|---|---|---|
| 0.70 (top-decile departures) vs 1.23 (rest) | **REMOVE as a standalone claim** | Sourced only to a permanently-lost pipeline (`git show 1d6aaab`); cannot be defended if its provenance is challenged. Superseded by this task's own gross-loss-decile mechanical diagnostic (Section A table, right-most columns) and by RQ3D's already-reported pooled raw-loss result (4.16 vs 2.58, direction-reversed, resolved) — both traceable and reproducible. |
| 18% / 32% / 34% (loss concentration) | **REPLACE** with the fresh per-band figures (18.2% / 18.9% / 27.7%), explicitly captioned as computed on RQ3D's own fresh rollout, NOT a verified reproduction of the original — with the partial-mismatch table above shown alongside, not hidden. | Original pipeline unrecoverable; two of three bands do not reproduce the historical figure, so presenting 18/32/34% as-is without this caveat would overstate confidence in numbers that cannot be checked. |
| 2.19 vs 2.45 (behavioural-residual decile departures, Addendum 7) | **RETAIN as a secondary/supporting diagnostic**, with its circularity caveat kept intact, explicitly demoted from primary to supporting evidence. | Independently re-verified exact reproduction in this task; still a legitimate (if construction-biased) cross-check, just not the most direct answer to the membership question anymore. |
| — (did not exist before) | **ADD as the new primary evidence**: per-band Jaccard overlap (0.233/0.355/0.520), % gross-worst-decile exit (65.0%/54.2%/40.9%), Spearman ρ (0.63/0.74/0.93) | This is the first same-episode, non-circular answer this project has produced to "does mechanical loss determine worst-decile membership" — directly responsive to the RQ3d question, unlike the group-mean comparisons that preceded it. |

## Outputs (committed alongside this record)
`compute_rq3d_ranking_overlap.py` (committed pre-run, `d4e82ef`), `run_output.log`, this summary.

## Not done in this task
No new rollout (`rq3d_rollout.py` not invoked). No retraining. No `.tex` file touched. No source
file modified — `event_episode.jsonl` data read via `git show` only, never re-committed here (it
already has a durable home on `attenuation-pooling-scale`, commit `3d8c9aa`). Definitions
(`gross_root_loss`, `mechanical_root_loss`, `behavioural_residual`) unchanged from the project's
own established FORMULA B.
