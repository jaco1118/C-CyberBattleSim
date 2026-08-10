# Task GRAPH-DEPTH STEP 4 — DIRECT/PROPAGATION decomposition on the real discovered graph

Source: `graphdepth_sweep/leaveembed_<band>/*/*.jsonl` (STEP 3 sweep, 15 manifest checkpoints,
71 seed×topology runs) joined against `graphdepth_sweep/drift_<band>.csv` on
`(run_id, seed, scenario_id, episode, step)`. Script: `compute_graphdepth_decomposition.py`
(committed `3323136`), reuses `probe_p.py`'s definitions unmodified (read via `git show
c05a16a:analysis/recovered_scripts_2026-08-04/taskF1/probe_p.py`, not checked out, not touched).

## Data used, and what was excluded (ARTIFACT)

| band | total leave-events logged | batch-excluded (n_touched_nodes≠1) | single-node | zero-2hop-neighbour (excluded) | used |
|---|---|---|---|---|---|
| 10-15 | 1437 | 76 | 1361 | 623 | 738 |
| 30-40 | 1923 | 415 | 1508 | 764 | 744 |
| 80-100 | 3591 | 772 | 2819 | 2197 | 622 |

Batch exclusion mirrors the RQ2C-1 precedent (`_rq2c_leaves` filter): a batch leave event's `hp`
snapshot already reflects every batch-mate having left simultaneously, confounding a per-node
decomposition. "Zero-2hop-neighbour" events are single-node leaves where the departing node had no
survivor within 2 (undirected) hops at all — i.e. it held no edge in `evolving_visible_graph`, so
PROPAGATION is undefined (nothing to propagate to). Both exclusions are counted, not silently
dropped. The zero-2hop rate itself rises sharply with band (46%/51%/78%), consistent with larger
discovered graphs having proportionally more distant/disconnected node pairs.

## Cross-checks (both required by the STEP 1 authorisation)

- **n_discovered vs N/total_survivors**: checked on every event that joined to a valid drift-CSV
  row (733/742/617 of 738/744/622 used events — the rest had `n_discovered_h2`/`h3` as NaN, all on
  attribution-only rows). **0 mismatches in every band**: `n_discovered_h2 == N` and
  `n_discovered_h3 == total_survivors` exactly, always. STEP 1's caution ("do not infer... without
  checking") is resolved: they are provably the same quantity, at least in this sweep.
- **hbar verification**: on the full-2-hop-coverage subset (below), `||(h_v + sum(pre_embeddings))
  / N||` was compared against the drift CSV's `norm_h2_mean` column (the `.slices["mean"]` norm
  computed by the production `encode()` path itself, joined on `(run_id, seed, scenario_id,
  episode, step)`). **All 148 checked events (142+4+2) matched to ≤1.83e-7 absolute difference** —
  well inside float32→float64 round-trip tolerance. hbar, as recovered from the logged pre-removal
  embeddings, is confirmed identical to the environment's own pooled-mean slice.

## PROPAGATION: exact for every used event, both denominators (FINDING)

The numerator only needs the 2-hop set (3+-hop delta is exactly 0, per STEP 0/1); the corrected
denominator (`total_survivors`, logged as its own field per Section A) makes PROPAGATION exactly
computable regardless of 2-hop coverage — no restriction needed here.

| band | prop, all-survivor denom (median) | prop, 2-hop-only denom (median) | denom ratio (median / mean / max) | 2-hop-only inflation factor (median) |
|---|---|---|---|---|
| 10-15 | 0.0727 | 0.1098 | 1.500 / 1.831 / 9.000 | **1.40x** |
| 30-40 | 0.0454 | 0.1532 | 3.444 / 4.332 / 23.000 | **3.00x** |
| 80-100 | 0.0174 | 0.1537 | 9.714 / 12.233 / 74.000 | **8.44x** |

This is the exact failure mode Section A warned about, now demonstrated on real data rather than
argued from first principles: the 2-hop-only denominator would have inflated PROPAGATION's
magnitude by a factor that grows sharply with band size — 1.4x at 10-15, rising to 8.4x at 80-100
— and would have done so in a way indistinguishable from a genuine scaling finding unless the two
denominators were logged separately, which they were.

## DIRECT and the propagation-to-direct ratio: reliable only at band 10-15 (FINDING + explicit limitation)

DIRECT needs `hbar` = mean of raw pre-removal embeddings over **all** N nodes; 3+-hop nodes' raw
values (not just their zero delta) were never logged, so DIRECT is only exactly recoverable when
the 2-hop neighbourhood happens to cover literally every other node.

| band | full-2-hop-coverage rate | n full-coverage | direct (median) | prop/direct ratio, correct denom (median / mean) | frac(prop>direct) |
|---|---|---|---|---|---|
| 10-15 | 19.2% | 142 | 0.1154 | 2.328 / 2.885 | 0.655 |
| 30-40 | 0.5% | **4** | 0.2009 | 0.000 / 1.456 | 0.250 |
| 80-100 | 0.3% | **2** | 0.2601 | 0.543 / 0.543 | 0.500 |

**10-15 is the only band where this is a reportable finding.** n=142 (19.2% of used events):
PROPAGATION exceeds DIRECT in 65.5% of these events; median ratio 2.33, mean 2.89 — propagation
dominates direct on the real discovered graph at this band, consistent with the synthetic-proxy
result in `evidence_taskP.md`, now confirmed on real trajectory data for the first time.

**30-40 and 80-100 are NOT reportable as findings.** Not only is n too small (4 and 2 events) to
support a median/mean of anything, but inspecting the actual rows shows why coverage is so rare at
these bands: 3 of the 4 (30-40) full-coverage events have N∈{2,4} — i.e. the "full coverage"
condition is only met when the discovered graph is still trivially small (1-3 other nodes total,
early in an episode before it has grown toward the band's true node count), not when a genuine
30-40 or 80-100-node graph happens to be 2-hop-complete. The apparent prop/direct ratio of ~0 for
two of those four rows is a real computation (`prop_allsurv_norm` ≈ 2-4e-7, i.e. genuinely near-
zero propagation on an N=2 graph with one survivor) reflecting the degenerate size, not noise or a
bug — but it says nothing about propagation-vs-direct behaviour at an actual 30-40 or 80-100-node
graph. **The 2-hop-restricted logging design (Section B) cannot answer the DIRECT question at
these two bands**; only the PROPAGATION-only, denominator-inflation result above is trustworthy at
scale.

## Bottom line

- The propagation-to-direct decomposition (`probe_p.py`'s own metric) is now confirmed on real
  trajectory data for the first time — but only at band 10-15, where PROPAGATION exceeds DIRECT
  (median ratio 2.33), matching the synthetic-proxy direction of the earlier finding.
- The denominator correction from the STEP 1 authorisation was load-bearing: using the 2-hop-only
  denominator instead of the logged all-survivor count would have inflated PROPAGATION by a factor
  growing from 1.4x to 8.4x across the three bands — exactly the "looks like a scaling finding"
  trap the correction was written to prevent.
- DIRECT cannot be recovered from this logging design except on a small, size-biased subset
  (trivial-N early-episode events); this is a genuine limitation of the 2-hop-restricted design at
  30-40/80-100, not a bug, and should not be papered over by reporting the n=4/n=2 medians as if
  they generalised.

Outputs: `graphdepth_decomposition_summary.csv` (one row per band), `graphdepth_decomposition_events_<band>.csv`
(one row per used event, for anyone re-deriving).
