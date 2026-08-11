# Task GRAPH-DEPTH-WIDE STEP 4 RESULT — real-graph decomposition, full population, no coverage gate

Source: `graphdepth_sweep_wide/leaveembed_<band>/*/*.jsonl` (widened logging, commit `1522b71`; STEP 3
sweep, 71 seed×topology runs, same 15 manifest checkpoints and change generator as the original
sweep) joined against nothing else — the widened records are now self-sufficient. Script:
`compute_graphdepth_decomposition_wide.py` (committed `c87ef91`), a separate file from, and does not
modify, `compute_graphdepth_decomposition.py` or its committed band-10-15 output (n=142, median ratio
2.33), which stands unchanged and remains reproducible.

## 1–3. Coverage gate, counts, exclusions

| band | total events | batch-excluded | single-node | coverage-gate fail | used for ratio |
|---|---|---|---|---|---|
| 10-15 | 1437 | 76 | 1361 | **0** | **1361** |
| 30-40 | 1923 | 415 | 1508 | **0** | **1508** |
| 80-100 | 3591 | 772 | 2819 | **0** | **2819** |

**The coverage gate excludes nothing on embedding grounds, at any band** (`n_coverage_fail=0`
everywhere, checked against `len(pre_embeddings)==N-1` and `len(post_embeddings)==total_survivors`
for every single-node event). This was the expected outcome and is stated as such, not just
assumed: pre/post embeddings are built unconditionally from `h`/`hp` under the widened logger
(`cyberbattle_env_compressed.py:1290-1291`), so there is no longer any node set to fall short of.
Zero events excluded for `N<=1` and zero for `direct==0-but-N>1` either — every single-node leave
event in the sweep enters the ratio statistics. This is a complete population, not a sample of one.

## 4. Episodes and seeds behind each band's figure

| band | distinct (seed, scenario_id, episode) | distinct seeds |
|---|---|---|
| 10-15 | 200 | 5 |
| 30-40 | 150 | 5 |
| 80-100 | 220 | 5 |

## 5. Median-N selection check

| band | median N, used-for-ratio | median N, ALL single-node events in band |
|---|---|---|
| 10-15 | 7.0 | 7.0 |
| 30-40 | 25.0 | 25.0 |
| 80-100 | 73.0 | 73.0 |

**Identical, not just close, in every band.** The selection problem the GRAPH-DEPTH follow-up
identified in the 2-hop-restricted design (included-vs-excluded median N gaps of 5-vs-7 up to
2.5-vs-71) is completely gone: with every single-node event now usable, "used for ratio" and "all
single-node events" are the same population by construction.

## 6 & 9. The headline numbers, and the band 10-15 comparison — reported plainly, not reconciled

| band | PROPAGATION median | DIRECT median | ratio median | ratio mean | frac(prop>direct) |
|---|---|---|---|---|---|
| 10-15 | 0.0000 | 0.1195 | **0.0000** | 0.857 | 0.243 |
| 30-40 | 0.0000 | 0.0315 | **0.0000** | 0.951 | 0.279 |
| 80-100 | 0.0000 | 0.0085 | **0.0000** | 0.357 | 0.123 |

**Band 10-15: WIDE median ratio = 0.0000 (n=1361) vs the previous 2-hop-restricted median ratio =
2.328 (n=142).** This is not a rounding artefact of the print statement — verified directly against
the per-event data: **64.95% of band 10-15 events have `prop_norm` exactly 0.0** (bit-exact, not
just small; 70.98% are below 1e-6). This contradicts the earlier 2.33 finding outright, and is
reported as such rather than reconciled toward it, per the task's own instruction.

**Why, established from the same data, not asserted**: `departing_node_degree` (undirected degree
of the departing node in the pre-removal graph) has **median 0 or 1 across all three bands**, and
`prop_norm==0.0` is essentially deterministic when degree is 0:

| band | frac(departing_node_degree==0) | frac(prop==0 \| degree==0) | frac(prop==0 \| degree>0) |
|---|---|---|---|
| 10-15 | 45.3% | 96.6% (n=617) | 38.7% (n=744) |
| 30-40 | 50.5% | 99.87% (n=762) | 31.4% (n=746) |
| 80-100 | 77.9% | 99.95% (n=2196) | 31.1% (n=623) |

A departing node with zero undirected edges in `evolving_visible_graph` cannot, under the encoder's
directed incoming-message aggregation (established mechanism from GRAPH-DEPTH STEP 1's own
debugging), change any other node's embedding when removed — propagation is mechanically exactly
zero for these events, not attenuated-to-near-zero. These degree-0-departure fractions
(45.3%/50.5%/77.9%) match the OLD 2-hop-restricted sweep's own "zero_2hop_neighbours" exclusion
rates (45.8%/50.7%/77.9% of the same n_single denominators) almost exactly — the same population,
now named and quantified by its actual cause (departing-node degree) rather than only by its
symptom (no 2-hop neighbours).

**Restricting to structurally-connected departures (degree>0) — a second, complementary finding,
not a correction of the first** — the pooled-zero result is not the whole story either:

| band | n (degree>0) | median ratio | mean ratio | frac(prop>direct) |
|---|---|---|---|---|
| 10-15 | 744 | 0.744 | 1.567 | 0.445 |
| 30-40 | 746 | 1.358 | 1.923 | 0.563 |
| 80-100 | 623 | 1.265 | 1.614 | 0.559 |

Among departures that actually touch the graph structure, propagation is roughly balanced against
direct — near parity at 10-15 (slightly direct-favouring), mildly propagation-favouring at the other
two bands — nowhere near the 2.33-2.89 or the synthetic proxy's ~2.5 dominance, but also not the
degenerate zero the pooled figure alone would suggest. **Both figures are reported: the TRUE
population-pooled ratio is ~0 (because most real leave events remove near-isolated nodes under this
environment's own inverse-degree-weighted leave-selection rule, an already-established mechanism
from Task F3), and the ratio conditional on the departure actually touching graph structure is close
to parity.** Neither is discarded in favour of the other.

## 7. Depth/degree distribution (exhaustive, past 2 hops for the first time)

| band | degree median | degree mean | 1-hop mean\|shift\| | 2-hop mean\|shift\| | max hop tested, mean\|shift\| | unreachable-from-v survivors |
|---|---|---|---|---|---|---|
| 10-15 | 1 | 1.386 | 0.4028 (n=1428) | 0.0926 (n=1233) | 6-hop: 0.0000 (n=2) | 62.29% (n=5271/8462) |
| 30-40 | 0 | 1.348 | 0.3920 (n=1825) | 0.1090 (n=3030) | 9-hop: 0.0000 (n=6) | 67.80% (n=23405/34520) |
| 80-100 | 0 | 0.582 | 0.4060 (n=1546) | 0.1123 (n=2957) | 10-hop: 0.0000 (n=32) | 91.43% (n=179366/196169) |

The "3-plus-hop delta is exactly zero" claim, previously verified only up to a cutoff of 2 hops, is
now confirmed **exhaustively at every depth actually reached** (up to 10 hops at band 80-100,
thousands of survivor-events per hop bin at the lower depths) — every hop bin from 3 upward reports
`mean|shift|=0.0000`, and unreachable-from-v survivors (no path at all in the discovered graph) show
the same. The dominant qualitative fact at scale is how sparse the discovered graph actually is: the
fraction of survivors with no path to the departing node at all rises from 62% to 91% across the
three bands.

## Cross-checks

- Coverage gate: 0 failures at any band (item 2 above).
- `n_discovered`-equivalent fields were not re-checked here (already verified exactly, 0 mismatches,
  in the original GRAPH-DEPTH STEP 4 report on the 2-hop-restricted sweep; the widened logger did not
  change how `N`/`total_survivors` are computed, only what gets logged alongside them).

## Bottom line

The wider, unbiased population **reverses** the earlier band 10-15 finding rather than confirming it
at higher precision. The 2.33 median ratio was a real number on its own n=142 subsample, but that
subsample was shown (GRAPH-DEPTH follow-up) to be the smallest graphs in the band; the true
population is dominated by near-isolated departing nodes for which propagation is mechanically zero,
driving the pooled median to exactly 0. Restricted to departures that touch graph structure at all,
propagation and direct are roughly comparable in magnitude — a materially different, and much less
dramatic, story than either the original 2-hop-restricted real-data result or the synthetic
BFS/DFS-tree proxy in `evidence_taskP.md`. Both the pooled and the degree-conditional figures are
reported above; neither has been adjusted toward the other or toward the earlier figures.

Outputs: `graphdepth_wide_summary.csv` (one row per band), `graphdepth_wide_events_<band>.csv` (one
row per used event, with identifying metadata this time — run_id/seed/scenario_id/episode/step —
unlike the original decomposition script's per-event CSV), `graphdepth_wide_depth_distribution.csv`
(hop-binned shift magnitudes per band).
