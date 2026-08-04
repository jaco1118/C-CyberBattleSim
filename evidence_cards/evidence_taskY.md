# Task Y — the equal-degree scenario set (STEP 0 pilot -> STEP 1 executed)

Zero-shot design (no retraining): three sizes (30/60/90 nodes) at ONE held-constant mean degree (~22), giving
SAME-degree-varying-size (30/60/90 @ deg 22) and SAME-size-varying-degree (90 @ 22 vs existing 80-100 @ 54.9).
STEP 0 is a small pilot with a hard gate. Ran alongside the 750k training.

**STATUS: STEP 1 (execution) complete -- see bottom of card.** N=30 CONVERGED, N=60 and N=90 both
NOT CONVERGED (hard cap / borderline). No RQ1(c) conclusion drawn (out of scope). Full per-cell tables,
the N=90 correction, and the N=60 degree-metric bugfix are in STEP 1 below.

## 0.1 CLARIFICATION — the DEGREE reconciliation table (read-only) [FINDING]

At least four distinct quantities are all called "degree" in this work. They are **not mutually comparable**,
and the RQ3 structural predictor "degree" cannot be specified until this is pinned. Measured on the real
scenario files (15 graphs/band) and cross-checked against the cards. **No figure changed.**

| # | figure | graph | in/out/total | band(s), N, averaged over | comparable to others? |
|---|---|---|---|---|---|
| **1** | **7.9 / 22.3 / 54.9** | **knows_graph** | **out-degree** (≡ undirected; the knows graph is near-symmetric — measured out 21.0/57.2, total = 2×) | 10-15 / 30-40 / 80-100; N=12.5/34.7/89.3; **all nodes**, 25 scenarios/band | the **canonical "scenario mean degree"** the scaling curve uses; the **Task Y target of 22 is expressed in THIS** |
| **2** | **8.9 / 24.2 / 61.8** | **access_graph** | **out-degree** | same as #1 (all nodes, 25 scen/band) | comparable in KIND to #1 (both scenario out-degree, all nodes), ≈**1.09× #1**; DIFFERENT graph (access ⊇ knows-reachability) |
| **3** | **156.9 / 94.5** (`evidence_taskN.md:18`) | **static access_graph** | **TOTAL** (in+out; `access_graph.degree(n)`, `evidence_taskF3.md:28`) | **80-100 ONLY**; per **DEPARTED** node, conditioned owned (n=3,603) / not-owned (n=11,936); over **leave EVENTS**, not nodes | **NOT comparable** to #1/#2: total-not-out (~2×), access-not-knows (~1.1×), event-conditioned subset, one band. (Mean access-total over all 80-100 nodes is 133.1 — 156.9 is the high-degree owned subset, 94.5 the low.) |
| **4** | corr(prop, degree) **+0.66 / +0.79 / +0.39** (`evidence_taskP.md:67`) | **DFS spanning-tree PROXY** (`Gu`, undirected) — NOT a scenario graph | **undirected** | proxy per removal trial; mean proxy degree ≈2 (`evidence_taskP.md:16`) | **NOT comparable**: the RQ3 propagation correlations' "degree" is the **proxy tree degree**, an entirely different object from the scenario knows/access degree |

**Two naming errors surfaced (report, do not edit the manuscript):**
- **`evidence_taskP.md:13` calls #1 "the dense access graph whose degrees (7.9/22.3/54.9)".** They are
  **knows_graph** out-degrees (Task H `:42`, dissertation log `:264`), not access. Access out-degrees are
  8.9/24.2/61.8 (#2). Relabel #1 as knows-degree wherever Task P cites it.
- **#1 is OUT-degree of knows; #3 is TOTAL-degree of access.** Both are written as "degree." So N1's 156.9
  vs the band's 54.9 is NOT "≈3× denser for owned departures" as a like-for-like reading would suggest — it
  mixes total-vs-out (~2×), access-vs-knows (~1.1×), AND owned-conditioning. Any cross-use of #1 and #3 must
  state the directionality and graph.

**Which "degree" the RQ3 predictor should be, and what Task Y controls:** the scaling curve and the Task Y
control both use **#1 (knows out-degree)** — `knows_neighbor_probability` directly governs the knows graph
(`generate_network.py:190-194`), so calibrating it to knows out-degree 22 makes the control mean exactly what
the design intends. Access out-degree (#2) tracks knows at ≈1.09× and will be reported alongside. The
propagation correlations (#4) are on the proxy and must stay labelled as proxy-degree (Task Q Q2.6). **The
RQ3 analysis must fix "degree" = #1 (scenario knows out-degree) and state it, or it is unspecified.**

## 0.2 THE THRESHOLD — analysed; empirical check pending pilot [ARTIFACT → FINDING on pilot completion]

Rejection is `access/knows/dos_connectivity < 0.5` (`generation_utils.py:43-45`). "Connectivity" is
**path-length/reachability based**, NOT density: `connectivity = 1 − avg_shortest_path / (2·(N−1))`
(`networkx_utils.py`; unreachable pairs contribute 2·(N−1)). A degree-22 graph on 90 nodes is well-connected
(avg degree 22 ≫ ln 90 ≈ 4.5 → short paths, few/no unreachable pairs) → connectivity ≈ 0.99 ≫ 0.5, so it
should **pass**. **The sparse-large-N kill risk therefore looks low — but this is the load-bearing check and
is verified empirically on the pilot 90-node graphs below.** The threshold is NOT altered.

## 0.3 WHAT ELSE MOVED with p — pending pilot [to compare vuln/service/value distributions vs existing]

## 0.4 GENERATION COST — pending pilot [wall clock per graph, projected total vs N4's 45–75 min]

## 0.5 CONTENTION — confirmed clear [ARTIFACT]

The pilot is a single generation process capped at 2 threads (`OMP_NUM_THREADS=2`, `PYTHONHASHSEED=0`); the
750k run uses 5 procs (~20 threads) of 24 cores, leaving headroom. Verified: the 750k continued at 12 procs
throughout pilot launch. No interference.

## Design / calibration setup [ARTIFACT]

Pilot configs (`y_pilot_configs/gen_{30,60,90}.yaml`), based on `generation_config_m.yaml` with
`num_nodes_range=[N,N]` and `knows_neighbor_probability_range` scaled ∝ 29/(N−1) to hold knows out-degree ~22:
**30 → [0.2,0.8]** (identical to existing 30-40 → the ANCHOR), **60 → [0.098,0.394]**, **90 → [0.065,0.261]**.
3 graphs/size, SecureBERT only, no split. Degree measured empirically (never the configured value alone).

## STEP 0 GATE — RESULTS (pilot complete; supersedes the PENDING markers above) [FINDING]

Two calibration iterations (3 graphs, then 5). **iter-1 hit the wrong degrees (18/23/27), so its numbers are
NOT carried forward** — all figures below are at the iter-2 (final calibrated) p.

**0.1 — calibration to knows out-degree 22 SUCCEEDS at 30/60 but FAILS at 90 [FINDING]:**

| size | configured `knows_neighbor_probability_range` | MEASURED knows out-deg (spread) | access_conn | vulns/node |
|---|---|---|---|---|
| 30 | [0.244, 0.976] | **21.8** (19.4–24.3) | 0.726 | 20.1 |
| 60 | [0.094, 0.377] | **23.4** (13.1–30.3 — WIDE) | 0.640 | 15.5 |
| 90 | [0.053, 0.213] | **28.1** (23.1–33.9) | 0.648 | 27.9 |

**30 and 60 reach ~22** (60's within-size spread is large, a caveat for "held constant"). **90 does NOT:
it floors at ~28.** Lowering p from iter-1→iter-2 ([0.065,0.261]→[0.053,0.213]) left degree at 27→28 while
**rejections rose 1→6** — the threshold culls the low-degree 90-node candidates, so the surviving set is
**degree-floored** at ~28. **Degree 22 is not achievable at 90 nodes.**

**0.2 — the threshold's real effect [FINDING, corrects the earlier "passes" read]:** the connectivity metric
is path-length based (§0.2 above), so a degree-**28** 90-node graph passes (access_conn 0.648) — but to reach
degree **22** at 90 requires a p at which **55% of candidates fall below 0.5 and are rejected**, and the
survivors are the degree-~28 ones. So the threshold does not reject degree-22 graphs by *name*, but it makes
degree-22 90-node graphs **unreachable in practice** by culling them. A **higher constant-degree target (~28)
is achievable at all three sizes and still separates from 80-100's 54.9** — but it breaks the degree-22 anchor
with the existing 30-40 band. (Threshold NOT altered.)

**0.3 — the vulns/node confound is SURVIVORSHIP, confirmed by all-generated vs accepted [FINDING]:** captured
every candidate G's vulns/node + accept/reject (temporary `SURV_LOG` line in `generate_valid_probabilities`,
**since reverted — git clean**; the graph G, hence its vulns, is fixed per candidate and only edges re-roll,
so this measures G-level selection):

| size | rejection rate | vulns/node ALL-generated | ACCEPTED | REJECTED | verdict |
|---|---|---|---|---|---|
| 90 | **55%** (6/11) | 18.0 | **20.4** | 16.0 | accepted **richer** → SURVIVORSHIP |
| 60 | 72% (13/18) | 14.4 | 15.1 | 14.2 | mildly richer → weak survivorship |

**Accepted graphs are systematically vuln-richer than rejected → survivorship, NOT direct dependence** (and
mechanistically: vulns derive from sampled services × CVE data, config-fixed, so p cannot move them directly).
The bias is strong at 90 (accepted 20.4 vs rejected 16.0) and the accepted set is far above the existing
80-100 baseline (12.4). So holding degree constant by lowering p at 90 **trades the degree confound for a
survivorship-driven vulns/node confound.**

**Remedy analysis (stated, not chosen — a design decision outside this task):** the generator has **no
parameter that holds vulns/node fixed while p varies.** The only vuln-adjacent config key is
`num_services_range: [1, 2]` (`generation_config_m.yaml`), which sets services per node; vulns/node then comes
from the sampled services' CVE data, not a direct count, and the survivorship operates through the
connectivity threshold regardless. So no clean config-level remedy exists; any fix would change what the
scenarios are.

**0.4 — cost [ARTIFACT]:** ~40 s–1.5 min/graph (SecureBERT only). iter-1 (9 graphs) ~10 min; iter-2 (15
graphs) ~15 min. A full set (15 graphs) ≈ **15–25 min** including rejections, **well under N4's 45–75 min**
estimate (N4 likely assumed all 8 extractors).

**0.5 — no contention** (confirmed above).

## GATE — the design is COMPROMISED at the 90-node cell; decision is the user's

**0.2 removed the largest risk (threshold does not reject equal-degree graphs by name), and the degree
reconciliation table (0.1) surfaced that RQ3's predictor is unspecified** (four "degree" quantities; the
propagation correlations are on the proxy — see `open_items.md` OI-1). **But the pilot shows the equal-degree
target of 22 cannot be met at 90 nodes: it floors at ~28 via connectivity-threshold survivorship, and that
same survivorship inflates vulns/node (accepted 20.4 vs rejected 16.0 vs existing 12.4).** So:
- **SAME-degree-varying-size (30/60/90 @ 22)** — broken at 90 (achievable only at ~28, with a vulns/node
  confound). Achievable cleanly only at 30 and 60.
- **SAME-size-varying-degree (90 @ ~22–28 vs 80-100 @ 54.9)** — the 90 cell exists at ~28, still separates
  from 54.9, but carries the vulns/node confound.

**Design decisions (outside this task):** (a) raise the held-constant target to ~28 (reachable at all three,
separates from 54.9, but breaks the 22-anchor with 30-40); (b) invoke the pre-registered STEP 3.3 fallback
(report the scale relation as joint size-and-degree, name degree as an unseparated component). **Reporting and
STOPPING; no full set generated. Pilot/diagnostic graphs (`graphs_ypilot*`, `graphs_ysurv*` in env_samples)
are diagnostic-only and must NOT enter the experiment.**

## STEP 0 VERIFICATION (target 28 + existing-band survivorship) — GATE STOP [FINDING]

Per the user's target-28 decision, verified before generating the full set (temporary `SURV_LOG` in
`generate_valid_probabilities`, since **reverted — git clean**).

**Target-28 calibration (a=rejection, b=acc/rej vulns/node, c=nodes, d=degree):**
| cell | rejection | vulns/node acc vs rej | nodes | measured knows out-deg |
|---|---|---|---|---|
| 30 | 62% | 20.4 vs 9.6 | 30 | **21.8 (NOT 28)** |
| 60 | 74% | 17.3 vs 15.6 | 60 | 27.5 |
| 90 | 50% | **17.7 vs 17.8 (no survivorship)** | 90 | 29.8 |

- **Degree 28 is UNACHIEVABLE at 30 nodes:** `[0.35,1.0]` gave 21.8, identical to the degree-22 config
  `[0.244,0.976]` (21.8) — degree saturates at ~22 at 30 nodes regardless of p (recon-vuln edge cap). So
  **no single degree is achievable across 30/60/90** (30 caps ~22, 90 floors ~28 — disjoint windows). The
  held-constant-degree design is infeasible across the full range for BOTH target 22 and target 28.
  > **[2026-08-01 CORRECTION — the "disjoint windows / 90 floors ~28" claim is WRONG; see the STEP 0 GATE
  > RE-RUN section below.]** The re-run pushed knows_neighbor_probability *lower* than this pilot did and
  > N=90 reaches degree **13** at `[0.0,0.1]` and **19.5** at `[0.05,0.15]` (conn 0.77–0.80, rej 0.44–0.56 —
  > generatable). N=90 does **not** floor at 28. The achievable windows **overlap at [13, 24]**, and a common
  > degree of **~20** is feasible across all three sizes (and roughly vulns/node-flat). What remains true: **28
  > specifically is unreachable at N=30** (ceiling ~24), so the target-28 gate still STOPs — but the equal-degree
  > design is revivable at target ~20, not dead.
- **The survivorship hypothesis is CONFIRMED at 90 for target 28:** acc 17.7 ≈ rej 17.8 (no bias) — nothing
  pushed below the connectivity floor. 60 mild (17.3 vs 15.6). So a **60/90-only** pair at degree ~28 would be
  a clean 2-point same-degree contrast; 30 cannot join.
- **GATE outcome: STOP** — the four-cell design cannot be built (30 can't reach 28). No full set generated.
  Decision (user's, outside this task): run a reduced **60/90-only** same-degree contrast (+ anchor +
  90-vs-80-100 degree contrast), or invoke the pre-registered STEP 3.3 fallback.

**Existing-band survivorship (the study-wide question) [FINDING — affects every result]:**
| existing band | rejection | vulns/node acc vs rej | nodes | degree |
|---|---|---|---|---|
| 10-15 | 67% | 18.3 vs 10.7 | 13 | 7.3 |
| 30-40 | 44% | 15.3 vs 12.1 | 35 | 19.6 |
| 80-100 | 38% | 13.7 vs 11.9 | 86 | 53.1 |

**At the existing bands' own `[0.2,0.8]` probabilities, rejection is 38–67% and accepted graphs are
systematically vuln-richer than rejected ones.** So **every scenario in this study was drawn from a
survivorship-filtered population — the reported networks are systematically more vulnerability-rich than the
generator's nominal distribution.** This is a **study-wide external-validity disclosure**, not a Task Y issue,
and the bias is **strongest at the smallest band** (10-15: 18.3 vs 10.7), which could itself contribute to
cross-band differences. Recorded in `open_items.md` (OI-2). No remedy chosen (design decision, outside this
task). Diagnostic graphs (`graphs_yv28_*`, `graphs_yvexist_*`) are diagnostic-only, excluded from any experiment.

## STEP 0 GATE RE-RUN — target-28 decision + achievable-degree windows [FINDING, 2026-08-01]

Per the user's four-cell decision (30@~22 anchor + 30/60/90@~28 experiment). Clean re-measurement against the
RQ3 predictor (scenario **knows out-degree**), sweeping `knows_neighbor_probability_range` both up and down
per size. Harness: `<jobdir>/taskY_calib.py` (throwaway candidates, writes nothing to the study; reuses the
generation functions + `generate_valid_probabilities`). K=20–25 accepted graphs per cell. Ran alongside — and
did not contend with — the Task L STEP 3 sweep (GPU idle, 12/24 cores free).

**Correction to the user's premise.** Natural knows out-degree at the study probability `[0.2,0.8]` scales
~**0.62×N** (measured 19.6 / 36.6 / 54.6 at N=30/60/90 — matching the reconciled 22.3/54.9 band scaling), NOT
flat ~22. Consequences: reaching 28 requires **raising** p at N=30 but **lowering** p at N=60/90 (their natural
degree is already 37/55, well above 28). And 28 is **not** N=90's natural floor — N=90's floor is ~13.

**Achievable knows-out-degree windows** (mean [lowest-config, highest-config], with frac of nodes carrying a
recon vuln):

| N | floor (low p) | ceiling (p→1) | frac_nz | 28 reachable? |
|---|---|---|---|---|
| 30 | ≤19.6 | **~24** (`[0.95,1.0]`=23.4) | ~0.80 | **NO — ceiling ~24 < 28** |
| 60 | ~9 (`[0.0,0.1]`) | ~45 | ~0.77 | yes (p≈[0.13,0.4]) |
| 90 | ~13 (`[0.0,0.1]`) | ~67 | ~0.78 | yes (p≈[0.08,0.25]) |

- **Why 28 caps out at N=30 (mechanism):** only ~80% of nodes carry a Reconnaissance vulnerability
  (`frac_nz`≈0.80), and a node with no recon vuln has knows out-degree 0. So mean out-degree ≤ 0.80·(30−1) ≈
  **23** even at p=1 — a **vuln-availability cap, not a p limit**. Raising p cannot break it. (`[1.0,1.0]` gave
  22.0 at higher rejection — the plateau, not an increase.)

**Gate items (a)–(d) at/near target 28:**
- **(a) rejection rate:** 0.28–0.72 across cells (K-noisy), no monotone size trend; typical ~0.3–0.5.
- **(b) vulns/node accepted vs rejected:** accepted ≥ rejected in most cells (survivorship direction holds),
  e.g. N=90 `[0.2,0.8]` 16.9 vs 13.1; N=30 `[0.95,1.0]` 20.4 vs 14.0. A few small-n cells reverse (noise).
- **(c) node counts:** pinned exactly to 30/60/90 (`num_nodes_range=[N,N]`).
- **(d) vulns/node across sizes — roughly FLAT and size-independent.** At `[0.2,0.8]`: 19.3/19.1/17.6
  (N=30/60/90); at the common feasible degree ~20: 19.3/16.9/16.7. Vuln sampling is per-node, so vulns/node
  does not track N. The confound the amendment worried about (differential survivorship inflating vulns/node
  at one cell) does **not** appear strongly. But **(d) cannot be evaluated AT degree 28** because the N=30@28
  cell cannot be built at all.

**GATE OUTCOME: STOP on target-28** — the pre-registered stop fires, but one step earlier than the vulns/node
check: degree 28 itself is unreachable at N=30 (ceiling ~24). No full set generated.

**NEW — the design is revivable at a lower target (supersedes the pilot's "disjoint/infeasible" verdict).**
The windows overlap at **[13, 24]**, so a **common degree ~20 is feasible across all three sizes** with
acceptable rejection and roughly flat vulns/node:

| cell | p-range | deg | rej | vulns/node |
|---|---|---|---|---|
| 30 | `[0.2,0.8]` | 19.6 | 0.30 | 19.3 |
| 60 | `[0.1,0.3]` | 20.4 | 0.32 | 16.9 |
| 90 | `[0.05,0.15]` | 19.5 | 0.44 | 16.7 |

A 3-cell equal-degree design at **~20** preserves BOTH contrasts RQ1(c) needs: node count 30→60→90 at fixed
degree ~20, and degree ~20 (90-node cell) vs 54.9 (80-100 band) at fixed size — a ~2.7× degree separation,
wider than the target-28 plan's would have been. **Decision is the user's** (nothing generated): (i) equal-degree
at target ~20 across 30/60/90; (ii) 60/90-only at ~28; or (iii) the pre-registered STEP 3.3 fallback.
**Amendments 1 (realised discovered-subgraph degree) and 2 (static-cell reporting) apply at STEP 2, if a design
is chosen** — and STEP 2 stays queued behind Task L STEP 3 + OI-1 probe + RQ2(c) counterfactual regardless.

## STEP 1 — DECISION + EXECUTION: equal-degree ~20 across N=30/60/90 [FINDING]

Decision taken: option (i) from the GATE OUTCOME above -- equal-degree design at target ~20 across all
three sizes, using the p-ranges from the revivability table (30 `[0.2,0.8]`, 60 `[0.1,0.3]`, 90's
existing `[0.05,0.15]`-family pilot topologies reused/extended). Branch: `taskY-probe-n90`.

### 1.1 N=90 cell: probe -> 5-seed batch -> extension -> FINAL corrected verdict

N=90 seed42 trained first as a probe (static500k, then resumed to 750k), then escalated to a full
5-seed batch (seeds 100/123/200/300, static500k each), with seeds failing the F4 band rule at 750k
extended further. Convergence rule (F4, from `evidence_taskF4.md`): metric `train/Root owned nodes`,
50k windows, per-seed within-band iff |Delta%|<5%, band CONVERGED iff mean|Delta%|<5% AND >=4/5 seeds
within-band.

**Per-seed final training leg and F4 result (recomputed fresh 2026-08-03/04, script-reproducible):**

| seed | final step | Delta% | within? |
|---|---|---|---|
| 42  | 1.25M (750k + 500k ext) | +1.09%  | YES |
| 100 | 1.25M (750k + 500k ext) | -13.90% | no  |
| 123 | 500k (never extended)   | -4.10%  | YES |
| 200 | 500k (never extended)   | +1.28%  | YES |
| 300 | 750k (one extension)    | -5.01%  | no  |

mean|Delta%| = 5.08% (>=5%), within-band = 3/5 (need >=4) -> **NOT CONVERGED**. seed300 misses by a
near-boundary margin (-5.01% vs the 5.00% threshold).

**CORRECTION (logged 2026-08-04):** an earlier informal, never-committed claim of "N=90 CONVERGED
(4/5)" was made in conversation before this session's context was compacted. That number does not
appear in any commit or artifact. The table above is a fresh, script-reproducible recomputation
(verified against raw tfevents data, no missing run folders) and **supersedes** the earlier claim.
Treat any reference to "N=90 converged" predating this card update as stale.

Command/script: `cyberbattle/agents/compute_convergence_check.py` (committed 4173c53, "Task Y-N90-EXTEND:
commit F4 convergence-rule window-extraction script" -- reproduces the prior inline 750k numbers
seed42 -9.61%/seed100 +10.00% with `--stop 250000`), re-run at each seed's actual final leg. Output
preserved: `cyberbattle/agents/y_n30n60/verdicts/N90_final_recomputed.txt` (committed c455dd9).
Run folders: `cyberbattle/agents/logs/yprobe_n90*` (gitignored, on-machine).

**N=90 degree bookkeeping:** the cell's degree span is genuinely WIDE per-topology (knows out-degree
18.9-27.4 across the 5 instances, mean 21.53, SD 3.39) -- confirmed via a separate check
(`cyberbattle/agents/y_n30n60/verify_n90_degree.py`, committed 8258f1f) that this span uses the correct
control metric (knows out-degree, #1 in the 0.1 reconciliation table above) and predates -- is NOT
affected by -- the N=60 degree-metric bug described in 1.2 below.

### 1.2 Y-N30-N60: crossed cells (N=30, N=60) at the same target

10 fresh topologies generated (5 seeds x {N=30, N=60}), p-ranges per the table above. Measured on the
CALIBRATED control metric (knows_graph out-degree, #1): **N=30 = 19.69 +/- 1.18, N=60 = 19.93 +/- 2.82**
-- degree IS approximately held constant across these two cells on the metric the design controls.

**Degree-metric bug + fix (mid-session correction):** an early version of the topology-measurement
script (`measure_topos.py`) reported mean UNDIRECTED degree of the `access_graph` (a different graph,
inflated by reachability, and a different convention -- undirected vs out-degree) and mislabelled it
"degree", producing a spurious "N=60=44.5 vs N=30=22.4, degree not held constant" alarm. Investigated
and fixed (committed a2c8285): the calibrated metric (knows out-degree) shows the two cells match
closely (~19.9 vs ~19.7); the 44.5 figure was an artifact of the knows graph being only ~34%
reciprocal at N=60 (vs ~68% at N=30) plus the access graph's reachability inflation. No generation
error, no regeneration needed. Script now reports knows out-degree as the primary control metric.
Committed: `cyberbattle/agents/y_n30n60/measure_topos.py` + `topo_measurements.csv` (a2c8285, 158f4a8).

**Pre-registered checkpoint stopping rule:** train in 250k increments ("stages"); F4-check each cell
after every stage; stop a cell at the first stage that CONVERGES; hard cap 1.25M (stage 5). Orchestrator:
`cyberbattle/agents/y_n30n60/run_stage.sh` (committed 158f4a8; thread-capped at 8db44d3 after an initial
launch hit 60x CPU-oversubscription -- 10 uncapped procs on 32 cores, load ~116, fps 3-5/proc vs 245
single-proc -- fixed via `OMP/MKL/OPENBLAS/NUMEXPR/VECLIB_NUM_THREADS=1`, restoring ~84 fps mean).

**N=30 result: CONVERGED at stage 1 (250k).** mean|Delta%|=4.18%, 4/5 within band. Cell done, no
further training. Verdict: `y_n30n60/verdicts/stage1_N30.txt` (e621f6c).

**N=60 result: NOT CONVERGED, ran all 5 stages to the 1.25M hard cap.**

| stage | absolute step | mean\|Delta%\| | within-band | verdict |
|---|---|---|---|---|
| 1 | 250k  | 6.97% | 2/5 | NOT CONVERGED |
| 2 | 500k  | 6.55% | 2/5 | NOT CONVERGED |
| 3 | 750k  | 7.61% | 3/5 | NOT CONVERGED |
| 4 | 1M    | 8.62% | 2/5 | NOT CONVERGED (worst) |
| 5 | 1.25M | 4.52% | 3/5 | NOT CONVERGED (best mean, still <4/5) |

Non-monotonic trajectory (not steadily approaching or receding from convergence): per-seed pass/fail
flips almost every stage (e.g. seed123 converged at stage1, failed stages 2-3, recovered stage3-5;
seed300 within-band at every stage 1-4, missed only the final stage 5). Only the count criterion
(>=4/5) ever failed at stage 5 -- the mean criterion (<5%) actually passed there. Verdicts:
`y_n30n60/verdicts/stage{1,2,3,4,5}_N60.txt` (e621f6c, 3ab40d4, f65ea5b, 3a0bcac, c455dd9).

### 1.3 FINAL three-way comparison (N=30 / N=60 / N=90) -- no RQ1(c) conclusion drawn

| | N=30 | N=60 | N=90 |
|---|---|---|---|
| control metric (knows out-deg) | 19.69+/-1.18 | 19.93+/-2.82 | 18.9-27.4 (wide, per-topology) |
| mean\|Delta%\| | 4.18% | 4.52% | 5.08% |
| within-band | 4/5 | 3/5 | 3/5 |
| **verdict** | **CONVERGED (250k)** | **NOT CONVERGED (hit 1.25M cap)** | **NOT CONVERGED (borderline)** |
| training used | 1.25M total (5x250k) | 6.25M total (5 seeds x 1.25M) | 4.75M total (mixed per-seed) |

**No RQ1(c) conclusion drawn from this table (explicitly out of scope for this task).** The three cells
differ in topology count, degree-spread, and stopping history and are NOT a controlled N-only sweep --
N=30's fast convergence, N=60's persistent instability, and N=90's near-boundary miss are reported as
observations, not as evidence of any N-vs-convergence-difficulty relationship.

### 1.4 Artifacts (all committed, branch `taskY-probe-n90` unless noted)

- `cyberbattle/agents/compute_convergence_check.py` -- F4 rule engine (4173c53)
- `cyberbattle/agents/y_n30n60/run_stage.sh` -- staged-training orchestrator, thread-capped (158f4a8, 8db44d3)
- `cyberbattle/agents/y_n30n60/y_base.yaml` -- N=30/N=60 base training config (158f4a8)
- `cyberbattle/agents/y_n30n60/measure_topos.py` + `topo_measurements.csv` -- degree/vuln measurement, bug-fixed (158f4a8, a2c8285)
- `cyberbattle/agents/y_n30n60/verify_n90_degree.py` -- N=90 degree-bug-scope check (8258f1f)
- `cyberbattle/agents/y_n30n60/generation_configs/` -- the 10 topology-generation recipes (158f4a8)
- `cyberbattle/agents/y_n30n60/seeds/`, `seeds_all/` -- in-repo stable seed folders (158f4a8)
- `cyberbattle/agents/y_n30n60/verdicts/` -- every stage's F4 output, N=30/N=60/N=90 (e621f6c..c455dd9)
- Topology binaries (234MB, 10 folders) gitignored per repo convention; off-machine backup only.
- Y-N30-N60/N=90 Part 2 (O8 Arm 4) work continues on `rq2b-10-15`, unrelated to this card's scope.
