# Task Y — the equal-degree scenario set (STEP 0 pilot)

Zero-shot design (no retraining): three sizes (30/60/90 nodes) at ONE held-constant mean degree (~22), giving
SAME-degree-varying-size (30/60/90 @ deg 22) and SAME-size-varying-degree (90 @ 22 vs existing 80-100 @ 54.9).
STEP 0 is a small pilot with a hard gate. Ran alongside the 750k training.

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
