# Task M — does two-hop coverage explain why degree stops predicting propagation?

Offline, frozen-encoder forward passes on graphs that already exist. No training, eval, episodes, or env
changes. Ran in a 17-second, 2-thread job in idle headroom beside the Task Z STEP 2 eval (0.4). **Verdict:
option (d) — coverage is inseparable from scale with the available trials; the hypothesis has no trial-level
support. RQ3(c) should be DROPPED.**

## STEP 0

**0.1 Which graph the propagation was computed on — the two descriptions are genuinely different, and the
CODE settles it.** `probe_p.py`'s docstring says "SPARSE = BFS tree over access_graph" (`probe_p.py:4`), but
the **code builds a DFS SPANNING TREE**: `edges = list(nx.dfs_tree(A, nodes[0]).edges())`
(`probe_p.py:60`), where `A = net.access_graph.subgraph(nodes)` (`:55`) and `nodes` is a BFS-collected node
set (`bfs_nodes`, `:76`). So the propagation ran on a **DFS spanning tree over the access-subgraph induced
on a BFS-selected node set** — deep/chain-like (inline comment `:57-59`: "DFS spanning tree edges ... DEEP
(chain-like, diameter>2, mean deg ~2)"). **It is NOT the dense access graph and NOT literally a BFS tree.**
This is exactly the conflation the task exists to resolve: node *selection* is BFS, edge *structure* is DFS.
Task M computes coverage on this same DFS-tree graph.

**0.2 Deterministic + reproducible; trial counts.** Global seed fixed once (`probe_p.py:16`
`random.seed(0); np.random.seed(0)`); `probe_m.py` uses the identical seed and identical graph construction.
Re-running reproduces Task P bit-for-directional: prop medians **0.1272 / 0.0659 / 0.0215** (P card:
0.127/0.065/0.022) and corr(prop,deg) **+0.657 / +0.796 / +0.383** (premise: +0.66/+0.79/+0.39) at
10-15/30-40/80-100. **Trials per band: 33 / 100 / 120** (253 total), matching the P card. [ARTIFACT]

**0.3 Hop distance already computed — reused, not re-implemented.** `probe_p.py:100`
`hops = nx.single_source_shortest_path_length(Gu, v)` (from the removed node `v` over the undirected sparse
graph `Gu`). Task M reuses exactly this to count nodes within 1 and 2 hops (`probe_m.py`: `within1/within2`,
`cov1 = within1/N`, `cov2 = within2/N`). No second implementation.

**0.4 No contention with Task Z.** Task Z *training* is complete; its STEP 2 *eval* is running at MAXJOBS=5
(20 of 24 cores). The Task M probe needs encoder forward passes, but on tiny graphs (N=11–69) it is a
17-second job; capped at `torch.set_num_threads(2)` it used only idle headroom. Verified: Z's completed-CSV
count was unchanged across the probe (2→2) and load stayed ~20. Non-contending in practice.

## STEP 1 — coverage on the CORRECT (sparse) graph — SUPERSEDES gate N6 [FINDING]

Two-hop coverage per band, on the DFS-tree graph the propagation runs on:

| band | N | cov2 mean | cov2 variance | cov2 range |
|---|---|---|---|---|
| 10-15 | 11 | **0.411** | 0.0078 | 0.250–0.600 |
| 30-40 | 25 | **0.202** | 0.0016 | 0.120–0.320 |
| 80-100 | 69 | **0.071** | 0.0001 | 0.043–0.087 |

**This CONTRADICTS gate N6 and supersedes it.** N6 (computed on the **dense access graph**) reported 2-hop
coverage mean **0.70–0.82 at every band** with **variance non-monotone, peaking at 30-40 (0.095)**. On the
correct sparse graph: coverage is far **lower** (0.41→0.20→0.07) and **monotone decreasing**, and its
variance is also **monotone decreasing** (0.0078→0.0016→0.0001), not peaked at 30-40. **N6's coverage
figures and its "variance peaks at 30-40, aligning with the degree-correlation peak" claim are artefacts of
the wrong graph and are hereby labelled SUPERSEDED.** The saturation the hypothesis needs (coverage → ~1)
never occurs on the encoder's real input; coverage tops out at ~0.41 in the smallest band.

## STEP 2 — the trial-level test (n=253 pooled) [FINDING]

**Confound structure (2.4 context):** corr(cov2, N) = **−0.852** (coverage is nearly collinear with graph
size — small N ⇒ high coverage), corr(deg, N) = +0.090, corr(cov2, deg) = +0.113, corr(cov2, prop) = +0.826,
corr(deg, prop) = +0.332.

**2.2 — the falsifiable prediction FAILS.** OLS `prop ~ deg + cov2 + deg:cov2` (standardized). The hypothesis
predicts a **negative** `deg:cov2` (degree stops predicting as coverage rises). Observed:
**deg:cov2 = +0.0115, CI95 [−0.0018, +0.0183]** — **positive point estimate, CI includes 0 (null).** Not the
predicted negative. (deg = +0.0119 [+0.0077,+0.0166]; cov2 = +0.0449 [+0.0382,+0.0515].)

**2.3 — within-coverage-tercile corr(prop,deg) is NON-monotone and just re-traces the bands:**
- low-cov (0.043–0.072): n=116, corr = **+0.381** — band composition **100% 80-100**
- mid-cov (0.087–0.200): n=76, corr = **+0.696** — **95% 30-40**
- high-cov (0.240–0.600): n=61, corr = **+0.191** — **54% 10-15, 46% 30-40**

The hypothesis predicts strong→weak as coverage rises. Instead it is weak (0.38) → strong (0.70) → weak
(0.19): it **peaks in the middle**, exactly reproducing the band-level non-monotone pattern (0.66/0.80/0.38
peaks at 30-40). The binning is not revealing a coverage mechanism; it is re-drawing the band axis.

**2.4 — coverage stands in for scale.** partial corr(prop, deg | N) = +0.491; partial corr(prop, deg | cov2)
= +0.427 (degree's link to propagation survives controlling for either — it is not explained by coverage).
Augmenting the model with N and deg:N: **deg:cov2 → +0.0121, CI95 [−0.0149, +0.0204] — NULL.** Once scale is
controlled, the (already null) interaction carries no signal. Coverage cannot be credited with anything
beyond N.

**2.5 — the design cannot separate coverage from band.** The coverage terciles are essentially single bands
(low = 116/116 from 80-100; mid = 72/76 from 30-40; high = 33 from 10-15 + 28 from 30-40), and per-band cov2
ranges barely overlap (10-15: 0.25–0.60; 30-40: 0.12–0.32; 80-100: 0.043–0.087 — only 30-40/10-15 touch at
the top). With coverage ≈ −0.85 collinear with N and bins ≈ bands, **coverage and scale are not separable
with these trials.**

## STEP 3 — VERDICT

**3.1 (one sentence):** On the correct (sparse) graph, two-hop coverage does **not** explain the weakening of
the degree effect — the predicted negative degree×coverage interaction is absent (null/slightly positive),
and in any case coverage is inseparable from scale (cov2 ≈ −0.85 with N, coverage terciles are essentially
single bands, and the interaction vanishes once N is controlled).

**3.2 — named choice: (d), inseparable from scale** — the binding, more-informative negative. Coverage is
collinear with N, the coverage bins are the bands, and the deg:cov2 interaction dies under N-control. **(c)
also holds** (the interaction is null even before controlling for N, so there is no positive evidence for a
coverage mechanism), but (d) is the deeper reason the question cannot be answered as posed. Options (a) and
(b) are excluded: not trial-level (a), and even the band-level directional support N6 offered is gone once
coverage is measured on the correct graph (monotone, not peaked).

**3.3 — RQ3(c) should be DROPPED, plainly.** It cannot be reworded into a supported mechanism: coverage does
not do the work, and the trials cannot separate it from scale. The non-monotone degree–propagation
correlation (0.66/0.80/0.38, peaking at 30-40) remains a **real but UNEXPLAINED observation** — it should be
reported as such (an observation, not a mechanism), and N6's coverage-based account of it withdrawn. Keeping
RQ3(c) because it is already drafted would be worse than never adding it.

## GATE — reported, stop

STEP 0–3 reported. Verdict **(d)** (with (c) corroborating): **drop RQ3(c)**; report the non-monotone
correlation as an unexplained observation; **gate N6's coverage figures are SUPERSEDED** by the correct-graph
values (mean 0.41/0.20/0.07 monotone, variance 0.0078/0.0016/0.0001 monotone). Nothing trained, evaluated,
or modified except this card; Task Z STEP 2 eval untouched and unaffected.
