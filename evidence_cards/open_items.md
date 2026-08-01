# Open items register

Cross-cutting items that block or qualify a claim and are not owned by a single task card.

## OI-1 — RQ3 has no measured predictor→propagation link on the real graph  [OPEN; blocks RQ3 cross-band structural claims]

**What:** the only measured propagation–degree correlations (+0.66/+0.80/+0.38, `evidence_taskP.md` STEP 3.2)
were computed against the **DFS-proxy undirected degree (mean ≈ 2)**, NOT the scenario knows out-degree
(7.9/22.3/54.9) that RQ3 names as its structural predictor. See the degree reconciliation table
(`evidence_taskY.md` 0.1): four distinct "degree" quantities, only #1 (scenario knows out-degree) is the RQ3
predictor, and the correlations use #4 (proxy).
**Consequence:** **no cross-band structural claim in RQ3 has measured support at present.**
**Resolution:** re-run the encoder probe against **knows out-degree on the REAL `evolving_visible_graph`**
once **Task L STEP 3** logs its per-event structure (edges + node identities). Until then, RQ3 cross-band
structural claims must be marked unsupported. Depends on: Task L STEP 3 (which waits on the 750k run).

## OI-2 — all study scenarios are survivorship-filtered (external-validity disclosure)  [OPEN; affects every result]

**What:** the scenario generator rejects any graph whose access/knows/dos connectivity < 0.5
(`generation_utils.py:43-45`) and resamples. Measured rejection at the existing bands' own `[0.2,0.8]`
probabilities: **10-15 = 67%, 30-40 = 44%, 80-100 = 38%**, and **accepted graphs are systematically
vulnerability-richer than rejected ones** (10-15: 18.3 vs 10.7 vulns/node; 30-40: 15.3 vs 12.1; 80-100:
13.7 vs 11.9). So **every reported scenario is drawn from a survivorship-filtered population — the networks
are systematically more vuln-rich than the generator's nominal distribution, most strongly at the small
band.** Consequence: a study-wide external-validity caveat; the small-band vuln-richness is a candidate
contributor to cross-band differences, unseparated. Discovered in Task Y STEP 0 verification
(`evidence_taskY.md`). No remedy chosen (design decision). Threshold NOT altered.
