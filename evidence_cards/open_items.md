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

## OI-3 — no convergence check on record for the checkpoints that actually back the reported results, and the one borrowed "80-100 never converged" claim understates the problem  [OPEN; qualifies attenuation/pooling/SNR results at all three bands]

**What:** the 15 manifest checkpoints of 2026-07-26
(`trpo_250k_tuned_compressed_band{10-15,30-40,80-100}_seed{42,100,123,200,300}`) back essentially every
reported attenuation, pooling and SNR figure in this project, and had never had a convergence check run
against them (confirmed: `evidence_taskW.md:65-75`, "**NOT** the F1/F2 single-topology agents that Task F4
retrained... unaffected by F4"). The thesis's only convergence evidence — "the largest band did not
converge," used to qualify four separate results — comes from a **different** checkpoint population
(F1_static/F2_static, Task F4), and Table IV.2 reports a **third**, unrelated population (Task Y's
N=30/N=60/N=90 degree-controlled grid, confirmed exactly, no hedge — `evidence_taskY.md` STEP 1+2).

Applying Table IV.2's own exact method (`compute_convergence_check.py`, unmodified, already committed —
`train/Root owned nodes`, 50k windows, mean\|Δ%\|<5% AND ≥4/5 seeds) directly to the 15 manifest
checkpoints' own tfevents logs (pure read, no training/resumption/new evaluation — Task
CONVERGENCE-PROVENANCE) gives: **10-15 mean\|Δ%\|=14.55% (2/5 seeds within tolerance), 30-40 mean\|Δ%\|=5.29%
(4/5), 80-100 mean\|Δ%\|=19.91% (1/5) — NOT CONVERGED, all three bands**, not only the largest one.

**Consequence:** the "largest band did not converge" qualifier, as currently used in five places, both
(a) imports a measurement made on a different checkpoint population than the one it qualifies, and (b)
understates the problem on the population it actually needs to describe — 10-15 and 80-100 are comparably
far from the criterion, and 30-40 misses on the mean threshold alone. A candidate (not established) partial
explanation: the 15-manifest checkpoints trained under 8-topology switching (`RandomSwitchEnv`), unlike
F1_static/F2_static's single/few-topology setup — episode-to-episode topology switching is a real, distinct
source of training-curve variance this metric cannot separate from genuine policy instability.
**Resolution:** none chosen yet — the qualifier needs to either name the correct population explicitly
(F1_static/F2_static, not the 15-manifest) everywhere it is used, or the 15-manifest checkpoints' own
result above needs to be reported and the five downstream qualifications revised to match it. Discovered
in Task THREE-LOOKUPS, resolved to source in Task CONVERGENCE-PROVENANCE
(`analysis/convergence_provenance_2026-08-15/RUN_RECORD.md`). No checkpoint retrained; nothing computed
requires re-running anything further to confirm.
