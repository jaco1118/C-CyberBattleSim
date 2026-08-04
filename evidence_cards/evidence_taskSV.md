# Task SV — what the scenario generator can and cannot produce

Read-only audit (STEP 0), then a rejection-rate measurement at DEFAULT parameters (STEP 1, waits for CX),
then a methodology declaration (STEP 3). Question: does the generation-time connectivity rejection filter
(Task Y measured survivorship at 55% rejection / 20.4-vs-16.0 vulns/node) touch the REPORTED corpus, which
is generated at the DEFAULT probability (degree ~54.9 at 80-100) rather than Task Y's lowered-probability
regime? Both answers publishable; only one true. All file:line refs to branch `attenuation-pooling-scale`.

## STEP 0 — the full generation audit (READ-ONLY) [ARTIFACT], 2026-08-02

### 0.1 The rejection criterion — `generation_utils.py:39-51`, quoted
```
def generate_valid_probabilities(G, config, max_attempts = 100):
    while True:
        model = wrap_networkx_to_model(G.copy(), **config)
        max_attempts -= 1
        if not (model.access_connectivity < config['minimum_access_connectivity_threshold'] or
                model.knows_connectivity < config['minimum_knows_connectivity_threshold'] or
                model.dos_connectivity < config['minimum_dos_connectivity_threshold']):
            break
        if max_attempts == 0:
            break
    if max_attempts == 0:
        return None
    return model
```
Three connectivity measures tested, each against **0.5** (`generation_config_m.yaml:20-22`,
`minimum_{access,knows,dos}_connectivity_threshold`): **access** (over the access_graph — who can access
whom), **knows** (knows_graph — who knows whom), **dos** (dos_graph). A candidate graph G is REJECTED only
if, after **100 re-rolls of the edge probabilities on the fixed G** (`wrap_networkx_to_model` each loop),
*no* roll clears all three thresholds → returns `None`. Rejection is at the graph-G level, not per-roll.

### 0.2 What "connectivity" means — `networkx_utils.py:19-55`, confirmed PATH-LENGTH not density
`connectivity_metric = 1 - (average_shortest_path_length / (2*max_possible_value))` (`:44`), where
`max_possible_value = len(nodes) - 1` (`:22`) and **unreachable pairs contribute `max_possible_value*2`
= 2(N−1)** (`:36`, "use double the maximum possible value"). So connectivity = **1 − avg_shortest_path /
(2(N−1))**, matching Task Y (evidence_taskY.md). This is an **average-shortest-path-length** measure — a
path-length quantity, NOT edge density. That is exactly why **sparse** graphs (long paths → low
connectivity) are culled and **dense** graphs (short paths → connectivity → 1) clear it easily; it is the
mechanism behind the Task-Y-vs-reported-corpus distinction this task exists to settle.

### 0.3 Every other generation constraint — `generation_config_m.yaml` (30-40 band) + `generate_network.py`
All at RELEASED DEFAULT except `num_nodes_range` (band-specific; the file header says "Copy of
generation_config.yaml with num_nodes_range widened"):

| parameter | value | default? | shapes |
|---|---|---|---|
| num_nodes_range | [30,40] | band-specific | node count |
| num_services_range | [1,2] | default | services/node → vuln count |
| homogeneity_range | [0.1,0.1] | default | service-combination diversity |
| knows_neighbor_probability_range | [0.2,0.8] | default | knows graph density (the degree knob) |
| data_presence / partial_visibility / need_to_escalate | [0.2,0.8] each | default | node data/visibility/escalation |
| firewall in/out, service_shutdown, probing/exploit detection | [0,0] | default | off |
| success_rate_probability_range | [0.6,1] | default | exploit success |
| value_range | [1,100] | default | node value |
| min_presence_each_category | 0.05 | default | category mix floor |
| connectivity thresholds | 0.5 ×3 | default | the rejection filter (0.1) |

**Vulnerabilities per node are NOT set directly** — they DERIVE from the sampled services (`num_services_range`
[1,2] services/node) and real CVE data: `sample_services`→`clean_sampled_services`→`clean_vulnerabilities`
(`generate_graphs.py`, `generation_utils.py`) attach each service's CVE-derived vulns; knows/access/dos edges
then arise from those vulns' Reconnaissance/exploit outcomes (`generate_network.py:189-267`). **No hard cap**
on vulnerability or credential count (bounded only by services × CVE availability). The only retry/resample
loops: (a) the 100-attempt edge-probability re-roll inside `generate_valid_probabilities` (0.1), and (b) the
whole-graph regenerate on rejection (0.4).

### 0.4 Is the rejected set recoverable for the existing scenarios? — NO, discarded in-process
`generate_graphs.py:213-217`: `model = generate_valid_probabilities(G, config); if not model: continue` —
a rejected candidate returns `None` and the loop **`continue`s to a fresh graph**; the rejected G is never
written to disk. So rejected candidates for the reported scenarios are **gone**. STEP 1 must **regenerate**
at the default settings and measure the rejection rate on fresh draws — it cannot re-read the specific
rejected companions of the scenarios in use. (Consequence for claims: STEP 1 estimates the filter's rate at
the reported settings, not the exact filtered-out siblings of each shipped scenario.)

### 0.5 Cost estimate for STEP 1
Task Y recorded ~40 s – 1.5 min per graph. To estimate a rejection rate per band with a usable interval,
~50–100 candidates/band × 3 bands ≈ 150–300 graphs → **~2.5–5 h** at ~1 min/graph. STEP 1 also builds full
service/vuln candidates (heavier at 80-100). It runs **only after CX finishes** (rule 5), single-process,
CPU-light — no contention with the eval arms.

**GATE: STEP 0 reported. STOPPING.** STEP 1 (measure rejection rate at DEFAULT params, with a
zero-randomness log line verified byte-identical against a known scenario, then reverted) waits for CX to
finish. STEP 2/3 per the decision rules in the handoff.

## STEP 1 — rejection rate at DEFAULT params (2026-08-02, autonomous) [FINDING — contradicts the hypothesis]
Standalone harness (`<jobdir>/sv_step1.py`; imports the reported generation path read-only, modifies NOTHING —
generator RNG and reported scenarios untouched by construction, so the in-generator-log-line byte-identical
check is N/A). K=100 candidates/band at default `knows_neighbor_probability_range [0.2,0.8]`; accept/reject via
the reported 100-edge-reroll `generate_valid_probabilities`; connectivity distribution from a single roll/candidate.

| band | mean nodes | rejection rate | Wilson 95% | n_rej/K |
|---|---|---|---|---|
| 10-15 | 12.6 | **0.52** | [0.42, 0.62] | 52/100 |
| 30-40 | 34.7 | **0.45** | [0.36, 0.55] | 45/100 |
| 80-100 | 89.7 | **0.60** | [0.50, 0.69] | 60/100 |

**1.4 trend: NON-MONOTONE / roughly FLAT** (0.52 / 0.45 / 0.60; 30-40 lowest, 80-100 highest, intervals
overlap). Not a clean rise or fall with band size.

**THE HYPOTHESIS IS WRONG — rejection is MATERIAL at the default params, not near-zero.** The handoff reasoned
that default-probability graphs (degree ~54.9 at 80-100) are dense → short paths → clear the 0.5 threshold
easily → near-zero rejection. **Measured: ~45–60% rejection at every band.** So the survivorship the task hoped
to rule out for the reported corpus is **present and substantial**. This CONFIRMS Task Y's OI-2
(evidence_taskY.md:165, "rejection 38–67% at the existing bands' own [0.2,0.8] probabilities") rather than
distinguishing the corpus from it — the reported corpus **is** drawn from a survivorship-filtered population.

**Connectivity distribution — which measure drives rejection:** the binding constraint is **dos and access**
connectivity, not knows. Fraction of candidates with connectivity < 0.5: at 80-100 **dos 0.57, access 0.42,
knows 0.17**; at 30-40 access 0.43, dos 0.25, knows 0.09; at 10-15 access 0.43, dos 0.41, knows 0.25. knows
connectivity clusters well above 0.5 (median 0.61–0.67), so the "dense knows graph clears it" intuition holds
for knows — but dos/access connectivity (also path-length over their own graphs) frequently fall below 0.5 and
drive the rejections. Distributions in `<jobdir>/sv_step1_result.json`.

**Decision-rule outcome = "MATERIAL BUT FLAT ACROSS BANDS":** the accepted population differs from the
generated one, but every band is filtered materially and roughly equally (no rising gradient), so it does not
by itself distort the cross-band comparison as a rival explanation — it is a **study-wide external-validity
scope note** (every scenario is from the accepted, vuln-/connectivity-richer population). NOT the "rejection
rises with band size" serious case.

**STOPPED after STEP 1** (rule 6: contradicts the pre-registered near-zero hypothesis; SV STEP 2/3 not covered
by the overnight authorization). STEP 2 (accepted-vs-rejected property comparison) and STEP 3 (the methodology
declaration) await the user. Tree clean — the standalone harness modified no repo file (nothing to revert).
