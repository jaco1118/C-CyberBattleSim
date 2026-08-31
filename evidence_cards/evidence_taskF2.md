# Task F2 — band 80-100 (RQ1 classification, RQ2/RQ3 second point)

Numbers and provenance only. No thesis wording. Same design as F1-R, one band larger, STATIC
agent only (the ADAPTED arm was not trained — at 30-40 its difference-in-differences was not
established, so training on the change did not measurably reduce its cost).

> **PROVISIONAL (donor-pool confound, Task G pending):** `membership_join` draws from a shared
> donor pool ~2.2x weaker at the large band; every join-related number inherits this caveat.

## Provenance / setup

- 5 STATIC-only TRPO agents, 250k steps, tuned config, one topology per replicate (0.3 pairing,
  all from the band's existing 8): seed42→topo5(85 nodes), 100→topo100(88), 123→topo18(95),
  200→topo2(84), 300→topo67(80). Named `trpo_250k_F2_static_band80-100_seed<SEED>`. All 5 trained
  exit 0. **No repo code changes** (HEAD `e714f53`); scripts in job scratch.
- Evaluation STOCHASTIC, F1-validated terminal capture (`switch_env.root_owned_nodes` /
  `.reachable_count` / `current_env._episode_count` at the terminal step before reset; never
  `get_statistics()` after done; never join on the post-reset counter).
- **Episode budget: 200 episodes per cell**, 4 conditions × 5 replicates = 20 cells, all exit 0,
  200/200 each. **Zero dropped, zero join failures** (1000/1000 joined per membership/property).
- Conditions (never two change types together): static; membership (natural rate); membership_matched
  (`change_interval=33`, calibrated); property.

## STEP 2.2 ceiling gate — PASSED [ARTIFACT]

Static-eval score per replicate (200 ep each): seed42/topo5 mean=0.265; seed100/topo100 0.426;
seed123/topo18 0.311; seed200/topo2 0.395; seed300/topo67 0.461. **Pooled mean 0.372**, median
0.374, zero-score fraction 0.006 (0–1.5% per replicate), `all_exactly_0=False` (not the capture
bug). Ceiling below 30-40's 0.727 (root-control is harder at 80–100 nodes) but far above near-zero
and <½ zeros → **continue** per the gate.

## Matched-rate calibration [ARTIFACT]

`change_interval` governs the leave rate. Calibrated against the trained policy (seed42/topo5):
ci=20→14.87, 25→13.40, 30→12.40, 33→**11.33**, 35→10.20 leave/ep. **Used `change_interval=33`**;
achieved leave/ep across the 5 replicates: 11.23/11.40/11.68/11.93/12.42 (pooled ~11.7), matching
30-40's 11.2. Natural-rate leave/ep at 80-100 was 15.2–16.4 (pooled ~15.7).

## STEP 3.1 within-agent cost = mean(static) − mean(condition) [FINDING]

| condition | cost | CI95 | verdict | per-replicate (sd) |
|---|---|---|---|---|
| membership | **+0.0111** | [+0.0023, +0.0201] | excludes 0 | 0.016/0.010/0.002/0.014/0.014 (0.006) |
| membership_matched | **+0.0132** | [+0.0046, +0.0218] | excludes 0 | 0.012/0.012/0.003/0.009/0.029 (0.010) |
| property | **+0.0161** | [+0.0074, +0.0246] | excludes 0 | 0.003/0.024/0.013/0.016/0.025 (0.009) |

**FINDING: all three change types impose a real but small within-agent cost** (intervals exclude 0).
**FINDING: membership cost is not event-rate-driven** — matched (+0.0132 at 11.7 leave/ep) ≈ natural
(+0.0111 at 15.7 leave/ep); cutting the event rate ~26% did not reduce the cost (if anything a hair
higher, intervals overlap). The per-event confound is settled empirically, not by assumption.

## STEP 3.2 robustness ratio = mean(condition)/mean(static) [FINDING — carries across bands]

| condition | 80-100 ratio | CI95 | 30-40 ratio (F1-R) |
|---|---|---|---|
| membership | **0.9700** | [0.9469, 0.9939] | 0.9156 |
| membership_matched | 0.9644 | [0.9419, 0.9875] | — |
| property | 0.9568 | [0.9340, 0.9797] | 0.9761 |

**FINDING (the RQ3 second point): the membership robustness ratio IMPROVES from 30-40 (0.916) to
80-100 (0.970) — the agent loses proportionally LESS to membership change at the larger band, and
the absolute cost also falls (0.0614 → 0.0111).** Property is roughly flat (0.976 → 0.957). Overall
performance drops with N (ceiling 0.727 → 0.372), but the *incremental* cost of change does not grow
with N on this axis — it shrinks for membership and is flat for property.

### STEP 3.2b robustness on the CHURN-INVARIANT metric — root-owned COUNT (added post-F3) [FINDING]

The ratio metric above (root_owned/reachable) inflates under membership churn because departures
shrink the `reachable` denominator (see F3 appendix). Recomputing robustness on root-owned COUNT
(count(change)/count(static), within band; no denominator), from the same episodes' `root_owned`
column — analysis only, no new runs:

| change type | metric | 30-40 robustness | 80-100 robustness |
|---|---|---|---|
| membership | **root-owned COUNT** | **0.665** [0.644, 0.684] | **0.874** [0.864, 0.883] |
| membership | ratio (for reference) | 0.916 | 0.970 |
| membership_matched (80-100) | root-owned COUNT | — | 0.916 [0.902, 0.933] |
| property | **root-owned COUNT** | 0.976 [0.957, 0.992] | 0.957 [0.948, 0.971] |
| property | ratio (for reference) | 0.976 | 0.957 |

**FINDING: the membership opposite-directions result is STRONGER on the count metric** — robustness
improves 0.665 → 0.874 across bands (a larger gap than the ratio's 0.916 → 0.970). The ratio metric
*understated* the membership cost, especially at 30-40 (ratio 0.916 vs count 0.665), because the
denominator inflation there is larger. **Property is metric-invariant** (0.976 → 0.957 on BOTH
metrics) — a clean check, since property removes vulnerabilities, not nodes, so `reachable` is
unchanged and there is no denominator effect. On the count metric the direction is unchanged (80-100
more robust to membership than 30-40), so the conclusion holds and is reinforced. Metric labelling:
count for the churn/scale comparison; the ratio numbers in this card's other sections stay as-is for
continuity with F1-R and the Condition-B replication. Full four-condition count analysis: F3 card
appendix.

## STEP 3.3 cost per event (ASSUMES linear additivity — labelled) [FINDING]

membership: 0.0111/13.78 = +0.00081/event; membership_matched: 0.0132/9.75 = +0.00136/event;
property: 0.0161/14.74 = +0.00109/event. (Denominator = immediate-phase event count from the drift
log.) 30-40 for comparison: membership +0.00548/event, property +0.00117/event. Per-event membership
cost is ~7× smaller at 80-100 than 30-40.

## STEP 3.4 per-slice response rate at τ=0 (max/min = the attenuation-bearing slices) [ARTIFACT]

| change type | band | full | mean | max (sd) | min (sd) |
|---|---|---|---|---|---|
| membership_leave | 80-100 | 1.000 | 1.000 | 0.644 (0.048) | 0.657 (0.042) |
| membership_leave | 30-40 (F1-R) | 1.000 | 1.000 | 0.813 | 0.827 |
| property | 80-100 | 1.000 | 1.000 | 0.419 (0.047) | 0.409 (0.050) |
| property | 30-40 (F1-R) | 1.000 | 1.000 | 0.718 | 0.713 |

**FINDING: attenuation INCREASES with scale** — max/min response rate falls from 30-40 to 80-100
(leave 0.81/0.83 → 0.64/0.66; property 0.72/0.71 → 0.42/0.41). mean/full = 1.000 by construction
(no attenuation info). membership_join fire-time response 0.024 (visible@fire=0.000) — COVERAGE, not
attenuation, reported separately. **Set against 3.2, attenuation and behavioural cost move in
OPPOSITE directions between these two bands: the change signal is attenuated MORE at 80-100, yet the
agent loses LESS to it.** (Two points only — no trend fitted, no extrapolation.) Note: the gate's
80-100 figures (max 43.0 / min 36.1) are for its adapted multi-topology agents; the F2 static agent
discovers a smaller effective graph, so less dilution and a higher max/min here — the clean
like-for-like comparison is F1-R↔F2, both single-topology static agents.

## STEP 3.5 classification [FINDING]

| change type | perceived full/mean/max/min | cost | classification |
|---|---|---|---|
| membership_leave | 1.000/1.000/0.644/0.657 | +0.0111 [+0.0023,+0.0201] | perceived + real cost → HANDLED-candidate |
| property | 1.000/1.000/0.419/0.409 | +0.0161 [+0.0074,+0.0246] | perceived + real cost → HANDLED-candidate |

**FINDING: still 0 BLIND, 0 ABSORBED even at 80-100.** Despite stronger attenuation, the max/min
slices still respond on 41–66% of events, so no change type is "not perceived"; and every cost
interval excludes 0. **HANDLED is NOT MEASURABLE** (score captured once/episode; within-episode
dip-and-recovery needs per-step ownership logging) — reported as such, not approximated. RQ1's
classification did not yield a BLIND case at this scale: the fraction of events perceived on the
max/min slices declined (0.81→0.64 leave; 0.72→0.42 property) but did not reach zero.

## STEP 3.6 membership cost — mechanical vs behavioural [FINDING]

Per membership_leave event, whether the departing node was already OWNED at fire time (captured via
a pre-step ownership snapshot; no drift-schema change):

| condition | leave events | was_owned_frac | owned-departures/ep | un-owned-departures/ep |
|---|---|---|---|---|
| membership | 15714 | 0.233 | 3.66 | 12.06 |
| membership_matched | 11733 | 0.199 | 2.33 | 9.40 |

**FINDING:** ~20–23% of departing nodes were already owned (~2.3–3.7/episode); ~77–80% were un-owned.
A precise counterfactual cost split is not identifiable from observational data (no per-event
counterfactual score), so this is reported as the departure composition, not a partitioned cost.
Read together with (a) the small total cost (+0.011), (b) no BLIND events (everything perceived),
and (c) the cost's independence from event rate (matched ≈ natural), the membership loss at this band
is consistent with being substantially structural/mechanical (owned nodes vanishing, denominator
churn) rather than a perception failure — echoing the zero difference-in-differences at 30-40. Stated
as consistency, not proof.

## STEP 3.7 side-by-side with 30-40 (F1-R) — two points, NO trend, NO extrapolation

| quantity | 30-40 (F1-R) | 80-100 (F2) |
|---|---|---|
| static-eval ceiling (mean) | 0.727 | 0.372 |
| cost(membership) | +0.0614 [+0.051,+0.072] | +0.0111 [+0.002,+0.020] |
| robustness(membership) | 0.9156 | 0.9700 |
| cost(property) | +0.0174 [+0.008,+0.027] | +0.0161 [+0.007,+0.025] |
| robustness(property) | 0.9761 | 0.9568 |
| leave max/min response τ=0 | 0.813 / 0.827 | 0.644 / 0.657 |
| property max/min response τ=0 | 0.718 / 0.713 | 0.419 / 0.409 |
| BLIND / ABSORBED count | 0 / 0 | 0 / 0 |

Gate attenuation reference at 80-100 (adapted multi-topology agents): max 43.0 / min 36.1 per cent —
not directly comparable to the F2 static figures above (different agent/discovery), included per the
task's request. **No trend is fitted through these two points and no band beyond 80-100 is inferred.**

## Filter stages / counts

20 cells in, 20 out, 200 episodes each, 0 dropped, 0 non-zero exits, 0 join failures. Zero-score
fraction per condition: static 0.006, membership 0.020, membership_matched 0.014, property 0.024 —
all far below any dynamic-range concern.

> **PROVISIONAL (donor-pool confound, Task G pending):** `membership_join` draws from a shared donor
> pool ~2.2x weaker at the large band; every join-related number inherits this caveat.
