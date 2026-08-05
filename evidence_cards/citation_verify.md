# Citation verification — thesis claims checked against source

Same pattern as `claims_audit.md`: the claim as currently written, what the code actually does,
and the disposition. A CLOSED item means the check is finished; residual action (if any) is a
writing/relabel task, not a recompute.

**Status: fresh, 2026-08-05.** No `citation_verify.md` existed before this file — checked the
whole repo (all branches, full git history, job scratch) before creating it. Nothing here is
carried over from a prior version.

---

## CV-1 — "leave events become more frequent as scenarios grow, and joins are capped per episode" [CLOSED — qualitative claim confirmed, but incomplete]

**As currently written (Methodology chapter, per this task's brief):** node-membership change is
described only qualitatively — no formula, no mechanism, no config values.

**What the code actually does** (verified `cyberbattle/_env/cyberbattle_env.py`, Task MC-VERIFY;
full detail in `membership-spec.md`):

1. **Leave/join do not fire on a fixed schedule** the way property change does (property:
   `:459`, `num_iterations % change_interval == 0`). They fire via a **per-step Bernoulli
   probability draw per eligible node**, calibrated so the *expected* per-step count matches
   `1/change_interval` at full ramp, tapering toward 0 as the network approaches a size-proportional
   floor (`:570-649` leave, `:658-723` join). This replaced an earlier deterministic "one node
   every change_interval steps" scheme (per the code's own comment, `:573-574`) — worth knowing
   since it means the CURRENT mechanism is not what an intuitive reading of "capped/scheduled"
   language would suggest.
2. **"More frequent as scenarios grow" is confirmed directionally correct, but not for the reason
   a reader would likely assume.** The per-step trigger rate contains no direct N-term and the
   expected per-step leave *count* is pool-size-invariant by construction (`:616-618` comment).
   The rise with N is a THIRD, emergent mechanism: floor scales with N (`≈0.5×N`), so larger
   networks have proportionally more absolute "room" to sustain a higher ramp for more of a
   *fixed*-length episode before tapering suppresses further leaves. See `membership-spec.md`'s
   N-dependence section for the worked numeric example (5.5 / 16.7 / 44.3 nodes of room across the
   three bands).
3. **"Joins are capped per episode" is confirmed exactly as written** — `dynamic_max_joins_per_episode
   = 3` (`:74`, matching `train_config.yaml:48`), unconditionally active for every reported run (the
   flag that would remove it, `uncapped_join`, postdates the F-series runs entirely — introduced at
   commit `1b42a2c`, tagged `env-baseline-2026-08-01`, after those runs already happened).

**Disposition:** the qualitative claim in the thesis is **not wrong**, but it describes an outcome
without a mechanism, and a reader could reasonably infer a much simpler mechanism (e.g. a fixed
per-N schedule, matching property change's own description) than what's actually implemented. **Do
not change any figure** — this is a completeness/precision gap, not an error. **Residual action
(writing only, not this task's job, per its own explicit restriction):** if the Methodology chapter
is expanded to give leave/join the same level of mechanistic detail as property change already has,
it should describe the Bernoulli-per-node-plus-ramp-plus-floor mechanism (not a fixed interval),
name the two axes of selection weighting (degree for leave, ownership+value for join), and state
the join cap value (3) explicitly with its config citation.

## CV-2 — node-departure selection rule ("eligible if discovered, running and unprotected, weighted by 1/(1+degree), no dependence on node value or ownership") [CLOSED — confirmed exactly, re-verified fresh]

Re-checked against current source rather than trusting the prior informal note (which, per this
task's own STEP 0, could not be located anywhere to check against in the first place). Matches
exactly: `_get_removal_eligible_nodes` (`:520-532`) for eligibility, `_apply_dynamic_leave`
(`:601-613`) for the 1/(1+degree) weighting, no value/ownership term present anywhere in either
function. Full quotes in `membership-spec.md`.

## CV-3 — join cap removal ("PROPOSED... to remove that cap so leave and join would share the same schedule") [CLOSED — proposal confirmed never applied to any reported run]

The removal mechanism (`uncapped_join` flag) exists in source but is **off by default**
(`cyberbattle_env.py:84`) and was introduced (commit `1b42a2c`, `env-baseline-2026-08-01`) after
every reported F-series run had already completed — those runs' code did not contain this flag at
all. Confirmed: the cap was never removed for any run whose numbers are currently reported. If this
proposal is acted on for a *future* run, that run's figures would not be directly comparable to the
existing reported ones without disclosure (same-shape caveat as this project's other
config-changed-mid-study disclosures, e.g. the donor-pool confound note).
