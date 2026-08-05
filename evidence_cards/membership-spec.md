# Membership-change (leave/join) mechanism spec — verified from source

**Status: fresh, source-verified 2026-08-05 (Task MC-VERIFY).** No `membership-spec.md` existed
before this file — the task that produced this checked the whole repo (all branches, full git
history, job scratch) and found no prior version, contrary to the note this task's own brief
described as already on record from "30 July." Written fresh from source; nothing here is carried
over from an unverified prior draft.

All line numbers refer to `cyberbattle/_env/cyberbattle_env.py` on branch `taskY2-pilot-n50`
(identical content to `attenuation-pooling-scale`/`rq2b-10-15`/`taskY-probe-n90`; `main` differs —
it is the pre-CX baseline and does not have the Task CX relaxation flags, but the membership-change
mechanism described here predates and is unaffected by those flags when they are off, which is the
default and what every reported run used).

## Call site [ARTIFACT]

`maybe_apply_dynamic_step()` (`:443-473`), called every step from `CyberBattleCompressedEnv.step()`
(`cyberbattle_env_compressed.py:701`, inside `def step(self, action_vector):` opening at `:558`) —
part of the normal training/eval rollout path, not a diagnostic-only or opt-in code path.

Property change fires on a fixed schedule: `:459` — `if self.patch_service_dynamic_enabled and
self.num_iterations % self.change_interval == 0:`. Membership leave/join do **not** share this
mechanism — dispatch is unconditional per qualifying step (`:465-472`); the actual rate logic is
inside `_apply_dynamic_leave()`/`_apply_dynamic_join()` themselves.

## Leave: mechanism [ARTIFACT/FINDING]

`_apply_dynamic_leave()`, `:575-649`. Own header comment (`:570-574`):

> "Probabilistic node-leave: per-node Bernoulli draw each step (probability weighted down for
> high-degree/critical nodes), plus a low-rate Poisson-triggered batch removal, calibrated so the
> expected removal rate matches change_interval as a sanity anchor, and ramped down as the eligible
> pool approaches the safety floor. **Replaces the old deterministic "exactly one node every
> change_interval steps, forever, uncapped" trigger.**"

**Eligibility** (`_get_removal_eligible_nodes`, `:520-532`): discovered (or, under Task CX's
`allow_undiscovered_removal` flag — off by default, not used by any reported run — the whole
topology) ∩ status `Running` ∩ not starter/source/target/interest node.

**Selection weighting** (`:601-613`): degree-based only.
```python
degrees = {n: (self.access_graph.degree(n) if n in self.access_graph else 0) for n in eligible}
weights = {n: 1.0 / (1.0 + degrees[n]) for n in eligible}
```
No value or ownership term anywhere in this function.

**Rate** (`:594-623`):
```python
ramp = min(1.0, room / max(1, self.num_nodes - floor))          # :594
target_rate = ramp * (1.0 / self.change_interval)               # :599
probabilities = {n: min(P_MAX, target_rate * weights[n] / weight_sum) for n in eligible}  # :619-622
hits = [n for n in eligible if random.random() < probabilities[n]]                        # :623
```
where `floor = max(dynamic_min_alive_nodes, ceil(dynamic_min_alive_fraction * self.num_nodes))`
(`:539`) and `room = alive - floor` (`:589`). Comment at `:616-618`: *"sum(p) == target_rate by
construction... degree weighting only reallocates WHICH node is likely removed, not the expected
COUNT removed per step."* Plus an independent low-rate Poisson batch-removal trigger (`:625-635`,
`p_batch_trigger = ramp / self.dynamic_batch_interval`).

**Config values behind the reported runs** (`cyberbattle/agents/config/train_config.yaml`):
`change_interval: 20` (`:19`), `dynamic_min_alive_nodes: 5`, `dynamic_min_alive_fraction: 0.5`
(constructor defaults `cyberbattle_env.py:66-67`, matching the config file).

## Join: mechanism [ARTIFACT]

`_apply_dynamic_join()`, `:658-723` — structural mirror of leave. Budget cap (`:670-672`):
```python
remaining_budget = self.dynamic_max_joins_per_episode - len(self._dynamic_joined_this_episode)
if remaining_budget <= 0 or not self.dynamic_join_donor_pool:
    return []
```
`dynamic_max_joins_per_episode = 3` (constructor default `:74`, matching
`cyberbattle/agents/config/train_config.yaml:48` and `train_config_static.yaml:48` verbatim). This
is the `uncapped_join=False` branch (default, `:84`) — see the CX-flag note below for why the
reported runs could not have used the alternative.

Rate/selection otherwise mirrors leave: `ramp`/`target_rate` structurally identical
(`:685-686`, using `dynamic_join_rate_interval` in place of `change_interval`), but weighting is by
**ownership + node value**, not degree (`:688-694`):
```python
weights = {n: (2.0 if n in self.owned_nodes else 1.0) * (1.0 + self.get_node(n).value / 100.0)
           for n in eligible_parents}
```
(self.environment carries no edges for join-eligible "parent" nodes, so degree is not a meaningful
signal there — this is why join and leave use different weighting axes.)

## N-dependence [FINDING — neither of the two intuitive explanations alone]

The per-step rate formula (`target_rate = ramp / change_interval`) contains **no direct N-term** —
`change_interval` is a fixed config constant (20), identical across bands. And per the `:616-618`
comment, the expected per-step leave *count* is pool-size-invariant by construction (degree
weighting redistributes *which* node is hit, not how many). So a naive "more eligible nodes -> more
leaves per step" explanation is not what the code does.

The actual driver is `ramp`'s dependence on **absolute room before the proportional floor**:
`floor ≈ 0.5 * N` for N well above the absolute floor of 5, so `room = N - floor ≈ 0.5 * N` at
episode start. Since the base per-step rate is N-independent, it takes proportionally more
cumulative leave-events -- hence more of a *fixed*-length 300-step episode -- for a larger network
to traverse its larger absolute room before `ramp` meaningfully decays toward 0. Worked example
using this project's own established per-band mean N (12.5 / 34.7 / 89.3):

| band | N | floor = max(5, ceil(0.5N)) | absolute room (N - floor) |
|---|---|---|---|
| 10-15 | 12.5 | 7 | 5.5 |
| 30-40 | 34.7 | 18 | 16.7 |
| 80-100 | 89.3 | 45 | 44.3 |

`ramp` starts at 1.0 identically for every band (by construction, `room == N-floor` at episode
start regardless of N) -- the difference is entirely in how many cumulative leaves it takes to
decay ramp toward 0, and larger bands can sustain more leaves (and hence more of the fixed episode
length at close-to-full rate) before that tapering meaningfully suppresses further leaves. **This
is a third, emergent mechanism -- not a direct rate-formula N-dependence, and not simply
eligible-pool-size scaling the trigger probability. Both of the task's proposed single-mechanism
explanations are incomplete on their own.**

## Join cap status for the reported (F-series) runs [FINDING]

`uncapped_join` (the flag that would remove this cap) was introduced at commit `1b42a2c`
(`git log --all -S "uncapped_join"`), tagged `env-baseline-2026-08-01` -- Task CX's own baseline.
The F-series runs behind the 69/72/76% mechanical-share figures ran in late July, before this
commit existed at all -- **the flag was not present in their code**, so those runs could not have
had it enabled even in principle. The cap of 3 was unconditionally active for every reported run.
No contradiction with "joins are capped per episode."

## Measured rates on record [ARTIFACT, from this task's own brief -- not independently re-measured here]

Leave events/episode ≈ 7.09 / 11.23 / 14.04 (10-15 / 30-40 / 80-100); join capped ≈ 2.8/episode.
This task (MC-VERIFY) verified the *mechanism* producing these numbers, not the numbers themselves
-- re-measuring them was out of scope (no new data-collection job was run, per the task's own
restriction).
