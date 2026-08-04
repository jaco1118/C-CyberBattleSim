# Standing rules — project-wide methodology, not tied to one task

Rules recorded here apply to every future task in this project, not just the one that prompted
them. Each entry: the rule, why it exists (the concrete incident that motivated it), and how to
apply it.

---

## SR-1 — Commit manifests and configs, not just scripts [adopted 2026-08-04]

**Rule:** any manifest, config dump, or run recipe that determines what a script actually did
(checkpoint paths, seeds, hyperparameters, flags) gets committed in the same commit as the script
or evidence card it supports — never left in a per-job scratch directory. A committed script with
an uncommitted manifest is incomplete, the same way an evidence card with a missing manifest is
incomplete.

**Why:** Task RQ3D needed to renormalize Task CX PART 3's headline figures (0.70 vs 1.23 root
departures). The producing script (`compute_attenuation_analysis.py`) was safely committed, but
the MANIFEST that told it which checkpoints/flags/episode-counts to use for that specific run was
not — it lived only in a job's `.claude/jobs/<id>/tmp/` scratch directory, which is not part of
the git repo and is not guaranteed to persist. By the time RQ3D needed it, it was gone (commit
`1d6aaab`'s own message: "the CX compute pipeline that produced those figures is NOT in the stash
and remains lost (ephemeral scratch)"). This is not a one-off: F1's card records the same failure
("All F1 scripts... live in the job scratch dir, not the repo"), as does F2/F3's cost-eval
harness. Same root cause each time — the script survives, the recipe that drove it doesn't.

**How to apply:** when a manifest-driven script (or any script taking a config/manifest argument)
produces output that gets cited in an evidence card, dissertation log entry, or committed results
file, commit a copy of that exact manifest/config alongside the script or the results — not just
a description of it in prose. `cyberbattle/agents/rq3d_manifest.yaml` is the concrete example this
rule produced: a clean, self-contained copy of the checkpoint/topology recipe RQ3D's rollout used,
committed specifically because half of what it needed (`grid_topology_id_map.json`) was sitting in
the entirely-untracked `attenuation_gate_archive/` directory.

## SR-2 — Default `event_graph_logging=True` (+ `drift_logging=True`) for any real, non-smoke run [adopted 2026-08-04]

**Rule:** for any run whose output might get analyzed or reported on later (i.e., not a quick,
disposable sanity-check run), enable `event_graph_logging=True` and its prerequisite
`drift_logging=True` by default. Only skip it for throwaway smoke tests whose output will not be
kept.

**Why:** confirmed from source (`cyberbattle/_env/cyberbattle_env_compressed.py`, traced for Task
RQ3D's STEP 0.5.4) that `event_graph_logging` is read-only and I/O-only — every code path it gates
only reads already-computed state and appends to its own side files; no RNG draw, no mutation of
reward/observation/done, nothing that could affect the trajectory. Given that, there is no real
cost to leaving it on. The original Task CX PART 3 run did NOT have it on (its `event_episode.jsonl`
— the file that would carry per-episode `root_owned_departures` and `final_root_owned_count` — is
missing from all 3 bands' output directories), which is exactly what forced RQ3D into a fresh
rollout instead of a pure recompute over existing logs. Nobody anticipated needing that signal at
the time CX PART 3 ran; a secondary flag that costs nothing sat off, and the data was gone by the
time a later task needed it.

**How to apply:** when writing or launching any evaluation/analysis rollout (not training — this
is an evaluation-side logging flag) that isn't a disposable smoke test, pass
`event_graph_logging=True, drift_logging=True` by default rather than leaving them at their
`False` defaults. If a specific run genuinely doesn't need this (e.g., truly throwaway), that's
fine — the point is not to reflexively skip it just because it wasn't asked for.
