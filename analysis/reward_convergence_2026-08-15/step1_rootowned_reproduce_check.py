"""Task RECOMPUTE CONVERGENCE ON EPISODE REWARD, STEP 1 ADDITION: recompute root-owned-count
delta_pct for every population using the SAME convention as this task's reward check (--stop left
at default, resolving to each file's own final logged step), and compare against the value already
on record for that cell.

Reuses the identical run-directory mapping from step0_reward_signcheck.py (same 35 rows, same
individually-traced N=90/lower-neighbour-N=30 disambiguation). Only the metric changes:
"train/Root owned nodes" instead of "rollout/ep_rew_mean". Imports compute_convergence_check.py
directly, unmodified.

RECORDED values, quoted from source, per row:
  (a) 15-manifest: analysis/convergence_provenance_2026-08-15/manifest_convergence_output.log
      (this project's own prior run, committed 2108f6d/780b65a -- that run used --stop 250000
      EXPLICITLY, not the default, so a mismatch here would reflect that known difference, not an
      unknown one).
  (b) N=30: evidence_cards/dissertation_log_v2.md:918-922 (per-seed "250k" column).
  (b) N=60: evidence_cards/evidence_taskY.md:383 (stage 7 final per-seed line).
  (b) N=90: evidence_cards/evidence_taskY.md:399-404 (STEP 2.2 per-seed table).
  (c) lower-neighbour N=30: evidence_cards/evidence_taskY3.md (branch taskY2-pilot-n30, commit
      5193e32, fetched via `git show` -- not on this branch), STEP 1 result table.

No experiment re-run. Pure re-read of already-logged tfevents scalars.
"""
import glob
import os
import sys

REPO = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim"
AGENTS = os.path.join(REPO, "cyberbattle/agents")
sys.path.insert(0, AGENTS)

import compute_convergence_check as ccc  # noqa: E402

METRIC = "train/Root owned nodes"
WINDOW = 50000

RUNS = []
for band in ["10-15", "30-40", "80-100"]:
    for seed in [42, 100, 123, 200, 300]:
        hits = glob.glob(os.path.join(AGENTS, "logs",
                                       f"trpo_250k_tuned_compressed_band{band}_seed{seed}_2026-07-26_*"))
        if hits:
            RUNS.append(("a-manifest", band, seed,
                         os.path.join(sorted(hits)[-1], "TRPO_x_control_SecureBERT", "TRPO_1")))
for seed in [42, 100, 123, 200, 300]:
    hits = glob.glob(os.path.join(AGENTS, "logs", f"yN30_s{seed}_stg1_2026-08-03_*"))
    if hits:
        RUNS.append(("b-N30", "N30", seed, os.path.join(sorted(hits)[-1], "TRPO_x_control_SecureBERT", "TRPO_1")))
for seed in [42, 100, 123, 200, 300]:
    hits = glob.glob(os.path.join(AGENTS, "logs", f"yN60_s{seed}_stg7_2026-08-05_*"))
    if hits:
        RUNS.append(("b-N60", "N60", seed, os.path.join(sorted(hits)[-1], "TRPO_x_control_SecureBERT", "TRPO_1")))
N90_FINAL = {
    42: "yprobe_n90_s42_ext125M_2026-08-03_13-42-07",
    100: "yN90ext_s100_stg1_2026-08-05_22-55-50",
    123: "yprobe_n90_s123_static500k_2026-08-03_08-43-27",
    200: "yprobe_n90_s200_static500k_2026-08-03_08-43-27",
    300: "yN90ext_s300_stg1_2026-08-05_22-55-50",
}
for seed, folder in N90_FINAL.items():
    RUNS.append(("b-N90", "N90", seed,
                 os.path.join(AGENTS, "logs", folder, "TRPO_x_control_SecureBERT", "TRPO_1")))
for seed in [42, 100, 123, 200, 300]:
    hits = glob.glob(os.path.join(AGENTS, "logs", f"yN30_s{seed}_stg1_2026-08-05_*"))
    if hits:
        RUNS.append(("c-lowdeg-N30", "N30(lowdeg)", seed,
                     os.path.join(sorted(hits)[-1], "TRPO_x_control_SecureBERT", "TRPO_1")))

RECORDED = {
    ("a-manifest", "10-15", 42): +7.20, ("a-manifest", "10-15", 100): +21.98,
    ("a-manifest", "10-15", 123): -4.81, ("a-manifest", "10-15", 200): +36.87,
    ("a-manifest", "10-15", 300): -1.92,
    ("a-manifest", "30-40", 42): -0.34, ("a-manifest", "30-40", 100): -18.20,
    ("a-manifest", "30-40", 123): +4.55, ("a-manifest", "30-40", 200): +3.05,
    ("a-manifest", "30-40", 300): -0.33,
    ("a-manifest", "80-100", 42): +23.87, ("a-manifest", "80-100", 100): +1.68,
    ("a-manifest", "80-100", 123): +15.56, ("a-manifest", "80-100", 200): -10.09,
    ("a-manifest", "80-100", 300): +48.37,
    ("b-N30", "N30", 42): +4.78, ("b-N30", "N30", 100): +8.47, ("b-N30", "N30", 123): -1.70,
    ("b-N30", "N30", 200): -1.86, ("b-N30", "N30", 300): +4.08,
    ("b-N60", "N60", 42): +1.83, ("b-N60", "N60", 100): -9.22, ("b-N60", "N60", 123): -2.17,
    ("b-N60", "N60", 200): -14.37, ("b-N60", "N60", 300): -0.92,
    ("b-N90", "N90", 42): +1.09, ("b-N90", "N90", 100): -1.25, ("b-N90", "N90", 123): -4.10,
    ("b-N90", "N90", 200): +1.28, ("b-N90", "N90", 300): -3.15,
    ("c-lowdeg-N30", "N30(lowdeg)", 42): +5.88, ("c-lowdeg-N30", "N30(lowdeg)", 100): +3.14,
    ("c-lowdeg-N30", "N30(lowdeg)", 123): +2.45, ("c-lowdeg-N30", "N30(lowdeg)", 200): +1.44,
    ("c-lowdeg-N30", "N30(lowdeg)", 300): -0.43,
}


def main():
    print(f"=== recompute {METRIC!r}, window={WINDOW}, --stop defaults to each file's own max ===\n")
    n_reproduce = 0
    n_not_reproduce = 0
    mismatches = []

    for pop, cell, seed, run_dir in RUNS:
        r = ccc.delta_pct(run_dir, METRIC, WINDOW, None)
        recomputed = r["delta_pct"]
        key = (pop, cell, seed)
        recorded = RECORDED.get(key)
        if recorded is None:
            print(f"  {pop:14s} {cell:12s} seed={seed:<4} NO RECORDED VALUE TO COMPARE")
            continue
        diff = recomputed - recorded
        reproduces = abs(diff) < 0.01  # bitwise-equal-to-2dp tolerance, not a fudge factor
        if reproduces:
            n_reproduce += 1
        else:
            n_not_reproduce += 1
            mismatches.append((pop, cell, seed, recorded, recomputed, diff, r["stop"]))
        print(f"  {pop:14s} {cell:12s} seed={seed:<4} stop={r['stop']:>8}  "
              f"recorded={recorded:+7.2f}%  recomputed={recomputed:+7.2f}%  "
              f"diff={diff:+.4f}pp  reproduces={'yes' if reproduces else 'NO'}")

    print(f"\n=== summary ===")
    print(f"  rows compared: {n_reproduce + n_not_reproduce}")
    print(f"  reproduces record: {n_reproduce}")
    print(f"  does NOT reproduce: {n_not_reproduce}")
    if mismatches:
        print(f"\n  MISMATCHES (do not draw conclusions from these beyond reporting them):")
        for pop, cell, seed, recorded, recomputed, diff, stop in mismatches:
            print(f"    {pop} {cell} seed={seed}: recorded={recorded:+.2f}% recomputed={recomputed:+.2f}% "
                  f"diff={diff:+.4f}pp (recompute used stop={stop})")


if __name__ == "__main__":
    main()
