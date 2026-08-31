"""Task RECOMPUTE CONVERGENCE ON EPISODE REWARD, STEP 0 Q1/Q2/Q6: read-only inspection of the
"rollout/ep_rew_mean" tensorboard scalar for every population that currently carries a
convergence verdict. Imports compute_convergence_check.py directly (series/delta_pct, unmodified)
so window/mean computation matches the existing tool exactly; adds min/max-across-run and the
sign/zero classification the existing tool's CLI does not report.

No training, no resumption, no new evaluation -- pure re-read of already-logged tfevents scalars.
--stop is left at its default (None -> each file's own max logged step) throughout, matching the
tool's own documented convention for resumed/extended runs (see its docstring, "anchored at the
run's own final logged step") -- NOT hardcoded to 250000, since several of the N=90 folders span a
longer local range (see Q6 in the reply this script supports).
"""
import glob
import os
import sys

REPO = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim"
AGENTS = os.path.join(REPO, "cyberbattle/agents")
sys.path.insert(0, AGENTS)

import compute_convergence_check as ccc  # noqa: E402

METRIC = "rollout/ep_rew_mean"
WINDOW = 50000

# (population_label, cell_label, seed, run_dir) -- run_dir resolved to the exact final-stage
# folder for each cell, per direct verification (see reply for how each was traced).
RUNS = []

# (a) 15 manifest checkpoints, 2026-07-26, single stage, 250k -- Table IV.1's population
for band in ["10-15", "30-40", "80-100"]:
    for seed in [42, 100, 123, 200, 300]:
        hits = glob.glob(os.path.join(AGENTS, "logs",
                                       f"trpo_250k_tuned_compressed_band{band}_seed{seed}_2026-07-26_*"))
        if hits:
            RUNS.append(("a-manifest", band, seed,
                         os.path.join(sorted(hits)[-1], "TRPO_x_control_SecureBERT", "TRPO_1")))

# (b) Task Y N=30 (original, 2026-08-03, converged stage 1)
for seed in [42, 100, 123, 200, 300]:
    hits = glob.glob(os.path.join(AGENTS, "logs", f"yN30_s{seed}_stg1_2026-08-03_*"))
    if hits:
        RUNS.append(("b-N30", "N30", seed, os.path.join(sorted(hits)[-1], "TRPO_x_control_SecureBERT", "TRPO_1")))

# (b) Task Y N=60, final stage 7 (1.75M cap), all 5 seeds group-lockstep
for seed in [42, 100, 123, 200, 300]:
    hits = glob.glob(os.path.join(AGENTS, "logs", f"yN60_s{seed}_stg7_2026-08-05_*"))
    if hits:
        RUNS.append(("b-N60", "N60", seed, os.path.join(sorted(hits)[-1], "TRPO_x_control_SecureBERT", "TRPO_1")))

# (b) Task Y N=90, per-seed final stage (irregular stopping, traced individually)
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

# (c) Task Y2-pilot lower-neighbour N=30 (degree ~12.35), 2026-08-05, single stage
for seed in [42, 100, 123, 200, 300]:
    hits = glob.glob(os.path.join(AGENTS, "logs", f"yN30_s{seed}_stg1_2026-08-05_*"))
    if hits:
        RUNS.append(("c-lowdeg-N30", "N30(lowdeg)", seed,
                     os.path.join(sorted(hits)[-1], "TRPO_x_control_SecureBERT", "TRPO_1")))


def main():
    print(f"=== Q1/Q2/Q6: metric={METRIC!r}, window={WINDOW}, --stop defaults to each file's own max ===\n")
    n_total = 0
    n_readable = 0
    n_unreadable = 0
    n_pre_negative = 0
    n_pre_near_zero = 0  # within 1.0 of zero, excluding exactly negative already counted separately if also <1

    for pop, cell, seed, run_dir in RUNS:
        n_total += 1
        try:
            steps, vals = ccc.series(run_dir, METRIC)
        except (FileNotFoundError, KeyError) as e:
            n_unreadable += 1
            print(f"  {pop:14s} {cell:12s} seed={seed:<4} UNREADABLE: {e}")
            continue

        r = ccc.delta_pct(run_dir, METRIC, WINDOW, None)
        n_readable += 1
        vmin, vmax = float(vals.min()), float(vals.max())
        pre, fin = r["pre"], r["fin"]
        abs_diff = fin - pre
        near_zero = abs(pre) < 1.0
        negative = pre < 0
        if negative:
            n_pre_negative += 1
        elif near_zero:
            n_pre_near_zero += 1

        status = "undefined, counted" if (negative or near_zero) else f"{r['delta_pct']:+.2f}%"
        print(f"  {pop:14s} {cell:12s} seed={seed:<4} stop={r['stop']:>8} "
              f"min={vmin:9.4f} max={vmax:9.4f}  pre={pre:9.4f} fin={fin:9.4f} "
              f"abs_diff={abs_diff:+9.4f}  delta%={status:>20}  "
              f"(npre={r['npre']}, nfin={r['nfin']}, n_total_points={len(steps)})")

    print(f"\n=== summary ===")
    print(f"  total (population,cell,seed) rows attempted: {n_total}")
    print(f"  readable: {n_readable}")
    print(f"  unreadable: {n_unreadable}")
    print(f"  pre negative: {n_pre_negative}")
    print(f"  pre within 1.0 of zero (and not negative): {n_pre_near_zero}")
    print(f"  pre safely away from zero and positive: {n_readable - n_pre_negative - n_pre_near_zero}")


if __name__ == "__main__":
    main()
