"""Task RECOMPUTE CONVERGENCE ON EPISODE REWARD, STEP 1 (full) + Additions 1 and 2.

For every (population, cell, seed) row already established, computes delta_pct for BOTH metrics
("train/Root owned nodes" and "rollout/ep_rew_mean") under BOTH --stop conventions (250000
explicit, and default = each file's own final logged step), reusing compute_convergence_check.py's
own series/delta_pct functions unmodified. Reports:
  - the full per-row table for population (a) under both conventions (Addition 1), plus band-level
    mean|delta%| and seeds-within-tolerance under each convention
  - the same convention-difference column for the reward metric, every population (Addition 2)
  - the final STEP 1 deliverable: per-seed table (both metrics) + summary table (population-level
    verdict comparison), all under the default convention, which this task designates the
    project's standard from here on

No experiment re-run. Pure re-read of already-logged tfevents scalars.
"""
import glob
import math
import os
import statistics
import sys

REPO = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim"
AGENTS = os.path.join(REPO, "cyberbattle/agents")
sys.path.insert(0, AGENTS)

import compute_convergence_check as ccc  # noqa: E402

WINDOW = 50000
NODE_METRIC = "train/Root owned nodes"
REW_METRIC = "rollout/ep_rew_mean"
THRESHOLD = 5.0

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


def get(metric, run_dir, stop):
    return ccc.delta_pct(run_dir, metric, WINDOW, stop)["delta_pct"]


def main():
    rows = []
    for pop, cell, seed, run_dir in RUNS:
        node_250k = get(NODE_METRIC, run_dir, 250000)
        node_default = get(NODE_METRIC, run_dir, None)
        rew_250k = get(REW_METRIC, run_dir, 250000)
        rew_default = get(REW_METRIC, run_dir, None)
        rows.append(dict(pop=pop, cell=cell, seed=seed,
                          node_250k=node_250k, node_default=node_default,
                          node_diff=node_default - node_250k,
                          rew_250k=rew_250k, rew_default=rew_default,
                          rew_diff=rew_default - rew_250k))

    print("=== ADDITION 1: population (a), root-owned-count, both conventions ===\n")
    a_rows = [r for r in rows if r["pop"] == "a-manifest"]
    max_abs_diff_row = max(a_rows, key=lambda r: abs(r["node_diff"]))
    print(f"Largest convention-induced difference: band {max_abs_diff_row['cell']} seed "
          f"{max_abs_diff_row['seed']}: stop=250000 -> {max_abs_diff_row['node_250k']:+.2f}%, "
          f"default -> {max_abs_diff_row['node_default']:+.2f}%, diff = "
          f"{max_abs_diff_row['node_diff']:+.2f}pp\n")
    print(f"{'band':>8} {'seed':>5} {'delta%(stop=250k)':>18} {'delta%(default)':>16} {'diff(pp)':>10}")
    for r in a_rows:
        print(f"{r['cell']:>8} {r['seed']:>5} {r['node_250k']:>17.2f}% {r['node_default']:>15.2f}% "
              f"{r['node_diff']:>+9.2f}")

    print(f"\n--- band-level summary, root-owned-count, both conventions ---")
    print(f"{'band':>8} {'mean|d%|(250k)':>16} {'within(250k)':>13} {'mean|d%|(default)':>18} {'within(default)':>16}")
    for band in ["10-15", "30-40", "80-100"]:
        band_rows = [r for r in a_rows if r["cell"] == band]
        mean_250k = statistics.mean(abs(r["node_250k"]) for r in band_rows)
        within_250k = sum(1 for r in band_rows if abs(r["node_250k"]) < THRESHOLD)
        mean_default = statistics.mean(abs(r["node_default"]) for r in band_rows)
        within_default = sum(1 for r in band_rows if abs(r["node_default"]) < THRESHOLD)
        print(f"{band:>8} {mean_250k:>15.2f}% {within_250k:>10}/5   {mean_default:>17.2f}% {within_default:>13}/5")

    print("\n\n=== ADDITION 2: reward-based delta_pct, both conventions, every population ===\n")
    print(f"{'pop':>14} {'cell':>12} {'seed':>5} {'rew d%(250k)':>13} {'rew d%(default)':>16} {'diff(pp)':>10}")
    for r in rows:
        print(f"{r['pop']:>14} {r['cell']:>12} {r['seed']:>5} {r['rew_250k']:>12.2f}% "
              f"{r['rew_default']:>15.2f}% {r['rew_diff']:>+9.2f}")

    node_diffs = [abs(r["node_diff"]) for r in rows]
    rew_diffs = [abs(r["rew_diff"]) for r in rows]
    print(f"\n--- cross-convention swing, both metrics, all 35 rows ---")
    print(f"  root-owned-count |diff|: mean={statistics.mean(node_diffs):.3f}pp  "
          f"median={statistics.median(node_diffs):.3f}pp  max={max(node_diffs):.3f}pp")
    print(f"  reward           |diff|: mean={statistics.mean(rew_diffs):.3f}pp  "
          f"median={statistics.median(rew_diffs):.3f}pp  max={max(rew_diffs):.3f}pp")

    print("\n\n=== STEP 1 (default convention, the project standard from here on) ===\n")
    print(f"{'pop':>14} {'cell':>12} {'seed':>5} {'node d%':>10} {'node within':>12} "
          f"{'rew d%':>10} {'rew within':>11}")
    for r in rows:
        node_within = "YES" if abs(r["node_default"]) < THRESHOLD else "no"
        rew_within = "YES" if abs(r["rew_default"]) < THRESHOLD else "no"
        print(f"{r['pop']:>14} {r['cell']:>12} {r['seed']:>5} {r['node_default']:>+9.2f}% {node_within:>12} "
              f"{r['rew_default']:>+9.2f}% {rew_within:>11}")

    print(f"\n--- population/cell-level summary, default convention ---")
    cells = []
    seen = set()
    for r in rows:
        key = (r["pop"], r["cell"])
        if key not in seen:
            seen.add(key)
            cells.append(key)
    print(f"{'population/cell':>28} {'mean|d%|(node)':>15} {'within(node)':>13} {'verdict(node)':>15} "
          f"{'mean|d%|(rew)':>14} {'within(rew)':>12} {'verdict(rew)':>14} {'AGREE?':>8}")
    for pop, cell in cells:
        cell_rows = [r for r in rows if r["pop"] == pop and r["cell"] == cell]
        n = len(cell_rows)
        need = math.ceil(0.8 * n)
        mean_node = statistics.mean(abs(r["node_default"]) for r in cell_rows)
        within_node = sum(1 for r in cell_rows if abs(r["node_default"]) < THRESHOLD)
        verdict_node = "CONVERGED" if (mean_node < THRESHOLD and within_node >= need) else "NOT CONVERGED"
        mean_rew = statistics.mean(abs(r["rew_default"]) for r in cell_rows)
        within_rew = sum(1 for r in cell_rows if abs(r["rew_default"]) < THRESHOLD)
        verdict_rew = "CONVERGED" if (mean_rew < THRESHOLD and within_rew >= need) else "NOT CONVERGED"
        agree = "AGREE" if verdict_node == verdict_rew else "DISAGREE"
        print(f"{pop + '/' + cell:>28} {mean_node:>14.2f}% {within_node:>10}/{n}   {verdict_node:>15} "
              f"{mean_rew:>13.2f}% {within_rew:>9}/{n}   {verdict_rew:>14} {agree:>8}")


if __name__ == "__main__":
    main()
