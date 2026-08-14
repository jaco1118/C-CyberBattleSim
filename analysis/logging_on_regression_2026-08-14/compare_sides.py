"""Task LOGGING-ON-REGRESSION, STEP 3/4: compare Side A (pre-instrumentation, commit 7cdfb2b,
drift_logging concept absent) against Side B (current HEAD, drift_logging=True), per seed.

"Identical" means bitwise equal (np.array_equal), per this task's explicit numerics convention --
no tolerance is applied anywhere in this script.

Usage: python compare_sides.py <out_dir> <seed1> <seed2> [<seed3> ...]
"""
import pickle
import sys
import os

import numpy as np

ITEMS_EXACT = ["source_node", "target_node", "vulnerability", "outcome",
               "min_distance_action", "reward", "done", "cumulative_reward",
               "n_discovered", "n_root_owned"]


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    out_dir = sys.argv[1]
    seeds = [int(s) for s in sys.argv[2:]]

    report_lines = []

    def emit(s):
        print(s)
        report_lines.append(s)

    emit("=== LOGGING-ON-REGRESSION: STEP 3/4 comparison, Side A (pre-instrumentation, 7cdfb2b) "
         "vs Side B (current HEAD, drift_logging=True) ===\n")

    overall_verdict = None
    for seed in seeds:
        path_a = os.path.join(out_dir, f"side_A_seed{seed}.pkl")
        path_b = os.path.join(out_dir, f"side_B_seed{seed}.pkl")
        rows_a = load(path_a)
        rows_b = load(path_b)

        emit(f"--- seed {seed} ---")
        emit(f"  Side A rows: {len(rows_a)}")
        emit(f"  Side B rows: {len(rows_b)}")

        if len(rows_a) != len(rows_b):
            emit(f"  STEP COUNT MISMATCH: {len(rows_a)} (A) vs {len(rows_b)} (B) -- "
                 f"this IS the answer for this seed; not truncating, not aligning.")
            overall_verdict = "differs"
            continue

        n_steps = len(rows_a)
        n_item_diff = {k: 0 for k in ITEMS_EXACT}
        first_divergence = None

        obs_max_abs_diff = 0.0
        obs_max_abs_diff_step = None
        obs_max_rel_diff = 0.0
        obs_max_rel_diff_step = None
        n_obs_bitwise_equal = 0

        for i in range(n_steps):
            a, b = rows_a[i], rows_b[i]
            step_diverges = False
            for k in ITEMS_EXACT:
                if a[k] != b[k]:
                    n_item_diff[k] += 1
                    step_diverges = True

            obs_a = np.asarray(a["obs_pre_step"], dtype=np.float64)
            obs_b = np.asarray(b["obs_pre_step"], dtype=np.float64)
            bitwise_equal = np.array_equal(obs_a, obs_b)
            if bitwise_equal:
                n_obs_bitwise_equal += 1
            else:
                step_diverges = True
                abs_diff = np.abs(obs_a - obs_b)
                max_abs = float(np.max(abs_diff))
                denom = np.where(np.abs(obs_a) > 0, np.abs(obs_a), 1.0)
                max_rel = float(np.max(abs_diff / denom))
                if max_abs > obs_max_abs_diff:
                    obs_max_abs_diff = max_abs
                    obs_max_abs_diff_step = i
                if max_rel > obs_max_rel_diff:
                    obs_max_rel_diff = max_rel
                    obs_max_rel_diff_step = i

            if step_diverges and first_divergence is None:
                first_divergence = i

        n_diff_any_item = sum(1 for k in ITEMS_EXACT if n_item_diff[k] > 0)
        emit(f"  steps compared: {n_steps}")
        for k in ITEMS_EXACT:
            status = "zero differing" if n_item_diff[k] == 0 else f"{n_item_diff[k]} differing"
            emit(f"    {k}: {status}")
        emit(f"  observation: bitwise-equal steps = {n_obs_bitwise_equal}/{n_steps}")
        if n_obs_bitwise_equal < n_steps:
            emit(f"    max abs diff = {obs_max_abs_diff} at step {obs_max_abs_diff_step}")
            emit(f"    max rel diff = {obs_max_rel_diff} at step {obs_max_rel_diff_step}")

        if first_divergence is None:
            emit(f"  ZERO DIFFERING across all {n_steps} steps, all items, observation bitwise-equal.")
            if overall_verdict is None:
                overall_verdict = "identical"
        else:
            i = first_divergence
            a, b = rows_a[i], rows_b[i]
            emit(f"  FIRST DIVERGENCE at step {i} (episode {a['episode']}, step_in_episode {a['step_in_episode']}):")
            emit(f"    Side A: source={a['source_node']} target={a['target_node']} vuln={a['vulnerability']} "
                 f"outcome={a['outcome']} reward={a['reward']} done={a['done']} "
                 f"n_discovered={a['n_discovered']} n_root_owned={a['n_root_owned']}")
            emit(f"    Side B: source={b['source_node']} target={b['target_node']} vuln={b['vulnerability']} "
                 f"outcome={b['outcome']} reward={b['reward']} done={b['done']} "
                 f"n_discovered={b['n_discovered']} n_root_owned={b['n_root_owned']}")
            obs_a = np.asarray(a["obs_pre_step"], dtype=np.float64)
            obs_b = np.asarray(b["obs_pre_step"], dtype=np.float64)
            if not np.array_equal(obs_a, obs_b):
                diff = np.abs(obs_a - obs_b)
                emit(f"    obs elementwise max abs diff at this step = {float(np.max(diff))}, "
                     f"at index {int(np.argmax(diff))}")
            overall_verdict = "differs"
        emit("")

    emit(f"=== OVERALL VERDICT: {overall_verdict} ===")
    with open(os.path.join(out_dir, "comparison_report.txt"), "w") as f:
        f.write("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
