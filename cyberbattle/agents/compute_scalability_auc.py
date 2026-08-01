import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# Verbatim from Terranova's plot_scalability_spaces.py (extract_auc_from_tensorboard)
def extract_auc_from_tensorboard(tensorboard_path, metric_name, normalizer=None):
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    if metric_name in ea.Tags()['scalars']:
        events = ea.Scalars(metric_name)
        values = [event.value for event in events]
        if normalizer:
            values = [((value * normalizer) - 1) / (normalizer - 1) for value in values]
            values = [value if value >= 0 else 0 for value in values]
        auc = np.trapz(values, dx=1)
        perfect_values = [1 for _ in values]
        ideal_auc = np.trapz(perfect_values, dx=1)
        auc /= ideal_auc
        return auc, len(values)
    else:
        return None, ea.Tags()['scalars']

LOGS = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents/logs"

runs = {
    "Local, 10-15, dynamic, single-topology (matched)": f"{LOGS}/local_10-15_dynamic_single_trpo_matched_2026-07-17_02-11-45/TRPO_x_control_SecureBERT/TRPO_1",
    "Local, 10-15, static, single-topology (matched)": f"{LOGS}/local_10-15_static_single_trpo_matched_2026-07-17_17-40-29/TRPO_x_control_SecureBERT/TRPO_1",
    "Compressed, 10-15, dynamic, single-topology (matched)": f"{LOGS}/compressed_10-15_dynamic_single_trpo_matched_2026-07-17_17-11-52/TRPO_x_control_SecureBERT/TRPO_1",
    "Compressed, 10-15, static, single-topology (matched)": f"{LOGS}/compressed_10-15_static_single_trpo_matched_2026-07-17_17-40-29/TRPO_x_control_SecureBERT/TRPO_1",
    "Compressed, 10-15, dynamic, 500-graph (PPO, PE-fixed)": f"{LOGS}/[useless but backup] compressed_10-15_dynamic_multi500_ppo_pefix_2026-07-17_13-06-04/PPO_x_control_SecureBERT/PPO_1",
    "Compressed, 10-15, dynamic, 500-graph (TRPO, PE-fixed)": f"{LOGS}/compressed_10-15_dynamic_multi500_trpo_pefix_2026-07-17_13-06-09/TRPO_x_control_SecureBERT/TRPO_1",
    "Local, 10-15, dynamic, 500-graph (fair multi-topology attempt)": f"{LOGS}/local_10-15_dynamic_multi500_trpo_fair_2026-07-17_15-50-21/TRPO_x_control_SecureBERT/TRPO_1",
}

extra_runs = {
    "Local, 30-40, dynamic, single-topology (H2 scale)": f"{LOGS}/local_30-40_dynamic_single_trpo_h2scale_2026-07-17_02-11-47/TRPO_x_control_SecureBERT/TRPO_1",
    "Local, 80-100, dynamic, single-topology (H2 scale)": f"{LOGS}/local_80-100_dynamic_single_trpo_h2scale_2026-07-17_02-11-50/TRPO_x_control_SecureBERT/TRPO_1",
}

metric_name = "train/Relative owned nodes percentage"

if __name__ == "__main__":
    for label, path in runs.items():
        auc, extra = extract_auc_from_tensorboard(path, metric_name)
        if auc is None:
            print(f"{label}: metric not found. Available tags: {extra}")
        else:
            print(f"{label}: AUC={auc:.4f}  (n_samples={extra})")

    print()
    print("--- other H2 scales (no matched Compressed run yet at these scales) ---")
    for label, path in extra_runs.items():
        auc, extra = extract_auc_from_tensorboard(path, metric_name)
        if auc is None:
            print(f"{label}: metric not found. Available tags: {extra}")
        else:
            print(f"{label}: AUC={auc:.4f}  (n_samples={extra})")
