"""
compile_appendix_data.py
Compiles raw per-seed values and per-checkpoint curves behind the three retention_results*.csv
summaries into dissertation-appendix-ready tables. Pure data compilation from already-computed
results (no new experiments) -- reuses the loader functions from the three retention-analysis
scripts already written this session. Run from cyberbattle/agents/. Output: appendix_raw_data/.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_retention_analysis import load_seed_score  # noqa: E402
from compute_auc_retention_analysis import load_seed_auc  # noqa: E402
from compute_test_auc_retention_analysis import load_seed_test_auc  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "appendix_raw_data")
DET_METRIC = "root_owned_nodes_percentage"
TRAIN_AUC_METRIC = "train/Relative owned nodes percentage"
TEST_AUC_METRIC = "root_owned_nodes_percentage"

LOGS = "logs"

# Same seed -> path mappings used in the three manifests built earlier this session
# (/tmp/manifest_100ep_full.yaml, /tmp/manifest_train_auc_final.yaml, /tmp/manifest_test_auc_final.yaml),
# written in directly since /tmp files aren't guaranteed to persist.

DET_CSV_PATTERNS = {
    "local_dynamic": {
        42: f"{LOGS}/local_10-15_dynamic_single_trpo_matched_2026-07-17_02-11-45/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/1/average_performances_*100_0_2026071[89]*.csv",
        100: f"{LOGS}/local_dynamic_5seed_2026-07-18_01-15-51/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/1/average_performances_*100_0_2026071[89]*.csv",
        123: f"{LOGS}/local_dynamic_5seed_2026-07-18_01-15-51/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/2/average_performances_*100_0_2026071[89]*.csv",
        200: f"{LOGS}/local_dynamic_5seed_2026-07-18_01-15-51/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/3/average_performances_*100_0_2026071[89]*.csv",
        300: f"{LOGS}/local_dynamic_5seed_2026-07-18_01-15-51/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/4/average_performances_*100_0_2026071[89]*.csv",
    },
    "local_static": {
        42: f"{LOGS}/local_10-15_static_single_trpo_matched_2026-07-17_17-40-29/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/1/average_performances_*100_0_2026071[89]*.csv",
        100: f"{LOGS}/local_static_5seed_2026-07-18_01-15-51/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/1/average_performances_*100_0_2026071[89]*.csv",
        123: f"{LOGS}/local_static_5seed_2026-07-18_01-15-51/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/2/average_performances_*100_0_2026071[89]*.csv",
        200: f"{LOGS}/local_static_5seed_2026-07-18_01-15-51/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/3/average_performances_*100_0_2026071[89]*.csv",
        300: f"{LOGS}/local_static_5seed_2026-07-18_01-15-51/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/4/average_performances_*100_0_2026071[89]*.csv",
    },
    "compressed_dynamic": {
        42: f"{LOGS}/compressed_dynamic_seed42_ppo_2026-07-18_14-52-50/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/1/average_performances_*100_0_2026071[89]*.csv",
        100: f"{LOGS}/compressed_dynamic_5seed_2026-07-18_01-09-06/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/1/average_performances_*100_0_2026071[89]*.csv",
        123: f"{LOGS}/compressed_dynamic_5seed_2026-07-18_01-09-06/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/2/average_performances_*100_0_2026071[89]*.csv",
        200: f"{LOGS}/compressed_dynamic_5seed_2026-07-18_01-09-06/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/3/average_performances_*100_0_2026071[89]*.csv",
        300: f"{LOGS}/compressed_dynamic_5seed_2026-07-18_01-09-06/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/4/average_performances_*100_0_2026071[89]*.csv",
    },
    "compressed_static": {
        42: f"{LOGS}/compressed_static_seed42_ppo_2026-07-18_14-52-50/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/1/average_performances_*100_0_2026071[89]*.csv",
        100: f"{LOGS}/compressed_static_5seed_2026-07-18_01-15-51/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/1/average_performances_*100_0_2026071[89]*.csv",
        123: f"{LOGS}/compressed_static_5seed_2026-07-18_01-15-51/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/2/average_performances_*100_0_2026071[89]*.csv",
        200: f"{LOGS}/compressed_static_5seed_2026-07-18_01-15-51/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/3/average_performances_*100_0_2026071[89]*.csv",
        300: f"{LOGS}/compressed_static_5seed_2026-07-18_01-15-51/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/4/average_performances_*100_0_2026071[89]*.csv",
    },
}

TRAIN_AUC_DIRS = {
    "local_dynamic": {
        42: f"{LOGS}/local_dynamic_seed42_retrain_2026-07-19_15-03-58/TRPO_x_control_SecureBERT/TRPO_1",
        100: f"{LOGS}/local_dynamic_5seed_2026-07-18_01-15-51/TRPO_x_control_SecureBERT/TRPO_1",
        123: f"{LOGS}/local_dynamic_5seed_2026-07-18_01-15-51/TRPO_x_control_SecureBERT/TRPO_2",
        200: f"{LOGS}/local_dynamic_5seed_2026-07-18_01-15-51/TRPO_x_control_SecureBERT/TRPO_3",
        300: f"{LOGS}/local_dynamic_5seed_2026-07-18_01-15-51/TRPO_x_control_SecureBERT/TRPO_4",
    },
    "local_static": {
        42: f"{LOGS}/local_10-15_static_single_trpo_matched_2026-07-17_17-40-29/TRPO_x_control_SecureBERT/TRPO_1",
        100: f"{LOGS}/local_static_5seed_2026-07-18_01-15-51/TRPO_x_control_SecureBERT/TRPO_1",
        123: f"{LOGS}/local_static_5seed_2026-07-18_01-15-51/TRPO_x_control_SecureBERT/TRPO_2",
        200: f"{LOGS}/local_static_5seed_2026-07-18_01-15-51/TRPO_x_control_SecureBERT/TRPO_3",
        300: f"{LOGS}/local_static_5seed_2026-07-18_01-15-51/TRPO_x_control_SecureBERT/TRPO_4",
    },
    "compressed_dynamic": {
        42: f"{LOGS}/compressed_dynamic_seed42_ppo_2026-07-18_14-52-50/PPO_x_control_SecureBERT/PPO_1",
        100: f"{LOGS}/compressed_dynamic_5seed_2026-07-18_01-09-06/PPO_x_control_SecureBERT/PPO_1",
        123: f"{LOGS}/compressed_dynamic_5seed_2026-07-18_01-09-06/PPO_x_control_SecureBERT/PPO_2",
        200: f"{LOGS}/compressed_dynamic_5seed_2026-07-18_01-09-06/PPO_x_control_SecureBERT/PPO_3",
        300: f"{LOGS}/compressed_dynamic_5seed_2026-07-18_01-09-06/PPO_x_control_SecureBERT/PPO_4",
    },
    "compressed_static": {
        42: f"{LOGS}/compressed_static_seed42_ppo_2026-07-18_14-52-50/PPO_x_control_SecureBERT/PPO_1",
        100: f"{LOGS}/compressed_static_5seed_2026-07-18_01-15-51/PPO_x_control_SecureBERT/PPO_1",
        123: f"{LOGS}/compressed_static_5seed_2026-07-18_01-15-51/PPO_x_control_SecureBERT/PPO_2",
        200: f"{LOGS}/compressed_static_5seed_2026-07-18_01-15-51/PPO_x_control_SecureBERT/PPO_3",
        300: f"{LOGS}/compressed_static_5seed_2026-07-18_01-15-51/PPO_x_control_SecureBERT/PPO_4",
    },
}

TEST_AUC_DIRS = {
    "local_dynamic": {
        42: f"{LOGS}/local_dynamic_seed42_retrain_2026-07-19_15-03-58/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/1",
        100: f"{LOGS}/local_dynamic_5seed_2026-07-18_01-15-51/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/1",
        123: f"{LOGS}/local_dynamic_5seed_2026-07-18_01-15-51/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/2",
        200: f"{LOGS}/local_dynamic_5seed_2026-07-18_01-15-51/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/3",
        300: f"{LOGS}/local_dynamic_5seed_2026-07-18_01-15-51/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/4",
    },
    "local_static": {
        42: f"{LOGS}/local_10-15_static_single_trpo_matched_2026-07-17_17-40-29/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/1",
        100: f"{LOGS}/local_static_5seed_2026-07-18_01-15-51/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/1",
        123: f"{LOGS}/local_static_5seed_2026-07-18_01-15-51/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/2",
        200: f"{LOGS}/local_static_5seed_2026-07-18_01-15-51/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/3",
        300: f"{LOGS}/local_static_5seed_2026-07-18_01-15-51/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/4",
    },
    "compressed_dynamic": {
        42: f"{LOGS}/compressed_dynamic_seed42_ppo_2026-07-18_14-52-50/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/1",
        100: f"{LOGS}/compressed_dynamic_5seed_2026-07-18_01-09-06/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/1",
        123: f"{LOGS}/compressed_dynamic_5seed_2026-07-18_01-09-06/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/2",
        200: f"{LOGS}/compressed_dynamic_5seed_2026-07-18_01-09-06/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/3",
        300: f"{LOGS}/compressed_dynamic_5seed_2026-07-18_01-09-06/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/4",
    },
    "compressed_static": {
        42: f"{LOGS}/compressed_static_seed42_ppo_2026-07-18_14-52-50/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/1",
        100: f"{LOGS}/compressed_static_5seed_2026-07-18_01-15-51/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/1",
        123: f"{LOGS}/compressed_static_5seed_2026-07-18_01-15-51/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/2",
        200: f"{LOGS}/compressed_static_5seed_2026-07-18_01-15-51/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/3",
        300: f"{LOGS}/compressed_static_5seed_2026-07-18_01-15-51/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/4",
    },
}

CONDITION_META = {
    "local_dynamic": ("local", "dynamic"),
    "local_static": ("local", "static"),
    "compressed_dynamic": ("compressed", "dynamic"),
    "compressed_static": ("compressed", "static"),
}


def build_summary_table():
    rows = []
    values_by_metric_cond = {}
    for metric_name, patterns_by_cond, loader in [
        ("deterministic_100ep_last_checkpoint", DET_CSV_PATTERNS, lambda p: load_seed_score(p, DET_METRIC, BASE_DIR)[0]),
        ("training_curve_auc", TRAIN_AUC_DIRS, lambda p: load_seed_auc(p, TRAIN_AUC_METRIC, BASE_DIR)[0]),
        ("test_data_auc", TEST_AUC_DIRS, lambda p: load_seed_test_auc(p, TEST_AUC_METRIC, BASE_DIR)[0]),
    ]:
        for cond_name, seed_paths in patterns_by_cond.items():
            rep, churn = CONDITION_META[cond_name]
            for seed, path in seed_paths.items():
                value = loader(path)
                rows.append({"metric": metric_name, "representation": rep, "churn": churn, "seed": seed, "value": value})
                values_by_metric_cond[(metric_name, rep, churn, seed)] = value

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "appendix_raw_seed_values.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")

    ratio_rows = []
    for metric_name in ["deterministic_100ep_last_checkpoint", "training_curve_auc", "test_data_auc"]:
        for rep in ["local", "compressed"]:
            for seed in [42, 100, 123, 200, 300]:
                dyn = values_by_metric_cond[(metric_name, rep, "dynamic", seed)]
                stat = values_by_metric_cond[(metric_name, rep, "static", seed)]
                ratio_rows.append({
                    "metric": metric_name, "representation": rep, "seed": seed,
                    "dynamic_value": dyn, "static_value": stat,
                    "retention_ratio": dyn / stat if stat else float("nan"),
                })
    ratio_df = pd.DataFrame(ratio_rows)
    ratio_path = os.path.join(OUT_DIR, "appendix_retention_ratios.csv")
    ratio_df.to_csv(ratio_path, index=False)
    print(f"Wrote {len(ratio_df)} rows to {ratio_path}")


def build_checkpoint_curves_table():
    import glob
    rows = []
    for cond_name, seed_paths in TEST_AUC_DIRS.items():
        rep, churn = CONDITION_META[cond_name]
        for seed, pattern in seed_paths.items():
            full_pattern = pattern if os.path.isabs(pattern) else os.path.join(BASE_DIR, pattern)
            dirs = sorted(glob.glob(full_pattern))
            run_dir = sorted(dirs, key=os.path.getmtime)[-1]
            csv_paths = glob.glob(os.path.join(run_dir, "average_performances_*.csv"))
            csv_paths.sort(key=os.path.getmtime)
            for idx, csv_path in enumerate(csv_paths):
                df = pd.read_csv(csv_path)
                agent_rows = df[df["agent"] == "agent"]
                mean_val = agent_rows[TEST_AUC_METRIC].mean()
                # checkpoint step embedded in the model file the CSV was generated from isn't
                # stored in the CSV itself; approximate ordering via checkpoint_index (0-based,
                # in ascending step order, matching how test_agent.py processes checkpoints)
                rows.append({
                    "representation": rep, "churn": churn, "seed": seed,
                    "checkpoint_index": idx, "mean_root_owned_nodes_percentage": mean_val,
                })
    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "appendix_test_auc_checkpoint_curves.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    build_summary_table()
    build_checkpoint_curves_table()
