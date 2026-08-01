"""
compute_test_auc_retention_analysis.py
Test-data (held-out) AUC-based retention-ratio and paired-significance analysis for the
Local-vs-Compressed dynamic-churn comparison (dissertation H1). Unlike
compute_auc_retention_analysis.py (which integrates the in-sample TensorBoard training
curve), this integrates a curve of *deterministic test-time* root-owned% across a strided
sample of checkpoints spanning training (test_agent.py --checkpoint_stride N, no
--last_checkpoint) -- a genuinely held-out-episode signal rather than a training-time
self-report. Same np.trapz/ideal-AUC normalization as the training-curve version, same
downstream retention-ratio + bootstrap significance logic, reusing bootstrap_ci from
cyberbattle.utils.math_utils. Run from cyberbattle/agents/.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from cyberbattle.utils.math_utils import bootstrap_ci  # noqa: E402

METRIC_DEFAULT = "root_owned_nodes_percentage"

EXAMPLE_MANIFEST = """\
metric: root_owned_nodes_percentage

conditions:
  local_dynamic:
    representation: local
    churn: dynamic
    seeds:
      42: logs/local_10-15_dynamic_single_trpo_matched_2026-07-17_02-11-45/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/1
      100: logs/local_dynamic_5seed_*/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/1
      123: logs/local_dynamic_5seed_*/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/2
      200: logs/local_dynamic_5seed_*/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/3
      300: logs/local_dynamic_5seed_*/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/4
  local_static:
    representation: local
    churn: static
    seeds:
      42: logs/local_10-15_static_single_trpo_matched_2026-07-17_17-40-29/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/1
      100: logs/local_static_5seed_*/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/1
      123: logs/local_static_5seed_*/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/2
      200: logs/local_static_5seed_*/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/3
      300: logs/local_static_5seed_*/TRPO_x_control_SecureBERT/test/local_baseline_single_topology/train/4
  compressed_dynamic:
    representation: compressed
    churn: dynamic
    seeds:
      42: logs/compressed_dynamic_seed42_ppo_*/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/1
      100: logs/compressed_dynamic_5seed_*/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/1
      123: logs/compressed_dynamic_5seed_*/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/2
      200: logs/compressed_dynamic_5seed_*/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/3
      300: logs/compressed_dynamic_5seed_*/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/4
  compressed_static:
    representation: compressed
    churn: static
    seeds:
      42: logs/compressed_static_seed42_ppo_*/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/1
      100: logs/compressed_static_5seed_*/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/1
      123: logs/compressed_static_5seed_*/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/2
      200: logs/compressed_static_5seed_*/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/3
      300: logs/compressed_static_5seed_*/PPO_x_control_SecureBERT/test/local_baseline_single_topology/train/4
"""


def load_seed_test_auc(pattern, metric, base_dir):
    """A 'seed' here points at a run_id's test folder containing one average_performances_*.csv
    per strided checkpoint. Each CSV's mean(metric) over its 'agent' rows becomes one point on
    the test-time curve; points are ordered by CSV file modification time, which matches
    checkpoint/step order since test_agent.py evaluates checkpoints in ascending step order."""
    full_pattern = pattern if os.path.isabs(pattern) else os.path.join(base_dir, pattern)
    # The directory portion may contain a literal "[...]" (e.g. a backup-tag rename), which
    # glob.glob() would otherwise interpret as character-class syntax rather than literal text;
    # the filename portion's wildcards are intentional and must stay functional -- escape only
    # the directory prefix in each call, leaving the wildcard filename pattern untouched.
    dir_part, file_part = os.path.split(full_pattern)
    dirs = sorted(glob.glob(os.path.join(glob.escape(dir_part), file_part)))
    if not dirs:
        raise FileNotFoundError(f"No directory matched pattern: {full_pattern}")
    run_dir = sorted(dirs, key=os.path.getmtime)[-1]

    csv_paths = glob.glob(os.path.join(glob.escape(run_dir), "average_performances_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No average_performances_*.csv found in {run_dir}")
    csv_paths.sort(key=os.path.getmtime)

    values = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        agent_rows = df[df["agent"] == "agent"]
        if metric not in agent_rows.columns:
            raise KeyError(f"Metric '{metric}' not found in {csv_path}. Columns: {list(agent_rows.columns)}")
        values.append(agent_rows[metric].mean())

    auc = np.trapz(values, dx=1)
    ideal_auc = np.trapz([1] * len(values), dx=1)
    return auc / ideal_auc, len(values), run_dir


def run_test_auc_retention(manifest_path):
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)
    metric = manifest.get("metric", METRIC_DEFAULT)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    condition_scores = {}
    condition_meta = {}

    for cond_name, cond in manifest["conditions"].items():
        condition_meta[cond_name] = {"representation": cond["representation"], "churn": cond["churn"]}
        scores = {}
        for seed, pattern in cond["seeds"].items():
            auc, n_checkpoints, run_dir = load_seed_test_auc(pattern, metric, base_dir)
            scores[seed] = auc
            print(f"  [{cond_name}] seed={seed} -> test-AUC={auc:.4f}  (n_checkpoints={n_checkpoints}, {run_dir})")
        condition_scores[cond_name] = scores

    print(f"\nMetric: {metric} (test-time, held-out episodes)\n")
    print("=== Per-condition test-AUC ===")
    summary_rows = []
    for cond_name, scores in condition_scores.items():
        values = list(scores.values())
        mean, lo, hi = bootstrap_ci(values)
        print(f"{cond_name}: n_seeds={len(values)}  mean_test_AUC={mean:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")
        summary_rows.append({"condition": cond_name, "n_seeds": len(values), "mean": mean, "ci_lower": lo, "ci_upper": hi})

    def cond_for(rep, churn):
        return next(c for c, m in condition_meta.items() if m["representation"] == rep and m["churn"] == churn)

    representations = sorted(set(m["representation"] for m in condition_meta.values()))
    retention_ratios = {}
    print("\n=== Test-AUC retention ratios (dynamic / static), paired by seed ===")
    for rep in representations:
        dyn_scores = condition_scores[cond_for(rep, "dynamic")]
        stat_scores = condition_scores[cond_for(rep, "static")]
        common_seeds = sorted(set(dyn_scores) & set(stat_scores))
        if set(dyn_scores) != set(stat_scores):
            print(f"WARNING: {rep} has mismatched seed sets between dynamic and static -- using intersection: {common_seeds}")
        ratios = {s: dyn_scores[s] / stat_scores[s] for s in common_seeds if stat_scores[s] != 0}
        retention_ratios[rep] = ratios
        mean, lo, hi = bootstrap_ci(list(ratios.values()))
        print(f"{rep}: n_seeds={len(ratios)}  retention_mean={mean:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]  ({(1 - mean) * 100:.1f}% degradation)")
        summary_rows.append({"condition": f"{rep}_retention_ratio", "n_seeds": len(ratios), "mean": mean, "ci_lower": lo, "ci_upper": hi})

    if len(representations) == 2:
        rep_a, rep_b = representations
        common = sorted(set(retention_ratios[rep_a]) & set(retention_ratios[rep_b]))
        diffs = [retention_ratios[rep_a][s] - retention_ratios[rep_b][s] for s in common]
        mean, lo, hi = bootstrap_ci(diffs)
        significant = not (lo <= 0 <= hi)
        print(f"\n=== Significance: test-AUC retention_ratio({rep_a}) - retention_ratio({rep_b}), paired by seed ===")
        print(f"n_seeds={len(diffs)}  mean_diff={mean:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]  "
              f"{'SIGNIFICANT (CI excludes 0)' if significant else 'not significant (CI includes 0)'}")
        summary_rows.append({"condition": f"diff_{rep_a}_minus_{rep_b}_retention", "n_seeds": len(diffs), "mean": mean, "ci_lower": lo, "ci_upper": hi})

    out_path = os.path.join(base_dir, "retention_results_test_auc.csv")
    pd.DataFrame(summary_rows).to_csv(out_path, index=False)
    print(f"\nSaved summary to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Test-data (held-out) AUC-based retention-ratio and significance analysis.")
    parser.add_argument("command", nargs="?", default=None, choices=["retention"], help="Subcommand to run")
    parser.add_argument("--manifest", type=str, help="Path to the manifest YAML")
    parser.add_argument("--example", action="store_true", help="Print an example manifest to stdout and exit")
    args = parser.parse_args()

    if args.example:
        print(EXAMPLE_MANIFEST)
        return

    if args.command == "retention":
        if not args.manifest:
            parser.error("retention requires --manifest <path>")
        run_test_auc_retention(args.manifest)
    else:
        parser.error("specify --example, or the 'retention' subcommand with --manifest")


if __name__ == "__main__":
    main()
