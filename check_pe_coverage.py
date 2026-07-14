import sys
import os
import pickle
import argparse

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from cyberbattle.simulation import model  # noqa: E402

def check_pe_coverage(env_path, n_resets=20):
    with open(env_path, 'rb') as f:
        env = pickle.load(f)

    coverage_ratios = []
    for i in range(n_resets):
        env.reset()  # picks a new random starter node each time, sets up ownable_count / shortest_paths_starter_control

        ownable_nodes = [
            node for node, dist in env.shortest_paths_starter_control.items()
            if dist is not None
        ]

        if not ownable_nodes:
            continue

        nodes_with_pe = 0
        for node in ownable_nodes:
            node_info = env.get_node(node)
            has_pe = any(
                isinstance(result.outcome, model.PrivilegeEscalation)
                for vuln in node_info.vulnerabilities.values()
                for result in vuln.results
            )
            if has_pe:
                nodes_with_pe += 1

        ratio = nodes_with_pe / len(ownable_nodes)
        coverage_ratios.append(ratio)
        print(f"Reset {i+1}: starter={env.starter_node}, "
              f"ownable={len(ownable_nodes)}, with_PE_vuln={nodes_with_pe}, "
              f"ratio={ratio:.2f}")

    if coverage_ratios:
        avg = sum(coverage_ratios) / len(coverage_ratios)
        print(f"\nAverage PE-coverage ratio across {len(coverage_ratios)} resets: {avg:.3f}")
        print(f"Min: {min(coverage_ratios):.3f}, Max: {max(coverage_ratios):.3f}")
    else:
        print("No valid resets produced ownable nodes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_path", type=str, help="Path to a processed environment .pkl file")
    parser.add_argument("--n_resets", type=int, default=20)
    args = parser.parse_args()
    check_pe_coverage(args.env_path, args.n_resets)
