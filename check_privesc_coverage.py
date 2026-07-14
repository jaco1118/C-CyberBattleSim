#!/usr/bin/env python3
"""
check_privesc_coverage.py

Dry-run validator: reports, over an EXISTING set of generated graphs, which ones
would pass/fail a privilege-escalation coverage filter. Generates nothing and
modifies nothing -- it only reads the vulnerabilities_distribution.yaml files
that generate_graphs.py already wrote (via collect_vulnerability_data()).

Use this BEFORE wiring the filter into generate_graphs.py, to (a) confirm it
flags the graphs you already know are bad, and (b) see your rejection rate so
you know whether filtering will starve your training set.

Usage:
    python check_privesc_coverage.py /path/to/graphs_<name>_<timestamp>
    python check_privesc_coverage.py /path/to/graphs_<...> --min-privesc 3
    python check_privesc_coverage.py /path/to/graphs_<...> --verbose

The folder passed should be the PARENT that contains numbered subdirs
(1/, 2/, ... each holding a vulnerabilities_distribution.yaml).
"""

import argparse
import glob
import os
import sys

try:
    import yaml
except ImportError:
    sys.exit("PyYAML not found. Run inside your project env (it uses yaml already).")

# Literal outcome key as written by result.outcome_str -> confirmed 'privilege escalation'
PRIVESC_KEY = "privilege escalation"


def node_privesc_count(node_outcomes):
    """Number of distinct privesc vulns recorded on one node."""
    return len(node_outcomes.get(PRIVESC_KEY, []))


def check_graph(yaml_path, min_privesc, privesc_key=PRIVESC_KEY):
    """
    Returns (passed: bool, offending: list[(node, count)], total_nodes: int).
    A graph passes iff EVERY node has >= min_privesc privesc vulns.
    """
    with open(yaml_path, "r") as f:
        stats = yaml.safe_load(f)

    per_node = (stats or {}).get("outcomes_nodes_presence", {})
    if not per_node:
        # No node data at all -> treat as failing so it surfaces loudly.
        return False, [("<no outcomes_nodes_presence>", 0)], 0

    offending = []
    for node, outcomes in per_node.items():
        cnt = len(outcomes.get(privesc_key, []))
        if cnt < min_privesc:
            offending.append((node, cnt))

    return (len(offending) == 0), offending, len(per_node)


def graph_id_of(yaml_path):
    """The numbered subdir name, e.g. '.../graphs_x/8/vulnerabilities_distribution.yaml' -> '8'."""
    return os.path.basename(os.path.dirname(yaml_path))


def main():
    ap = argparse.ArgumentParser(description="Dry-run privesc-coverage validator.")
    ap.add_argument("graphs_dir", help="Parent folder containing numbered graph subdirs.")
    ap.add_argument("--min-privesc", type=int, default=3,
                    help="Minimum privesc vulns required per node (default: 3).")
    ap.add_argument("--privesc-key", default=PRIVESC_KEY,
                    help=f"Outcome key to count (default: '{PRIVESC_KEY}').")
    ap.add_argument("--verbose", action="store_true",
                    help="List every offending node, not just the per-graph summary.")
    args = ap.parse_args()

    pattern = os.path.join(args.graphs_dir, "*", "vulnerabilities_distribution.yaml")
    yaml_paths = sorted(
        glob.glob(pattern),
        key=lambda p: (len(graph_id_of(p)), graph_id_of(p)),  # natural-ish sort
    )

    if not yaml_paths:
        sys.exit(f"No vulnerabilities_distribution.yaml found under: {pattern}")

    total = len(yaml_paths)
    passed_ids, failed_ids = [], []

    print(f"Checking {total} graph(s) in: {args.graphs_dir}")
    print(f"Rule: every node must have >= {args.min_privesc} '{args.privesc_key}' vuln(s)\n")

    for yp in yaml_paths:
        gid = graph_id_of(yp)
        try:
            passed, offending, n_nodes = check_graph(yp, args.min_privesc, args.privesc_key)
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"  graph {gid:>4}: ERROR reading ({e})")
            failed_ids.append(gid)
            continue

        if passed:
            passed_ids.append(gid)
            print(f"  graph {gid:>4}: PASS  ({n_nodes} nodes)")
        else:
            failed_ids.append(gid)
            worst = min((c for _, c in offending), default=0)
            print(f"  graph {gid:>4}: FAIL  ({len(offending)}/{n_nodes} node(s) short, "
                  f"min count = {worst})")
            if args.verbose:
                for node, cnt in sorted(offending, key=lambda x: x[1]):
                    print(f"                 - {node}: {cnt} privesc vuln(s)")

    n_pass, n_fail = len(passed_ids), len(failed_ids)
    print("\n" + "=" * 52)
    print(f"PASS: {n_pass}/{total}   FAIL: {n_fail}/{total}   "
          f"rejection rate: {100.0 * n_fail / total:.1f}%")
    print("=" * 52)

    if n_fail:
        print("Failing graph IDs:", ", ".join(failed_ids))
    # Non-zero exit if anything failed, handy for scripting.
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
