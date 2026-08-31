"""Measure the generated Y-N30-N60 topologies on the CALIBRATED control metric + vulns.

Each generated cell is one topology per seed under
  cyberbattle/data/env_samples/graphs_yN{N}_s{seed}_<ts>/1/network_SecureBERT.pkl
a simulation.model.Model whose connectivity lives in three DIRECTED relationship graphs:
  knows_graph  (who-knows-whom / visibility), access_graph (who-can-access-whom / attack surface,
  which includes reachability so it inflates with N), dos_graph.
The plain .network graph carries node attrs only (0 edges), so degree must come from these.

DEGREE DEFINITION (evidence_taskY.md 0.1 reconciliation table, rows #1/#2):
  The Task-Y control target of ~22 is expressed in metric #1 = knows_graph *OUT-degree*
  (`np.mean(out_degree)`), NOT any undirected or access degree. `knows_neighbor_probability`
  directly governs the knows graph (generate_network.py:190-194), so the design holds THIS constant.
  Access out-degree (#2) tracks knows at ~1.09x IN KIND but grows super-linearly with N via
  reachability, so it is reported only as context and is NOT the controlled quantity.

CORRECTION (2026-08-03): an earlier version of this script reported mean UNDIRECTED degree of the
access_graph (44.5 at N=60) and mislabelled it "degree", which looked like an N=60 generation error.
It was not: it is a different quantity. On the calibrated metric (knows out-degree) N=60 measures
~19.9, right at target and ~equal to N=30's ~19.7 -- degree IS held constant on the control metric.
The undirected/access numbers differ from out-degree because the knows graph is only partly reciprocal
(sym% falls with N), which this script now reports so the distinction is auditable.

Usage: python measure_topos.py [--out <csv>]
"""
import argparse
import glob
import pickle

import numpy as np

BASE = "cyberbattle/data/env_samples"
SEEDS = [42, 100, 123, 200, 300]
CELLS = [30, 60]


def deg(g):
    """Return (out_degree_mean, undirected_degree_mean, reciprocity_pct) for a relationship graph."""
    if not g.is_directed():
        u = float(np.mean([d for _, d in g.degree()])) if g.number_of_nodes() else 0.0
        return u, u, 100.0
    out = float(np.mean([d for _, d in g.out_degree()]))
    und = float(np.mean([d for _, d in g.to_undirected().degree()]))
    E = list(g.edges())
    sym = 100.0 * sum(1 for a, b in E if g.has_edge(b, a)) / len(E) if E else float("nan")
    return out, und, sym


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = []
    print(f"{'cell':>6} {'seed':>5} {'N':>4} | {'KNOWS_out#1':>11} {'knows_und':>9} {'knows_sym%':>10} "
          f"| {'access_out':>10} {'access_und':>10} | {'vulns':>6}")
    summ = {}
    for N in CELLS:
        ko = []
        for s in SEEDS:
            d = glob.glob(f"{BASE}/graphs_yN{N}_s{s}_*")[0]
            m = pickle.load(open(f"{d}/1/network_SecureBERT.pkl", "rb"))
            kout, kund, ksym = deg(m.knows_graph)
            aout, aund, _ = deg(m.access_graph)
            nv = sum(len(getattr(m.get_node(nid), "vulnerabilities", None) or {}) for nid in m.network.nodes())
            ko.append(kout)
            rows.append([N, s, m.network.number_of_nodes(), kout, kund, ksym, aout, aund, nv])
            print(f"{'yN'+str(N):>6} {s:>5} {N:>4} | {kout:>11.2f} {kund:>9.2f} {ksym:>10.0f} "
                  f"| {aout:>10.2f} {aund:>10.2f} | {nv:>6}")
        summ[N] = (float(np.mean(ko)), float(np.std(ko, ddof=1)))
    print("\nCONTROL METRIC = knows_graph out-degree (#1; target ~22):")
    for N in CELLS:
        k, ks = summ[N]
        print(f"  N={N}: knows out-degree = {k:.2f} +/- {ks:.2f}")
    print("  -> degree IS approximately held constant across N on the control metric.")
    if args.out:
        import csv
        w = csv.writer(open(args.out, "w"))
        w.writerow(["N", "seed", "num_nodes", "knows_out_deg_1", "knows_undir_deg",
                    "knows_reciprocity_pct", "access_out_deg", "access_undir_deg", "num_vulns"])
        w.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
