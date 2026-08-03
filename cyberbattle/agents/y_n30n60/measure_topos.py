"""Measure the generated Y-N30-N60 topologies: node count, relationship-graph degree, vulns.

Each generated cell is one topology per seed under
  cyberbattle/data/env_samples/graphs_yN{N}_s{seed}_<ts>/1/network_SecureBERT.pkl
a simulation.model.Model whose connectivity lives in three relationship graphs:
  knows_graph  (who-knows-whom / visibility), access_graph (who-can-access-whom / attack surface),
  dos_graph    (denial-of-service reachability).
The plain .network graph carries node attrs only (0 edges), so degree must come from these.

Reports per (cell, seed): mean undirected degree of each relationship graph + total vulnerabilities
across nodes, and the per-cell mean+/-SD. NOTE (caveat for the write-up): this crossed design does
NOT hold degree constant across N -- report the measured degrees honestly; do not read a pure-N
contrast into an N-vs-degree confound.

Usage: python measure_topos.py [--out <csv>]
"""
import argparse
import glob
import pickle

import networkx as nx
import numpy as np

BASE = "cyberbattle/data/env_samples"
SEEDS = [42, 100, 123, 200, 300]
CELLS = [30, 60]


def mean_degree(g):
    ug = g.to_undirected() if g.is_directed() else g
    if ug.number_of_nodes() == 0:
        return 0.0
    return float(np.mean([d for _, d in ug.degree()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = []
    hdr = f"{'cell':>6} {'seed':>5} {'N':>4} | {'knows_deg':>9} {'access_deg':>10} {'dos_deg':>8} | {'vulns':>6}"
    print(hdr)
    summ = {}
    for N in CELLS:
        kd, ad, dd = [], [], []
        for s in SEEDS:
            d = glob.glob(f"{BASE}/graphs_yN{N}_s{s}_*")[0]
            m = pickle.load(open(f"{d}/1/network_SecureBERT.pkl", "rb"))
            k, a, o = mean_degree(m.knows_graph), mean_degree(m.access_graph), mean_degree(m.dos_graph)
            nv = sum(len(getattr(m.get_node(nid), "vulnerabilities", None) or {}) for nid in m.network.nodes())
            kd.append(k); ad.append(a); dd.append(o)
            rows.append([N, s, m.network.number_of_nodes(), k, a, o, nv])
            print(f"{'yN'+str(N):>6} {s:>5} {N:>4} | {k:>9.2f} {a:>10.2f} {o:>8.2f} | {nv:>6}")
        summ[N] = (np.mean(kd), np.std(kd, ddof=1), np.mean(ad), np.std(ad, ddof=1))
    print()
    for N in CELLS:
        k, ks, a, asd = summ[N]
        print(f"N={N}: knows_deg {k:.2f}+/-{ks:.2f}   access_deg {a:.2f}+/-{asd:.2f}   (degree NOT held constant across N)")
    if args.out:
        import csv
        w = csv.writer(open(args.out, "w"))
        w.writerow(["N", "seed", "num_nodes", "knows_deg", "access_deg", "dos_deg", "num_vulns"])
        w.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
