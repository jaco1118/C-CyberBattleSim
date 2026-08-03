"""Bookkeeping check: was N=90's already-reported degree span (18.9-27.4) measured with the
undirected-access-degree bug that affected this session's N=30/N=60 measure_topos.py (fixed a2c8285)?

Re-measures the 5 EXISTING N=90 topology instances (no retraining, no regeneration) both ways:
  - knows_graph OUT-degree  (metric #1, the calibrated Task-Y control; evidence_taskY.md 0.1)
  - access_graph UNDIRECTED (the bug: undirected-union of a reachability-inflated graph)
and prints which one reproduces the previously-reported 18.9-27.4 span.

Instances: seed 42 = the original probe folder graphs_yprobe_n90_<ts>; seeds 100/123/200/300 =
graphs_yprobe_n90_s<seed>_<ts> (the 5-seed batch), the same folders the batch runs loaded via load_envs.

RESULT (2026-08-03): knows out-degree reproduces 18.89-27.42 exactly (mean 21.53, SD 3.39); the buggy
metric would have read ~53.5-83.8. So N=90 was ALREADY measured correctly on knows out-degree -- the bug
was new to this session and only ever touched N=30/N=60. No N=90 correction needed; the 18.9-27.4 banner
is the true knows-out-degree spread and stands.

Usage: python verify_n90_degree.py
"""
import glob
import pickle

import numpy as np

BASE = "cyberbattle/data/env_samples"


def folder(seed):
    pat = f"{BASE}/graphs_yprobe_n90_2026-*" if seed == 42 else f"{BASE}/graphs_yprobe_n90_s{seed}_*"
    return sorted(glob.glob(pat))[-1]


def main():
    print(f"{'seed':>5} {'N':>4} | {'KNOWS_out#1':>11} {'access_UND(bug)':>15}")
    ko, au = [], []
    for s in (42, 100, 123, 200, 300):
        m = pickle.load(open(f"{folder(s)}/1/network_SecureBERT.pkl", "rb"))
        kout = float(np.mean([d for _, d in m.knows_graph.out_degree()]))
        aund = float(np.mean([d for _, d in m.access_graph.to_undirected().degree()]))
        ko.append(kout); au.append(aund)
        print(f"{s:>5} {m.network.number_of_nodes():>4} | {kout:>11.2f} {aund:>15.2f}")
    ko, au = np.array(ko), np.array(au)
    print(f"\nKNOWS out-degree (#1, correct): mean {ko.mean():.2f}  spread {ko.min():.1f}-{ko.max():.1f}  SD {ko.std(ddof=1):.2f}")
    print(f"access undirected (the bug):    mean {au.mean():.2f}  spread {au.min():.1f}-{au.max():.1f}  SD {au.std(ddof=1):.2f}")
    print(f"\n-> previously-reported 18.9-27.4 matches the {'KNOWS-out (already correct)' if ko.min() < 19 < ko.max() and 27 < ko.max() < 28 else 'OTHER'} metric.")


if __name__ == "__main__":
    main()
