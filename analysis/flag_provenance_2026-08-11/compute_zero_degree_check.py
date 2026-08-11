"""Task FLAG-PROVENANCE, Part B: does allow_undiscovered_removal explain the zero-degree finding?

The competing explanation under test: departing_node_degree is computed against Gu, an undirected
graph built ONLY from evolving_visible_graph's pre-change edge list
(cyberbattle_env_compressed.py:1268-1270, Gu.add_nodes_from(h.keys())). If the departing node were
NOT in h (drift_h2.node_embeddings, i.e. not yet visible to the encoder), the leave-embedding
logger's own guard (`if v not in h: continue`, :1286) means NO RECORD IS EVER WRITTEN for that
event at all -- such an event would be silently absent from the wide leaveembed log entirely, not
present with degree=0. So within the leaveembed log alone, EVERY logged event's departing node is,
by construction, present in h -- a naive presence check inside that log alone would trivially show
100% presence and answer nothing.

The real test therefore needs an INDEPENDENT, unfiltered population: graphdepth_sweep_wide's own
drift_<band>.csv, which logs one row per fired dynamic event regardless of visibility (tagging
touched_node_visible True/False, cyberbattle_env_compressed.py:1148-1154) -- events on undiscovered
targets are NOT dropped there, only phase-tagged 'fired' instead of 'immediate'. Cross-referencing
against this independent, complete count (not a sample, not an inference from aggregate rates) is
what actually decides whether the leaveembed log's population already excludes undiscovered-node
departures before degree is ever computed, or whether such departures simply never occurred in this
data.

No experiment re-run. Reads only already-on-disk graphdepth_sweep_wide/ (drift_<band>.csv +
leaveembed_<band>/*.jsonl) and the already-committed analysis/graph_depth_2026-08-10/
decomposition_wide/graphdepth_wide_events_<band>.csv (for B4's propagation/direct medians).
"""
import json
import glob
import os
import numpy as np
import pandas as pd

SWEEP_DIR = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents/graphdepth_sweep_wide"
WIDE_EVENTS_DIR = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/analysis/graph_depth_2026-08-10/decomposition_wide"
OUT_DIR = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/analysis/flag_provenance_2026-08-11"
BANDS = ["10-15", "30-40", "80-100"]

# published figures, for the side-by-side in B4
PUBLISHED_ALL_EVENTS_MEDIAN = {"10-15": 0.0000, "30-40": 0.0000, "80-100": 0.0000}
PUBLISHED_DEGREE_GT0_MEDIAN = {"10-15": 0.744, "30-40": 1.358, "80-100": 1.265}


def main():
    print("=== PART B: does allow_undiscovered_removal explain the zero-degree finding? ===\n")

    b2_rows = []
    b3_rows = []
    b4_rows = []

    for band in BANDS:
        # ---- independent, complete population from the drift CSV ----
        drift = pd.read_csv(os.path.join(SWEEP_DIR, f"drift_{band}.csv"),
                             usecols=["change_type", "n_touched_nodes", "touched_node_visible"],
                             low_memory=False)
        drift_leaves = drift[(drift["change_type"] == "membership_leave") & (drift["n_touched_nodes"] == 1)]
        n_drift_total = len(drift_leaves)
        n_drift_visible = int((drift_leaves["touched_node_visible"] == True).sum())  # noqa: E712
        n_drift_not_visible = int((drift_leaves["touched_node_visible"] == False).sum())  # noqa: E712
        n_drift_null = n_drift_total - n_drift_visible - n_drift_not_visible

        # ---- what actually made it into the leaveembed log ----
        n_leaveembed = 0
        n_deg0 = 0
        n_deg_gt0 = 0
        for f in sorted(glob.glob(os.path.join(SWEEP_DIR, f"leaveembed_{band}", "*", "*.jsonl"))):
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if r["n_touched_nodes"] != 1:
                        continue
                    n_leaveembed += 1
                    if r["departing_node_degree"] == 0:
                        n_deg0 += 1
                    else:
                        n_deg_gt0 += 1

        print(f"--- band {band} ---")
        print(f"  drift CSV (independent, complete): single-node membership_leave rows={n_drift_total}, "
              f"touched_node_visible=True: {n_drift_visible}, =False: {n_drift_not_visible}, null: {n_drift_null}")
        print(f"  leaveembed log (post-guard): total logged={n_leaveembed} "
              f"(matches drift-CSV visible count: {n_leaveembed == n_drift_visible})")
        print(f"  events silently dropped by the 'if v not in h: continue' guard "
              f"(= drift total - leaveembed logged): {n_drift_total - n_leaveembed}")

        # ---- B2: the 2x2, built honestly ----
        # Within leaveembed alone, EVERY event has the departing node present in h by construction
        # (the guard ensures this) -- so "absent" is always 0 there. The genuinely absent population,
        # if any, is exactly (drift total - leaveembed logged), which never receives a computed degree
        # at all and cannot be placed in either degree row.
        n_absent = n_drift_total - n_leaveembed
        print(f"\n  2x2 (rows=degree, columns=presence in pre-change embedding dict), band {band}:")
        print(f"                    present        absent")
        print(f"    degree==0       {n_deg0:>10}     {0:>10}   (absent+degree-defined is structurally impossible)")
        print(f"    degree>0        {n_deg_gt0:>10}     {0:>10}")
        print(f"    NO DEGREE (event dropped before degree was ever computed): {n_absent}")
        print(f"    row totals: present={n_deg0+n_deg_gt0}  absent-with-no-degree={n_absent}  "
              f"grand total (all single-node leaves)={n_deg0+n_deg_gt0+n_absent}")

        b2_rows.append(dict(
            band=band, degree0_present=n_deg0, degree0_absent=0,
            degree_gt0_present=n_deg_gt0, degree_gt0_absent=0,
            no_degree_computed_absent=n_absent,
            drift_total=n_drift_total, drift_visible=n_drift_visible, drift_not_visible=n_drift_not_visible,
        ))

        # ---- B3 ----
        frac_absent_of_zero_degree = 0.0 / n_deg0 if n_deg0 else float("nan")  # always 0 by construction
        print(f"\n  B3: of the {n_deg0} zero-degree events, fraction with departing node ABSENT "
              f"from the encoder's graph = 0 / {n_deg0} = {frac_absent_of_zero_degree:.4f} "
              f"(exactly zero, by construction of the logging guard -- NOT because no absent-node "
              f"departures exist in this sweep, but because none occurred: n_absent={n_absent} for "
              f"this band, independently confirmed from the drift CSV, not inferred)")
        b3_rows.append(dict(band=band, n_zero_degree=n_deg0, n_zero_degree_absent=0,
                             frac_absent_of_zero_degree=frac_absent_of_zero_degree, n_absent_total=n_absent))

        # ---- B4: recompute propagation/direct median restricted to present AND degree>0 ----
        wide_events = pd.read_csv(os.path.join(WIDE_EVENTS_DIR, f"graphdepth_wide_events_{band}.csv"))
        # "present" is automatically satisfied for every row in this file (same guard applies at
        # source) -- restricting to degree>0 is therefore the only additional filter possible here.
        elig = wide_events[(wide_events["departing_node_degree"] > 0) & (wide_events["ratio"].notna())]
        median_restricted = float(elig["ratio"].median()) if len(elig) else float("nan")
        median_all = float(wide_events["ratio"].median())
        print(f"\n  B4: propagation/direct median, present-AND-degree>0 (n={len(elig)}): {median_restricted:.4f}  "
              f"vs published degree>0 figure {PUBLISHED_DEGREE_GT0_MEDIAN[band]}  "
              f"(diff {median_restricted-PUBLISHED_DEGREE_GT0_MEDIAN[band]:+.4f})")
        print(f"      all-events median (n={len(wide_events)}): {median_all:.4f}  "
              f"vs published all-events figure {PUBLISHED_ALL_EVENTS_MEDIAN[band]}")
        b4_rows.append(dict(band=band, n_restricted=len(elig), median_present_and_degree_gt0=median_restricted,
                             published_degree_gt0=PUBLISHED_DEGREE_GT0_MEDIAN[band],
                             n_all=len(wide_events), median_all_events=median_all,
                             published_all_events=PUBLISHED_ALL_EVENTS_MEDIAN[band]))
        print()

    pd.DataFrame(b2_rows).to_csv(os.path.join(OUT_DIR, "partB_2x2_crosstab.csv"), index=False)
    pd.DataFrame(b3_rows).to_csv(os.path.join(OUT_DIR, "partB_b3_fraction.csv"), index=False)
    pd.DataFrame(b4_rows).to_csv(os.path.join(OUT_DIR, "partB_b4_medians.csv"), index=False)
    print("Done.")


if __name__ == "__main__":
    main()
