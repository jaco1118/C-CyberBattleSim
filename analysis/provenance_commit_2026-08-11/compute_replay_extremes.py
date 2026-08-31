"""Task PROVENANCE-COMMIT, Item 1: the forced-replay extremal figures (holding rate A,
conditional move probability B), from cx_step2_replay/probe/*.jsonl.

QUANTITY A -- holding rate: fraction of membership_leave events where the departing node held a
coordinate extreme (>=1 of 64 dims) in at least one of max/min. Computed under TWO denominators,
since STEP 0 found most probe leave records carry no holding field at all (the node was not
discovered pre-change, so the argmax/argmin computation in _cx_replay_probe never ran for it):
  A_all        = holds / ALL membership_leave records in the probe
  A_discovered = holds / records with discovered==1 (the only records where "held" is even defined)
Reported separately for held_max, held_min, and their union -- never averaged.

QUANTITY B -- conditional move probability: among events where the departing node held an extreme,
the fraction where the corresponding pooling slice actually moved. The probe records no "moved"
flag (STEP 0, Q1.2) -- this is filled by an explicitly authorised join to cx_step2_replay's own
drift_<band>.csv on (run_id, scenario, episode, step), using change_drift_max/change_drift_min as
the moved signal (moved = value != 0.0 exactly, no tolerance; NaN counted and excluded from both
numerator and denominator). held_max is matched only to change_drift_max, held_min only to
change_drift_min -- never cross-matched.

Both A and B are also reported with and without batch events (n_touched > 1), since the probe only
ever inspects node_ids[0] for a batch leave -- a real methodological gap, not folded silently into
the headline figure.

No tolerance anywhere. No epsilon floor. No new data source beyond the probe and the one
explicitly authorised drift-CSV join.
"""
import json
import glob
import os
import numpy as np
import pandas as pd

AG = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents"
PROBE_DIR = os.path.join(AG, "cx_step2_replay", "probe")
DRIFT_DIR = os.path.join(AG, "cx_step2_replay")
OUT_DIR = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/analysis/provenance_commit_2026-08-11"
BANDS = ["10-15", "30-40", "80-100"]

TARGET_A = {"10-15": 0.89, "30-40": 0.61, "80-100": 0.34}


def load_probe_leaves(band):
    recs = []
    for f in sorted(glob.glob(os.path.join(PROBE_DIR, f"probe_{band}_seed*_*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r["change_type"] == "membership_leave":
                    recs.append(r)
    return recs


def held_union(r):
    return bool(r.get("departing_held_max")) or bool(r.get("departing_held_min"))


def rate_A(recs, key):
    """key: 'departing_held_max', 'departing_held_min', or 'union'."""
    n_all = len(recs)
    n_disc = sum(1 for r in recs if r.get("discovered") == 1)
    if key == "union":
        n_hold = sum(1 for r in recs if held_union(r))
    else:
        n_hold = sum(1 for r in recs if bool(r.get(key)))
    a_all = n_hold / n_all if n_all else float("nan")
    a_disc = n_hold / n_disc if n_disc else float("nan")
    return dict(n_all=n_all, n_discovered=n_disc, n_hold=n_hold, A_all=a_all, A_discovered=a_disc)


def main():
    print("=== ITEM 1: forced-replay extremal figures ===\n")
    a_rows = []
    b_rows = []
    join_rows = []

    for band in BANDS:
        recs = load_probe_leaves(band)
        n_batch = sum(1 for r in recs if r.get("n_touched", 1) > 1)
        single = [r for r in recs if r.get("n_touched", 1) == 1]

        print(f"--- band {band} ---")
        print(f"  probe leave records: total={len(recs)}  batch(n_touched>1)={n_batch}  single-node={len(single)}")

        # ===== QUANTITY A =====
        for label, pop in [("including batch", recs), ("excluding batch", single)]:
            for key in ["departing_held_max", "departing_held_min", "union"]:
                res = rate_A(pop, key)
                row = dict(band=band, population=label, slice=key, **res)
                a_rows.append(row)
                print(f"  A [{label}] {key}: n_all={res['n_all']} n_discovered={res['n_discovered']} "
                      f"n_hold={res['n_hold']}  A_all={res['A_all']:.4f}  A_discovered={res['A_discovered']:.4f}")

        target = TARGET_A[band]
        a_disc_union = rate_A(recs, "union")["A_discovered"]
        a_all_union = rate_A(recs, "union")["A_all"]
        print(f"  target A={target}  A_all(union)={a_all_union:.4f} (diff {a_all_union-target:+.4f})  "
              f"A_discovered(union)={a_disc_union:.4f} (diff {a_disc_union-target:+.4f})")

        batch_diff_all = abs(rate_A(recs, "union")["A_all"] - rate_A(single, "union")["A_all"])
        batch_diff_disc = abs(rate_A(recs, "union")["A_discovered"] - rate_A(single, "union")["A_discovered"])
        print(f"  batch-inclusion difference: A_all delta={batch_diff_all:.4f}  A_discovered delta={batch_diff_disc:.4f}")

        # ===== QUANTITY B: join to drift CSV =====
        drift = pd.read_csv(os.path.join(DRIFT_DIR, f"drift_{band}.csv"),
                             usecols=["run_id", "scenario_id", "episode", "step", "change_drift_max", "change_drift_min"],
                             low_memory=False)
        drift["scenario_id"] = drift["scenario_id"].astype(str)
        drift_key = drift.set_index(["run_id", "scenario_id", "episode", "step"])
        drift_key = drift_key[~drift_key.index.duplicated(keep="first")]

        n_probe_in = len(recs)
        n_drift_in = len(drift)
        matched = 0
        unmatched_probe = 0
        move_counts = {"max": {"held": 0, "moved": 0, "nan": 0}, "min": {"held": 0, "moved": 0, "nan": 0}}

        for r in recs:
            k = (r["run_id"], str(r["scenario"]), r["episode"], r["step"])
            if k not in drift_key.index:
                unmatched_probe += 1
                continue
            matched += 1
            row = drift_key.loc[k]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            for slicekey, heldkey, driftcol in [("max", "departing_held_max", "change_drift_max"),
                                                  ("min", "departing_held_min", "change_drift_min")]:
                if bool(r.get(heldkey)):
                    move_counts[slicekey]["held"] += 1
                    val = row[driftcol]
                    if pd.isna(val):
                        move_counts[slicekey]["nan"] += 1
                    elif float(val) != 0.0:
                        move_counts[slicekey]["moved"] += 1

        unmatched_rate = unmatched_probe / n_probe_in if n_probe_in else float("nan")
        print(f"  JOIN: probe_records_in={n_probe_in}  drift_rows_in={n_drift_in}  "
              f"matched={matched}  unmatched={unmatched_probe} ({unmatched_rate*100:.2f}%)")
        join_rows.append(dict(band=band, probe_records_in=n_probe_in, drift_rows_in=n_drift_in,
                               matched=matched, unmatched=unmatched_probe, unmatched_rate=unmatched_rate))

        if unmatched_rate > 0.02:
            print(f"  *** unmatched rate exceeds 2% -- STOPPING per instruction, key likely wrong ***")
            b_rows.append(dict(band=band, slice="max", STOPPED=True))
            b_rows.append(dict(band=band, slice="min", STOPPED=True))
            continue

        for slicekey in ["max", "min"]:
            c = move_counts[slicekey]
            n_eval = c["held"] - c["nan"]
            b_val = c["moved"] / n_eval if n_eval else float("nan")
            print(f"  B [{slicekey}]: held={c['held']}  nan_excluded={c['nan']}  moved={c['moved']}  "
                  f"n_evaluated={n_eval}  B={b_val:.4f}")
            b_rows.append(dict(band=band, slice=slicekey, held=c["held"], nan_excluded=c["nan"],
                                moved=c["moved"], n_evaluated=n_eval, B=b_val))
        print()

    pd.DataFrame(a_rows).to_csv(os.path.join(OUT_DIR, "item1_quantity_A.csv"), index=False)
    pd.DataFrame(b_rows).to_csv(os.path.join(OUT_DIR, "item1_quantity_B.csv"), index=False)
    pd.DataFrame(join_rows).to_csv(os.path.join(OUT_DIR, "item1_join_diagnostics.csv"), index=False)
    print("Done.")


if __name__ == "__main__":
    main()
