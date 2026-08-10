"""Task GRAPH-DEPTH STEP 4: DIRECT/PROPAGATION decomposition on the REAL discovered graph at
membership_leave events, using the leave_embedding_logging side-channel (STEP 3 sweep output,
graphdepth_sweep/leaveembed_<band>/*/*.jsonl) joined against the drift CSV.

Reuses probe_p.py's definitions (commit c05a16a, analysis/recovered_scripts_2026-08-04/taskF1/
probe_p.py -- NOT modified, only read) without change:
    direct = (hbar - h[v]) / (N - 1)
    prop   = mean(hp[n] - h[n] for n in ALL surviving nodes)   <- STEP 1 authorisation Section A:
             denominator is total_survivors (ALL survivors), not the 2-hop-restricted count.

What the logged data permits exactly, and what it does not:
  - PROPAGATION's *numerator* only needs 2-hop nodes: 3+-hop shift is proven exactly 0 (STEP 0/1
    finding), so sum(hp[n]-h[n] over ALL survivors) == sum(... over 2-hop survivors) exactly.
    The corrected denominator (total_survivors, logged as its own field) makes PROPAGATION exactly
    computable for every logged event, regardless of 2-hop coverage.
  - DIRECT needs hbar = mean of h[n] over ALL N pre-removal nodes (raw values, not deltas) --
    3+-hop nodes' raw pre-embeddings were never logged (only their absence of *change* is provable,
    not their value). DIRECT is therefore only exactly recoverable from logged data on events where
    the 2-hop neighbourhood happens to cover literally every other node ("full coverage": len(
    pre_embeddings) == N-1). This restriction is reported explicitly, not silently dropped.

Restricted to n_touched_nodes==1 (single-node leave) events, mirroring the RQ2C-1 precedent
(cyberbattle_env_compressed.py's own _rq2c_leaves filter) -- batch leave events confound the
per-node decomposition, since hp already reflects every batch-mate having left too, not just v.
Batch-excluded counts are reported, not silently dropped.
"""
import json
import glob
import os
import numpy as np
import pandas as pd

SWEEP_DIR = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents/graphdepth_sweep"
OUT_DIR = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/analysis/graph_depth_2026-08-10"
BANDS = ["10-15", "30-40", "80-100"]
HBAR_TOL = 1e-4  # floating-point tolerance for the hbar cross-check (float32 embeddings -> float64 sums)


def load_leaveembed(band):
    files = sorted(glob.glob(os.path.join(SWEEP_DIR, f"leaveembed_{band}", "*", "*.jsonl")))
    recs = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return recs, len(files)


def load_drift_lookup(band):
    df = pd.read_csv(
        os.path.join(SWEEP_DIR, f"drift_{band}.csv"),
        usecols=["run_id", "seed", "scenario_id", "episode", "step", "norm_h2_mean", "n_discovered_h2", "n_discovered_h3"],
    )
    df = df.drop_duplicates(subset=["run_id", "seed", "scenario_id", "episode", "step"], keep="first")
    lookup = {}
    for row in df.itertuples(index=False):
        key = (row.run_id, int(row.seed), str(row.scenario_id), int(row.episode), int(row.step))
        lookup[key] = (row.norm_h2_mean, row.n_discovered_h2, row.n_discovered_h3)
    return lookup


def process_band(band):
    recs, nfiles = load_leaveembed(band)
    lookup = load_drift_lookup(band)

    n_total = len(recs)
    n_batch_excluded = sum(1 for r in recs if r["n_touched_nodes"] != 1)
    single = [r for r in recs if r["n_touched_nodes"] == 1]

    n_zero_2hop = 0
    rows = []
    n_joined = 0
    n_disc_h2_mismatch = 0
    n_disc_h3_mismatch = 0
    n_disc_checked = 0

    for r in single:
        N = r["N"]
        total_survivors = r["total_survivors"]
        two_hop_survivors = r["two_hop_survivors"]
        h_v = np.array(r["h_v"], dtype=np.float64)
        pre = {k: np.array(v, dtype=np.float64) for k, v in r["pre_embeddings"].items()}
        post = {k: np.array(v, dtype=np.float64) for k, v in r["post_embeddings"].items()}

        surv2 = [k for k in pre if k in post]
        if total_survivors <= 0 or two_hop_survivors <= 0 or not surv2:
            n_zero_2hop += 1
            continue

        dh_sum = np.array([post[k] - pre[k] for k in surv2]).sum(axis=0)
        prop_allsurv = dh_sum / total_survivors
        prop_2hoponly = dh_sum / two_hop_survivors
        prop_allsurv_norm = float(np.linalg.norm(prop_allsurv))
        prop_2hoponly_norm = float(np.linalg.norm(prop_2hoponly))
        denom_ratio = total_survivors / two_hop_survivors

        full_coverage = (N > 1) and (len(pre) == N - 1)
        direct_norm = np.nan
        hbar_check_diff = np.nan
        key = (r["run_id"], int(r["seed"]), str(r["scenario_id"]), int(r["episode"]), int(r["step"]))
        drift_row = lookup.get(key)
        if drift_row is not None:
            n_joined += 1
            norm_h2_mean, n_disc_h2, n_disc_h3 = drift_row
            if not (pd.isna(n_disc_h2) or pd.isna(n_disc_h3)):
                n_disc_checked += 1
                if int(n_disc_h2) != N:
                    n_disc_h2_mismatch += 1
                if int(n_disc_h3) != total_survivors:
                    n_disc_h3_mismatch += 1

        if full_coverage:
            hbar_pre = (h_v + sum(pre.values())) / N
            direct_vec = (hbar_pre - h_v) / (N - 1)
            direct_norm = float(np.linalg.norm(direct_vec))
            if drift_row is not None and not pd.isna(drift_row[0]):
                hbar_check_diff = abs(float(np.linalg.norm(hbar_pre)) - float(drift_row[0]))

        rows.append(dict(
            N=N, total_survivors=total_survivors, two_hop_survivors=two_hop_survivors,
            coverage_frac=two_hop_survivors / total_survivors,
            full_coverage=full_coverage,
            prop_allsurv_norm=prop_allsurv_norm, prop_2hoponly_norm=prop_2hoponly_norm,
            denom_ratio=denom_ratio, direct_norm=direct_norm, hbar_check_diff=hbar_check_diff,
        ))

    df = pd.DataFrame(rows)
    return dict(
        band=band, nfiles=nfiles, n_total=n_total, n_batch_excluded=n_batch_excluded,
        n_single=len(single), n_zero_2hop=n_zero_2hop, n_used=len(df),
        n_joined=n_joined, n_disc_checked=n_disc_checked,
        n_disc_h2_mismatch=n_disc_h2_mismatch, n_disc_h3_mismatch=n_disc_h3_mismatch,
        df=df,
    )


def main():
    band_results = {}
    for band in BANDS:
        print(f"=== band {band} ===")
        res = process_band(band)
        band_results[band] = res
        df = res["df"]
        print(f"  files={res['nfiles']} n_total_events={res['n_total']} batch_excluded(n_touched!=1)={res['n_batch_excluded']} "
              f"single-node={res['n_single']} zero_2hop_neighbours={res['n_zero_2hop']} used={res['n_used']}")
        print(f"  joined to drift CSV: {res['n_joined']}/{res['n_used']}; "
              f"n_discovered cross-check: {res['n_disc_h2_mismatch']}/{res['n_disc_checked']} h2 mismatches, "
              f"{res['n_disc_h3_mismatch']}/{res['n_disc_checked']} h3 mismatches")

        print(f"  coverage_frac (two_hop_survivors/total_survivors): "
              f"median={df['coverage_frac'].median():.3f} mean={df['coverage_frac'].mean():.3f} "
              f"full_coverage_rate={df['full_coverage'].mean():.3f} ({df['full_coverage'].sum()}/{len(df)})")

        print(f"  PROPAGATION (all {res['n_used']} single-node events, exact regardless of coverage):")
        print(f"    all-survivor denom:  median={df['prop_allsurv_norm'].median():.4f} mean={df['prop_allsurv_norm'].mean():.4f}")
        print(f"    2-hop-only denom:    median={df['prop_2hoponly_norm'].median():.4f} mean={df['prop_2hoponly_norm'].mean():.4f}")
        print(f"    denom ratio (total_survivors/two_hop_survivors): median={df['denom_ratio'].median():.3f} mean={df['denom_ratio'].mean():.3f} max={df['denom_ratio'].max():.3f}")
        inflation = df["prop_2hoponly_norm"] / df["prop_allsurv_norm"].replace(0, np.nan)
        print(f"    2-hop-only/all-survivor PROPAGATION-magnitude inflation: median={inflation.median():.3f}x mean={inflation.mean():.3f}x")

        fc = df[df["full_coverage"]]
        print(f"  DIRECT (full-2-hop-coverage subset only, n={len(fc)}, {len(fc)/len(df)*100:.1f}% of used events):")
        if len(fc):
            print(f"    direct: median={fc['direct_norm'].median():.4f} mean={fc['direct_norm'].mean():.4f}")
            ratio = fc["prop_allsurv_norm"] / fc["direct_norm"].replace(0, np.nan)
            print(f"    prop/direct ratio (correct, all-survivor denom): median={ratio.median():.3f} mean={ratio.mean():.3f} frac(prop>direct)={float((ratio>1).mean()):.3f}")
            ratio_2h = fc["prop_2hoponly_norm"] / fc["direct_norm"].replace(0, np.nan)
            print(f"    prop/direct ratio (2-hop-only denom, for comparison; == correct here by construction since full coverage): median={ratio_2h.median():.3f} mean={ratio_2h.mean():.3f}")
            hb = fc["hbar_check_diff"].dropna()
            print(f"    hbar verification (||computed hbar|| vs drift CSV norm_h2_mean), n_checked={len(hb)}: "
                  f"max_abs_diff={hb.max() if len(hb) else float('nan'):.6g} all_within_tol({HBAR_TOL})={bool((hb < HBAR_TOL).all()) if len(hb) else 'n/a'}")
        else:
            print("    no full-coverage events in this band -- DIRECT not computable from logged data")

    # combined summary CSV
    summary_rows = []
    for band, res in band_results.items():
        df = res["df"]
        fc = df[df["full_coverage"]]
        summary_rows.append(dict(
            band=band, n_total_events=res["n_total"], n_batch_excluded=res["n_batch_excluded"],
            n_single_node=res["n_single"], n_zero_2hop=res["n_zero_2hop"], n_used=res["n_used"],
            full_coverage_rate=float(df["full_coverage"].mean()) if len(df) else np.nan,
            n_full_coverage=int(df["full_coverage"].sum()) if len(df) else 0,
            coverage_frac_median=float(df["coverage_frac"].median()) if len(df) else np.nan,
            prop_allsurv_median=float(df["prop_allsurv_norm"].median()) if len(df) else np.nan,
            prop_2hoponly_median=float(df["prop_2hoponly_norm"].median()) if len(df) else np.nan,
            denom_ratio_median=float(df["denom_ratio"].median()) if len(df) else np.nan,
            denom_ratio_mean=float(df["denom_ratio"].mean()) if len(df) else np.nan,
            direct_median_fullcov=float(fc["direct_norm"].median()) if len(fc) else np.nan,
            prop_direct_ratio_correct_median_fullcov=float((fc["prop_allsurv_norm"] / fc["direct_norm"].replace(0, np.nan)).median()) if len(fc) else np.nan,
            prop_direct_ratio_2hoponly_median_fullcov=float((fc["prop_2hoponly_norm"] / fc["direct_norm"].replace(0, np.nan)).median()) if len(fc) else np.nan,
            n_joined_drift_csv=res["n_joined"],
            n_disc_h2_mismatch=res["n_disc_h2_mismatch"], n_disc_h3_mismatch=res["n_disc_h3_mismatch"],
            n_disc_checked=res["n_disc_checked"],
        ))
    out_csv = os.path.join(OUT_DIR, "graphdepth_decomposition_summary.csv")
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    # per-event rows too, for anyone re-deriving
    for band, res in band_results.items():
        res["df"].to_csv(os.path.join(OUT_DIR, f"graphdepth_decomposition_events_{band}.csv"), index=False)
    print(f"wrote per-event CSVs to {OUT_DIR}/graphdepth_decomposition_events_<band>.csv")


if __name__ == "__main__":
    main()
