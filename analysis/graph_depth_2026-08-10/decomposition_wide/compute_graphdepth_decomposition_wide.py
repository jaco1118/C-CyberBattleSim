"""Task GRAPH-DEPTH-WIDE STEP 4: re-run the DIRECT/PROPAGATION decomposition on the WIDENED
leave-embedding logging (every present node's embedding, not the 2-hop-restricted subset --
cyberbattle_env_compressed.py commit 1522b71). A separate script from, and does not modify,
analysis/graph_depth_2026-08-10/compute_graphdepth_decomposition.py or its committed output; the
band 10-15 result that script produced (n=142, median ratio 2.33) stands and remains reproducible.

Conventions (unchanged from the task spec):
    DIRECT      = norm( (hbar - h_v) / max(1, N - 1) )
    PROPAGATION = norm( mean_{n in ALL survivors}(hp[n] - h[n]) )   -- no longer needs the
                  "3+-hop delta is zero" argument: every survivor's shift is now logged directly.
    If N - 1 == 0, emit no ratio for that event; count it separately. No denominator is
    epsilon-floored anywhere -- an exact-zero DIRECT (only possible for N>1 by genuine coincidence)
    is also excluded from ratio stats and counted separately, not floored.
"""
import json
import glob
import os
import statistics as st
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

SWEEP_DIR = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents/graphdepth_sweep_wide"
OUT_DIR = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/analysis/graph_depth_2026-08-10/decomposition_wide"
BANDS = ["10-15", "30-40", "80-100"]
PREV_10_15_MEDIAN_RATIO = 2.328   # from compute_graphdepth_decomposition.py's committed output, n=142
PREV_10_15_N = 142


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


def process_band(band):
    recs, nfiles = load_leaveembed(band)
    n_total = len(recs)
    single = [r for r in recs if r["n_touched_nodes"] == 1]
    n_batch_excluded = n_total - len(single)

    # STEP 4.2: coverage-gate check under the widened design -- pre/post should now equal the
    # FULL node set unconditionally; report how many single-node events fail that, and why.
    n_coverage_fail = 0
    coverage_fail_reasons = Counter()

    rows = []
    n_no_survivors = 0
    n_direct_zero_Ngt1 = 0
    n_Nle1 = 0

    for r in single:
        N = r["N"]
        total_survivors = r["total_survivors"]
        h_v = np.array(r["h_v"], dtype=np.float64)
        pre = {k: np.array(v, dtype=np.float64) for k, v in r["pre_embeddings"].items()}
        post = {k: np.array(v, dtype=np.float64) for k, v in r["post_embeddings"].items()}

        # coverage check: under the widened logger, pre should be exactly N-1 entries (every other
        # pre-removal node) and post should be exactly total_survivors entries (every survivor).
        ok_pre = (len(pre) == max(N - 1, 0))
        ok_post = (len(post) == max(total_survivors, 0))
        if not (ok_pre and ok_post):
            n_coverage_fail += 1
            if not ok_pre:
                coverage_fail_reasons["pre_embeddings count != N-1"] += 1
            if not ok_post:
                coverage_fail_reasons["post_embeddings count != total_survivors"] += 1

        surv = [k for k in post if k in pre]
        has_survivors = total_survivors > 0 and len(surv) > 0
        if not has_survivors:
            n_no_survivors += 1

        prop_norm = np.nan
        if has_survivors:
            dh_sum = np.array([post[k] - pre[k] for k in surv]).sum(axis=0)
            prop_norm = float(np.linalg.norm(dh_sum / total_survivors))

        hbar = (h_v + sum(pre.values())) / N if N > 0 else h_v
        denom = max(1, N - 1)
        direct_vec = (hbar - h_v) / denom
        direct_norm = float(np.linalg.norm(direct_vec))

        eligible_for_ratio = has_survivors
        if N <= 1:
            n_Nle1 += 1
            eligible_for_ratio = False
        elif direct_norm == 0.0:
            n_direct_zero_Ngt1 += 1
            eligible_for_ratio = False

        ratio = (prop_norm / direct_norm) if (eligible_for_ratio and direct_norm > 0) else np.nan

        rows.append(dict(
            run_id=r["run_id"], seed=r["seed"], scenario_id=r["scenario_id"], episode=r["episode"],
            step=r["step"], departing_node=r["departing_node"],
            N=N, total_survivors=total_survivors,
            departing_node_degree=r.get("departing_node_degree", None),
            prop_norm=prop_norm, direct_norm=direct_norm, ratio=ratio,
            eligible_for_ratio=eligible_for_ratio,
            hop_distance=r.get("hop_distance", {}),
            pre_keys=list(pre.keys()), post_keys=list(post.keys()),
        ))

    df = pd.DataFrame(rows)
    return dict(
        band=band, nfiles=nfiles, n_total=n_total, n_batch_excluded=n_batch_excluded,
        n_single=len(single), n_coverage_fail=n_coverage_fail, coverage_fail_reasons=coverage_fail_reasons,
        n_no_survivors=n_no_survivors, n_Nle1=n_Nle1, n_direct_zero_Ngt1=n_direct_zero_Ngt1,
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
              f"single-node={res['n_single']}")
        print(f"  coverage-gate check (pre==N-1 and post==total_survivors, unconditional under widened logging): "
              f"fail={res['n_coverage_fail']}/{res['n_single']}  reasons={dict(res['coverage_fail_reasons'])}")
        print(f"  exclusions from ratio stats: no_survivors(total_survivors<=0 or empty overlap)={res['n_no_survivors']} "
              f"N<=1={res['n_Nle1']} direct==0-but-N>1={res['n_direct_zero_Ngt1']}")

        elig = df[df["eligible_for_ratio"]]
        print(f"  used for ratio: n={len(elig)}/{len(df)}")
        if len(elig):
            eps = set(zip(elig["seed"], elig["scenario_id"], elig["episode"]))
            seeds = Counter(elig["seed"])
            print(f"  distinct (seed,scenario_id,episode) episodes = {len(eps)}; distinct seeds = {len(seeds)}; per-seed counts = {dict(seeds)}")
            print(f"  PROPAGATION: median={elig['prop_norm'].median():.4f} mean={elig['prop_norm'].mean():.4f}")
            print(f"  DIRECT:      median={elig['direct_norm'].median():.4f} mean={elig['direct_norm'].mean():.4f}")
            print(f"  ratio (prop/direct): median={elig['ratio'].median():.4f} mean={elig['ratio'].mean():.4f} "
                  f"frac(prop>direct)={float((elig['ratio']>1).mean()):.4f}")
            print(f"  median N: used-for-ratio={elig['N'].median()}  ALL single-node events in band={df['N'].median()}")
        else:
            print("  no events eligible for ratio in this band")

        if band == "10-15" and len(elig):
            print(f"  >>> band 10-15 comparison: WIDE median ratio={elig['ratio'].median():.4f} (n={len(elig)}) "
                  f"vs previous 2-hop-restricted median ratio={PREV_10_15_MEDIAN_RATIO} (n={PREV_10_15_N})")

    # depth/degree distribution -- needs raw pre/post vectors, so recompute directly from the raw
    # JSONL here rather than from the summary df (which drops the embedding arrays to keep memory
    # reasonable across all bands).
    print("\n### depth/degree distribution (per band) ###")
    depth_summary_rows = []
    for band in BANDS:
        recs, _ = load_leaveembed(band)
        single = [r for r in recs if r["n_touched_nodes"] == 1]
        degs = [r.get("departing_node_degree") for r in single if r.get("departing_node_degree") is not None]
        by_hop_norms = defaultdict(list)
        n_unreachable = 0
        n_survivor_total = 0
        for r in single:
            pre = r["pre_embeddings"]; post = r["post_embeddings"]; hop_distance = r["hop_distance"]
            surv = [k for k in post if k in pre]
            for k in surv:
                n_survivor_total += 1
                d = hop_distance.get(k)
                shift = float(np.linalg.norm(np.array(post[k], dtype=np.float64) - np.array(pre[k], dtype=np.float64)))
                if d is None:
                    n_unreachable += 1
                    by_hop_norms["unreachable"].append(shift)
                else:
                    by_hop_norms[int(d)].append(shift)
        print(f"  {band}: departing_node_degree median={st.median(degs) if degs else float('nan')} "
              f"mean={(sum(degs)/len(degs) if degs else float('nan')):.3f} (n={len(degs)})")
        hop_keys = sorted([k for k in by_hop_norms if isinstance(k, int)])
        line = f"    by hop: " + "  ".join(
            f"{k}hop:mean|shift|={np.mean(by_hop_norms[k]):.4f}(n={len(by_hop_norms[k])})" for k in hop_keys[:10])
        print(line)
        if "unreachable" in by_hop_norms:
            u = by_hop_norms["unreachable"]
            print(f"    unreachable-from-v survivors: n={len(u)}/{n_survivor_total} "
                  f"({len(u)/n_survivor_total*100:.2f}%) mean|shift|={np.mean(u):.4f}")
        ge3 = [m for k in hop_keys if k >= 3 for m in by_hop_norms[k]]
        print(f"    >=3hop survivors: n={len(ge3)}  mean|shift|={ (np.mean(ge3) if ge3 else float('nan')):.6f}  "
              f"max|shift|={ (max(ge3) if ge3 else float('nan')):.6f}")
        for k in hop_keys:
            depth_summary_rows.append(dict(band=band, hop=k, n=len(by_hop_norms[k]),
                                            mean_shift=float(np.mean(by_hop_norms[k])),
                                            median_shift=float(np.median(by_hop_norms[k]))))
        if "unreachable" in by_hop_norms:
            depth_summary_rows.append(dict(band=band, hop="unreachable", n=len(by_hop_norms["unreachable"]),
                                            mean_shift=float(np.mean(by_hop_norms["unreachable"])),
                                            median_shift=float(np.median(by_hop_norms["unreachable"]))))

    pd.DataFrame(depth_summary_rows).to_csv(os.path.join(OUT_DIR, "graphdepth_wide_depth_distribution.csv"), index=False)

    # combined summary CSV
    summary_rows = []
    for band, res in band_results.items():
        df = res["df"]
        elig = df[df["eligible_for_ratio"]]
        summary_rows.append(dict(
            band=band, n_total_events=res["n_total"], n_batch_excluded=res["n_batch_excluded"],
            n_single_node=res["n_single"], n_coverage_fail=res["n_coverage_fail"],
            n_no_survivors=res["n_no_survivors"], n_Nle1=res["n_Nle1"], n_direct_zero_Ngt1=res["n_direct_zero_Ngt1"],
            n_used_for_ratio=len(elig),
            median_N_used=float(elig["N"].median()) if len(elig) else np.nan,
            median_N_all_single=float(df["N"].median()) if len(df) else np.nan,
            prop_median=float(elig["prop_norm"].median()) if len(elig) else np.nan,
            direct_median=float(elig["direct_norm"].median()) if len(elig) else np.nan,
            ratio_median=float(elig["ratio"].median()) if len(elig) else np.nan,
            ratio_mean=float(elig["ratio"].mean()) if len(elig) else np.nan,
            frac_prop_gt_direct=float((elig["ratio"] > 1).mean()) if len(elig) else np.nan,
            n_distinct_episodes=len(set(zip(elig["seed"], elig["scenario_id"], elig["episode"]))) if len(elig) else 0,
            n_distinct_seeds=elig["seed"].nunique() if len(elig) else 0,
        ))
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUT_DIR, "graphdepth_wide_summary.csv"), index=False)

    # per-event CSV, WITH identifying metadata this time (join key present)
    for band, res in band_results.items():
        out = res["df"].drop(columns=["hop_distance", "pre_keys", "post_keys"])
        out.to_csv(os.path.join(OUT_DIR, f"graphdepth_wide_events_{band}.csv"), index=False)

    print(f"\nwrote outputs to {OUT_DIR}/")


if __name__ == "__main__":
    main()
