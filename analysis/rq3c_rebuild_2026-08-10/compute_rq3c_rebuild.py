"""Task RQ3C-REBUILD v2: implement the recovered ITEM B specification (Task CX, 2026-08-02),
preserved at analysis/recovered_stash0_cardedits_2026-08-03/stash0_evidence_card_edits.patch on
branch rq2b-10-15 (commit 1d6aaab). The compute script behind that patch is confirmed lost
(RQ1C-MECHANISM STEP 0/0B); this is a fresh implementation of the recovered specification, not a
recovered script.

RESPONSE (paired behavioural residual, one row per CX registration episode, n=4,410):
    residual = (baseline_root_owned_mean_over_cell - actual_final_root_owned_count) - root_owned_departures
  baseline = mean(final_root_owned_count) over the SAME (band,seed,scenario_id) cell's cx_step2_static episodes.
  actual / root_owned_departures = the registration episode's own event_episode.jsonl fields, directly logged
  (NOT derived).

STRUCTURE predictors, episode-level aggregates over that episode's own logged events:
  - mean_2hop: for each event with changed_node_discovered==1 (event_graph.jsonl), count of nodes within
    2 (undirected) hops of changed_node_id in that step's pre_edges graph; averaged over such events per episode.
  - mean_degree: changed_node_degree for the SAME discovered==1 events, averaged per episode. VERIFIED
    (not assumed): changed_node_degree is defined if and only if changed_node_discovered==1 (0 counter-cases
    in either direction, checked over all 192,649 events) -- so mean_2hop and mean_degree share exactly the
    same per-episode defined-event population; there is no event where one is computable and the other isn't.
  - summed_propagation: sum over event_phase=='immediate' drift-CSV rows in that episode of
    delta_h_v_norm * (1/attenuation_ratio_full - 1) [ITEM B's own formula], with attenuation_ratio_full==0
    treated as contributing exactly 0 (delta_h_v_norm is also exactly 0 on every such row, checked).

TYPE predictor: property_fraction = event_episode.jsonl's own event_counts['property'] / sum(event_counts).

EVENT-DEFINITION DISCLOSURE (a specification choice this task's authorisation left open, documented here
rather than left implicit): "each event" for summed_propagation uses event_phase=='immediate' in the drift
CSV -- exactly the convention already established in this project for the same cx_step2_registration data
(compute_rq3a_gate.py, compute_rq3b_slice.py). The drift CSV also contains 'fired' rows (pending, placeholder
zero values) and 'attributed' rows (the delayed resolution of a 'fired' event once its node is discovered) --
summing across phases would double-count the same underlying event once as a zero placeholder and once as
its resolved value, confirmed by direct inspection (matching (event_id) pairs, 'fired' phase always 0/0,
paired 'attributed' phase carrying the real value). Restricting to 'immediate' avoids this entirely, at the
cost of excluding delayed-perception events (disclosed, not hidden) -- known from this project's own history
to affect membership_join predominantly. mean_2hop/mean_degree use event_graph.jsonl's own
changed_node_discovered==1 flag, the natural in-file equivalent; cross-checked against drift CSV's
'immediate' row count (band 10-15: 9236 vs 9127, 1.2% apart) -- close but not exact, reported as its own
finding rather than forced to match (Rule 8).
"""
import json
import glob
import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import brentq

AG = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents"
REG_DIR = os.path.join(AG, "cx_step2_registration")
STATIC_DIR = os.path.join(AG, "cx_step2_static")
BANDS = ["10-15", "30-40", "80-100"]
OUT_DIR = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/analysis/rq3c_rebuild_2026-08-10"
BOOT_SEED = 20260810
N_BOOT = 10000

RECOVERED = {  # ITEM B's own table, for the side-by-side reproduction check
    "structure_pooled": 0.155, "structure_10_15": 0.148, "structure_30_40": 0.365, "structure_80_100": 0.331,
    "type_pooled": 0.011, "type_10_15": 0.003, "type_30_40": 0.033, "type_80_100": 0.004,
}


# ---------- loaders ----------

def load_event_episode(base, band):
    rows = []
    for f in sorted(glob.glob(os.path.join(base, f"eventgraph_{band}", "*", "event_episode.jsonl"))):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def build_structure_from_event_graph(band):
    """Per (seed, scenario_id, episode): list of (two_hop_size, degree) for discovered==1 events."""
    per_ep = {}
    n_events_disc1 = 0
    n_events_total = 0
    for f in sorted(glob.glob(os.path.join(REG_DIR, f"eventgraph_{band}", "*", "event_graph.jsonl"))):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                key = (r["seed"], str(r["scenario_id"]), r["episode"])
                G = None
                for ev in r["events"]:
                    n_events_total += 1
                    if ev.get("changed_node_discovered") != 1:
                        continue
                    n_events_disc1 += 1
                    if G is None:
                        G = nx.Graph()
                        G.add_edges_from([tuple(e) for e in r["pre_edges"]])
                    nid = ev["changed_node_id"]
                    if nid in G:
                        hops = nx.single_source_shortest_path_length(G, nid, cutoff=2)
                        two_hop = sum(1 for d in hops.values() if d > 0)
                    else:
                        two_hop = 0  # node has no edges recorded but is "discovered" -- isolated, 0 neighbours
                    deg = ev.get("changed_node_degree")
                    per_ep.setdefault(key, []).append((two_hop, deg))
    return per_ep, n_events_total, n_events_disc1


def build_propagation_from_drift(band):
    """Per (seed, scenario_id, episode): summed propagation over event_phase=='immediate' rows."""
    usecols = ["seed", "scenario_id", "episode", "event_phase", "attenuation_ratio_full", "delta_h_v_norm"]
    df = pd.read_csv(os.path.join(REG_DIR, f"drift_{band}.csv"), usecols=usecols, low_memory=False)
    imm = df[df["event_phase"] == "immediate"].copy()
    att = imm["attenuation_ratio_full"].to_numpy()
    dhv = imm["delta_h_v_norm"].to_numpy()
    n_zero_att = int((att == 0).sum())
    n_zero_att_nonzero_dhv = int(((att == 0) & (dhv != 0)).sum())
    prop = np.where(att > 0, dhv * (1.0 / np.where(att > 0, att, 1.0) - 1.0), 0.0)
    imm["prop_event"] = prop
    imm["scenario_id"] = imm["scenario_id"].astype(str)
    grp = imm.groupby(["seed", "scenario_id", "episode"])["prop_event"].sum()
    return grp.to_dict(), len(imm), n_zero_att, n_zero_att_nonzero_dhv


# ---------- assembly ----------

def assemble(band):
    reg_ep = load_event_episode(REG_DIR, band)
    static_ep = load_event_episode(STATIC_DIR, band)

    baseline_by_cell = {}
    for r in static_ep:
        key = (r["seed"], str(r["scenario_id"]))
        baseline_by_cell.setdefault(key, []).append(r["final_root_owned_count"])
    baseline_mean = {k: float(np.mean(v)) for k, v in baseline_by_cell.items()}

    struct_events, n_ev_total, n_ev_disc1 = build_structure_from_event_graph(band)
    prop_by_ep, n_immediate_rows, n_zero_att, n_zero_att_nonzero_dhv = build_propagation_from_drift(band)

    rows = []
    n_no_static_pair = 0
    n_no_structure_events = 0
    for r in reg_ep:
        seed = r["seed"]
        sid = str(r["scenario_id"])
        ep = r["episode"]
        cell_key = (seed, sid)
        ep_key = (seed, sid, ep)

        if cell_key not in baseline_mean:
            n_no_static_pair += 1
            continue

        ec = r.get("event_counts", {})
        total_ec = sum(ec.values())
        prop_fraction = (ec.get("property", 0) / total_ec) if total_ec > 0 else np.nan

        residual = (baseline_mean[cell_key] - r["final_root_owned_count"]) - r["root_owned_departures"]

        pairs = struct_events.get(ep_key, [])
        if pairs:
            two_hops = [p[0] for p in pairs]
            degs = [p[1] for p in pairs if p[1] is not None]
            mean_2hop = float(np.mean(two_hops))
            mean_degree = float(np.mean(degs)) if degs else np.nan
        else:
            n_no_structure_events += 1
            mean_2hop = np.nan
            mean_degree = np.nan

        summed_prop = prop_by_ep.get(ep_key, 0.0)

        rows.append(dict(
            band=band, seed=seed, scenario_id=sid, episode=ep,
            residual=residual, mean_2hop=mean_2hop, mean_degree=mean_degree,
            summed_propagation=summed_prop, property_fraction=prop_fraction,
            n_structure_events=len(pairs), n_degree_defined=sum(1 for p in pairs if p[1] is not None),
        ))

    diag = dict(
        band=band, n_registration_episodes=len(reg_ep), n_no_static_pair=n_no_static_pair,
        n_no_structure_events=n_no_structure_events,
        n_event_graph_events_total=n_ev_total, n_event_graph_events_discovered1=n_ev_disc1,
        n_drift_immediate_rows=n_immediate_rows, n_zero_attenuation=n_zero_att,
        n_zero_attenuation_nonzero_dhv=n_zero_att_nonzero_dhv,
    )
    return pd.DataFrame(rows), diag


# ---------- regression helpers ----------

def r2(df, ycol, xcols):
    sub = df.dropna(subset=[ycol] + xcols)
    if len(sub) < len(xcols) + 2:
        return np.nan, len(sub)
    y = sub[ycol].to_numpy()
    X = sm.add_constant(sub[xcols].to_numpy())
    model = sm.OLS(y, X).fit()
    return float(model.rsquared), len(sub)


def detectable_r2(n, k, alpha=0.05, power=0.80):
    """Minimum detectable R^2 for a k-predictor multiple regression F-test, given n, at the stated
    alpha/power. lambda solved via the noncentral F distribution; R^2 = lambda / (lambda + n)."""
    df1, df2 = k, n - k - 1
    if df2 <= 0:
        return np.nan
    f_crit = stats.f.ppf(1 - alpha, df1, df2)

    def power_at(lam):
        return stats.ncf.sf(f_crit, df1, df2, lam) - power

    lo, hi = 1e-6, 10.0
    while power_at(hi) < 0 and hi < 1e6:
        hi *= 2
    lam = brentq(power_at, lo, hi)
    return lam / (lam + n)


# ---------- main ----------

def main():
    all_df = []
    diags = []
    for band in BANDS:
        df, diag = assemble(band)
        all_df.append(df)
        diags.append(diag)
        print(f"[{band}] assembled {len(df)} episodes "
              f"(registration={diag['n_registration_episodes']}, no_static_pair={diag['n_no_static_pair']}, "
              f"no_structure_events={diag['n_no_structure_events']})")
        print(f"        event_graph: total_events={diag['n_event_graph_events_total']} "
              f"discovered1={diag['n_event_graph_events_discovered1']}")
        print(f"        drift 'immediate' rows={diag['n_drift_immediate_rows']} "
              f"zero_attenuation={diag['n_zero_attenuation']} "
              f"(of which delta_h_v_norm also nonzero: {diag['n_zero_attenuation_nonzero_dhv']})")

    full = pd.concat(all_df, ignore_index=True)
    full.to_csv(os.path.join(OUT_DIR, "rq3c_rebuild_episodes.csv"), index=False)
    pd.DataFrame(diags).to_csv(os.path.join(OUT_DIR, "rq3c_rebuild_diagnostics.csv"), index=False)
    print(f"\npooled n={len(full)}")

    STRUCT_ALL = ["mean_degree", "mean_2hop", "summed_propagation"]
    STRUCT_NO_DEG = ["mean_2hop", "summed_propagation"]
    TYPE = ["property_fraction"]

    # ================= STEP 1: reproduce the recovered table =================
    print("\n" + "=" * 70)
    print("STEP 1 -- reproduction of the recovered table")
    print("=" * 70)
    step1_rows = []
    for label, xcols in [("STRUCTURE (deg+2hop+prop)", STRUCT_ALL), ("TYPE (property_fraction)", TYPE)]:
        r2_pooled, n_pooled = r2(full, "residual", xcols)
        row = {"predictor_set": label, "R2_pooled": r2_pooled, "n_pooled": n_pooled}
        for band in BANDS:
            r2_b, n_b = r2(full[full["band"] == band], "residual", xcols)
            row[f"R2_{band}"] = r2_b
            row[f"n_{band}"] = n_b
        step1_rows.append(row)
        print(f"{label}: pooled R2={r2_pooled:.4f} (n={n_pooled})  "
              + "  ".join(f"{b}={row[f'R2_{b}']:.4f}(n={row[f'n_{b}']})" for b in BANDS))
    step1_df = pd.DataFrame(step1_rows)
    step1_df.to_csv(os.path.join(OUT_DIR, "rq3c_step1_reproduction.csv"), index=False)

    print("\nReproduction check against the recovered table (no adjustment):")
    s = step1_df.iloc[0]
    t = step1_df.iloc[1]
    checks = [
        ("structure_pooled", s["R2_pooled"]), ("structure_10_15", s["R2_10-15"]),
        ("structure_30_40", s["R2_30-40"]), ("structure_80_100", s["R2_80-100"]),
        ("type_pooled", t["R2_pooled"]), ("type_10_15", t["R2_10-15"]),
        ("type_30_40", t["R2_30-40"]), ("type_80_100", t["R2_80-100"]),
    ]
    for name, got in checks:
        rec = RECOVERED[name]
        print(f"  {name}: recovered={rec}  rebuilt={got:.4f}  "
              f"{'MATCH' if abs(rec-got) < 0.01 else 'DIFFERS'} (diff={got-rec:+.4f})")

    # ================= SECTION A: degree nulls, three fits =================
    print("\n" + "=" * 70)
    print("SECTION A -- degree nulls, three fits")
    print("=" * 70)
    full["degree_defined"] = full["mean_degree"].notna()
    a2_rows = []
    for band in BANDS + ["pooled"]:
        sub = full if band == "pooled" else full[full["band"] == band]
        n_no_degree = int((~sub["degree_defined"]).sum())
        a2_rows.append({"band": band, "n_total": len(sub), "n_no_defined_degree": n_no_degree})
        print(f"  {band}: n_total={len(sub)}  episodes with NO defined degree at all={n_no_degree}")
    pd.DataFrame(a2_rows).to_csv(os.path.join(OUT_DIR, "rq3c_sectionA_degree_nulls.csv"), index=False)

    subset = full[full["degree_defined"]]
    a3_rows = []
    print(f"\nA3 three fits (degree-defined subset n={len(subset)} vs full n={len(full)}):")
    for fit_label, data, xcols in [
        ("(i) STRUCTURE with degree, degree-defined subset", subset, STRUCT_ALL),
        ("(ii) STRUCTURE without degree, SAME subset", subset, STRUCT_NO_DEG),
        ("(iii) STRUCTURE without degree, full 4410-equivalent", full, STRUCT_NO_DEG),
    ]:
        row = {"fit": fit_label}
        r2p, np_ = r2(data, "residual", xcols)
        row["R2_pooled"] = r2p
        row["n_pooled"] = np_
        for band in BANDS:
            r2b, nb = r2(data[data["band"] == band], "residual", xcols)
            row[f"R2_{band}"] = r2b
            row[f"n_{band}"] = nb
        a3_rows.append(row)
        print(f"  {fit_label}: pooled R2={r2p:.4f} (n={np_})  "
              + "  ".join(f"{b}={row[f'R2_{b}']:.4f}(n={row[f'n_{b}']})" for b in BANDS))
    # TYPE on the same three populations
    for fit_label, data in [
        ("TYPE, degree-defined subset", subset),
        ("TYPE, full population", full),
    ]:
        row = {"fit": fit_label}
        r2p, np_ = r2(data, "residual", TYPE)
        row["R2_pooled"] = r2p
        row["n_pooled"] = np_
        for band in BANDS:
            r2b, nb = r2(data[data["band"] == band], "residual", TYPE)
            row[f"R2_{band}"] = r2b
            row[f"n_{band}"] = nb
        a3_rows.append(row)
        print(f"  {fit_label}: pooled R2={r2p:.4f} (n={np_})  "
              + "  ".join(f"{b}={row[f'R2_{b}']:.4f}(n={row[f'n_{b}']})" for b in BANDS))
    pd.DataFrame(a3_rows).to_csv(os.path.join(OUT_DIR, "rq3c_sectionA_three_fits.csv"), index=False)

    n_no_degree_pooled = int((~full["degree_defined"]).sum())
    print(f"\nA4: recovered STRUCTURE reports pooled n=4410 (the FULL population). "
          f"Episodes lacking a defined degree, pooled: {n_no_degree_pooled}. "
          f"{'Fit (ii) and (iii) are therefore identical by construction.' if n_no_degree_pooled == 0 else 'Fits (ii)/(iii) differ.'}")

    # ================= STEP 2: three discrepancies =================
    print("\n" + "=" * 70)
    print("STEP 2 -- three discrepancies")
    print("=" * 70)

    # 2a: with vs without degree -- already covered by A3 (i) vs (ii); restate explicitly here
    print("2a. STRUCTURE with vs without degree (see Section A fits (i) and (ii) above).")

    # 2b: degree alone vs detectability floor
    print("\n2b. The two 0.002s:")
    deg_alone_rows = []
    r2p, np_ = r2(subset, "residual", ["mean_degree"])
    deg_alone_rows.append({"quantity": "R2 of degree alone", "band": "pooled", "value": r2p, "n": np_})
    print(f"  R2 of degree alone, pooled: {r2p:.4f} (n={np_})")
    for band in BANDS:
        r2b, nb = r2(subset[subset["band"] == band], "residual", ["mean_degree"])
        deg_alone_rows.append({"quantity": "R2 of degree alone", "band": band, "value": r2b, "n": nb})
        print(f"  R2 of degree alone, {band}: {r2b:.4f} (n={nb})")

    print()
    for k, label in [(3, "STRUCTURE (k=3)"), (1, "single predictor (k=1)")]:
        floor_pooled = detectable_r2(len(full), k)
        deg_alone_rows.append({"quantity": f"detectable R2 floor {label}", "band": "pooled", "value": floor_pooled, "n": len(full)})
        print(f"  detectable R2 floor, {label}, pooled n={len(full)}: {floor_pooled:.5f}")
        for band in BANDS:
            nb = len(full[full["band"] == band])
            floor_b = detectable_r2(nb, k)
            deg_alone_rows.append({"quantity": f"detectable R2 floor {label}", "band": band, "value": floor_b, "n": nb})
            print(f"    {band} (n={nb}): {floor_b:.5f}")
    pd.DataFrame(deg_alone_rows).to_csv(os.path.join(OUT_DIR, "rq3c_step2b_degree_vs_floor.csv"), index=False)

    # 2c: episode-level vs cell-level (117 cells)
    print("\n2c. Episode-level vs 117-cell-level fit:")
    cell_agg = full.groupby(["band", "seed", "scenario_id"], as_index=False).agg(
        residual=("residual", "mean"), mean_2hop=("mean_2hop", "mean"),
        mean_degree=("mean_degree", "mean"), summed_propagation=("summed_propagation", "mean"),
        property_fraction=("property_fraction", "mean"),
    )
    cell_agg.to_csv(os.path.join(OUT_DIR, "rq3c_cell_level_117.csv"), index=False)
    print(f"  cell-level n={len(cell_agg)} (expect 117)")

    step2c_rows = []
    for label, xcols in [("STRUCTURE", STRUCT_ALL), ("TYPE", TYPE)]:
        ep_r2, ep_n = r2(full, "residual", xcols)
        cell_r2, cell_n = r2(cell_agg, "residual", xcols)
        step2c_rows.append({"predictor_set": label, "R2_episode_level": ep_r2, "n_episode": ep_n,
                             "R2_cell_level": cell_r2, "n_cell": cell_n, "gap": ep_r2 - cell_r2})
        print(f"  {label}: episode-level R2={ep_r2:.4f} (n={ep_n})  vs  cell-level R2={cell_r2:.4f} (n={cell_n})  "
              f"gap={ep_r2-cell_r2:+.4f}")
    pd.DataFrame(step2c_rows).to_csv(os.path.join(OUT_DIR, "rq3c_step2c_episode_vs_cell.csv"), index=False)

    floor_4410 = detectable_r2(4410, 3)
    floor_117 = detectable_r2(117, 3)
    print(f"\n  detectable-R2 floor at n=4410 (treating episodes as independent): {floor_4410:.5f}")
    print(f"  detectable-R2 floor at n=117 (cells): {floor_117:.5f}")
    print(f"  ratio (117-floor / 4410-floor): {floor_117/floor_4410:.1f}x -- "
          f"a floor computed at n=4410 is this many times too optimistic if episodes are not independent.")

    # ================= STEP 3: cell-level bootstrap =================
    print("\n" + "=" * 70)
    print("STEP 3 -- cell-level bootstrap (10,000 resamples, seed={})".format(BOOT_SEED))
    print("=" * 70)
    rng = np.random.default_rng(BOOT_SEED)
    n_cells = len(cell_agg)
    boot_results = {"structure_pooled": [], "type_pooled": [], "ratio": []}
    for band in BANDS:
        boot_results[f"structure_{band}"] = []
        boot_results[f"type_{band}"] = []

    for _ in range(N_BOOT):
        idx = rng.integers(0, n_cells, n_cells)
        sample = cell_agg.iloc[idx]
        rs, _ = r2(sample, "residual", STRUCT_ALL)
        rt, _ = r2(sample, "residual", TYPE)
        boot_results["structure_pooled"].append(rs)
        boot_results["type_pooled"].append(rt)
        boot_results["ratio"].append(rs / rt if rt and rt > 0 else np.nan)
        for band in BANDS:
            bsub = sample[sample["band"] == band]
            rsb, _ = r2(bsub, "residual", STRUCT_ALL)
            rtb, _ = r2(bsub, "residual", TYPE)
            boot_results[f"structure_{band}"].append(rsb)
            boot_results[f"type_{band}"].append(rtb)

    def ci(a):
        a = np.array(a, dtype=float)
        a = a[~np.isnan(a)]
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    boot_summary = []
    for key, vals in boot_results.items():
        lo, hi = ci(vals)
        point = np.nanmedian(vals)
        boot_summary.append({"quantity": key, "median": point, "ci_lo": lo, "ci_hi": hi, "n_valid": int(np.sum(~np.isnan(vals)))})
        print(f"  {key}: median={point:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")
    pd.DataFrame(boot_summary).to_csv(os.path.join(OUT_DIR, "rq3c_step3_bootstrap_cell_level.csv"), index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
