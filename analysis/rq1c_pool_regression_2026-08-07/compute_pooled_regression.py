"""Task RQ1C-POOL STEP 1: pooled regression -- RQ1a's 117 main-study cells plus Task Y's cells.

Background: TASK RQ1C-POOL STEP 0 found this blocked in its cheap form -- the data already
collected by TASK Y-ROBUSTNESS / TASK Y-NEIGHBOUR (via taskF2_eval.py) never captured which
departing nodes held ROOT privilege, only whether they were owned at all, so relative_fragility's
mechanical term could not be computed for Task Y's cells. STEP 0B unblocked this two ways:
  (1) taskF3_mech_eval.py (already committed, from TASK Y-ROBUSTNESS) DOES capture was_root per
      removal and can be pointed at Task Y's checkpoints/topologies unmodified -- run via
      run_mech_taskY.sh (STEP 1.1), reusing the exact checkpoints/topologies/calibrated churn
      conditions already established (no new training, no change to the disturbance condition).
      Consistency-checked against the already-reported figures: EXACT match (diff=0.0000 on all
      14 seeds' mean score) -- same seeding convention, same underlying episodes.
  (2) ownable_count (needed for conquest) turned out NOT to need any new evaluation at all -- but
      not quite as simply as first hoped. It is precomputed and stored inside each topology's own
      pickle file (Model.access_shortest_paths[starter_node]), which reproduced the already-known
      value (9) for the RQ1a sample cell exactly from every starter node tested there. For Task Y's
      own 14 cells, though, this turned out to be starter-node BIMODAL (0 from some starters, a
      value near N-1 from others) -- not a single fixed topology property after all. Resolved using
      data already on hand: the static arm's own `reachable` column (already logged by
      TASK Y-ROBUSTNESS/Y-NEIGHBOUR) is fixed across all 200 episodes for every one of the 14 cells,
      and matches the non-zero (well-connected-starter) topology value exactly wherever checked --
      used directly rather than re-deriving from the topology file's ambiguous full starter set.

This script:
  - loads the 117-row RQ1a dataset from analysis/rq1a_regression_recovered_2026-08-07/rq1a_cells.csv
  - computes the same three quantities (conquest, network_size, relative_fragility -- FORMULA B)
    plus degree for Task Y's up to 14 rows, from: the STATIC arm's already-committed score CSVs
    (TASK Y-ROBUSTNESS/Y-NEIGHBOUR), the CHANGE arm's mech_*.csv (this task's STEP 1.1 run), and
    each cell's topology pickle (for ownable_count, num_nodes_start, and degree).
  - computes degree for the 117 main-study rows too (same knows_graph.out_degree() method used
    throughout this project), so degree is a genuine pooled predictor, not defined only for the
    14 Task Y rows.
  - fits relative_fragility ~ conquest + log(network_size) + degree + population_indicator via
    statsmodels OLS, WITH and WITHOUT the population indicator, to check whether any degree effect
    survives removing it (the critical confound check).

Usage: run from cyberbattle/agents/ (where the raw data and Y-ROBUSTNESS/Y-NEIGHBOUR output live):
  python ../../analysis/rq1c_pool_regression_2026-08-07/compute_pooled_regression.py
"""
import glob
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

NBOOT = 10000
BOOT_SEED = 11

RQ1A_CSV = "../../analysis/rq1a_regression_recovered_2026-08-07/rq1a_cells.csv"

# Task Y cell definitions: (population_label, band_label_in_mech_out, seed, static_score_dir,
# topology_pkl_relpath, N (intended node count))
TASKY_CELLS = []
for s in [42, 100, 123, 200, 300]:
    TASKY_CELLS.append(("taskY_n30hi", s, "n30", f"graphs_yN30_s{s}_2026-08-03_17-15-34/1", 30))
for s in [42, 100, 123, 200, 300]:
    if s == 42:
        topo = "graphs_yprobe_n90_2026-08-03_02-10-10/1"
    else:
        topo = f"graphs_yprobe_n90_s{s}_2026-08-03_08-38-04/1"
    TASKY_CELLS.append(("taskY_n90", s, "n90", topo, 90))
for s in [100, 123, 200, 300]:  # seed42 excluded -- never converged (TASK Y-NEIGHBOUR)
    TASKY_CELLS.append(("taskY_n30lo", s, "n30_lowdeg", f"graphs_yN30_s{s}_2026-08-05_21-{'16-40' if s==100 else '17-08' if s==123 else '17-46' if s==200 else '18-13'}/1", 30))


def topology_size(topo_relpath):
    import pickle
    path = os.path.join("../data/env_samples", topo_relpath, "network_SecureBERT.pkl")
    m = pickle.load(open(path, "rb"))
    return m.network.number_of_nodes()


def topology_degree(topo_relpath):
    import pickle
    path = os.path.join("../data/env_samples", topo_relpath, "network_SecureBERT.pkl")
    m = pickle.load(open(path, "rb"))
    return float(np.mean([d for _, d in m.knows_graph.out_degree()]))


def compute_tasky_row(pop_label, seed, static_dir, topo_relpath):
    static_df = pd.read_csv(f"y_robustness/out/{static_dir}/score_static_seed{seed}_evalstatic.csv")
    root_owned_static = static_df["root_owned"].mean()

    # ownable_count: NOT computable unambiguously from the topology file alone -- STEP 0B/1.3 found
    # access_shortest_paths (hence ownable-node reachability) is bimodal by starter node (0 or a
    # value near N-1), so which one is "correct" depends on which starter the actual episodes used.
    # Already-available data resolves this cleanly: the static arm's own `reachable` column is fixed
    # across all 200 episodes for every one of Task Y's 14 cells (verified directly, not assumed),
    # and matches the non-zero (well-connected-starter) topology-file value exactly on every cell
    # spot-checked -- using it directly rather than re-deriving from the topology file.
    reachable_vals = static_df["reachable"].unique()
    ownable_count_fixed = len(reachable_vals) == 1
    ownable_count = int(reachable_vals[0]) if ownable_count_fixed else None

    change_df = pd.read_csv(f"y_robustness/out/{static_dir}/score_static_seed{seed}_evalmembership_matched.csv")
    mean_change = change_df["root_owned"].mean()
    cost = root_owned_static - mean_change

    mech_df = pd.read_csv(f"y_robustness/out/mech_taskY/mech_{pop_label}_seed{seed}.csv")
    neps = mech_df["episode"].nunique()
    mechanical = mech_df[mech_df["was_root"] == 1].shape[0] / neps

    behavioural_residual = cost - mechanical
    relative_fragility = behavioural_residual / root_owned_static

    n_nodes = topology_size(topo_relpath)
    conquest = root_owned_static / ownable_count if ownable_count_fixed else None
    degree = topology_degree(topo_relpath)

    return {
        "source": "task_y", "population": pop_label, "seed": seed,
        "n_static_ep": len(static_df), "n_change_ep": len(change_df), "n_mech_removal_episodes": neps,
        "root_owned_static": root_owned_static, "cost": cost, "mechanical": mechanical,
        "behavioural_residual": behavioural_residual, "relative_fragility": relative_fragility,
        "ownable_count": ownable_count, "ownable_count_fixed": ownable_count_fixed,
        "conquest": conquest, "network_size": n_nodes, "degree": degree,
    }


def fit_ols(df, use_pop_indicator, use_degree=True):
    cols = [df["conquest"].to_numpy(), np.log(df["network_size"].to_numpy())]
    names = ["conquest", "log_size"]
    if use_degree:
        cols.append(df["degree"].to_numpy())
        names.append("degree")
    if use_pop_indicator:
        cols.append((df["source"] == "task_y").astype(float).to_numpy())
        names.append("is_task_y")
    X = sm.add_constant(np.column_stack(cols))
    y = df["relative_fragility"].to_numpy()
    model = sm.OLS(y, X).fit()
    return model, ["const"] + names


def bootstrap_ci(df, use_pop_indicator, use_degree=True, n_boot=NBOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    n = len(df)
    coefs = None
    r2s = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = df.iloc[idx]
        m, names = fit_ols(sample, use_pop_indicator, use_degree)
        if coefs is None:
            coefs = {name: [] for name in names}
        for i, name in enumerate(names):
            coefs[name].append(m.params[i])
        r2s.append(m.rsquared)
    def ci(a):
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    out = {name: ci(vals) for name, vals in coefs.items()}
    out["r2"] = ci(r2s)
    return out


def main():
    main_df = pd.read_csv(RQ1A_CSV)
    main_df["source"] = "main_study"
    print(f"Main-study rows loaded: {len(main_df)} (expected 117)")

    print("\nComputing degree for main-study topologies (band/seed/topo -> topology pickle)...")
    def main_topo_path(row):
        band_map = {"10-15": "scalability_10_15", "30-40": "scalability_30_40", "80-100": "scalability_80_100"}
        return f"{band_map[row['band']]}/{row['topo']}"
    main_df["degree"] = main_df.apply(lambda r: topology_degree(main_topo_path(r)), axis=1)
    print(f"  done. degree summary: mean={main_df['degree'].mean():.2f} sd={main_df['degree'].std(ddof=1):.2f} "
          f"min={main_df['degree'].min():.2f} max={main_df['degree'].max():.2f}")

    print("\nComputing Task Y's up to 14 rows...")
    tasky_rows = []
    for pop_label, seed, static_dir, topo_relpath, n in TASKY_CELLS:
        r = compute_tasky_row(pop_label, seed, static_dir, topo_relpath)
        tasky_rows.append(r)
        print(f"  {pop_label} seed{seed}: n_static_ep={r['n_static_ep']} n_change_ep={r['n_change_ep']} "
              f"n_mech_ep={r['n_mech_removal_episodes']} ownable_count={r['ownable_count']} "
              f"(fixed={r['ownable_count_fixed']}) network_size={r['network_size']} degree={r['degree']:.2f} "
              f"conquest={r['conquest']:.4f} relative_fragility={r['relative_fragility']:.4f}")
    tasky_df = pd.DataFrame(tasky_rows)
    print(f"\nTask Y rows computed: {len(tasky_df)} (expected up to 14)")

    pooled = pd.concat([main_df, tasky_df], ignore_index=True, sort=False)
    pooled.to_csv("../../analysis/rq1c_pool_regression_2026-08-07/pooled_cells.csv", index=False)
    print(f"\nPooled dataset written: {len(pooled)} rows ({len(main_df)} main study + {len(tasky_df)} Task Y)")

    print("\n" + "=" * 70)
    print("117-ROW BASELINE (no degree, no pop indicator) -- reproduce for reference")
    print("=" * 70)
    m0, n0 = fit_ols(main_df, use_pop_indicator=False, use_degree=False)
    print(f"R^2={m0.rsquared:.4f}  conquest={m0.params[1]:+.4f}  log_size={m0.params[2]:+.4f}  n={int(m0.nobs)}")

    for use_pop in [True, False]:
        print("\n" + "=" * 70)
        label = "WITH population indicator" if use_pop else "WITHOUT population indicator (critical confound check)"
        print(f"POOLED MODEL, n={len(pooled)}, {label}")
        print("=" * 70)
        model, names = fit_ols(pooled, use_pop_indicator=use_pop, use_degree=True)
        print(model.summary())
        for i, name in enumerate(names):
            print(f"  {name:12s} = {model.params[i]:+.4f}  (SE {model.bse[i]:.4f}, p={model.pvalues[i]:.4g})")
        print(f"  R^2 = {model.rsquared:.4f}  n = {int(model.nobs)}")

        boot = bootstrap_ci(pooled, use_pop_indicator=use_pop, use_degree=True)
        print(f"\n  Bootstrap 95% CI (10,000 resamples of the {len(pooled)} pooled rows):")
        for name in names:
            print(f"    {name:12s}: {boot[name]}")
        print(f"    {'r2':12s}: {boot['r2']}")


if __name__ == "__main__":
    main()
