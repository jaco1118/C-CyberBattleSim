"""Task RQ1-POWER PART B: minimum detectable effect for both RQ1(c) comparisons (node-count,
N=30 vs N=90; neighbour-count, high- vs low-degree at fixed N=30), using the SAME MDE convention
the chapter's ablation already uses -- so both sections of the chapter apply the same standard.

CONVENTION (fixed here, before computing, per compute_z_mde.py's own definition -- commit 91df45b,
lines 10-13, quoted verbatim):
  "MDE : the noise floor = pooled between-seed SD of the terminal root-owned COUNT under the
   STATIC condition (dynamic_mode=none), pooled over the three arms' 5 seeds each (15 static
   seeds). Static is the same env for all arms, so its between-seed spread is the pure seed/eval
   noise the design can resolve."
  mde = float(np.nanstd(static_pool, ddof=1))   # a raw sample SD, no z-multiplier, no power formula

ADAPTATION, stated explicitly because a literal reuse is not possible: the ablation's static_pool
pools a LITERALLY SHARED environment (the same static condition, identical across all three arms)
in RAW NODE-COUNT units. Neither RQ1(c) comparison has a literally-shared reference condition --
N=30 and N=90 are different topologies; high- and low-degree are different topologies at the same
N -- and the quantity actually being compared in both cases is ROBUSTNESS (a change/static RATIO,
unitless), not a raw node count. So the adaptation applied here, fixed before any number is seen:
POOL THE PER-SEED ROBUSTNESS VALUES FROM BOTH SIDES OF EACH COMPARISON TOGETHER (equal per-seed
weighting, exactly as the ablation pools all 15 static seeds with equal weight regardless of which
arm they came from), and take the sample SD (ddof=1) of that pooled set as the MDE for THAT
comparison. This keeps the same construction (a raw pooled between-seed SD, not a power-formula
MDE) applied to the metric this comparison actually reports, rather than forcing an artificial
raw-node-count analog that would not be in the same units as the reported +-0.0693/+-0.0409 etc.
intervals. This is NOT the identical procedure -- it is the closest faithful adaptation, and is
reported as such, not as a literal reproduction.

Per-seed robustness values (verified against the task brief / earlier committed work, TASK
NODECOUNT-CI, exact 4-decimal match):
  N=30 (~20 deg): seed42=0.6802 seed100=0.7686 seed123=0.6740 seed200=0.6983 seed300=0.7066
  N=90 (~20 deg): seed42=0.6805 seed100=0.6486 seed123=0.6755 seed200=0.6924 seed300=0.7194
  N=30 high-degree (same cell as N=30 above, reused per compute_neighbour_comparison.py)
  N=30 low-degree (~12.35 deg, 4 seeds, seed42 excluded -- never converged in taskY2-pilot-n30):
    read fresh from y_robustness/out/n30_lowdeg/, same formula as compute_robustness.py /
    compute_neighbour_comparison.py: robustness(seed) = mean(root_owned|membership_matched) /
    mean(root_owned|static)
"""
import os
import numpy as np
import pandas as pd

AG = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

N30_SEEDS = [42, 100, 123, 200, 300]
N90_SEEDS = [42, 100, 123, 200, 300]
LOWDEG_SEEDS = [100, 123, 200, 300]

# Reported intervals, for the comparison table only -- never adjusted, never used to pick a
# convention.
REPORTED = {
    "node_count": {"point": -0.022, "ci": (-0.0574, 0.0117)},
    "neighbour_count": {"point": 0.0168, "ci": (-0.0409, 0.0743)},
}
N30_MEAN_ROBUSTNESS = 0.7056


def load(cell, s, cond):
    return pd.read_csv(os.path.join(AG, f"y_robustness/out/{cell}/score_static_seed{s}_eval{cond}.csv"))["root_owned"].to_numpy()


def per_seed_robustness(cell, seeds):
    out = {}
    for s in seeds:
        st = load(cell, s, "static")
        ch = load(cell, s, "membership_matched")
        out[s] = ch.mean() / st.mean()
    return out


def pooled_mde(*robustness_dicts):
    """Pool every per-seed robustness value from every group passed in, equal per-seed weight,
    sample SD (ddof=1) -- the same construction as compute_z_mde.py's own MDE, applied to the
    metric this comparison reports."""
    pooled = np.concatenate([np.array(list(d.values())) for d in robustness_dicts])
    return float(np.std(pooled, ddof=1)), len(pooled), pooled


def main():
    print("=== SAFETY CONFIRMATION ===")
    print("Reads existing per-seed robustness values (already-logged score CSVs) and committed "
          "cell data; computes statistics from them. No training, no environment reset, no new "
          "episode, no checkpoint/encoder touched, no step()/encode()/reward path modified.\n")

    r30 = per_seed_robustness("n30", N30_SEEDS)
    r90 = per_seed_robustness("n90", N90_SEEDS)
    r_lowdeg = per_seed_robustness("n30_lowdeg", LOWDEG_SEEDS)

    print("N=30 (~20 deg, n=5):", {k: round(v, 4) for k, v in r30.items()})
    print("N=90 (~20 deg, n=5):", {k: round(v, 4) for k, v in r90.items()})
    print("N=30 low-degree (~12.35 deg, n=4):", {k: round(v, 4) for k, v in r_lowdeg.items()})
    print()

    # --- B.2: node-count comparison (N=30 vs N=90) ---
    mde_nc, n_nc, pooled_nc = pooled_mde(r30, r90)
    mde_nc_pct = 100 * mde_nc / N30_MEAN_ROBUSTNESS
    print(f"[B.2] node-count MDE: pooled n={n_nc} (5+5), SD(ddof=1)={mde_nc:.4f} robustness units "
          f"= {mde_nc_pct:.2f}% of N=30 mean robustness ({N30_MEAN_ROBUSTNESS})")
    rep_nc = REPORTED["node_count"]
    print(f"       reported point diff={rep_nc['point']:+.4f}, CI={rep_nc['ci']} "
          f"-> |point diff|/MDE = {abs(rep_nc['point'])/mde_nc:.3f}")

    # --- B.3: neighbour-count comparison (high- vs low-degree at N=30) ---
    mde_nb, n_nb, pooled_nb = pooled_mde(r30, r_lowdeg)  # r30 doubles as the "high-degree" cell
    mde_nb_pct = 100 * mde_nb / N30_MEAN_ROBUSTNESS
    print(f"\n[B.3] neighbour-count MDE: pooled n={n_nb} (5 high-degree + 4 low-degree, UNEQUAL, "
          f"equal per-seed weight -- the group with more seeds (high-degree) contributes "
          f"proportionally more to the pooled SD, same as the ablation's own per-seed pooling "
          f"scheme applied to unequal-sized groups), SD(ddof=1)={mde_nb:.4f} robustness units "
          f"= {mde_nb_pct:.2f}% of N=30 mean robustness ({N30_MEAN_ROBUSTNESS})")
    rep_nb = REPORTED["neighbour_count"]
    print(f"       reported point diff={rep_nb['point']:+.4f}, CI={rep_nb['ci']} "
          f"-> |point diff|/MDE = {abs(rep_nb['point'])/mde_nb:.3f}")

    out_df = pd.DataFrame([
        {"comparison": "node_count", "group_a": "N30(n=5)", "group_b": "N90(n=5)",
         "pooled_n": n_nc, "mde_robustness_units": round(mde_nc, 4),
         "mde_pct_of_n30_mean": round(mde_nc_pct, 2),
         "reported_point_diff": rep_nc["point"], "reported_ci_lo": rep_nc["ci"][0],
         "reported_ci_hi": rep_nc["ci"][1], "abs_point_over_mde": round(abs(rep_nc["point"]) / mde_nc, 3)},
        {"comparison": "neighbour_count", "group_a": "high-degree N30(n=5)", "group_b": "low-degree N30(n=4)",
         "pooled_n": n_nb, "mde_robustness_units": round(mde_nb, 4),
         "mde_pct_of_n30_mean": round(mde_nb_pct, 2),
         "reported_point_diff": rep_nb["point"], "reported_ci_lo": rep_nb["ci"][0],
         "reported_ci_hi": rep_nb["ci"][1], "abs_point_over_mde": round(abs(rep_nb["point"]) / mde_nb, 3)},
    ])
    out_path = os.path.join(OUT_DIR, "rq1c_mde_result.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
