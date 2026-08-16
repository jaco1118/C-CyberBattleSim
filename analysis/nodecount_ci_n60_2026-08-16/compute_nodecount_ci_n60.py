"""Task NODECOUNT-CI (N=60 extension): the two new node-count robustness-difference pairs,
N=30 vs N=60 and N=60 vs N=90, using the SAME method as analysis/nodecount_ci_2026-08-09/
compute_nodecount_ci.py (itself a reuse of y_robustness/scripts/compute_neighbour_comparison.py's
seed bootstrap) for the CI, and the SAME method as analysis/rq1c_mde_2026-08-10/
compute_rq1c_mde.py (pooled between-seed SD of raw per-seed robustness, ddof=1, no z-multiplier)
for the MDE. Neither existing script is modified.

Metric (identical to all three prior scripts):
  robustness(seed) = mean(root_owned count | membership_matched) / mean(root_owned count | static)

N=60 data source: y_robustness/out/n60_ci7/ -- the SELECTED run from Task N=60-ROBUSTNESS
Amendment 2 (ci=7, 42.23% achieved churn, in the declared 40.5-43.5% band). The ci=5 run
(y_robustness/out/n60/, 47.18% achieved churn, rejected out of band) and the ci=8 run
(y_robustness/out/n60_ci8/, 40.12%, rejected out of band) are NOT used here.

Resample count and unit: NB=10,000, RNG seed=11 -- identical to compute_nodecount_ci.py and
compute_neighbour_comparison.py's own SEED constant. Resampling unit: seed (5 seeds per cell
resampled with replacement, independently per cell; pooled_robustness then concatenates all
episodes across the resampled seeds and takes the ratio of pooled means -- the same two-level
construction used in both prior scripts, not an episode-level bootstrap).

Sign convention: (larger cell) minus (smaller cell), matching compute_nodecount_ci.py's own
N90-N30 convention and the reported "-0.022, the 90-node cell the lower" framing:
  N=30 vs N=60 -> N60 - N30
  N=60 vs N=90 -> N90 - N60
"""
import numpy as np
import pandas as pd

SEEDS = [42, 100, 123, 200, 300]
NB = 10000
SEED = 11

CELLS = {30: "n30", 60: "n60_ci7", 90: "n90"}


def load(cell_dir, s, cond):
    return pd.read_csv(f"y_robustness/out/{cell_dir}/score_static_seed{s}_eval{cond}.csv")["root_owned"].to_numpy()


_CACHE = {}


def _get(cell_dir, s, cond):
    key = (cell_dir, s, cond)
    if key not in _CACHE:
        _CACHE[key] = load(cell_dir, s, cond)
    return _CACHE[key]


def per_seed_robustness(cell_dir, seeds):
    out = {}
    for s in seeds:
        st = _get(cell_dir, s, "static")
        ch = _get(cell_dir, s, "membership_matched")
        out[s] = ch.mean() / st.mean()
    return out


def pooled_robustness(cell_dir, seeds):
    st = np.concatenate([_get(cell_dir, s, "static") for s in seeds])
    ch = np.concatenate([_get(cell_dir, s, "membership_matched") for s in seeds])
    return ch.mean() / st.mean()


def bootstrap_ci(cell_a_dir, cell_b_dir):
    """diff = pooled_robustness(b) - pooled_robustness(a), b assumed the larger cell."""
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(NB):
        ba = rng.choice(SEEDS, len(SEEDS), replace=True)
        bb = rng.choice(SEEDS, len(SEEDS), replace=True)
        diffs.append(pooled_robustness(cell_b_dir, bb) - pooled_robustness(cell_a_dir, ba))
    diffs = np.array(diffs)
    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def pooled_mde(*robustness_dicts):
    pooled = np.concatenate([np.array(list(d.values())) for d in robustness_dicts])
    return float(np.std(pooled, ddof=1)), len(pooled)


def main():
    print("=== SAFETY CONFIRMATION ===")
    print("Reads existing per-seed robustness values (already-logged score CSVs) and resamples "
          "them. No training, no environment reset, no new episode, no checkpoint/encoder touched, "
          "no step()/encode()/reward path modified.\n")

    r = {n: per_seed_robustness(CELLS[n], SEEDS) for n in (30, 60, 90)}
    for n in (30, 60, 90):
        v = np.array(list(r[n].values()))
        print(f"N={n} ({CELLS[n]}), n=5: " + ", ".join(f"{s}={r[n][s]:.4f}" for s in SEEDS)
              + f"  mean={v.mean():.4f} sd={v.std(ddof=1):.4f}")

    print()
    results = []
    for a, b in [(30, 60), (60, 90)]:
        va = np.array(list(r[a].values()))
        vb = np.array(list(r[b].values()))
        diff_point = vb.mean() - va.mean()
        lo, hi = bootstrap_ci(CELLS[a], CELLS[b])
        contains_zero = lo <= 0.0 <= hi
        mde, pooled_n = pooled_mde(r[a], r[b])
        print(f"N={a} vs N={b}: point diff (N{b}-N{a}) = {diff_point:+.4f}, "
              f"bootstrap 95% CI (NB={NB}, seed={SEED}, seed-resampled independently per cell) "
              f"= [{lo:+.4f}, {hi:+.4f}]  {'INCLUDES 0' if contains_zero else 'EXCLUDES 0'}")
        print(f"  MDE (pooled between-seed SD, ddof=1, n={pooled_n}) = {mde:.4f} robustness units, "
              f"|point diff|/MDE = {abs(diff_point)/mde:.3f}")
        results.append({
            "comparison": f"N{a}_vs_N{b}", "cell_a": f"N{a}({CELLS[a]})", "cell_b": f"N{b}({CELLS[b]})",
            "point_diff": diff_point, "ci_lo": lo, "ci_hi": hi, "contains_zero": contains_zero,
            "n_resamples": NB, "rng_seed": SEED, "resample_unit": "seed",
            "mde_robustness_units": mde, "pooled_n": pooled_n,
        })

    out_df = pd.DataFrame(results)
    out_path = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/analysis/nodecount_ci_n60_2026-08-16/nodecount_ci_n60_result.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
