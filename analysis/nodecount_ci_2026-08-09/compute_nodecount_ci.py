"""Task NODECOUNT-CI: rebuild the node-count (N=30 vs N=90) robustness-difference confidence
interval, the one reported CI in the dissertation with no surviving computation (TASK CI-BASIS).

Not a reconstruction from scratch: reuses the exact working method already in
y_robustness/scripts/compute_neighbour_comparison.py (seed bootstrap, 10,000 reps, percentile CI)
over the OTHER pair of cells that method was never applied to (N=30 vs N=90 at matched degree,
instead of high-degree vs low-degree at matched N=30). Neither existing script is modified.

Metric (identical to both compute_robustness.py and compute_neighbour_comparison.py):
  robustness(seed) = mean(root_owned count | membership_matched) / mean(root_owned count | static)

Per-seed inputs verified against the task brief's listed values before running (STEP 0.2, exact
match to 4 decimal places, same seed order):
  N=30, degree ~20: 0.6802, 0.7686, 0.6740, 0.6983, 0.7066  (mean 0.7056)
  N=90, degree ~20: 0.6805, 0.6486, 0.6755, 0.6924, 0.7194  (mean 0.6833)

RNG seed: 11, matching compute_neighbour_comparison.py's own SEED constant exactly (chosen for
consistency with the reused method, decided before running -- not tried against alternatives to
approach the reported [-0.0693, +0.0088] interval, per the task's explicit prohibition on
seed-shopping).
"""
import numpy as np
import pandas as pd

SEEDS = [42, 100, 123, 200, 300]
NB = 10000
SEED = 11


def load(cell, s, cond):
    return pd.read_csv(f"y_robustness/out/{cell}/score_static_seed{s}_eval{cond}.csv")["root_owned"].to_numpy()


# PERFORMANCE NOTE (not a methodology change): compute_neighbour_comparison.py's pooled_robustness
# re-reads each seed's CSV from disk on every call, which is fine at NB=10,000 for its own cell
# pair but timed out here after 2 minutes on n30/n90 (identical file sizes, just more redundant
# I/O across the run). Preloading each seed's static/change arrays ONCE below and resampling the
# in-memory arrays is mathematically identical -- same files, same values, same resampling logic,
# same RNG seed, same NB -- it only removes repeated disk reads of the exact same content.
_CACHE = {}


def _get(cell, s, cond):
    key = (cell, s, cond)
    if key not in _CACHE:
        _CACHE[key] = load(cell, s, cond)
    return _CACHE[key]


def per_seed_robustness(cell, seeds):
    out = {}
    for s in seeds:
        st = _get(cell, s, "static")
        ch = _get(cell, s, "membership_matched")
        out[s] = ch.mean() / st.mean()
    return out


def pooled_robustness(cell, seeds):
    st = np.concatenate([_get(cell, s, "static") for s in seeds])
    ch = np.concatenate([_get(cell, s, "membership_matched") for s in seeds])
    return ch.mean() / st.mean()


def main():
    print("=== SAFETY CONFIRMATION ===")
    print("Reads existing per-seed robustness values (already-logged score CSVs) and resamples "
          "them. No training, no environment reset, no new episode, no checkpoint/encoder touched, "
          "no step()/encode()/reward path modified.\n")

    r30 = per_seed_robustness("n30", SEEDS)
    r90 = per_seed_robustness("n90", SEEDS)

    print("N=30, degree ~20, n=5:")
    for s in SEEDS:
        print(f"  seed {s}: {r30[s]:.4f}")
    v30 = np.array(list(r30.values()))
    print(f"  mean={v30.mean():.4f} sd={v30.std(ddof=1):.4f}")

    print("\nN=90, degree ~20, n=5:")
    for s in SEEDS:
        print(f"  seed {s}: {r90[s]:.4f}")
    v90 = np.array(list(r90.values()))
    print(f"  mean={v90.mean():.4f} sd={v90.std(ddof=1):.4f}")

    diff_point = v30.mean() - v90.mean()
    print(f"\nPoint difference (N30 - N90) = {diff_point:+.4f}")

    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(NB):
        b30 = rng.choice(SEEDS, len(SEEDS), replace=True)
        b90 = rng.choice(SEEDS, len(SEEDS), replace=True)
        diffs.append(pooled_robustness("n30", b30) - pooled_robustness("n90", b90))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
    contains_zero = lo <= 0.0 <= hi
    verdict = "INCLUDES 0 (not established)" if contains_zero else "EXCLUDES 0"
    width = hi - lo
    reported_width = 0.0088 - (-0.0693)

    print(f"\nBootstrap 95% CI on (N30-N90) pooled difference, RNG seed={SEED}, {NB} reps "
          f"(5 N=30 seeds / 5 N=90 seeds resampled independently): [{lo:+.4f}, {hi:+.4f}]  {verdict}")
    print(f"Interval width = {width:.4f}  vs reported width = {reported_width:.4f}  "
          f"(ratio fresh/reported = {width/reported_width:.3f})")
    print(f"Reported interval: [-0.0693, +0.0088]  (also includes 0)")

    out_df = pd.DataFrame([{
        "seed": s, "n30_robustness": r30[s], "n90_robustness": r90[s],
    } for s in SEEDS])
    out_df["point_difference_n30_minus_n90"] = diff_point
    out_df["ci_lo"] = lo
    out_df["ci_hi"] = hi
    out_df["n_resamples"] = NB
    out_df["rng_seed"] = SEED
    out_df["contains_zero"] = contains_zero
    out_df["reported_ci_lo"] = -0.0693
    out_df["reported_ci_hi"] = 0.0088

    out_path = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/analysis/nodecount_ci_2026-08-09/nodecount_ci_result.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
