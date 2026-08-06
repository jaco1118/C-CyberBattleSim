"""Task Y-ROBUSTNESS STEP 1.4: count-based robustness (churn-invariant), N=30 vs N=90.

Methodology mirrors taskF3_count_recompute.py exactly: metric = root_owned COUNT (not the
root_owned/reachable ratio); robustness(seed) = mean(root_owned | membership_matched) /
mean(root_owned | static), computed per seed then aggregated across the 5 seeds.
"""
import numpy as np
import pandas as pd

SEEDS = [42, 100, 123, 200, 300]
NODE_COUNTS = {"n30": 30, "n90": 90}
BASE = "y_robustness/out"


def load(cell, seed, cond):
    return pd.read_csv(f"{BASE}/{cell}/score_static_seed{seed}_eval{cond}.csv")


results = {}
for cell, n in NODE_COUNTS.items():
    print(f"\n{'=' * 90}\n{cell.upper()} (N={n})\n{'=' * 90}")
    per_seed_robustness = []
    per_seed_static_mean = []
    per_seed_change_mean = []
    per_seed_churn_frac = []
    for s in SEEDS:
        static_df = load(cell, s, "static")
        change_df = load(cell, s, "membership_matched")
        static_count = static_df["root_owned"].to_numpy()
        change_count = change_df["root_owned"].to_numpy()
        static_mean = static_count.mean()
        change_mean = change_count.mean()
        robustness = change_mean / static_mean
        per_seed_robustness.append(robustness)
        per_seed_static_mean.append(static_mean)
        per_seed_change_mean.append(change_mean)
        print(f"  seed {s}: static_count_mean={static_mean:.3f} (n={len(static_count)})  "
              f"change_count_mean={change_mean:.3f} (n={len(change_count)})  "
              f"ROBUSTNESS={robustness:.4f}")
    per_seed_robustness = np.array(per_seed_robustness)
    results[cell] = {
        "robustness": per_seed_robustness,
        "static_mean": np.array(per_seed_static_mean),
        "change_mean": np.array(per_seed_change_mean),
    }
    print(f"\n  Per-seed robustness ({cell}): {[round(x, 4) for x in per_seed_robustness]}")
    print(f"  mean = {per_seed_robustness.mean():.4f}   sd = {per_seed_robustness.std(ddof=1):.4f}   "
          f"range = [{per_seed_robustness.min():.4f}, {per_seed_robustness.max():.4f}]")
    print(f"  pooled static count mean = {per_seed_static_mean and np.mean(per_seed_static_mean):.3f}  "
          f"pooled change count mean = {np.mean(per_seed_change_mean):.3f}")
    print(f"  pooled robustness (pooled change mean / pooled static mean) = "
          f"{np.mean(per_seed_change_mean) / np.mean(per_seed_static_mean):.4f}")

print(f"\n{'=' * 90}\nCOMPARISON: N=30 vs N=90\n{'=' * 90}")
r30 = results["n30"]["robustness"]
r90 = results["n90"]["robustness"]
diff = r90.mean() - r30.mean()
print(f"N=30 robustness: mean={r30.mean():.4f} sd={r30.std(ddof=1):.4f} range=[{r30.min():.4f},{r30.max():.4f}]")
print(f"N=90 robustness: mean={r90.mean():.4f} sd={r90.std(ddof=1):.4f} range=[{r90.min():.4f},{r90.max():.4f}]")
print(f"Difference (N90 - N30) = {diff:+.4f}")
print(f"N=30 within-cell spread (sd)  = {r30.std(ddof=1):.4f}")
print(f"N=90 within-cell spread (sd)  = {r90.std(ddof=1):.4f}")
print(f"Between-cell difference vs within-cell spread: "
      f"|diff|={abs(diff):.4f} vs N30_sd={r30.std(ddof=1):.4f}, N90_sd={r90.std(ddof=1):.4f}")
