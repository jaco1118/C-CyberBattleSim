"""Task Y-EARLYCKPT follow-up: recompute the pairwise MDE figures using the new N=60/N=90
per-seed robustness values (from the early-checkpoint evaluation, commit 7a03dfb), N=30
unchanged. Identical method to compute_rq1c_mde.py / compute_nodecount_ci_n60.py's own
pooled_mde(): pool every per-seed robustness value from both sides of a comparison (equal
per-seed weight, 5+5=10), sample SD (ddof=1) of the pooled set -- no z-multiplier, no power
target, the same construction as compute_z_mde.py's ablation MDE. No new evaluation, no
checkpoint access; reads only the already-committed score CSVs.
"""
import numpy as np
import pandas as pd

SEEDS = [42, 100, 123, 200, 300]
CELLS = {30: "n30", 60: "n60_stage3", 90: "n90_static500k"}


def load(cell_dir, s, cond):
    return pd.read_csv(f"y_robustness/out/{cell_dir}/score_static_seed{s}_eval{cond}.csv")["root_owned"].to_numpy()


def per_seed_robustness(cell_dir, seeds):
    return {s: load(cell_dir, s, "membership_matched").mean() / load(cell_dir, s, "static").mean() for s in seeds}


def pooled_mde(d_a, d_b):
    pooled = np.concatenate([np.array(list(d_a.values())), np.array(list(d_b.values()))])
    return float(np.std(pooled, ddof=1)), len(pooled)


def main():
    r = {n: per_seed_robustness(CELLS[n], SEEDS) for n in (30, 60, 90)}
    for n in (30, 60, 90):
        v = np.array(list(r[n].values()))
        print(f"N={n} ({CELLS[n]}): mean={v.mean():.4f} sd={v.std(ddof=1):.4f}  per-seed={dict(r[n])}")

    print()
    results = {}
    for a, b in [(30, 60), (30, 90), (60, 90)]:
        mde, n = pooled_mde(r[a], r[b])
        mean_a = np.mean(list(r[a].values()))
        pct = 100 * mde / mean_a
        results[(a, b)] = (mde, pct)
        print(f"N={a} vs N={b}: pooled n={n}, MDE={mde:.4f} robustness units "
              f"= {pct:.1f}% of N={a} mean robustness ({mean_a:.4f})")

    vals = [v[0] for v in results.values()]
    print(f"\nRange across the three pairs: {min(vals):.3f} to {max(vals):.3f}")
    pcts = [v[1] for v in results.values()]
    print(f"Percentage-of-mean range: {min(pcts):.1f}% to {max(pcts):.1f}%")


if __name__ == "__main__":
    main()
