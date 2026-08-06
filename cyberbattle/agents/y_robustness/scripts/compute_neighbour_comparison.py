"""Task Y-NEIGHBOUR STEP 1.2: neighbour-count (degree) robustness comparison at fixed N=30.

Compares HIGH-degree N=30 (~19.7, 5 seeds, REUSED verbatim from TASK Y-ROBUSTNESS's own
y_robustness/out/n30/) against LOW-degree N=30 (~12.35, 4 converged seeds -- seed42 excluded,
never converged in taskY2-pilot-n30 -- NEW data from this task, y_robustness/out/n30_lowdeg/).

Same count-based robustness metric as taskF3_count_recompute.py / compute_robustness.py:
robustness(seed) = mean(root_owned count | membership_matched) / mean(root_owned count | static).
Bootstrap resamples each side's own seeds independently (5 high-degree, 4 low-degree -- not
evened out) to get a 95% CI on the difference in means.
"""
import numpy as np
import pandas as pd

HIGH_SEEDS = [42, 100, 123, 200, 300]
LOW_SEEDS = [100, 123, 200, 300]
NB = 10000
SEED = 11


def load(cell, s, cond):
    return pd.read_csv(f"y_robustness/out/{cell}/score_static_seed{s}_eval{cond}.csv")["root_owned"].to_numpy()


def per_seed_robustness(cell, seeds):
    out = {}
    for s in seeds:
        st = load(cell, s, "static")
        ch = load(cell, s, "membership_matched")
        out[s] = ch.mean() / st.mean()
    return out


def pooled_robustness(cell, seeds):
    st = np.concatenate([load(cell, s, "static") for s in seeds])
    ch = np.concatenate([load(cell, s, "membership_matched") for s in seeds])
    return ch.mean() / st.mean()


def main():
    high_r = per_seed_robustness("n30", HIGH_SEEDS)
    low_r = per_seed_robustness("n30_lowdeg", LOW_SEEDS)

    print("HIGH-degree (~19.7), n=5, REUSED from Y-ROBUSTNESS:")
    for s in HIGH_SEEDS:
        print(f"  seed {s}: {high_r[s]:.4f}")
    hv = np.array(list(high_r.values()))
    print(f"  mean={hv.mean():.4f} sd={hv.std(ddof=1):.4f}")

    print("\nLOW-degree (~12.35), n=4, NEW this task (seed42 excluded, never converged):")
    for s in LOW_SEEDS:
        print(f"  seed {s}: {low_r[s]:.4f}")
    lv = np.array(list(low_r.values()))
    print(f"  mean={lv.mean():.4f} sd={lv.std(ddof=1):.4f}")

    diff_point = lv.mean() - hv.mean()
    print(f"\nPoint difference (LOW - HIGH) = {diff_point:+.4f}")

    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(NB):
        bh = rng.choice(HIGH_SEEDS, len(HIGH_SEEDS), replace=True)
        bl = rng.choice(LOW_SEEDS, len(LOW_SEEDS), replace=True)
        diffs.append(pooled_robustness("n30_lowdeg", bl) - pooled_robustness("n30", bh))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
    verdict = "EXCLUDES 0" if lo * hi > 0 else "INCLUDES 0 (not established)"
    print(f"Bootstrap 95% CI on (LOW-HIGH) pooled difference, {NB} reps "
          f"(5 high-degree / 4 low-degree seeds resampled independently): [{lo:+.4f}, {hi:+.4f}]  {verdict}")

    leave_per_ep = {}
    for s in LOW_SEEDS:
        df = pd.read_csv(f"y_robustness/out/n30_lowdeg/leaveown_static_seed{s}_evalmembership_matched.csv")
        n_episodes = len(load("n30_lowdeg", s, "membership_matched"))
        leave_per_ep[s] = len(df) / n_episodes
    mean_leave = np.mean(list(leave_per_ep.values()))
    print(f"\nAchieved churn (low-degree, this run, recomputed from leaveown CSVs): "
          f"per-seed leave/ep={[round(v,2) for v in leave_per_ep.values()]}  "
          f"mean={mean_leave:.2f} -> {mean_leave/30*100:.1f}%")


if __name__ == "__main__":
    main()
