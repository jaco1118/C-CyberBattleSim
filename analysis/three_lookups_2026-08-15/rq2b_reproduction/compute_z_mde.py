"""Task Z / RQ2B pooling-ablation analysis (band 10-15): MDE + arm contrasts + verdict.

Reads the per-seed eval CSVs written by taskZ_eval.py (columns: arm, topo, condition, run_id,
seed, mean_root_owned, n_episodes) for the three-arm design at 10-15, and reports, per topology:

  Arms   : 1 = full 256-d [mean,max,min]  (full extremal info, 256-wide input)
           2 = mean-only 128-d [mean]     (no extremal info, 128-wide input)
           3 = 256-d extremal-zeroed      (no extremal info, 256-wide input)  <- capacity control

  MDE    : the noise floor = pooled between-seed SD of the terminal root-owned COUNT under the
           STATIC condition (dynamic_mode=none), pooled over the three arms' 5 seeds each (15 static
           seeds). Static is the same env for all arms, so its between-seed spread is the pure
           seed/eval noise the design can resolve. Reported in nodes and as % of the static mean.

  Arm contrasts (evaluated under the CHANGE condition, paired by seed since all arms share the
  same 5 seeds 42/100/123/200/300):
     INFO      = arm1 - arm3   both 256-wide; differ ONLY in whether the extremal channels carry
                               information -> isolates the *information* the extremal pooling adds
     CAPACITY  = arm3 - arm2   both carry no extremal info; differ ONLY in input width (256 vs 128)
                               -> isolates the *capacity* (extra input dims) confound
     RAW       = arm1 - arm2   the naive full-vs-mean comparison, which confounds INFO + CAPACITY

  Per contrast: paired mean diff, paired SD, and a 10,000-sample paired bootstrap 95% CI.

  Verdict per contrast:
     EFFECT ABSENT (well-powered) : |mean diff| < MDE AND the 95% CI contains 0
                                    (design could resolve an effect of size MDE; none seen)
     EFFECT PRESENT               : 95% CI excludes 0
     UNDERPOWERED                 : |mean diff| < MDE but CI does not cleanly bracket 0, i.e. the
                                    noise floor is too large to call absence

Usage: python compute_z_mde.py [--eval-dir cyberbattle/agents/rq2b_10-15_eval] [--topos 44,34]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

SEED_ORDER = [42, 100, 123, 200, 300]
ARM_LABEL = {1: "full-256 [mean,max,min]", 2: "mean-only-128 [mean]", 3: "256 extremal-zeroed"}


def load(eval_dir):
    frames = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(eval_dir, "arm*_t*_*.csv")))]
    return pd.concat(frames, ignore_index=True)


def seed_vec(df, arm, topo, cond):
    """Return the 5 per-seed means in fixed SEED_ORDER (NaN if a seed missing)."""
    sub = df[(df.arm == arm) & (df.topo == topo) & (df.condition == cond)]
    m = dict(zip(sub.seed, sub.mean_root_owned))
    return np.array([m.get(s, np.nan) for s in SEED_ORDER], float)


def paired_bootstrap(diffs, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def contrast(name, a_vec, b_vec, mde):
    d = a_vec - b_vec
    md = float(np.nanmean(d))
    sd = float(np.nanstd(d, ddof=1))
    lo, hi = paired_bootstrap(d[~np.isnan(d)])
    ci_excl0 = (lo > 0) or (hi < 0)
    if ci_excl0:
        verdict = "EFFECT PRESENT (CI excludes 0)"
    elif abs(md) < mde and lo <= 0 <= hi:
        verdict = "EFFECT ABSENT (well-powered: |diff|<MDE, CI brackets 0)"
    else:
        verdict = "UNDERPOWERED (noise floor too large to call)"
    print(f"    {name:>9}: mean diff = {md:+.3f}  (paired SD {sd:.3f})  "
          f"95% CI [{lo:+.3f}, {hi:+.3f}]  |  {verdict}")
    return md, (lo, hi), verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="cyberbattle/agents/rq2b_10-15_eval")
    ap.add_argument("--topos", default="44,34")
    args = ap.parse_args()
    df = load(args.eval_dir)
    topos = [int(t) for t in args.topos.split(",")]

    for ti, topo in enumerate(topos):
        role = "PRIMARY" if ti == 0 else "SECONDARY (robustness)"
        print(f"\n{'='*78}\nTOPOLOGY #{topo}  [{role}]  band 10-15\n{'='*78}")

        # per-arm static/change means
        print("  per-arm mean terminal root-owned COUNT (5 seeds):")
        static_pool = []
        for arm in (1, 2, 3):
            sv = seed_vec(df, arm, topo, "static"); cv = seed_vec(df, arm, topo, "change")
            static_pool.append(sv)
            print(f"    arm{arm} {ARM_LABEL[arm]:<26}  static {np.nanmean(sv):6.3f}  "
                  f"change {np.nanmean(cv):6.3f}")
        static_pool = np.concatenate(static_pool)
        mde = float(np.nanstd(static_pool, ddof=1))
        static_mean = float(np.nanmean(static_pool))
        print(f"  MDE (pooled between-seed SD, static, n={np.sum(~np.isnan(static_pool))}): "
              f"{mde:.3f} nodes = {100*mde/static_mean:.1f}% of static mean ({static_mean:.3f})")

        a1 = seed_vec(df, 1, topo, "change")
        a2 = seed_vec(df, 2, topo, "change")
        a3 = seed_vec(df, 3, topo, "change")
        print("  arm contrasts under CHANGE (paired by seed):")
        contrast("INFO", a1, a3, mde)
        contrast("CAPACITY", a3, a2, mde)
        contrast("RAW", a1, a2, mde)


if __name__ == "__main__":
    main()
