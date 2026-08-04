"""
Task I-2 STEP 2: fit encode/full_rebuild/steady_state/match vs topology N, separately for
precise_flags False and True, using the v3 measurements (which correctly wire
sample_subset_samples through, unlike Task I's original scripts).
"""
import json, glob
import numpy as np
from scipy import stats

def load(flag_str):
    files = sorted(glob.glob(f'v3_N*_{flag_str}.json'), key=lambda f: int(f.split('N')[1].split('_')[0]))
    points = []
    for f in files:
        d = json.load(open(f))
        for r in d['outer_results']:
            points.append(dict(n=d['n_nodes_ground_truth'], **r))
    return points


def power_law_fit(xs_raw, ys_raw):
    xs, ys = [], []
    for x, y in zip(xs_raw, ys_raw):
        if x > 0 and y > 0:
            xs.append(np.log(x)); ys.append(np.log(y))
    xs = np.array(xs); ys = np.array(ys)
    slope, intercept, r, p, se = stats.linregress(xs, ys)
    dfree = len(xs) - 2
    tval = stats.t.ppf(0.975, dfree) if dfree > 0 else float('nan')
    return dict(exponent=slope, exponent_ci95=(slope - tval * se, slope + tval * se),
                intercept=intercept, r_squared=r**2, n_points=len(xs))


def report(label, points, key):
    xs = [p['n'] for p in points]
    ys = [p[key] for p in points]
    fit = power_law_fit(xs, ys)
    print(f"{label:45s} exponent={fit['exponent']:7.4f}  CI=({fit['exponent_ci95'][0]:7.4f},{fit['exponent_ci95'][1]:7.4f})  R2={fit['r_squared']:.4f}")
    return fit


for flag_str, flag_label in [('false', 'precise_flags=False (production default)'), ('true', 'precise_flags=True')]:
    print(f"\n=== {flag_label} ===")
    points = load(flag_str)
    for n in sorted(set(p['n'] for p in points)):
        rows = [p for p in points if p['n'] == n]
        print(f"  N={n}: n_actions={[r['n_actions_after_full_rebuild'] for r in rows]} "
              f"full_rebuild={[round(r['full_rebuild_mean_s'],6) for r in rows]} "
              f"steady_state={[round(r['steady_state_mean_s'],6) for r in rows]} "
              f"match={[round(r['match_mean_s'],6) for r in rows]}")
    print()
    report("encode() vs N", points, 'encode_mean_s')
    report("full_rebuild vs N", points, 'full_rebuild_mean_s')
    report("steady_state vs N", points, 'steady_state_mean_s')
    report("match vs N", points, 'match_mean_s')
