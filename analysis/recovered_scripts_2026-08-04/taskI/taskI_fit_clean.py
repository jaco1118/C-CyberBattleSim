"""
Task I STEP 2, corrected: fit each cost component against the quantity 0.2 identified it scales
with (not raw topology N), using the clean fixed-state measurements from taskI_profile_clean.py.
  - encode() vs n_visible_graph_nodes (discovered nodes actually in evolving_visible_graph)
  - create_continuous_action_space() vs n_owned_final * n_discovered_final (the nested-loop
    product the code actually iterates, cyberbattle_env_compressed.py:981-982)
  - find_closest_action_embedding() vs n_action_embeddings_final (candidate actions searched)
Also fits each against raw topology N for the "extrapolate to N=500/1000/2500" requirement,
since N is the practical quantity a reader can reason about, with the caveat that the fit is
against the discovered-quantity at a snapshot roughly proportional to N in this data.
"""
import json, glob
import numpy as np
from scipy import stats

files = sorted(glob.glob('clean_N*.json'), key=lambda f: int(f.split('N')[1].split('.')[0]))
points = []
for f in files:
    d = json.load(open(f))
    for r in d['outer_results']:
        points.append(dict(
            n_label=d['n_label'], n_topology=d['n_nodes_ground_truth'],
            n_discovered=r['n_discovered_final'], n_owned=r['n_owned_final'],
            n_edges=r['n_visible_graph_edges'], n_actions=r['n_action_embeddings_final'],
            encode_mean=r['encode_mean_s'], action_space_mean=r['action_space_mean_s'],
            match_mean=r['match_mean_s'],
        ))

print(f"Total points: {len(points)}")
for p in points:
    print(f"  {p['n_label']}: n_discovered={p['n_discovered']} n_owned={p['n_owned']} n_edges={p['n_edges']} "
          f"n_actions={p['n_actions']} encode={p['encode_mean']:.6f}s action_space={p['action_space_mean']:.6f}s match={p['match_mean']:.6f}s")


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


def linear_nlogn_quad_fit(xs_raw, ys_raw):
    xs = np.array([x for x in xs_raw]); ys = np.array([y for y in ys_raw])
    slope, intercept, r, p, se = stats.linregress(xs, ys)
    lin = dict(r_squared=r**2, slope=slope, intercept=intercept)
    x_nlogn = xs * np.log(np.maximum(xs, 1))
    slope2, intercept2, r2, p2, se2 = stats.linregress(x_nlogn, ys)
    nlogn = dict(r_squared=r2**2, slope=slope2, intercept=intercept2)
    x_quad = xs ** 2
    slope3, intercept3, r3, p3, se3 = stats.linregress(x_quad, ys)
    quad = dict(r_squared=r3**2, slope=slope3, intercept=intercept3)
    return lin, nlogn, quad


def report(label, xs, ys, x_label):
    lin, nlogn, quad = linear_nlogn_quad_fit(xs, ys)
    pw = power_law_fit(xs, ys)
    print(f"\n=== {label} vs {x_label} ===")
    print(f"  data points (x,y): {list(zip(xs, [round(y,6) for y in ys]))}")
    print(f"  linear     R2={lin['r_squared']:.4f} slope={lin['slope']:.6g} intercept={lin['intercept']:.6g}")
    print(f"  N log N    R2={nlogn['r_squared']:.4f} slope={nlogn['slope']:.6g} intercept={nlogn['intercept']:.6g}")
    print(f"  quadratic  R2={quad['r_squared']:.4f} slope={quad['slope']:.6g} intercept={quad['intercept']:.6g}")
    print(f"  power law  R2={pw['r_squared']:.4f} exponent={pw['exponent']:.4f} "
          f"95% CI=({pw['exponent_ci95'][0]:.4f},{pw['exponent_ci95'][1]:.4f}) n={pw['n_points']}")
    best = max([('linear', lin['r_squared']), ('n_log_n', nlogn['r_squared']),
                ('quadratic', quad['r_squared']), ('power_law', pw['r_squared'])], key=lambda t: t[1])
    print(f"  BEST FIT: {best[0]} (R2={best[1]:.4f})")
    return dict(linear=lin, n_log_n=nlogn, quadratic=quad, power_law=pw, best=best)


results = {}

# vs the actual scaling quantity from the code
results['encode_vs_n_discovered'] = report(
    "encode() mean time", [p['n_discovered'] for p in points], [p['encode_mean'] for p in points], "n_discovered (evolving_visible_graph nodes)")
results['action_space_vs_owned_x_discovered'] = report(
    "create_continuous_action_space() mean time", [p['n_owned'] * p['n_discovered'] for p in points],
    [p['action_space_mean'] for p in points], "n_owned * n_discovered")
results['match_vs_n_actions'] = report(
    "find_closest_action_embedding() mean time", [p['n_actions'] for p in points], [p['match_mean'] for p in points], "n_action_embeddings (candidates)")

# vs raw topology N, for the extrapolate-to-500/1000/2500 requirement
results['encode_vs_N'] = report(
    "encode() mean time", [p['n_topology'] for p in points], [p['encode_mean'] for p in points], "topology N (total nodes)")
results['action_space_vs_N'] = report(
    "create_continuous_action_space() mean time", [p['n_topology'] for p in points], [p['action_space_mean'] for p in points], "topology N (total nodes)")
results['match_vs_N'] = report(
    "find_closest_action_embedding() mean time", [p['n_topology'] for p in points], [p['match_mean'] for p in points], "topology N (total nodes)")

with open('taskI_clean_fit_results.json', 'w') as f:
    def clean(d):
        return {k: (clean(v) if isinstance(v, dict) else v) for k, v in d.items()}
    json.dump({k: clean(v) for k, v in results.items()}, f, indent=2, default=str)

print("\nSaved to taskI_clean_fit_results.json")
