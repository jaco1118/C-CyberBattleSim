"""
Task I STEP 2: fit cost-vs-N curves per component using the 21 raw (7 sizes x 3 repeats)
measurements from taskI_profile_one_topology.py. Fits linear, N log N, and quadratic forms to
mean-per-encode-call time and mean-per-step time, reports fitted exponent (power-law fit,
log-log OLS) with a 95% CI and R^2 for each candidate form, and extrapolates to N=500/1000/2500.
"""
import json, glob
import numpy as np
from scipy import stats

files = sorted(glob.glob('result_N*.json'), key=lambda f: int(f.split('N')[1].split('.')[0]))

data = []
for f in files:
    d = json.load(open(f))
    for r in d['repeats']:
        data.append(r)

Ns = sorted(set(r['n_nodes_ground_truth'] for r in data))
print("Sizes measured:", Ns)


def component_series(key):
    # returns dict N -> list of values across repeats
    out = {}
    for r in data:
        n = r['n_nodes_ground_truth']
        v = r[key]
        if v is None:
            continue
        out.setdefault(n, []).append(v)
    return out


def summarize(series):
    rows = []
    for n in sorted(series.keys()):
        vals = np.array(series[n])
        rows.append((n, vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0, vals.min(), vals.max(), len(vals)))
    return rows


def power_law_fit(series):
    # log-log OLS: log(y) = a + b*log(N) -> y = exp(a) * N^b
    xs, ys = [], []
    for n, vals in series.items():
        for v in vals:
            if v > 0:
                xs.append(np.log(n))
                ys.append(np.log(v))
    xs = np.array(xs); ys = np.array(ys)
    slope, intercept, r, p, se = stats.linregress(xs, ys)
    # 95% CI on slope: t-distribution, df = n-2
    dfree = len(xs) - 2
    tval = stats.t.ppf(0.975, dfree) if dfree > 0 else float('nan')
    ci = (slope - tval * se, slope + tval * se)
    return dict(exponent=slope, exponent_ci95=ci, intercept=intercept, r_squared=r**2, n_points=len(xs))


def fit_and_compare(series, label):
    means = {n: np.mean(v) for n, v in series.items() if any(x > 0 for x in v)}
    Ns_arr = np.array(sorted(means.keys()))
    Ys_arr = np.array([means[n] for n in Ns_arr])

    results = {}
    # Linear: y = a + b*N
    slope, intercept, r, p, se = stats.linregress(Ns_arr, Ys_arr)
    results['linear'] = dict(r_squared=r**2, slope=slope, intercept=intercept)
    # N log N: y = a + b*(N log N)
    x_nlogn = Ns_arr * np.log(Ns_arr)
    slope2, intercept2, r2, p2, se2 = stats.linregress(x_nlogn, Ys_arr)
    results['n_log_n'] = dict(r_squared=r2**2, slope=slope2, intercept=intercept2)
    # Quadratic: y = a + b*N^2
    x_quad = Ns_arr ** 2
    slope3, intercept3, r3, p3, se3 = stats.linregress(x_quad, Ys_arr)
    results['quadratic'] = dict(r_squared=r3**2, slope=slope3, intercept=intercept3)
    # Power law (log-log)
    pw = power_law_fit(series)
    results['power_law'] = pw

    best = max(
        [('linear', results['linear']['r_squared']), ('n_log_n', results['n_log_n']['r_squared']),
         ('quadratic', results['quadratic']['r_squared']), ('power_law', results['power_law']['r_squared'])],
        key=lambda t: t[1]
    )
    print(f"\n=== {label} ===")
    print(f"  means by N: {[(n, round(means[n], 6)) for n in Ns_arr]}")
    print(f"  linear     R2={results['linear']['r_squared']:.4f}  slope={results['linear']['slope']:.6g}  intercept={results['linear']['intercept']:.6g}")
    print(f"  N log N    R2={results['n_log_n']['r_squared']:.4f}  slope={results['n_log_n']['slope']:.6g}  intercept={results['n_log_n']['intercept']:.6g}")
    print(f"  quadratic  R2={results['quadratic']['r_squared']:.4f}  slope={results['quadratic']['slope']:.6g}  intercept={results['quadratic']['intercept']:.6g}")
    print(f"  power law  R2={results['power_law']['r_squared']:.4f}  exponent={results['power_law']['exponent']:.4f}  "
          f"95% CI=({results['power_law']['exponent_ci95'][0]:.4f}, {results['power_law']['exponent_ci95'][1]:.4f})  "
          f"n_points={results['power_law']['n_points']}")
    print(f"  BEST FIT: {best[0]} (R2={best[1]:.4f})")
    return results, best


components = {
    'mean_time_per_encode_call_s': 'GAE encode() -- mean wall-clock per call',
    'mean_time_per_action_match_s': 'find_closest_action_embedding -- mean wall-clock per call',
    'action_space_creation_time_total_s': 'create_continuous_action_space -- total per-episode (300 steps)',
    'mean_wallclock_per_step_s': 'TOTAL mean wall-clock per env step (sum of tracked components)',
    'mean_inner_step_time_s': 'step_attacker_env (base env simulation) -- mean wall-clock per step',
    'encode_calls_per_episode': 'encode() calls per episode (300-step episode)',
}

fit_summary = {}
for key, label in components.items():
    series = component_series(key)
    results, best = fit_and_compare(series, label)
    fit_summary[key] = dict(label=label, results=results, best=best)

# memory
print("\n=== Peak RSS after run (KB) ===")
rss_series = component_series('rss_peak_kb_after')
for n, vals in sorted(rss_series.items()):
    print(f"  N={n}: mean={np.mean(vals):.0f} KB, spread={vals}")

with open('taskI_fit_summary.json', 'w') as f:
    json.dump({k: {'label': v['label'], 'best_fit': v['best'][0], 'best_r2': v['best'][1],
                    'power_law_exponent': v['results']['power_law']['exponent'],
                    'power_law_ci95': v['results']['power_law']['exponent_ci95']}
               for k, v in fit_summary.items()}, f, indent=2)

print("\nSaved fit summary to taskI_fit_summary.json")
