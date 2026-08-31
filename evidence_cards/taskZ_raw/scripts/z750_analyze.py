"""Reproduction of the '750k ROBUSTNESS at 80-100' figures cited in evidence_taskZ.md, using the
identical method as z_step2_analyze.py (same MDE definition, same paired-seed bootstrap, NB=20000,
RNG seed=7), pointed at the preserved z750_eval/ per-seed CSVs instead of z_step2_out/.
Read-only: no training, no new evaluation episodes."""
import pandas as pd, numpy as np, os
B = "/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1/z750_eval"
SEEDS = [42, 100, 123, 200, 300]; T80 = {42:5,100:100,123:18,200:2,300:67}
RNG = np.random.default_rng(7); NB = 20000

def topo(s): return f"scalability_80_100_{T80[s]}"

def load(subdir, arm, s, cond):
    f = f"{B}/{subdir}/zscore_arm{arm}_{topo(s)}_seed{s}_{cond}.csv"
    return pd.read_csv(f)["root_owned"].to_numpy() if os.path.exists(f) else None

def perseed_means(subdir, arm, cond):
    return np.array([load(subdir, arm, s, cond).mean() for s in SEEDS])

STATIC = {arm: perseed_means("static", arm, "static") for arm in (1,2,3)}
CHG_REL = {arm: perseed_means("rel", arm, "membership") for arm in (1,2,3)}
CHG_ABS = {arm: perseed_means("abs", arm, "membership") for arm in (1,2,3)}

def ci(vals): return (float(np.percentile(vals,2.5)), float(np.percentile(vals,97.5)))
def boot_diff(a, b):
    out=[]
    for _ in range(NB):
        idx = RNG.integers(0,len(SEEDS),len(SEEDS)); out.append(a[idx].mean()-b[idx].mean())
    return a.mean()-b.mean(), ci(np.array(out))

print("=== 750k @ 80-100: static (per-arm, per-seed) ===")
for arm in (1,2,3):
    m = STATIC[arm]
    print(f"  arm{arm}: mean={m.mean():.3f} SD={m.std(ddof=1):.3f}  seeds={[round(x,2) for x in m]}")

paired_static_diff = STATIC[1] - STATIC[3]
thr_paired = paired_static_diff.std(ddof=1)
thr_arm = max(STATIC[1].std(ddof=1), STATIC[3].std(ddof=1))
mde = max(thr_paired, thr_arm)
print(f"\nMDE: paired(arm1-arm3) SD={thr_paired:.3f}; max single-arm SD={thr_arm:.3f} -> MDE={mde:.3f}")

for COND, CHG, tag in ((CHG_REL, CHG_REL, "fixed-REL (primary)"), (CHG_ABS, CHG_ABS, "fixed-ABS")):
    a1,a2,a3 = CHG[1], CHG[2], CHG[3]
    info, ci_i = boot_diff(a1,a3); cap, ci_c = boot_diff(a2,a3); raw, ci_r = boot_diff(a1,a2)
    print(f"\n=== {tag} ===")
    print(f"  INFO (a1-a3) = {info:+.3f}  CI95 [{ci_i[0]:+.3f},{ci_i[1]:+.3f}]")
    print(f"  CAP  (a2-a3) = {cap:+.3f}  CI95 [{ci_c[0]:+.3f},{ci_c[1]:+.3f}]")
    print(f"  RAW  (a1-a2) = {raw:+.3f}  CI95 [{ci_r[0]:+.3f},{ci_r[1]:+.3f}]")
