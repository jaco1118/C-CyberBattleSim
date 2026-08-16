"""Task Z STEP 2 measurement. Pre-registered order (NOT renegotiable): the 2.4 static null and the power
statement (MDE) are computed and printed BEFORE any arm difference. Primary condition = fixed-RELATIVE
(30-40 CI=20, 80-100 CI=8); secondary = fixed-ABSOLUTE (both CI=20). Metric = root-owned COUNT.
All differences bootstrapped over the 5 seeds (seed = unit of variance)."""
import pandas as pd, numpy as np, os
B = "/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1"
SEEDS = [42, 100, 123, 200, 300]; T80 = {42:5,100:100,123:18,200:2,300:67}
RNG = np.random.default_rng(7); NB = 20000
def topo(band, s): return "scalability_30_40_44" if band=="30-40" else f"scalability_80_100_{T80[s]}"
def load(subdir, band, arm, s, cond):
    f = f"{B}/z_step2_out/{subdir}/zscore_arm{arm}_{topo(band,s)}_seed{s}_{cond}.csv"
    return pd.read_csv(f)["root_owned"].to_numpy() if os.path.exists(f) else None
def perseed_means(subdir, band, arm, cond):
    return np.array([load(subdir, band, arm, s, cond).mean() for s in SEEDS])
# per-arm per-band per-source per-seed means
STATIC = {(band,arm): perseed_means("static", band, arm, "static") for band in ("30-40","80-100") for arm in (1,2,3)}
CHG = {}  # (condition, band, arm) -> per-seed means
for arm in (1,2,3):
    CHG[("rel","30-40",arm)] = perseed_means("chg_3040", "30-40", arm, "membership")
    CHG[("abs","30-40",arm)] = CHG[("rel","30-40",arm)]  # 30-40 identical (CI=20) for both
    CHG[("abs","80-100",arm)] = perseed_means("chg_80100_abs", "80-100", arm, "membership")
    CHG[("rel","80-100",arm)] = perseed_means("chg_80100_rel", "80-100", arm, "membership")
def ci(vals): return (float(np.percentile(vals,2.5)), float(np.percentile(vals,97.5)))
def boot_diff(a, b):  # per-seed arrays; bootstrap band-mean(a)-band-mean(b) over seeds (paired)
    out=[]
    for _ in range(NB):
        idx = RNG.integers(0,len(SEEDS),len(SEEDS)); out.append(a[idx].mean()-b[idx].mean())
    return a.mean()-b.mean(), ci(np.array(out))

print("="*94)
print("TASK Z STEP 2 — three-arm pooling ablation. Metric: root-owned COUNT. 250k (PRIMARY budget).")
print("Arm1=full 256 | Arm2=mean-only 128 (capacity ctrl) | Arm3=256 extremal-zeroed (info ctrl).")
print("="*94)

print("\n########## 2.4 NULL — static between-seed spread (THE NULL THRESHOLD), reported FIRST ##########")
print(f"  {'band':>7} {'arm':>4} {'static mean':>12} {'between-seed SD':>16} {'seed means':>34}")
for band in ("30-40","80-100"):
    for arm in (1,2,3):
        m = STATIC[(band,arm)]
        print(f"  {band:>7} {arm:>4} {m.mean():>12.3f} {m.std(ddof=1):>16.3f}   {[round(x,2) for x in m]}")

print("\n########## POWER STATEMENT (a,b reported BEFORE any difference) ##########")
# threshold for the info effect (arm1-arm3): between-seed SD of the paired static difference (arm1-arm3),
# and also the larger single-arm static SD, both reported. MDE = the null threshold.
POW = {}
for band in ("30-40","80-100"):
    paired_static_diff = STATIC[(band,1)] - STATIC[(band,3)]
    thr_paired = paired_static_diff.std(ddof=1)
    thr_arm = max(STATIC[(band,1)].std(ddof=1), STATIC[(band,3)].std(ddof=1))
    static_mean = STATIC[(band,1)].mean()
    mde = max(thr_paired, thr_arm)   # conservative: an effect must clear the larger noise estimate
    POW[band] = (mde, static_mean, thr_paired, thr_arm)
    print(f"  band {band}:")
    print(f"    (a) NULL THRESHOLD (static seed spread): paired(arm1-arm3) SD={thr_paired:.3f} ; "
          f"max single-arm SD={thr_arm:.3f}  -> using MDE={mde:.3f} nodes")
    print(f"    (b) MIN DETECTABLE EFFECT = {mde:.3f} root-owned nodes = {100*mde/static_mean:.1f}% of static mean ({static_mean:.2f})")

def label_null(band, observed_abs, mde, ref_effect):
    # ref_effect = the 30-40 observed info effect (a plausible expected magnitude)
    if observed_abs >= mde: return "MEASURABLE"
    if mde > abs(ref_effect): return "UNDERPOWERED (MDE exceeds a 30-40-sized effect; null carries no info)"
    return "EFFECT ABSENT (powered to detect a 30-40-sized effect; none found)"

for COND, tag in (("rel","PRIMARY (fixed-RELATIVE)"), ("abs","SECONDARY (fixed-ABSOLUTE)")):
    print("\n" + "="*94)
    print(f"CONDITION: {tag}")
    print("="*94)
    print("\n## 2.1 root-owned COUNT under change, per arm (mean over seeds, [per-seed]) ##")
    for band in ("30-40","80-100"):
        for arm in (1,2,3):
            m = CHG[(COND,band,arm)]
            print(f"  {band:>7} arm{arm}: change={m.mean():7.3f} SD={m.std(ddof=1):5.3f}  static={STATIC[(band,arm)].mean():7.3f}  [{[round(x,1) for x in m]}]")
    print("\n## 2.2 three differences (under change), bootstrap CI over seeds ##")
    info_by_band = {}
    for band in ("30-40","80-100"):
        a1,a2,a3 = (CHG[(COND,band,k)] for k in (1,2,3))
        info,ci_i = boot_diff(a1,a3); cap,ci_c = boot_diff(a2,a3); raw,ci_r = boot_diff(a1,a2)
        info_by_band[band] = info
        print(f"  band {band}:")
        print(f"    INFORMATION (arm1-arm3) = {info:+.3f}  CI95 [{ci_i[0]:+.3f},{ci_i[1]:+.3f}]")
        print(f"    CAPACITY    (arm2-arm3) = {cap:+.3f}  CI95 [{ci_c[0]:+.3f},{ci_c[1]:+.3f}]")
        print(f"    RAW GAP     (arm1-arm2) = {raw:+.3f}  CI95 [{ci_r[0]:+.3f},{ci_r[1]:+.3f}]")
    print("\n## 2.3 INFORMATION EFFECT: 30-40 vs 80-100 (THE ANSWER) ##")
    i30, i80 = info_by_band["30-40"], info_by_band["80-100"]
    print(f"    30-40 info effect = {i30:+.3f} nodes ; 80-100 info effect = {i80:+.3f} nodes")
    print(f"    direction: {'larger at 30-40' if abs(i30)>abs(i80) else 'larger at 80-100' if abs(i80)>abs(i30) else 'equal'} "
          f"(expected: larger at 30-40)")
    print("\n## null labelling (per 2.4 + power) ##")
    for band in ("30-40","80-100"):
        mde,_,_,_ = POW[band]; obs = abs(info_by_band[band])
        print(f"    {band}: |info|={obs:.3f} vs MDE={mde:.3f} -> {label_null(band, obs, mde, i30)}")

print("\n"+"="*94)
print("Reminder: 250k is PRIMARY. 750k-at-80-100 is a separate robustness check (undertraining artefact?).")
print("Fixed-relative is PRIMARY; if fixed-absolute disagrees, primary stands, disagreement is a finding.")
print("="*94)
