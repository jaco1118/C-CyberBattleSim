"""
Task F3 appendix: recompute on the CHURN-INVARIANT measure = root-owned COUNT (numerator only,
NOT divided by reachable). Robustness = count(change)/count(static) within a band (units cancel).
Analysis only, existing data. Mechanical/behavioural split redone on counts: on the count measure a
root-owned departure is a full -1 (no denominator offset), so mechanical = root-owned departures per
episode (from the mech runs' was_root), behavioural = cost_count - mechanical.

Data (root_owned COUNT from score CSVs; was_root per removal from mech CSVs):
  30-40  static: eval_out/score_*_evalstatic ; membership(~33%): eval_out/score_*_evalmembership ; mech: f3_mech_out/mech_30-40
  80-100 static: f2eval_out/score_*_evalstatic ; fixed-abs(~18%): f2eval_out/score_*_evalmembership ; mech: f3_mech_out/mech_80-100
  80-100 fixed-relative(32.3%): f3_rel_out/score_*_evalmembership_matched ; mech: f3_rel_out/mech_80-100rel
"""
import pandas as pd, numpy as np
BASE = "/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1"
SEEDS = [42, 100, 123, 200, 300]; RNG = np.random.default_rng(11); NB = 10000
BANNER = ("PROVISIONAL (donor-pool confound, Task G pending): membership_join draws from a shared "
          "donor pool ~2.2x weaker at the large band; all join-related numbers inherit this caveat.")

def col(path_tmpl, colname):
    return {s: pd.read_csv(path_tmpl.format(s=s))[colname].to_numpy() for s in SEEDS}
def rootdep_per_ep(mech_tmpl):
    # mean root-owned departures per episode, per seed
    out = {}
    for s in SEEDS:
        d = pd.read_csv(mech_tmpl.format(s=s))
        neps = d["episode"].nunique()
        out[s] = d["was_root"].sum() / neps
    return out

CONDS = {
 "30-40 static":            (f"{BASE}/eval_out/score_static_seed{{s}}_evalstatic.csv", None),
 "30-40 fixed-abs (~33%)":  (f"{BASE}/eval_out/score_static_seed{{s}}_evalmembership.csv", f"{BASE}/f3_mech_out/mech_30-40_seed{{s}}.csv"),
 "80-100 static":           (f"{BASE}/f2eval_out/score_static_seed{{s}}_evalstatic.csv", None),
 "80-100 fixed-abs (~18%)": (f"{BASE}/f2eval_out/score_static_seed{{s}}_evalmembership.csv", f"{BASE}/f3_mech_out/mech_80-100_seed{{s}}.csv"),
 "80-100 fixed-rel (32.3%)":(f"{BASE}/f3_rel_out/score_static_seed{{s}}_evalmembership_matched.csv", f"{BASE}/f3_rel_out/mech_80-100rel_seed{{s}}.csv"),
}
counts = {name: col(tmpl, "root_owned") for name,(tmpl,_) in CONDS.items()}
def ci(a): return (float(np.percentile(a,2.5)), float(np.percentile(a,97.5)))

print("="*88); print("F3 APPENDIX — CHURN-INVARIANT measure: root-owned COUNT (metric = ROOT-OWNED COUNT)")
print(BANNER); print("="*88)
print("\n### root-owned COUNT per condition (mean over 1000 ep) [ARTIFACT] ###")
for name in CONDS:
    c = np.concatenate([counts[name][s] for s in SEEDS])
    print(f"  {name:26s}: mean_count={c.mean():.3f}  median={np.median(c):.1f}  (per-seed {[round(counts[name][s].mean(),2) for s in SEEDS]})")

print("\n### robustness (count-based, WITHIN band) + cost + mechanical/behavioural split [FINDING] ###")
pairs = [("30-40 fixed-abs (~33%)","30-40 static"),
         ("80-100 fixed-abs (~18%)","80-100 static"),
         ("80-100 fixed-rel (32.3%)","80-100 static")]
for change, base in pairs:
    ch = counts[change]; bs = counts[base]
    chp = np.concatenate([ch[s] for s in SEEDS]); bsp = np.concatenate([bs[s] for s in SEEDS])
    cost = bsp.mean() - chp.mean(); robu = chp.mean()/bsp.mean()
    mech_dep = rootdep_per_ep(CONDS[change][1])
    mech = np.mean([mech_dep[s] for s in SEEDS]); behav = cost - mech
    # bootstrap over replicates
    bc=[];bm=[];bb=[];br=[]
    for _ in range(NB):
        chs = RNG.choice(SEEDS,5,replace=True)
        c2=np.concatenate([ch[s] for s in chs]); b2=np.concatenate([bs[s] for s in chs])
        m2=np.mean([mech_dep[s] for s in chs]); cst=b2.mean()-c2.mean()
        bc.append(cst); bm.append(m2); bb.append(cst-m2); br.append(c2.mean()/b2.mean())
    print(f"\n  {change}  vs  {base}")
    print(f"    static count={bsp.mean():.3f}  change count={chp.mean():.3f}")
    print(f"    COST (count) = {cost:+.3f} CI95 [{ci(bc)[0]:+.3f},{ci(bc)[1]:+.3f}]   ROBUSTNESS = {robu:.4f} CI95 [{ci(br)[0]:.4f},{ci(br)[1]:.4f}]")
    print(f"    MECHANICAL (root-owned departures/ep) = {mech:.3f} CI95 [{ci(bm)[0]:.3f},{ci(bm)[1]:.3f}]  ({mech/cost*100:.0f}% of cost)" if cost>0 else
          f"    MECHANICAL (root-owned departures/ep) = {mech:.3f} CI95 [{ci(bm)[0]:.3f},{ci(bm)[1]:.3f}]  (cost<=0, fraction n/a)")
    print(f"    BEHAVIOURAL = cost - mechanical = {behav:+.3f} CI95 [{ci(bb)[0]:+.3f},{ci(bb)[1]:+.3f}]")

print("\n### ratio-metric (root_owned/reachable) vs count-metric split — does the offset disappear? ###")
print("  ratio metric (Condition-B-comparable, KEPT for replication): 30-40 mech 32%, behav 68%; 80-100abs mech 19%, behav 81%.")
print("  On the COUNT metric there is NO denominator, so NO offsetting gain from non-root departures.")
print("\n"+"="*88); print(BANNER); print("="*88)
