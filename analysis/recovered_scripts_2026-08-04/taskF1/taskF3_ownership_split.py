"""
Task F3 ownership split: decompose the per-episode membership cost at BOTH bands into
MECHANICAL (departures of already-OWNED nodes) vs BEHAVIOURAL (departures of DISCOVERED-BUT-NOT-
OWNED nodes). Empirical attribution (the metric root_owned/reachable couples numerator and
denominator, so no closed-form arithmetic split): within membership episodes, OLS
    score_i = b0 + b_owned * owned_dep_i + b_unowned * unowned_dep_i
Mechanical = -b_owned * E[owned_dep]; Behavioural = -b_unowned * E[unowned_dep];
intercept gap = mean(static) - b0 (churn cost not linear in departure counts). Cluster-bootstrap
over replicates (refit each draw). Reports the split as a fraction of total cost, per band.

Data:
  30-40: f3_own_out/{leaveown,score}_static_seed*_evalmembership.csv ; static from eval_out/score_static_seed*_evalstatic.csv
  80-100: f2eval_out/{leaveown,score}_static_seed*_evalmembership.csv ; static from f2eval_out/score_static_seed*_evalstatic.csv
"""
import os, glob
import numpy as np, pandas as pd
BASE = "/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1"
SEEDS = [42, 100, 123, 200, 300]
RNG = np.random.default_rng(20260729); NBOOT = 10000
BANNER = ("PROVISIONAL (donor-pool confound, Task G pending): membership_join draws from a shared "
          "donor pool ~2.2x weaker at the large band; all join-related numbers inherit this caveat.")

BANDS = {
    "30-40": dict(own_dir=f"{BASE}/f3_own_out", static_dir=f"{BASE}/eval_out"),
    "80-100": dict(own_dir=f"{BASE}/f2eval_out", static_dir=f"{BASE}/f2eval_out"),
}

def per_episode(band):
    d = BANDS[band]
    rows = []
    for s in SEEDS:
        lo = pd.read_csv(f"{d['own_dir']}/leaveown_static_seed{s}_evalmembership.csv")
        sc = pd.read_csv(f"{d['own_dir']}/score_static_seed{s}_evalmembership.csv")
        agg = lo.groupby("episode").agg(owned_dep=("was_owned","sum"),
                                        total_dep=("was_owned","size")).reset_index()
        agg["unowned_dep"] = agg["total_dep"] - agg["owned_dep"]
        m = sc[["seed","episode","score"]].merge(agg, on="episode", how="left").fillna({"owned_dep":0,"unowned_dep":0,"total_dep":0})
        m["seed"] = s
        rows.append(m)
    return pd.concat(rows, ignore_index=True)

def static_mean(band):
    d = BANDS[band]
    vs = []
    for s in SEEDS:
        vs.append(pd.read_csv(f"{d['static_dir']}/score_static_seed{s}_evalstatic.csv")["score"].to_numpy())
    return np.concatenate(vs)

def ols(df):
    X = np.column_stack([np.ones(len(df)), df["owned_dep"].to_numpy(), df["unowned_dep"].to_numpy()])
    y = df["score"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta  # b0, b_owned, b_unowned

print("="*84); print("TASK F3 — OWNERSHIP SPLIT of the membership cost (mechanical vs behavioural)")
print(BANNER); print("="*84)
print("Method: OLS score ~ owned_dep + unowned_dep within membership episodes (empirical, not")
print("arithmetic — the metric couples numerator/denominator). Cluster-bootstrap over 5 replicates.")
print("CAVEAT: associational — owned_dep and score are jointly determined (a better episode owns more,")
print("so more owned departures are available), so coefficients are NOT clean causal effects.")

for band in ["30-40", "80-100"]:
    df = per_episode(band); stat = static_mean(band)
    cost = stat.mean() - df["score"].mean()
    beta = ols(df); b0, bo, bu = beta
    Eown = df["owned_dep"].mean(); Eun = df["unowned_dep"].mean()
    mech = -bo * Eown; behav = -bu * Eun; gap = stat.mean() - b0
    # cluster bootstrap over replicates
    bs_mech=[]; bs_behav=[]; bs_bo=[]; bs_bu=[]; bs_cost=[]
    seed_groups = {s: df[df.seed==s] for s in SEEDS}
    stat_groups = {s: pd.read_csv(f"{BANDS[band]['static_dir']}/score_static_seed{s}_evalstatic.csv")["score"].to_numpy() for s in SEEDS}
    for _ in range(NBOOT):
        chosen = RNG.choice(SEEDS, len(SEEDS), replace=True)
        dd = pd.concat([seed_groups[s] for s in chosen], ignore_index=True)
        ss = np.concatenate([stat_groups[s] for s in chosen])
        bb = ols(dd)
        bs_bo.append(bb[1]); bs_bu.append(bb[2])
        bs_mech.append(-bb[1]*dd["owned_dep"].mean()); bs_behav.append(-bb[2]*dd["unowned_dep"].mean())
        bs_cost.append(ss.mean()-dd["score"].mean())
    def ci(a): return (np.percentile(a,2.5), np.percentile(a,97.5))
    print(f"\n### {band} [FINDING] ###")
    print(f"  total cost = {cost:+.4f}  (static mean {stat.mean():.4f} - membership mean {df['score'].mean():.4f})")
    print(f"  E[owned_dep/ep]={Eown:.2f}  E[unowned_dep/ep]={Eun:.2f}  owned_frac={Eown/(Eown+Eun):.3f}")
    print(f"  b_owned  = {bo:+.5f} CI95 [{ci(bs_bo)[0]:+.5f},{ci(bs_bo)[1]:+.5f}]  (score change per owned departure)")
    print(f"  b_unowned= {bu:+.5f} CI95 [{ci(bs_bu)[0]:+.5f},{ci(bs_bu)[1]:+.5f}]  (score change per un-owned departure)")
    print(f"  MECHANICAL  component = {mech:+.4f} CI95 [{ci(bs_mech)[0]:+.4f},{ci(bs_mech)[1]:+.4f}]  ({mech/cost*100:.0f}% of total cost)")
    print(f"  BEHAVIOURAL component = {behav:+.4f} CI95 [{ci(bs_behav)[0]:+.4f},{ci(bs_behav)[1]:+.4f}]  ({behav/cost*100:.0f}% of total cost)")
    print(f"  intercept gap (static - b0) = {gap:+.4f}  [model zero-churn score b0={b0:.4f} vs static {stat.mean():.4f}]")
    # raw stratified association (robustness): mean score by owned_dep tercile
    q = df["owned_dep"].quantile([0.33,0.66]).values
    lowm = df[df.owned_dep<=q[0]]["score"].mean(); highm = df[df.owned_dep>q[1]]["score"].mean()
    print(f"  raw check: mean score at LOW owned_dep (<= {q[0]:.0f}) = {lowm:.3f} ; HIGH owned_dep (> {q[1]:.0f}) = {highm:.3f}")

print("\n### which quantity to compare against the attenuation curve ###")
print("  The BEHAVIOURAL component is the RQ3-relevant one: mechanical loss is independent of whether")
print("  the agent perceived anything. If behavioural is near zero at both bands, that is a clean")
print("  negative answer to RQ3's causal half.")
print("\n"+"="*84); print(BANNER); print("="*84)
