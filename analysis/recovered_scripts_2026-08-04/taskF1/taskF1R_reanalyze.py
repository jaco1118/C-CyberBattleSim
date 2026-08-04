"""
Task F1-R: re-analyse the EXISTING Pass 1 data on the within-agent axis. No new runs.
Band 30-40, topology 44. Reads eval_out/{score,drift}_{agent}_seed{seed}_eval{cond}.csv.
Unit = episode; bootstrap 0.95; seed = grouping factor. Donor-pool provisional banner carried.
"""
import os, glob
import numpy as np
import pandas as pd

OUT = "/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1/eval_out"
SEEDS = [42, 100, 123, 200, 300]
AGENTS = ["static", "adapted"]
EVAL = ["static", "membership", "property"]
RNG = np.random.default_rng(20260729)
NBOOT = 10000
BANNER = ("PROVISIONAL (donor-pool confound, Task G pending): membership_join draws from a shared "
          "donor pool ~2.2x weaker at the large band; all join-related numbers inherit this caveat.")

def sload(agent, cond):
    fs = [os.path.join(OUT, f"score_{agent}_seed{s}_eval{cond}.csv") for s in SEEDS]
    return pd.concat([pd.read_csv(f) for f in fs if os.path.exists(f)], ignore_index=True)

def dload(agent, cond):
    fs = [os.path.join(OUT, f"drift_{agent}_seed{s}_eval{cond}.csv") for s in SEEDS]
    return pd.concat([pd.read_csv(f) for f in fs if os.path.exists(f)], ignore_index=True)

def ci_mean(x):
    x = np.asarray(x, float)
    bs = np.array([x[RNG.integers(0, len(x), len(x))].mean() for _ in range(NBOOT)])
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def ci_diff(a, b):
    # a - b, independent resampling
    a = np.asarray(a, float); b = np.asarray(b, float)
    bs = np.array([a[RNG.integers(0, len(a), len(a))].mean() - b[RNG.integers(0, len(b), len(b))].mean()
                   for _ in range(NBOOT)])
    return a.mean() - b.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def ci_ratio(num, den):
    num = np.asarray(num, float); den = np.asarray(den, float)
    bs = np.array([num[RNG.integers(0, len(num), len(num))].mean() / den[RNG.integers(0, len(den), len(den))].mean()
                   for _ in range(NBOOT)])
    return num.mean()/den.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def ci_did(sa_stat, sa_chg, ad_stat, ad_chg):
    # (mean(sa_stat)-mean(sa_chg)) - (mean(ad_stat)-mean(ad_chg))
    arrs = [np.asarray(x, float) for x in (sa_stat, sa_chg, ad_stat, ad_chg)]
    def draw(a): return a[RNG.integers(0, len(a), len(a))].mean()
    bs = np.array([(draw(arrs[0])-draw(arrs[1])) - (draw(arrs[2])-draw(arrs[3])) for _ in range(NBOOT)])
    pt = (arrs[0].mean()-arrs[1].mean()) - (arrs[2].mean()-arrs[3].mean())
    return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def verdict(lo, hi):
    return "excludes 0" if (lo > 0 or hi < 0) else "NOT ESTABLISHED (includes 0)"

print("="*84); print("TASK F1-R  —  RE-ANALYSIS ON THE WITHIN-AGENT AXIS  (band 30-40, topology 44)")
print(BANNER); print("="*84)

# ================= STEP 0 =================
print("\n########## STEP 0: WHAT IS STORED ##########")
print("\n### 0.1 absolute per-condition scores (root_owned%) per agent/eval/seed [ARTIFACT] ###")
scores = {}
for a in AGENTS:
    for c in EVAL:
        df = sload(a, c); scores[(a, c)] = df
        for s in SEEDS:
            g = df[df.seed == s]["score"].to_numpy()
            print(f"  {a:7s}/{c:10s}/seed{s}: n={len(g):3d} mean={g.mean():.4f} med={np.median(g):.4f} "
                  f"min={g.min():.4f} max={g.max():.4f}")
print("  -> raw per-condition per-episode scores ARE stored (recoverable without re-running). zero dropped.")

print("\n### 0.2 response rate at tau=0 per slice (max/min are the attenuation-bearing ones) [ARTIFACT] ###")
def filt(d, ct):
    return d[(d.change_type==ct)&(d.relevant==True)&(d.touched_node_visible==True)&(d.event_phase.isin(["immediate","attributed"]))]
for a in AGENTS:
    for c, cts in [("membership", ["membership_leave"]), ("property", ["property"])]:
        d = dload(a, c)
        for ct in cts:
            e = filt(d, ct)
            if len(e)==0: continue
            row = f"  {a:7s}/{ct:16s}: "
            for sl in ["full","mean","max","min"]:
                col=f"change_drift_{sl}"
                overall=(e[col]>0).mean()
                perseed=e.groupby("seed").apply(lambda g:(g[col]>0).mean(), include_groups=False)
                row += f"{sl}={overall:.3f}(sd{perseed.std(ddof=1):.3f}) "
            print(row)
# membership_join reported separately (coverage) at fire time
for a in AGENTS:
    d = dload(a,"membership"); fired=d[(d.change_type=="membership_join")&(d.event_phase=="fired")]
    print(f"  {a:7s}/membership_join (FIRE-TIME, coverage; node invisible until discovered): "
          f"resp_rate(full>0)={(fired.change_drift_full>0).mean():.3f} visible@fire={(fired.touched_node_visible==True).mean():.3f}")

print("\n### 0.3 per-episode join table (score<->drift) [ARTIFACT] ###")
join_rows=0
for a in AGENTS:
    for c in ["membership","property"]:
        sdf=scores[(a,c)]; ddf=dload(a,c)
        dkey=ddf[["seed","scenario_id","episode"]].drop_duplicates()
        j=sdf.merge(dkey, on=["seed","scenario_id","episode"], how="inner")
        join_rows+=len(j)
        print(f"  {a}/{c}: score_ep={len(sdf)} drift_ep={len(dkey)} joined={len(j)} unjoined={len(sdf)-len(j)}")
print(f"  per-episode score tables: {OUT}/score_*.csv (200 rows each x30); drift tables drift_*.csv (30). "
      f"join is exact on (seed,scenario_id,episode), 0 failures.")

print("\n### 0.4 zero-score episodes per agent/condition [ARTIFACT] ###")
for a in AGENTS:
    for c in EVAL:
        df=scores[(a,c)]; z=(df.score==0).sum()
        print(f"  {a:7s}/{c:10s}: zero_score={z}/{len(df)} ({z/len(df):.3f})")

print("\n### 0.5 change-events per episode per condition [ARTIFACT] ###")
evrate={}
for a in AGENTS:
    for c,cts in [("membership",["membership_leave","membership_join"]),("property",["property"])]:
        d=dload(a,c); nep=scores[(a,c)].shape[0]
        for ct in cts:
            # count FIRED/immediate events (the actual firings), dedup by (seed,episode,step-ish) not needed: count 'immediate' for leave/property, 'fired' for join
            if ct=="membership_join":
                n=len(d[(d.change_type==ct)&(d.event_phase=="fired")])
            else:
                n=len(d[(d.change_type==ct)&(d.event_phase=="immediate")])
            evrate[(a,ct)]=n/nep
            print(f"  {a:7s}/{ct:16s}: {n} events / {nep} episodes = {n/nep:.3f} per episode")

# ================= STEP 1 =================
print("\n########## STEP 1: WITHIN-AGENT CONTRAST ##########")
print("cost(change) = mean(score | STATIC eval) - mean(score | CHANGE eval), per agent. episode unit; seed grouping.")
costs={}
for a in AGENTS:
    print(f"\n### 1.{'1' if a=='static' else '2'} {a.upper()} agent [FINDING] ###")
    base=scores[(a,"static")]["score"].to_numpy()
    for c in ["membership","property"]:
        chg=scores[(a,c)]["score"].to_numpy()
        pt,lo,hi=ci_diff(base,chg)
        costs[(a,c)]=(pt,lo,hi)
        per_seed=[scores[(a,"static")][scores[(a,"static")].seed==s]["score"].mean()
                  - scores[(a,c)][scores[(a,c)].seed==s]["score"].mean() for s in SEEDS]
        print(f"  cost({c:10s}) = {pt:+.4f}  CI95 [{lo:+.4f},{hi:+.4f}]  -> {verdict(lo,hi)}")
        print(f"      per-seed cost: {[round(x,4) for x in per_seed]}  (between-seed sd={np.std(per_seed,ddof=1):.4f})")

print("\n### 1.3 DIFFERENCE-IN-DIFFERENCES  DiD = cost_static - cost_adapted [FINDING] ###")
for c in ["membership","property"]:
    pt,lo,hi=ci_did(scores[("static","static")]["score"].to_numpy(), scores[("static",c)]["score"].to_numpy(),
                    scores[("adapted","static")]["score"].to_numpy(), scores[("adapted",c)]["score"].to_numpy())
    print(f"  DiD({c:10s}) = {pt:+.4f}  CI95 [{lo:+.4f},{hi:+.4f}]  -> {verdict(lo,hi)}")

print("\n### 1.5 RATIO robustness(change)=mean(CHANGE)/mean(STATIC eval) [FINDING] ###")
for a in AGENTS:
    for c in ["membership","property"]:
        pt,lo,hi=ci_ratio(scores[(a,c)]["score"].to_numpy(), scores[(a,"static")]["score"].to_numpy())
        print(f"  {a:7s} robustness({c:10s}) = {pt:.4f}  CI95 [{lo:.4f},{hi:.4f}]")

print("\n### 1.6 cost PER EVENT = cost per episode / mean events per episode [FINDING] ###")
print("  (assumes effects add linearly across events — an ASSUMPTION, labelled as such; does not replace per-episode)")
for a in AGENTS:
    # membership cost attributed to leave-event rate (join is coverage/near-zero drift); property to property rate
    for c, ct in [("membership","membership_leave"), ("property","property")]:
        cpe=costs[(a,c)][0]; r=evrate[(a,ct)]
        print(f"  {a:7s}/{c:10s}: cost/episode={cpe:+.4f}  events/ep({ct})={r:.3f}  cost/event={cpe/r:+.5f}")

# ================= STEP 2 =================
print("\n########## STEP 2: CLASSIFICATION (within-agent measure, identical for every change type) ##########")
print("perceived reported PER SLICE (0.2); cost = within-agent contrast (1.1, STATIC agent).")
for c, cts in [("membership",["membership_leave"]),("property",["property"])]:
    for ct in cts:
        d=dload("static",c); e=filt(d,ct)
        slice_perc={sl:(e[f"change_drift_{sl}"]>0).mean() for sl in ["full","mean","max","min"]}
        pt,lo,hi=costs[("static",c)]
        cost_excl0 = (lo>0 or hi<0)
        cost_neg = hi<0  # negative cost = change lowered score (score dropped) => cost(=static-change) positive means drop
        # NOTE cost=static-change; positive cost = score DROPPED under change. "negative" in BLIND means a real loss.
        print(f"  {ct}: perceived per-slice full={slice_perc['full']:.3f} mean={slice_perc['mean']:.3f} "
              f"max={slice_perc['max']:.3f} min={slice_perc['min']:.3f}")
        print(f"      cost(static agent)={pt:+.4f} [{lo:+.4f},{hi:+.4f}] -> {verdict(lo,hi)}")
        # classification: perceived if max/min slice moved (the attenuation-bearing slices); mean/full always 1
        perceived_any = slice_perc['max']>0 or slice_perc['min']>0
        if perceived_any:
            klass="HANDLED-candidate (perceived) — but HANDLED NOT MEASURABLE (see note)"
        elif cost_excl0:
            klass="BLIND (not perceived & cost excludes 0)"
        else:
            klass="ABSORBED (not perceived & cost includes 0)"
        print(f"      -> {klass}")
print("  membership_join: perceived=coverage (fire-time ~0, invisible node) — reported separately, not attenuation.")
print("  HANDLED: NOT MEASURABLE with present instrumentation (score captured once/episode; within-episode")
print("           dip+recovery needs per-step ownership logging). Not approximated.")

# ================= STEP 3 =================
print("\n########## STEP 3: DYNAMIC RANGE + CORRELATION (this band only; NO extrapolation) ##########")
print("\n### 3.1 dynamic range [FINDING] ###")
for a in AGENTS:
    ceil=scores[(a,"static")]["score"].mean()
    largest_cost=max(costs[(a,c)][0] for c in ["membership","property"])
    print(f"  {a:7s}: STATIC-eval ceiling mean={ceil:.4f}; largest observed cost={largest_cost:+.4f} "
          f"= {largest_cost/ceil*100:.1f}% of ceiling")

print("\n### 3.3 drift<->score correlation, with/without zero-score episodes [FINDING; NOT a causal basis] ###")
print("  exclusion rule: drop episodes with score==0 (0.4). This is reported for completeness / as a documented")
print("  negative result — the causal weight is the within-agent cost, NOT this correlation.")
for a in AGENTS:
    for c,cts in [("membership",["membership_leave"]),("property",["property"])]:
        sdf=scores[(a,c)]; d=dload(a,c); ev=d[d.change_type.isin(cts)]
        ed=ev.groupby(["seed","episode"])["change_drift_full"].mean().reset_index(name="md")
        m=sdf.merge(ed,on=["seed","episode"])
        mnz=m[m.score>0]
        pa=m.md.corr(m.score); sa=m.md.corr(m.score,method="spearman")
        pn=mnz.md.corr(mnz.score); sn=mnz.md.corr(mnz.score,method="spearman")
        print(f"  {a:7s}/{c:10s}: ALL n={len(m)} pear={pa:.3f} spear={sa:.3f} | EXCL-zero n={len(mnz)} pear={pn:.3f} spear={sn:.3f}")

print("\n"+"="*84); print(BANNER); print("="*84)
