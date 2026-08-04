"""
Task F2 STEP 3 analysis, band 80-100, static agent only. Identical definitions to F1-R.
Reads f2eval_out/{score,drift,leaveown}_static_seed{seed}_eval{cond}.csv. Replicate = seed (each
a different topology, 0.3). Unit=episode; bootstrap 0.95; replicate as grouping factor.
Conditions: static, membership, membership_matched (if present), property.
Adds 3.6 mechanical-vs-behavioural decomposition of the membership cost.
"""
import os, sys, glob
import numpy as np, pandas as pd
OUT = "/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1/f2eval_out"
SEEDS = [42, 100, 123, 200, 300]
RNG = np.random.default_rng(20260729); NBOOT = 10000
BANNER = ("PROVISIONAL (donor-pool confound, Task G pending): membership_join draws from a shared "
          "donor pool ~2.2x weaker at the large band; all join-related numbers inherit this caveat.")
# which change conditions are present
CONDS = ["static", "membership", "membership_matched", "property"]

def sload(cond):
    fs = [os.path.join(OUT, f"score_static_seed{s}_eval{cond}.csv") for s in SEEDS]
    fs = [f for f in fs if os.path.exists(f)]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()
def dload(cond):
    fs = [os.path.join(OUT, f"drift_static_seed{s}_eval{cond}.csv") for s in SEEDS]
    fs = [f for f in fs if os.path.exists(f)]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()
def lload(cond):
    fs = [os.path.join(OUT, f"leaveown_static_seed{s}_eval{cond}.csv") for s in SEEDS]
    fs = [f for f in fs if os.path.exists(f)]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()

def ci_diff(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    bs = [a[RNG.integers(0,len(a),len(a))].mean() - b[RNG.integers(0,len(b),len(b))].mean() for _ in range(NBOOT)]
    return a.mean()-b.mean(), np.percentile(bs,2.5), np.percentile(bs,97.5)
def ci_ratio(n, d):
    n = np.asarray(n, float); d = np.asarray(d, float)
    bs = [n[RNG.integers(0,len(n),len(n))].mean()/d[RNG.integers(0,len(d),len(d))].mean() for _ in range(NBOOT)]
    return n.mean()/d.mean(), np.percentile(bs,2.5), np.percentile(bs,97.5)
def verdict(lo, hi): return "excludes 0" if (lo>0 or hi<0) else "NOT ESTABLISHED (includes 0)"

present = [c for c in CONDS if not sload(c).empty]
print("="*84); print("TASK F2 STEP 3  —  band 80-100, STATIC agent, 5 topology-and-seed replicates")
print("conditions present:", present); print(BANNER); print("="*84)
scores = {c: sload(c) for c in present}

print("\n### score distribution + zero-score fraction (dynamic-range check) [ARTIFACT] ###")
for c in present:
    df = scores[c]
    per = df.groupby("seed")["score"].mean()
    print(f"  {c:20s}: n={len(df)} mean={df.score.mean():.4f} med={df.score.median():.4f} "
          f"frac0={(df.score==0).mean():.3f} | per-replicate {[round(v,3) for v in per.values]} "
          f"(sd={per.std(ddof=1):.4f})")

print("\n### JOIN CHECK [ARTIFACT] ###")
for c in [x for x in present if x != "static"]:
    sdf = scores[c]; ddf = dload(c)
    if ddf.empty: print(f"  {c}: no drift"); continue
    dk = ddf[["seed","scenario_id","episode"]].drop_duplicates()
    j = sdf.merge(dk, on=["seed","scenario_id","episode"], how="inner")
    print(f"  {c}: score_ep={len(sdf)} drift_ep={len(dk)} joined={len(j)} unjoined={len(sdf)-len(j)}")

base = scores["static"]["score"].to_numpy()
print("\n### 3.1 within-agent cost = mean(static) - mean(condition) [FINDING] ###")
costs = {}
for c in [x for x in present if x != "static"]:
    chg = scores[c]["score"].to_numpy()
    pt, lo, hi = ci_diff(base, chg); costs[c] = (pt, lo, hi)
    per = [scores["static"][scores["static"].seed==s].score.mean() - scores[c][scores[c].seed==s].score.mean() for s in SEEDS]
    print(f"  cost({c:20s}) = {pt:+.4f} CI95 [{lo:+.4f},{hi:+.4f}] -> {verdict(lo,hi)}")
    print(f"      per-replicate: {[round(x,4) for x in per]} (sd={np.std(per,ddof=1):.4f})")

print("\n### 3.2 robustness ratio = mean(condition)/mean(static) [FINDING; carries across bands] ###")
for c in [x for x in present if x != "static"]:
    pt, lo, hi = ci_ratio(scores[c]["score"].to_numpy(), base)
    print(f"  robustness({c:20s}) = {pt:.4f} CI95 [{lo:.4f},{hi:.4f}]")

print("\n### 3.3 cost per event (ASSUMES linear additivity — labelled) [FINDING] ###")
for c in [x for x in present if x != "static"]:
    ddf = dload(c)
    if c == "property":
        n = len(ddf[(ddf.change_type=="property")&(ddf.event_phase=="immediate")]); ct="property"
    else:
        n = len(ddf[(ddf.change_type=="membership_leave")&(ddf.event_phase=="immediate")]); ct="membership_leave"
    rate = n/len(scores[c]); cpe = costs[c][0]
    print(f"  {c:20s}: cost/ep={cpe:+.4f} events/ep({ct})={rate:.3f} cost/event={cpe/rate:+.5f}")

print("\n### 3.4 per-slice response rate at tau=0 (max/min = attenuation-bearing) [ARTIFACT] ###")
def filt(d, ct): return d[(d.change_type==ct)&(d.relevant==True)&(d.touched_node_visible==True)&(d.event_phase.isin(["immediate","attributed"]))]
for c in [x for x in present if x != "static"]:
    ddf = dload(c)
    cts = ["membership_leave"] if c in ("membership","membership_matched") else ["property"]
    for ct in cts:
        e = filt(ddf, ct)
        if len(e)==0: print(f"  {c}/{ct}: 0 events"); continue
        r = f"  {c}/{ct} (n={len(e)}): "
        for sl in ["full","mean","max","min"]:
            col=f"change_drift_{sl}"; overall=(e[col]>0).mean()
            ps=e.groupby("seed").apply(lambda g:(g[col]>0).mean(), include_groups=False)
            r += f"{sl}={overall:.3f}(sd{ps.std(ddof=1):.3f}) "
        print(r)
    if c in ("membership","membership_matched"):
        fired=ddf[(ddf.change_type=="membership_join")&(ddf.event_phase=="fired")]
        if len(fired): print(f"  {c}/membership_join FIRE-TIME (coverage): resp_rate(full>0)={(fired.change_drift_full>0).mean():.3f} visible@fire={(fired.touched_node_visible==True).mean():.3f}")

print("\n### 3.5 classification (per change type; perceived per slice) [FINDING] ###")
for c in [x for x in present if x in ("membership","membership_matched","property")]:
    ddf = dload(c); ct = "membership_leave" if c!="property" else "property"
    e = filt(ddf, ct)
    if len(e)==0: continue
    sl = {s:(e[f"change_drift_{s}"]>0).mean() for s in ["full","mean","max","min"]}
    pt,lo,hi = costs[c]
    perceived = sl["max"]>0 or sl["min"]>0
    if perceived: k = "HANDLED-candidate (perceived) — HANDLED NOT MEASURABLE (per-episode score only)"
    elif (lo>0 or hi<0): k = "BLIND (not perceived & cost excludes 0)"
    else: k = "ABSORBED (not perceived & cost includes 0)"
    print(f"  {c}/{ct}: perceived full={sl['full']:.3f} mean={sl['mean']:.3f} max={sl['max']:.3f} min={sl['min']:.3f} | cost {pt:+.4f}[{lo:+.4f},{hi:+.4f}] -> {k}")
print("  HANDLED not measurable (needs per-step ownership logging); not approximated.")

print("\n### 3.6 membership cost: mechanical vs behavioural decomposition [FINDING] ###")
for c in [x for x in present if x in ("membership","membership_matched")]:
    lo = lload(c)
    if lo.empty: print(f"  {c}: no leave-ownership log"); continue
    wof = lo.was_owned.mean(); nep = len(scores[c])
    owned_per_ep = lo.was_owned.sum()/nep; unowned_per_ep = (len(lo)-lo.was_owned.sum())/nep
    print(f"  {c}: leave_events={len(lo)} was_owned_frac={wof:.3f} | owned-departures/ep={owned_per_ep:.2f} "
          f"un-owned-departures/ep={unowned_per_ep:.2f}")
    print(f"      interpretation: was_owned_frac small -> membership cost is predominantly NOT direct")
    print(f"      mechanical loss of owned nodes; large -> more mechanical. (A precise counterfactual")
    print(f"      cost split is not identifiable from observational data — labelled.)")

print("\n### 3.7 side-by-side with 30-40 (F1-R). NO trend through two points; NO extrapolation. ###")
print("  30-40 (F1-R, static agent): cost(membership)=+0.0614 robustness=0.9156 | cost(property)=+0.0174 robustness=0.9761")
print("  30-40 per-slice RR: leave max=0.813 min=0.827 ; property max=0.718 min=0.713")
print("  gate attenuation figures at 80-100: max=43.0 min=36.1 (per cent) — compare to 3.4 max/min above.")
print("\n"+"="*84); print(BANNER); print("="*84)
