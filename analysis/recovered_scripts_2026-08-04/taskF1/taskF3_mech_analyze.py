"""
Task F3 mechanical-channel analysis. From the per-departure instantaneous score deltas
(mech_{band}_seed*.csv) and the mech-run terminal scores (mechscore_*), decompose the membership
cost into a directly-measured arithmetic (mechanical) channel and a behavioural residual.

Per removal: delta = score_after - score_before at the instant that node is removed (control metric
root_owned/ownable). Only ROOT-owned departures lower the score; un-owned / owned-non-root shrink
the denominator and RAISE it. Per-episode net arithmetic displacement = sum of deltas.

cost = mean(static-eval) - mean(mech-run membership terminal).
arithmetic loss = -mean_per_ep( sum delta )          (positive = net score pushed down by churn arithmetic)
  owned-root loss channel = -mean_per_ep( sum delta | was_root )
  denom-shrink gain       = +mean_per_ep( sum delta | not was_root )
behavioural residual = cost - arithmetic loss         (score the agent failed to reach beyond arithmetic)
Cluster-bootstrap over 5 replicates.
"""
import os, glob
import numpy as np, pandas as pd
BASE = "/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1"
SEEDS = [42, 100, 123, 200, 300]; RNG = np.random.default_rng(7); NB = 10000
BANNER = ("PROVISIONAL (donor-pool confound, Task G pending): membership_join draws from a shared "
          "donor pool ~2.2x weaker at the large band; all join-related numbers inherit this caveat.")
STAT = {"30-40": f"{BASE}/eval_out", "80-100": f"{BASE}/f2eval_out"}

def mech(band):
    return pd.concat([pd.read_csv(f"{BASE}/f3_mech_out/mech_{band}_seed{s}.csv") for s in SEEDS], ignore_index=True)
def mscore(band):
    return pd.concat([pd.read_csv(f"{BASE}/f3_mech_out/mechscore_{band}_seed{s}.csv") for s in SEEDS], ignore_index=True)
def static_scores(band):
    return {s: pd.read_csv(f"{STAT[band]}/score_static_seed{s}_evalstatic.csv")["score"].to_numpy() for s in SEEDS}

def ci(a): return (float(np.percentile(a,2.5)), float(np.percentile(a,97.5)))

print("="*84); print("TASK F3 — MECHANICAL CHANNEL (directly-measured per-departure arithmetic)")
print(BANNER); print("="*84)
print("Resolves the reverse-causation confound that made the cross-episode OLS invalid (there the")
print("owned-departure count was endogenous to episode quality). Here each departure's instantaneous")
print("arithmetic score impact is measured directly.")

for band in ["30-40", "80-100"]:
    d = mech(band); ms = mscore(band); ss = static_scores(band)
    # per-episode sums
    ep = d.groupby(["seed","episode"]).agg(sumdelta=("delta","sum"),
             sum_root=("delta", lambda x: x[d.loc[x.index,"was_root"]==1].sum()),
             sum_nonroot=("delta", lambda x: x[d.loc[x.index,"was_root"]==0].sum()),
             n_rem=("delta","size"), n_root=("was_root","sum")).reset_index()
    memb_mean = ms["score"].mean()
    stat_mean = np.concatenate([ss[s] for s in SEEDS]).mean()
    cost = stat_mean - memb_mean
    arith_loss = -ep["sumdelta"].mean()
    root_loss = -ep["sum_root"].mean(); denom_gain = ep["sum_nonroot"].mean()
    behav = cost - arith_loss
    # per-removal category means
    dr = d[d.was_root==1].delta.mean(); dnr_owned = d[(d.was_owned==1)&(d.was_root==0)].delta.mean(); dun = d[d.was_owned==0].delta.mean()
    # bootstrap over replicates
    bs_cost=[]; bs_arith=[]; bs_behav=[]; bs_root=[]
    eg = {s: ep[ep.seed==s] for s in SEEDS}; mg = {s: ms[ms.seed==s] for s in SEEDS}
    for _ in range(NB):
        ch = RNG.choice(SEEDS, len(SEEDS), replace=True)
        e2 = pd.concat([eg[s] for s in ch], ignore_index=True)
        m2 = np.concatenate([mg[s]["score"].to_numpy() for s in ch])
        s2 = np.concatenate([ss[s] for s in ch])
        cst = s2.mean() - m2.mean(); al = -e2["sumdelta"].mean()
        bs_cost.append(cst); bs_arith.append(al); bs_behav.append(cst-al); bs_root.append(-e2["sum_root"].mean())
    print(f"\n### {band} [FINDING] ###")
    print(f"  removals={len(d)}  per-ep: removals={ep.n_rem.mean():.1f} root-owned departures={ep.n_root.mean():.2f}")
    print(f"  per-removal mean delta: owned-ROOT={dr:+.5f}  owned-nonroot={dnr_owned:+.5f}  un-owned={dun:+.5f}")
    print(f"    (only root-owned departures lower the score; others shrink the denominator and raise it)")
    print(f"  total cost (static {stat_mean:.4f} - membership {memb_mean:.4f}) = {cost:+.4f} CI95 [{ci(bs_cost)[0]:+.4f},{ci(bs_cost)[1]:+.4f}]")
    print(f"  ARITHMETIC (mechanical) loss = {arith_loss:+.4f} CI95 [{ci(bs_arith)[0]:+.4f},{ci(bs_arith)[1]:+.4f}]  ({arith_loss/cost*100:.0f}% of cost)")
    print(f"     = owned-root loss {root_loss:+.4f} [{ci(bs_root)[0]:+.4f},{ci(bs_root)[1]:+.4f}] offset by denom-shrink gain {denom_gain:+.4f}")
    print(f"  BEHAVIOURAL residual = cost - arithmetic = {behav:+.4f} CI95 [{ci(bs_behav)[0]:+.4f},{ci(bs_behav)[1]:+.4f}]  ({behav/cost*100:.0f}% of cost)")

print("\n### which to compare to the attenuation curve ###")
print("  BEHAVIOURAL residual — the only component a perception failure could affect. Arithmetic loss")
print("  is independent of perception. Caveat: the arithmetic deltas are instantaneous (removal-time);")
print("  behavioural = cost - arithmetic is approximate (the two compound over the trajectory).")
print("\n"+"="*84); print(BANNER); print("="*84)
