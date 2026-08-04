"""Task D3 STEP 3: substitution vs removal (band 30-40), same definitions as taskF1R_reanalyze.
Response rate at tau=0 per slice, event rate, absolute/relative drift, cost+robustness on BOTH score
metrics, bootstrap 0.95 over episodes + between-seed spread."""
import numpy as np, pandas as pd, os
B = "/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1"
SEEDS = [42, 100, 123, 200, 300]
RNG = np.random.default_rng(0); NBOOT = 10000

def sload(folder, suf):
    return pd.concat([pd.read_csv(f"{B}/{folder}/score_static_seed{s}_eval{suf}.csv").assign(seed=s) for s in SEEDS],
                     ignore_index=True)
def dload(folder, suf):
    return pd.concat([pd.read_csv(f"{B}/{folder}/drift_static_seed{s}_eval{suf}.csv") for s in SEEDS], ignore_index=True)

static = sload("eval_out", "static")
removal = sload("eval_out", "property")
sub = sload("d3_eval_out", "property_substitution")

print("="*80); print("TASK D3 STEP 3 — vulnerability SUBSTITUTION vs removal, band 30-40 (5 static seeds, 200 ep)")
print("PROVISIONAL: none (donor drawn from same-scenario nodes, not the join pool)"); print("="*80)

# ---- cost + robustness, both metrics, bootstrap over episodes ----
def boot_mean(a):
    a = np.asarray(a, float)
    return np.array([a[RNG.integers(0, len(a), len(a))].mean() for _ in range(NBOOT)])

for metric, name in [("root_owned", "COUNT (root_owned) [PRIMARY, churn-invariant]"),
                     ("score", "RATIO (root_owned/reachable) [secondary]")]:
    st = static[metric].to_numpy(); rm = removal[metric].to_numpy(); su = sub[metric].to_numpy()
    print(f"\n### cost & robustness — {name} ###")
    for lbl, chg in [("removal", rm), ("substitution", su)]:
        bs_st, bs_ch = boot_mean(st), boot_mean(chg)
        rob = chg.mean()/st.mean(); rob_bs = bs_ch/bs_st
        cost = st.mean()-chg.mean(); cost_bs = bs_st-bs_ch
        # between-seed spread of robustness
        per = np.array([sub[sub.seed==s][metric].mean() if lbl=="substitution" else
                        (removal[removal.seed==s][metric].mean())
                        for s in SEEDS]) / np.array([static[static.seed==s][metric].mean() for s in SEEDS])
        print(f"  {lbl:12s}: robustness={rob:.4f} [95%CI {np.percentile(rob_bs,2.5):.4f},{np.percentile(rob_bs,97.5):.4f}]  "
              f"cost={cost:.4f} [{np.percentile(cost_bs,2.5):.4f},{np.percentile(cost_bs,97.5):.4f}]  "
              f"seed-spread(rob) sd={per.std(ddof=1):.4f}")
    # difference in cost: substitution - removal, paired bootstrap
    d_bs = (boot_mean(st)-boot_mean(su)) - (boot_mean(st)-boot_mean(rm))
    dpt = (st.mean()-su.mean()) - (st.mean()-rm.mean())
    lo, hi = np.percentile(d_bs, 2.5), np.percentile(d_bs, 97.5)
    print(f"  DIFFERENCE (substitution cost - removal cost) = {dpt:.4f} [95%CI {lo:.4f},{hi:.4f}]  "
          f"-> {'DIFFER (excludes 0)' if (lo>0 or hi<0) else 'not established (includes 0)'}")

# ---- per-slice response rate at tau=0 ----
print("\n### per-slice response rate at tau=0 (relevant+visible+immediate/attributed events) ###")
def filt(d, ct):
    return d[(d.change_type==ct)&(d.relevant==True)&(d.touched_node_visible==True)&(d.event_phase.isin(["immediate","attributed"]))]
for lbl, folder, suf, ct in [("removal", "eval_out", "property", "property"),
                             ("substitution", "d3_eval_out", "property_substitution", "property_substitution")]:
    d = dload(folder, suf); e = filt(d, ct)
    row = f"  {lbl:12s} (n_events={len(e)}): "
    for sl in ["full", "mean", "max", "min"]:
        col = f"change_drift_{sl}"
        overall = (e[col] > 0).mean()
        perseed = e.groupby("seed").apply(lambda g: (g[col] > 0).mean(), include_groups=False)
        row += f"{sl}={overall:.3f}(sd{perseed.std(ddof=1):.3f}) "
    print(row)

# ---- absolute & relative drift (within change type, 30-40) ----
print("\n### absolute & relative change-drift (event mean; relevant+visible+immediate/attributed) ###")
for lbl, folder, suf, ct in [("removal", "eval_out", "property", "property"),
                             ("substitution", "d3_eval_out", "property_substitution", "property_substitution")]:
    e = filt(dload(folder, suf), ct)
    rel = e["change_drift_full"].mean()
    absd = e["delta_h_v_norm"].mean() if "delta_h_v_norm" in e.columns else float("nan")
    print(f"  {lbl:12s}: relative(change_drift_full)={rel:.5f}  absolute(delta_h_v_norm)={absd:.4f}")

# ---- achieved event rate ----
print("\n### achieved event rate (events/episode) ###")
for lbl, folder, suf, ct, sc in [("removal", "eval_out", "property", "property", removal),
                                 ("substitution", "d3_eval_out", "property_substitution", "property_substitution", sub)]:
    d = dload(folder, suf); nev = len(d[(d.change_type==ct)&(d.event_phase=="immediate")]); nep = len(sc)
    print(f"  {lbl:12s}: {nev} events / {nep} eps = {nev/nep:.2f}/episode")

# ---- zero-score fraction ----
print("\n### zero-score fraction ###")
for lbl, sc in [("static", static), ("removal", removal), ("substitution", sub)]:
    print(f"  {lbl:12s}: {(sc.score==0).mean():.3f}")
