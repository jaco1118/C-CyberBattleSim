"""
Task F Pass 1 analysis. Consumes the STOCHASTIC-eval grid in eval_out/. Produces, numbers only:
  Condition A (2 agents x 5 seeds x {static,membership,property}):
    - join count report (score<->drift per episode on seed/scenario_id/episode; join failures)
    - per-episode score distribution per (agent, eval_cond) with between-seed spread
    - STATIC vs ADAPTED contrast per eval condition: bootstrap 0.95 CI on (adapted-static) mean
      score difference, pooled-over-episodes AND per-seed spread -> the "mattered" test
    - response rate at tau=0 per change type (primary cross-type metric), per slice, between-seed
    - per-slice absolute+relative drift per change type (within-type), bootstrap CI
    - per-episode drift<->score correlation (with the non-causal caveat stated in output)
    - classification counts ABSORBED / BLIND / HANDLED per change type
  Condition B (static agent x 5 seeds x pn{0.01,0.10,0.25,0.50}):
    - control-goal score vs pn, fitted slope + bootstrap CI, reported beside Terranova +0.09
      (shape/direction only; no ratio; reward configs differ)
Unit of analysis = EPISODE. bootstrap 0.95. seed = grouping factor. Never pool across change types.
PROVISIONAL: donor-pool confound (Task G not yet run) -- carried into the output.
"""
import os, glob, sys
import numpy as np
import pandas as pd

OUT = "/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1/eval_out"
SEEDS = [42, 100, 123, 200, 300]
RNG = np.random.default_rng(0)
BANNER = ("PROVISIONAL: donor-pool confound not yet controlled (Task G pending); membership_join "
          "events draw from a shared donor pool ~2.2x weaker at the large band. All join-related "
          "numbers below inherit this caveat.")

def boot_ci(x, nboot=10000, alpha=0.05, stat=np.mean):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    bs = [stat(RNG.choice(x, size=len(x), replace=True)) for _ in range(nboot)]
    return (float(stat(x)), float(np.percentile(bs, 100*alpha/2)), float(np.percentile(bs, 100*(1-alpha/2))))

def boot_ci_diff(a, b, nboot=10000, alpha=0.05):
    a = np.asarray(a, float); b = np.asarray(b, float)
    bs = [a[RNG.integers(0, len(a), len(a))].mean() - b[RNG.integers(0, len(b), len(b))].mean() for _ in range(nboot)]
    return (float(a.mean()-b.mean()), float(np.percentile(bs, 100*alpha/2)), float(np.percentile(bs, 100*(1-alpha/2))))

def load_scores(agent, eval_cond, pn=None):
    frames = []
    for s in SEEDS:
        suf = eval_cond if pn is None else f"defender_p{pn}"
        p = os.path.join(OUT, f"score_{agent}_seed{s}_eval{suf}.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def load_drift(agent, eval_cond):
    frames = []
    for s in SEEDS:
        p = os.path.join(OUT, f"drift_{agent}_seed{s}_eval{eval_cond}.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

print("="*80); print("TASK F PASS 1 -- CONDITION A & B ANALYSIS"); print(BANNER); print("="*80)

# ---------------- Condition A: score distributions + join ----------------
print("\n### SCORE DISTRIBUTION per (agent, eval_cond), with per-seed spread ###")
score_tables = {}
for eval_cond in ["static", "membership", "property"]:
    for agent in ["static", "adapted"]:
        df = load_scores(agent, eval_cond)
        score_tables[(agent, eval_cond)] = df
        if df.empty:
            print(f"  {agent:7s} eval={eval_cond:10s}: NO DATA"); continue
        s = df["score"].to_numpy()
        per_seed = df.groupby("seed")["score"].mean()
        print(f"  {agent:7s} eval={eval_cond:10s}: n_ep={len(df):4d} "
              f"min={s.min():.3f} med={np.median(s):.3f} mean={s.mean():.3f} max={s.max():.3f} "
              f"frac0={(s==0).mean():.3f} | per-seed means={[round(v,3) for v in per_seed.values]} "
              f"(between-seed sd={per_seed.std(ddof=1):.3f})")

print("\n### JOIN CHECK (score<->drift per episode) ###")
for eval_cond in ["membership", "property"]:
    for agent in ["static", "adapted"]:
        sdf = score_tables[(agent, eval_cond)]
        ddf = load_drift(agent, eval_cond)
        if sdf.empty or ddf.empty:
            print(f"  {agent}/{eval_cond}: missing data"); continue
        skey = set(map(tuple, sdf[["seed","scenario_id","episode"]].values))
        dkey = set(map(tuple, ddf[["seed","scenario_id","episode"]].drop_duplicates().values))
        only_score = len(skey - dkey); only_drift = len(dkey - skey)
        print(f"  {agent:7s}/{eval_cond:10s}: score_ep={len(skey)} drift_ep={len(dkey)} "
              f"score-without-drift={only_score} drift-without-score={only_drift} joined={len(skey & dkey)}")

# ---------------- STATIC vs ADAPTED contrast ("mattered") ----------------
print("\n### STATIC vs ADAPTED contrast per eval condition (mattered test) ###")
print("  (causal weight lives HERE, in the matched-condition contrast, not in the drift-score correlation)")
mattered = {}
for eval_cond in ["static", "membership", "property"]:
    st = score_tables[("static", eval_cond)]; ad = score_tables[("adapted", eval_cond)]
    if st.empty or ad.empty:
        print(f"  eval={eval_cond}: missing"); continue
    d, lo, hi = boot_ci_diff(ad["score"].to_numpy(), st["score"].to_numpy())
    per_seed_diff = [ad[ad.seed==s]["score"].mean() - st[st.seed==s]["score"].mean() for s in SEEDS]
    excludes0 = (lo > 0) or (hi < 0)
    verdict = ("MATTERED (adapted>static)" if lo > 0 else
               "MATTERED (adapted<static)" if hi < 0 else "NOT ESTABLISHED (CI includes 0)")
    mattered[eval_cond] = verdict
    print(f"  eval={eval_cond:10s}: mean(adapted-static)={d:+.4f} CI95=[{lo:+.4f},{hi:+.4f}] -> {verdict}")
    print(f"      per-seed (adapted-static): {[round(x,3) for x in per_seed_diff]} "
          f"(between-seed sd={np.std(per_seed_diff, ddof=1):.4f})")

# ---------------- Response rate at tau=0 + per-slice drift ----------------
SLICES = ["full", "mean", "max", "min"]
def filt(ddf, ct):
    m = ((ddf["change_type"]==ct) & (ddf["relevant"]==True) & (ddf["touched_node_visible"]==True)
         & (ddf["event_phase"].isin(["immediate","attributed"])))
    return ddf[m]

print("\n### RESPONSE RATE at tau=0 (primary cross-type metric) + per-slice drift (within-type) ###")
for eval_cond, cts in [("membership", ["membership_leave","membership_join"]), ("property", ["property"])]:
    for agent in ["static", "adapted"]:
        ddf = load_drift(agent, eval_cond)
        if ddf.empty:
            continue
        for ct in cts:
            e = filt(ddf, ct)
            if len(e)==0:
                print(f"  {agent:7s}/{eval_cond}/{ct}: 0 events"); continue
            print(f"  {agent:7s}/{eval_cond}/{ct}: n_events={len(e)}")
            for sl in SLICES:
                col = f"change_drift_{sl}"
                perceived = (e[col] > 0).mean()
                # between-seed spread of response rate
                rr_seed = e.groupby("seed").apply(lambda g: (g[col]>0).mean())
                m, lo, hi = boot_ci(e[col].to_numpy())
                print(f"      slice={sl:4s} resp_rate(tau=0)={perceived:.3f} "
                      f"(between-seed sd={rr_seed.std(ddof=1):.3f}) | drift mean={m:.5f} CI95=[{lo:.5f},{hi:.5f}]")

# ---------------- drift<->score correlation ----------------
print("\n### per-episode DRIFT<->SCORE correlation (NOT causal on its own -- see note) ###")
print("  NOTE: a cross-episode correlation is associational; the causal claim rests on the")
print("  STATIC-vs-ADAPTED matched-condition contrast above, not on this correlation.")
for eval_cond, cts in [("membership", ["membership_leave","membership_join"]), ("property", ["property"])]:
    for agent in ["static", "adapted"]:
        sdf = score_tables[(agent, eval_cond)]; ddf = load_drift(agent, eval_cond)
        if sdf.empty or ddf.empty: continue
        ev = ddf[ddf["change_type"].isin(cts)]
        epi_drift = ev.groupby(["seed","episode"])["change_drift_full"].mean().reset_index(name="mean_drift")
        merged = sdf.merge(epi_drift, on=["seed","episode"], how="inner")
        if len(merged) < 5:
            print(f"  {agent}/{eval_cond}: too few joined episodes ({len(merged)})"); continue
        pear = merged["mean_drift"].corr(merged["score"])
        spear = merged["mean_drift"].corr(merged["score"], method="spearman")
        print(f"  {agent:7s}/{eval_cond:10s}: n_ep={len(merged)} pearson={pear:.3f} spearman={spear:.3f}")

# ---------------- classification ----------------
print("\n### CLASSIFICATION per change type (ABSORBED / BLIND / HANDLED) ###")
print("  perceived(event) = change_drift_full > 0 (tau=0), on relevant+visible+immediate/attributed events")
print("  mattered(change type) = STATIC-vs-ADAPTED contrast (above) for the matching eval condition")
print("  HANDLED operationalized at EPISODE granularity (per-step score not captured): perceived events")
print("  whose ADAPTED agent kept a near-ceiling terminal score (recovered by episode end).")
for eval_cond, cts in [("membership", ["membership_leave","membership_join"]), ("property", ["property"])]:
    verdict = mattered.get(eval_cond, "NOT ESTABLISHED")
    for ct in cts:
        # use STATIC agent's events for the "blind" baseline (it never trained on the change)
        ddf = load_drift("static", eval_cond)
        e = filt(ddf, ct)
        if len(e)==0:
            print(f"  {ct}: 0 events"); continue
        n = len(e); n_perc = int((e["change_drift_full"]>0).sum()); n_notperc = n - n_perc
        did_matter = verdict.startswith("MATTERED")
        not_established = verdict.startswith("NOT ESTABLISHED")
        if not_established:
            absorbed_label = "NOT ESTABLISHED (not 'did not matter')"
        else:
            absorbed_label = "ABSORBED" if not did_matter else "(perceived split below)"
        print(f"  {ct}: n={n} perceived={n_perc} not_perceived={n_notperc} | change-type mattered? {verdict}")
        if did_matter:
            print(f"      BLIND (not perceived & mattered)  = {n_notperc}")
            print(f"      HANDLED (perceived)               = {n_perc}")
        elif not_established:
            print(f"      NOT ESTABLISHED whether it mattered -> ABSORBED/BLIND split not asserted; "
                  f"perceived={n_perc}, not_perceived={n_notperc}")
        else:
            print(f"      ABSORBED (not perceived & did not matter) = {n_notperc}")
            print(f"      HANDLED (perceived)                       = {n_perc}")

# ---------------- Condition B ----------------
print("\n### CONDITION B (replication): static agent, control score vs defender pn ###")
print("  Comparison to Terranova +0.09 is SHAPE and DIRECTION only (reward configs differ; no ratio).")
PNS = [0.01, 0.10, 0.25, 0.50]
rows = []
for pn in PNS:
    df = load_scores("static", "defender", pn=pn)
    if df.empty:
        print(f"  pn={pn}: NO DATA"); continue
    per_seed = df.groupby("seed")["score"].mean()
    m, lo, hi = boot_ci(df["score"].to_numpy())
    print(f"  pn={pn:.2f}: n_ep={len(df)} mean={m:.4f} CI95=[{lo:.4f},{hi:.4f}] "
          f"per-seed={[round(v,3) for v in per_seed.values]} (sd={per_seed.std(ddof=1):.4f})")
    for s in SEEDS:
        rows.append((pn, s, df[df.seed==s]["score"].mean()))
if rows:
    rdf = pd.DataFrame(rows, columns=["pn","seed","mean_score"])
    # slope via OLS on per-seed means (pn as regressor); bootstrap over seeds
    def slope_of(d):
        x = d["pn"].to_numpy(); y = d["mean_score"].to_numpy()
        return np.polyfit(x, y, 1)[0]
    obs_slope = slope_of(rdf)
    bs = []
    for _ in range(10000):
        # cluster bootstrap over seeds (seed = grouping factor): resample seeds with replacement
        chosen = RNG.choice(SEEDS, len(SEEDS), replace=True)
        parts = [rdf[rdf.seed==s] for s in chosen]
        bs.append(slope_of(pd.concat(parts, ignore_index=True)))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"\n  FITTED slope (control score vs pn): {obs_slope:+.4f} bootstrap95=[{lo:+.4f},{hi:+.4f}]")
    print(f"  Terranova reported +0.09 for the control goal (no measurable degradation).")
    print(f"  SHAPE/DIRECTION comparison only: this slope is {'negative' if obs_slope<0 else 'positive'} "
          f"(sign {'matches' if (obs_slope>0)== (0.09>0) else 'differs from'} the reported +0.09 sign).")

print("\n" + "="*80); print(BANNER); print("="*80)
