"""
JOB 1 analysis: Condition B on the multi-topology gate checkpoints (band 30-40).
Reports (a) fitted score-vs-pn slope with bootstrap 0.95, beside F1's specialist -0.262
[-0.304,-0.216]; (b) the undisturbed baseline (pn=0) per gate agent on its own topologies.
Slope fit over the 4 defender probabilities {0.01,0.10,0.25,0.50} to match F1's Condition B.
"""
import os, glob
import numpy as np, pandas as pd
OUT = "/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1/condB_multitopo_out"
SEEDS = [42, 100, 123, 200, 300]
PNS_FIT = [0.01, 0.10, 0.25, 0.50]
RNG = np.random.default_rng(20260729)
BANNER = ("PROVISIONAL (donor-pool confound, Task G pending): unrelated to Condition B here "
          "(no join events), but carried for consistency across F-series outputs.")

def load(seed, pn):
    p = os.path.join(OUT, f"condB_multitopo_seed{seed}_pn{pn}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

print("="*82); print("JOB 1 — CONDITION B on MULTI-TOPOLOGY GATE checkpoints (band 30-40)"); print(BANNER); print("="*82)

print("\n### (b) UNDISTURBED baseline (pn=0) per gate agent on its own 8 topologies [ARTIFACT] ###")
base_ok = True
for s in SEEDS:
    d = load(s, 0.0)
    if d is None: print(f"  seed{s}: MISSING pn=0"); base_ok=False; continue
    sc = d["score"].to_numpy()
    print(f"  seed{s}: n={len(d)} mean={sc.mean():.4f} median={np.median(sc):.4f} frac0={(sc==0).mean():.3f} "
          f"min={sc.min():.4f} max={sc.max():.4f}")

print("\n### score vs pn per seed (mean) [ARTIFACT] ###")
rows = []
for s in SEEDS:
    line = f"  seed{s}: "
    for pn in [0.0]+PNS_FIT:
        d = load(s, pn)
        if d is None: line += f"pn{pn}=NA "; continue
        m = d["score"].mean(); line += f"pn{pn}={m:.3f} "
        if pn in PNS_FIT: rows.append((s, pn, m))
    print(line)

rdf = pd.DataFrame(rows, columns=["seed","pn","mean_score"])
def slope_of(d):
    return np.polyfit(d["pn"].to_numpy(), d["mean_score"].to_numpy(), 1)[0]
obs = slope_of(rdf)
bs = []
for _ in range(10000):
    chosen = RNG.choice(SEEDS, len(SEEDS), replace=True)
    bs.append(slope_of(pd.concat([rdf[rdf.seed==s] for s in chosen], ignore_index=True)))
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"\n### (a) FITTED slope (score vs pn over {PNS_FIT}) [FINDING] ###")
print(f"  MULTI-TOPOLOGY gate slope = {obs:+.4f}  bootstrap95 [{lo:+.4f}, {hi:+.4f}]")
print(f"  F1 SPECIALIST slope       = -0.2620  [-0.3037, -0.2160]  (single-topology agents, from F1)")
print(f"  F1 specialist undisturbed baseline on topology 44 ~ 0.72 (stochastic).")
print("  Interpretation is SHAPE/DIRECTION only. NOT a refutation of the released study")
print("  (reward config + training budget still differ). This isolates ONE factor (multi- vs")
print("  single-topology training) between two of OUR OWN agents.")
print("\n"+"="*82); print(BANNER); print("="*82)
