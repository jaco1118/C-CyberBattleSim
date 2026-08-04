"""STEP 1.2 reproduction check: does the fresh-retrained Arm 1 reproduce the reported static root-owned
COUNT within seed spread? Three columns per seed:
  NEW  = fresh Z arm1 (taskZ_eval, static)      -> tests the TRAINING harness
  OLD  = reported F1/F2 checkpoints (taskZ_eval) -> same eval harness, isolates training from eval drift
  REP  = the numbers already reported (eval_out / f2eval_out score CSVs, produced by taskF1/F2_eval)
NEW vs OLD isolates the training harness (identical eval both sides). OLD vs REP sanity-checks that
taskZ_eval matches the reported eval harness. Metric = root_owned COUNT (primary, per Task Z 2.1)."""
import pandas as pd, numpy as np, os
B = "/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1"
SEEDS = [42, 100, 123, 200, 300]
T80 = {42: 5, 100: 100, 123: 18, 200: 2, 300: 67}

def mean_count(path):
    if not os.path.exists(path): return None
    return pd.read_csv(path)["root_owned"].to_numpy()

def zpath(sub, band, topo, s):
    return f"{B}/z_repro_out/{sub}_{band}/zscore_arm1_{topo}_seed{s}_static.csv"

REP = {
    "30-40":  lambda s: f"{B}/eval_out/score_static_seed{s}_evalstatic.csv",
    "80-100": lambda s: f"{B}/f2eval_out/score_static_seed{s}_evalstatic.csv",
}
BANDS = {"30-40": lambda s: "scalability_30_40_44", "80-100": lambda s: f"scalability_80_100_{T80[s]}"}

print("="*92)
print("STEP 1.2 REPRODUCTION CHECK — Arm 1 static, root-owned COUNT (mean over 200 ep/seed)  [FINDING]")
print("="*92)
verdicts = {}
for band in ("30-40", "80-100"):
    print(f"\n### band {band} ###")
    print(f"  {'seed':>5} {'NEW(fresh)':>11} {'OLD(rep-ckpt)':>13} {'REP(reported)':>13} "
          f"{'NEW-OLD':>8} {'NEW-REP':>8}")
    newm, oldm, repm = [], [], []
    for s in SEEDS:
        topo = BANDS[band](s)
        a = mean_count(zpath("new", band, topo, s))
        o = mean_count(zpath("old", band, topo, s))
        r = mean_count(REP[band](s))
        an, on, rn = (x.mean() if x is not None else float("nan") for x in (a, o, r))
        newm.append(an); oldm.append(on); repm.append(rn)
        print(f"  {s:>5} {an:>11.3f} {on:>13.3f} {rn:>13.3f} {an-on:>+8.3f} {an-rn:>+8.3f}")
    newm, oldm, repm = map(np.array, (newm, oldm, repm))
    # band-level means + between-seed spread
    print(f"  {'BAND':>5} {newm.mean():>11.3f} {oldm.mean():>13.3f} {repm.mean():>13.3f}")
    print(f"        between-seed SD: NEW={newm.std(ddof=1):.3f}  OLD={oldm.std(ddof=1):.3f}  REP={repm.std(ddof=1):.3f}")
    # reproduction criterion: band-level NEW within OLD's between-seed spread, AND per-seed |NEW-OLD| within OLD spread
    old_sd = oldm.std(ddof=1)
    band_diff = abs(newm.mean() - oldm.mean())
    per_seed_ok = np.all(np.abs(newm - oldm) <= (old_sd + 0.05*np.abs(oldm) + 1e-9) + old_sd)  # lenient band; primary is band-level
    band_ok = band_diff <= old_sd
    # primary published criterion: band means agree within the larger between-seed SD
    ref_sd = max(old_sd, repm.std(ddof=1))
    reproduces = abs(newm.mean() - oldm.mean()) <= ref_sd and abs(newm.mean() - repm.mean()) <= ref_sd
    verdicts[band] = reproduces
    print(f"        |NEW.mean - OLD.mean| = {band_diff:.3f}  vs between-seed SD ref = {ref_sd:.3f}  "
          f"-> {'WITHIN spread' if band_ok else 'OUTSIDE spread'}")
    print(f"        |NEW.mean - REP.mean| = {abs(newm.mean()-repm.mean()):.3f}")
    print(f"        REPRODUCES (band mean within seed spread of both OLD and REP): {'YES' if reproduces else 'NO'}")

print("\n" + "="*92)
print("VERDICT:", "ALL BANDS REPRODUCE" if all(verdicts.values()) else f"CHECK — {verdicts}")
print("If any band does NOT reproduce, per Task Z STEP 1.2: STOP and report; do not interpret arms 2/3.")
print("="*92)
