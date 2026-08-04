"""RECALIBRATE the convergence threshold on the eval signal (checkpoint-eval mean root_owned), using
30-40 (converged) vs 80-100-at-250k (not converged) as references. Windows 150/200/250k:
  preceding [150,200] = (m150+m200)/2 ; final [200,250] = (m200+m250)/2 ; Delta% = (final-prec)/prec*100.
Report the eval-signal noise floor (30-40) and the not-converged signal (80-100), and whether 5%
separates them. Runs BEFORE any 80-100 500k number is computed."""
import pandas as pd, numpy as np
B = "/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1"
d = pd.read_csv(f"{B}/conv_calib.csv")
print("="*76); print("F4 THRESHOLD RECALIBRATION ON THE EVAL SIGNAL  (before the 80-100 500k verdict)")
print("eval signal = mean root_owned per checkpoint, static, N per row; windows 150/200/250k"); print("="*76)
res = {}
for band, role in [("30-40", "KNOWN CONVERGED (noise floor)"), ("80-100", "KNOWN NOT-CONVERGED @250k")]:
    b = d[d.band == band]; deltas = []
    print(f"\n### {band}  — {role} ###")
    for seed in sorted(b.seed.unique()):
        g = b[b.seed == seed].set_index("ckpt")["mean_root"]
        try:
            m1, m2, m3 = g[150000], g[200000], g[250000]
        except KeyError:
            print(f"  seed{seed}: MISSING -> {dict(g)}"); continue
        prec = (m1 + m2) / 2; fin = (m2 + m3) / 2; dpct = (fin - prec) / prec * 100
        deltas.append(dpct)
        print(f"  seed{seed}: root_owned 150k={m1:.2f} 200k={m2:.2f} 250k={m3:.2f} | window Δ%={dpct:+.1f}")
    deltas = np.array(deltas); res[band] = deltas
    print(f"  --> across-seed mean|Δ%| = {np.abs(deltas).mean():.2f}% ; mean Δ% = {deltas.mean():+.2f}% ; "
          f"per-seed |Δ%|: max={np.abs(deltas).max():.1f}% p50={np.median(np.abs(deltas)):.1f}%")

print("\n" + "="*76); print("SEPARATION CHECK")
c = np.abs(res["30-40"]); nc = np.abs(res["80-100"])
print(f"  converged (30-40)  |Δ%|: mean {c.mean():.2f}%  max {c.max():.2f}%  within-5%: {int((c<5).sum())}/{len(c)}")
print(f"  not-conv (80-100)  |Δ%|: mean {nc.mean():.2f}%  min {nc.min():.2f}%  within-5%: {int((nc<5).sum())}/{len(nc)}")
sep5 = (c.mean() < 5) and (nc.mean() >= 5)
print(f"  does 5% separate (converged mean<5% AND not-conv mean>=5%)? {'YES' if sep5 else 'NO'}")
# what threshold the same logic gives: midpoint between the two bands' mean|Δ%| (or converged max + margin)
mid = (c.mean() + nc.mean()) / 2
print(f"  calibration-implied threshold options: converged mean|Δ%|={c.mean():.2f}%, converged max={c.max():.2f}%, "
      f"not-conv mean={nc.mean():.2f}%; midpoint={mid:.2f}%")
print("  -> If 5% cleanly separates, KEEP 5% (pre-registered on the new signal). Else adopt the "
      "calibration-implied threshold (converged-max rounded up, still below the not-conv signal).")
