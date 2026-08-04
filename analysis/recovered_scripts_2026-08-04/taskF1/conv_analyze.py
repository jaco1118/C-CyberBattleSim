"""Apply the F4 convergence criterion (fixed in evidence_taskF4.md) to conv_results.csv.
Window reconstruction from checkpoint evals at 400k/450k/500k:
  preceding 50k window [400,450] mean = (m400+m450)/2 ; final 50k window [450,500] mean = (m450+m500)/2
  Delta% = (final - preceding)/preceding * 100 , per seed.
CONVERGED iff across-seed mean|Delta%| < 5% AND >=4/5 seeds |Delta%| < 5%. Report per band + per seed,
including any failing seed (do not drop it)."""
import pandas as pd, numpy as np
B = "/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1"
d = pd.read_csv(f"{B}/conv_results.csv")
print("="*78); print("F4 CONVERGENCE CHECK  (criterion fixed in evidence_taskF4.md, applied to checkpoint-eval)")
print("metric: mean root_owned per checkpoint (static eval, N per row); windows 400/450/500k"); print("="*78)
for band in ["30-40", "80-100", "10-15"]:
    b = d[d.band == band]
    if b.empty:
        print(f"\n{band}: NO DATA"); continue
    deltas = []
    print(f"\n### band {band} ###")
    for seed in sorted(b.seed.unique()):
        g = b[b.seed == seed].set_index("ckpt")["mean_root"]
        try:
            m4, m45, m5 = g[400000], g[450000], g[500000]
        except KeyError:
            print(f"  seed{seed}: MISSING checkpoint(s) -> {dict(g)}"); continue
        prec = (m4 + m45) / 2; fin = (m45 + m5) / 2
        dpct = (fin - prec) / prec * 100
        inc_final = (m5 - m45) / m45 * 100
        deltas.append(dpct)
        flag = "OK" if abs(dpct) < 5 else "**>5%**"
        print(f"  seed{seed}: root_owned 400k={m4:.2f} 450k={m45:.2f} 500k={m5:.2f} | "
              f"window Δ%={dpct:+.1f} {flag}  (final-window inc {inc_final:+.1f}%)")
    deltas = np.array(deltas)
    mean_abs = np.abs(deltas).mean()
    n_within = int((np.abs(deltas) < 5).sum())
    conv = (mean_abs < 5) and (n_within >= 4)
    print(f"  --> across-seed mean|Δ%| = {mean_abs:.1f}%  ; seeds within 5% = {n_within}/{len(deltas)}")
    print(f"  --> {'CONVERGED' if conv else 'NOT CONVERGED'} (criterion: mean|Δ%|<5% AND >=4/5 within 5%)")
