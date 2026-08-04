"""Re-apply the F4 5% convergence criterion to the 80-100 750k runs (checkpoint-eval, N=60).
Windows 650/700/750k: preceding [650,700]=(m650+m700)/2, final [700,750]=(m700+m750)/2,
Delta%=(final-prec)/prec*100. CONVERGED iff mean|Delta%|<5% AND >=4/5 within 5%."""
import pandas as pd, numpy as np
B = "/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1"
d = pd.read_csv(f"{B}/conv_results_750k.csv")
print("="*74); print("F4 80-100 RE-CHECK at 750k (second resume; same 5% criterion, N=60)")
print("windows 650/700/750k"); print("="*74)
b = d[d.band == "80-100"]; deltas = []
for seed in sorted(b.seed.unique()):
    g = b[b.seed == seed].set_index("ckpt")["mean_root"]
    try:
        m1, m2, m3 = g[650000], g[700000], g[750000]
    except KeyError:
        print(f"  seed{seed}: MISSING -> {dict(g)}"); continue
    prec = (m1 + m2) / 2; fin = (m2 + m3) / 2; dpct = (fin - prec) / prec * 100
    incf = (m3 - m2) / m2 * 100
    deltas.append(dpct)
    flag = "OK" if abs(dpct) < 5 else "**>5%**"
    print(f"  seed{seed}: root_owned 650k={m1:.2f} 700k={m2:.2f} 750k={m3:.2f} | window Δ%={dpct:+.1f} {flag} (final-win {incf:+.1f}%)")
deltas = np.array(deltas)
mean_abs = np.abs(deltas).mean(); n_within = int((np.abs(deltas) < 5).sum())
conv = (mean_abs < 5) and (n_within >= 4)
print(f"  --> across-seed mean|Δ%| = {mean_abs:.1f}% ; seeds within 5% = {n_within}/{len(deltas)}")
print(f"  --> {'CONVERGED' if conv else 'NOT CONVERGED'} at 750k (criterion: mean|Δ%|<5% AND >=4/5 within 5%)")
if not conv:
    print("  --> pre-registered ceiling reached (750k). STOP and report: 80-100 undertrained is the finding.")
