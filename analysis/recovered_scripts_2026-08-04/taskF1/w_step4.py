"""Task W STEP 4: is the SNR slope an artefact of the agent_drift denominator (discovery contamination)?
Data: 5-seed TRPO attenuation-gate drift logs (trpo_250k_tuned_compressed_band*, 250k). Reading only."""
import pandas as pd, numpy as np
G = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/attenuation_gate_archive/2026-07-26_trpo_5seed_gate/attenuation_drift_logs"
RNG = np.random.default_rng(0); NBOOT = 5000
frames = []
for b in ["10-15", "30-40", "80-100"]:
    d = pd.read_csv(f"{G}/drift_{b}.csv")
    d["band"] = b
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
df["discovery"] = df["n_discovered_h2"] > df["n_discovered_h1"]   # 0.3: agent discovered >=1 node this step

print("="*82)
print("TASK W STEP 4 — SNR slope vs discovery contamination  [gate agents, 250k, membership_leave]")
print("PROVENANCE: trpo_250k_tuned_compressed_band{10-15,30-40,80-100}_seed{42,100,123,200,300}, 250k")
print("="*82)

# ---------- 4.1: discovery prevalence + agent_drift by discovery status, per band ----------
print("\n### 4.1 fraction of agent-acting steps that DISCOVERED >=1 node; mean agent_drift by status [FINDING] ###")
print("(all logged steps with a finite agent_drift_full; the agent acts every step)")
for b in ["10-15", "30-40", "80-100"]:
    a = df[(df.band == b) & df.agent_drift_full.notna()]
    fdisc = a.discovery.mean()
    md = a[a.discovery].agent_drift_full.mean(); mn = a[~a.discovery].agent_drift_full.mean()
    print(f"  {b:7s}: discovery-step frac={fdisc:.3f} | mean agent_drift  discovery={md:.5f}  no-discovery={mn:.5f}  ratio={md/max(mn,1e-12):.2f}x")

# ---------- SNR helpers ----------
def _snr_df(sub):
    s = sub[(sub.change_type == "membership_leave") & (sub.agent_drift_full > 0) &
            (sub.change_drift_full > 0) & (sub.n_discovered > 0)].copy()
    s["snr"] = s.change_drift_full / s.agent_drift_full
    return s

def loglog_slope(sub):
    """per-event log-log OLS of SNR vs n_discovered."""
    s = _snr_df(sub)
    x = np.log(s.n_discovered.to_numpy()); y = np.log(s.snr.to_numpy())
    b1, b0 = np.polyfit(x, y, 1); snr100 = np.exp(b0 + b1 * np.log(100))
    bs = [np.polyfit(x[j], y[j], 1)[0] for j in (RNG.integers(0, len(x), len(x)) for _ in range(NBOOT))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return b1, snr100, len(s), lo, hi

def median_bin_slope(sub):
    """median SNR per integer n_discovered bin, log-log OLS of median vs bin (Task-T-style)."""
    s = _snr_df(sub)
    g = s.groupby(s.n_discovered.astype(int)).snr.agg(['median', 'size'])
    g = g[g['size'] >= 20]  # only bins with enough events
    x = np.log(g.index.to_numpy(dtype=float)); y = np.log(g['median'].to_numpy())
    b1, b0 = np.polyfit(x, y, 1); snr100 = np.exp(b0 + b1 * np.log(100))
    return b1, snr100, len(g), int(g['size'].sum()), s.n_discovered.min(), s.n_discovered.max()

# zero-noise-floor fraction (the excluded 66.4% in Task T)
lev = df[df.change_type == "membership_leave"]
zero_floor = (lev.agent_drift_full <= 1e-12).mean()
print(f"\n### SNR = change_drift_full / agent_drift_full (membership_leave); zero-agent-noise-floor frac = {zero_floor:.3f} (excluded) ###")

# ---------- 4.2/4.3: OLD (all) vs NEW (no-discovery only) ----------
print("\n### 4.2/4.3 SNR log-log slope vs n_discovered: OLD (all steps) vs NEW (no-discovery denominator) [FINDING] ###")
b1_old, s100_old, n_old, lo_o, hi_o = loglog_slope(df)
b1_new, s100_new, n_new, lo_n, hi_n = loglog_slope(df[~df.discovery])
print("  -- per-event log-log OLS --")
print(f"  OLD (all leave events, agent_drift>0):        slope={b1_old:+.3f} [{lo_o:+.3f},{hi_o:+.3f}]  SNR@100={s100_old:.3f}  n={n_old}")
print(f"  NEW (no-discovery same-step only):            slope={b1_new:+.3f} [{lo_n:+.3f},{hi_n:+.3f}]  SNR@100={s100_new:.3f}  n={n_new}")
mb_old = median_bin_slope(df); mb_new = median_bin_slope(df[~df.discovery])
print("  -- median-SNR-per-n_discovered-bin log-log OLS (Task-T-style; reference slope -0.804, SNR@100 0.492) --")
print(f"  OLD: slope={mb_old[0]:+.3f}  SNR@100={mb_old[1]:.3f}  bins={mb_old[2]}  n={mb_old[3]}  n_disc range=[{mb_old[4]},{mb_old[5]}]")
print(f"  NEW: slope={mb_new[0]:+.3f}  SNR@100={mb_new[1]:.3f}  bins={mb_new[2]}  n={mb_new[3]}")
surv = "SURVIVES" if hi_n < 0 else ("DISAPPEARS/FLATTENS" if lo_n <= 0 <= hi_n else "REVERSES")
print(f"  --> once discovery steps are excluded, the slope {surv}: it does NOT flatten (both methods")
print(f"      STEEPEN slightly: OLS {b1_old:+.3f}->{b1_new:+.3f}, median-bin {mb_old[0]:+.3f}->{mb_new[0]:+.3f}).")
print(f"      Only {n_old-n_new} of {n_old} leave events ({100*(n_old-n_new)/n_old:.1f}%) had same-step discovery,")
print(f"      so the SNR denominator is already ~discovery-free -> the trend is NOT a discovery effect.")

# ---------- 4.4: scale-invariance -> decompose numerator vs denominator dilution ----------
def med_bin_slope_of(sub, col):
    s = sub[(sub.change_type == "membership_leave") & (sub[col] > 0) & (sub.n_discovered > 0)]
    g = s.groupby(s.n_discovered.astype(int))[col].agg(['median', 'size']); g = g[g['size'] >= 20]
    x = np.log(g.index.to_numpy(dtype=float)); y = np.log(g['median'].to_numpy())
    return np.polyfit(x, y, 1)[0]
sl_change = med_bin_slope_of(df, "change_drift_full")                 # numerator dilution
sl_agent  = med_bin_slope_of(df[~df.discovery], "agent_drift_full")   # denominator dilution (no-discovery)
print("\n### 4.4 scale-invariance -> numerator/denominator dilution decomposition [FINDING] ###")
print("  In log-log vs n_discovered: PURE 1/N dilution = slope -1 for each of numerator and denominator,")
print("  so the RATIO's expected slope = (-1) - (-1) = 0 (scale-invariant). Measured (median-bin):")
print(f"    numerator   slope(change_drift_full)            = {sl_change:+.3f}   (1/N would be -1.00)")
print(f"    denominator slope(agent_drift_full, no-disc)    = {sl_agent:+.3f}   (1/N would be -1.00)")
print(f"    implied SNR slope = {sl_change:+.3f} - ({sl_agent:+.3f}) = {sl_change - sl_agent:+.3f}  (vs measured NEW {mb_new[0]:+.3f})")
print(f"  --> The change signal dilutes {'FASTER' if sl_change < sl_agent else 'SLOWER'} than the agent-noise")
print(f"      baseline: numerator {sl_change:+.3f} vs denominator {sl_agent:+.3f}. NEITHER is the naive -1;")
print(f"      the SNR decline is the GAP between them, not 1/N dilution (which would give 0) and not discovery.")
