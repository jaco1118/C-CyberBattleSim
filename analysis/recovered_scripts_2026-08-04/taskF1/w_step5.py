"""Task W STEP 5: measure the dilution reference (do not assume -1.0). Gate drift logs, reading only.
Absolute per-slice drift = relative change_drift_slice * norm_h2_slice. Slopes are log-log vs
n_discovered via MEAN-absolute-drift-per-integer-bin (mean INCLUDES zero-drift/silent events, so the
extremal slices' growing silence is captured); bootstrap 95% CI over events."""
import pandas as pd, numpy as np
G = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/attenuation_gate_archive/2026-07-26_trpo_5seed_gate/attenuation_drift_logs"
RNG = np.random.default_rng(0); NBOOT = 1000
frames = []
for b in ["10-15", "30-40", "80-100"]:
    d = pd.read_csv(f"{G}/drift_{b}.csv"); d["band"] = b; frames.append(d)
df = pd.concat(frames, ignore_index=True)
lev = df[(df.change_type == "membership_leave") & (df.n_discovered >= 2)].copy()
# absolute per-slice change drift = relative * norm_h2_slice
lev["abs_mean"] = lev.change_drift_mean * lev.norm_h2_mean
lev["abs_max"]  = lev.change_drift_max  * lev.norm_h2_max
lev["abs_min"]  = lev.change_drift_min  * lev.norm_h2_min
lev["abs_full"] = lev.change_drift_full * lev.norm_h2
lev["hbar_minus_hk"] = lev.abs_mean * (lev.n_discovered - 1.0)   # ||h_bar-h_k|| recovered (circular, see 5.1)
nd = lev.n_discovered.to_numpy(dtype=float)

def meanbin_slope(y, x=nd, mincount=20):
    """log-log OLS of MEAN(y) per integer-x bin (mean includes zeros); returns slope."""
    s = pd.DataFrame({"x": x, "y": y})
    g = s.groupby(s.x.astype(int)).y.agg(['mean', 'size']); g = g[(g['size'] >= mincount) & (g['mean'] > 0)]
    lx = np.log(g.index.to_numpy(dtype=float)); ly = np.log(g['mean'].to_numpy())
    return np.polyfit(lx, ly, 1)[0]

def medbin_slope(y, x=nd, mincount=20):
    s = pd.DataFrame({"x": x, "y": y}); s = s[s.y > 0]
    g = s.groupby(s.x.astype(int)).y.agg(['median', 'size']); g = g[g['size'] >= mincount]
    lx = np.log(g.index.to_numpy(dtype=float)); ly = np.log(g['median'].to_numpy())
    return np.polyfit(lx, ly, 1)[0]

def boot_ci(y, fn):
    yv = y.to_numpy() if hasattr(y, 'to_numpy') else np.asarray(y)
    pt = fn(yv)
    bs = []
    for _ in range(NBOOT):
        j = RNG.integers(0, len(yv), len(yv)); bs.append(fn(yv[j], nd[j]))
    return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

print("="*84)
print("TASK W STEP 5 — the dilution reference, measured  [gate agents 250k, membership_leave]")
print("PROVENANCE: trpo_250k_tuned_compressed_band*_seed*, 250k; n_leave_events=%d" % len(lev))
print("="*84)

# 5.1 components: is ||h_bar-h_k|| N-dependent?
print("\n### 5.1 the reference, measured (||h_bar-h_k|| is NOT directly loggable -> see note) [ARTIFACT] ###")
for name, col in [("||h_bar|| (norm_h2_mean)", lev.norm_h2_mean), ("||h_k|| (delta_h_v_norm)", lev.delta_h_v_norm)]:
    pt, lo, hi = boot_ci(col, meanbin_slope)
    print(f"    slope({name}) vs n_discovered = {pt:+.3f} [{lo:+.3f},{hi:+.3f}]")
pt_hk, lo_hk, hi_hk = boot_ci(lev.hbar_minus_hk, meanbin_slope)
print(f"    slope(||h_bar-h_k|| recovered = abs_mean*(N-1)) = a = {pt_hk:+.3f} [{lo_hk:+.3f},{hi_hk:+.3f}]")
print(f"    => dilution reference for the mean slice = a-1 = {pt_hk-1:+.3f}  (assumed value was -1.000)")
print("    NOTE: h_k's vector is not logged (only ||h_k|| and ||h_bar||); ||h_bar-h_k|| needs the")
print("    cross term h_bar.h_k, absent. It is recovered only AS abs_mean*(N-1), so 'a' and the")
print("    mean-slice absolute slope below are the same measurement (a = mean_abs_slope + 1), not")
print("    an independent check. The ||h_bar|| / ||h_k|| slopes above ARE independent evidence on a.")

# 5.2 absolute drift slopes, all slices
print("\n### 5.2 ABSOLUTE change-drift log-log slope vs n_discovered, per slice (mean-per-bin) [FINDING] ###")
abs_sl = {}
for s, col in [("mean", lev.abs_mean), ("max", lev.abs_max), ("min", lev.abs_min), ("full", lev.abs_full)]:
    pt, lo, hi = boot_ci(col, meanbin_slope); abs_sl[s] = pt
    print(f"    {s:5s}: {pt:+.3f} [{lo:+.3f},{hi:+.3f}]")

# 5.3 decomposition restated
print("\n### 5.3 decomposition restated on MEASURED quantities [FINDING] ###")
print(f"    empirical mean-slice (dilution) reference = {abs_sl['mean']:+.3f}   (was assumed -1.000)")
print(f"    full-vector absolute slope                = {abs_sl['full']:+.3f}")
print(f"    EXTREMAL contribution = full - mean       = {abs_sl['full']-abs_sl['mean']:+.3f}   (was inferred -0.46 vs assumed -1.0)")

# 5.4 consistency with response rates
print("\n### 5.4 consistency: extremal slopes vs per-slice response rates [FINDING] ###")
for b in ["10-15", "30-40", "80-100"]:
    sub = lev[lev.band == b]
    rr_max = (sub.change_drift_max > 0).mean(); rr_min = (sub.change_drift_min > 0).mean(); rr_mean = (sub.change_drift_mean > 0).mean()
    print(f"    {b:7s}: response-rate mean={rr_mean:.3f} max={rr_max:.3f} min={rr_min:.3f}")
print(f"    slopes: mean={abs_sl['mean']:+.3f}  max={abs_sl['max']:+.3f}  min={abs_sl['min']:+.3f}")
ok = (abs_sl['max'] < abs_sl['mean']) and (abs_sl['min'] < abs_sl['mean'])
print(f"    max/min slopes steeper (more negative) than mean? {ok}  (expected: silent-slices decline faster)")

# 5.5 relative vs absolute
print("\n### 5.5 RELATIVE change-drift slopes (denominator = slice norm, itself N-dependent) [FINDING] ###")
for s, col in [("mean", lev.change_drift_mean), ("max", lev.change_drift_max), ("min", lev.change_drift_min), ("full", lev.change_drift_full)]:
    pt, lo, hi = boot_ci(col, meanbin_slope)
    print(f"    {s:5s} relative: {pt:+.3f} [{lo:+.3f},{hi:+.3f}]")
pt_med, lo_med, hi_med = boot_ci(lev.change_drift_full, medbin_slope)
print(f"    (full relative, median-per-bin to match STEP 4's -1.461: {pt_med:+.3f} [{lo_med:+.3f},{hi_med:+.3f}])")
