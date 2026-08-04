"""Task M STEP 2-3: pooled trial-level test of the two-hop-coverage hypothesis on the SPARSE (DFS-tree)
graph. Reads m_out/m_trials.csv (253 trials). Predictors standardized. Bootstrap over trials (n=253) for
CIs; scenario-level robustness noted. The hypothesis predicts a NEGATIVE deg x cov2 interaction (degree
stops predicting propagation as coverage rises)."""
import pandas as pd, numpy as np
B = "/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1"
df = pd.read_csv(f"{B}/m_out/m_trials.csv")
RNG = np.random.default_rng(3); NB = 20000
def z(x): return (x - x.mean()) / x.std(ddof=0)
def ols(y, X):  # X includes intercept col; returns beta
    return np.linalg.lstsq(X, y, rcond=None)[0]
def design(d):
    zd, zc = z(d["deg"].values), z(d["cov2"].values)
    return np.column_stack([np.ones(len(d)), zd, zc, zd*zc]), zd, zc
y = df["prop"].values

print("="*92); print("TASK M STEP 2 — pooled trial-level test (SPARSE / DFS-tree graph). n=%d trials." % len(df))
print("bands pooled: %s" % df.groupby('band').size().to_dict()); print("="*92)

print("\n## confound structure (2.4 context): how coverage, degree, N relate ##  [ARTIFACT]")
for a,b in [("cov2","N"),("deg","N"),("cov2","deg"),("cov2","prop"),("deg","prop")]:
    print(f"  corr({a:>4},{b:>4}) = {df[a].corr(df[b]):+.3f}")
print("  band means:")
print(df.groupby('band').agg(n=('prop','size'), prop_med=('prop','median'),
      corr_prop_deg=('prop', lambda s: s.corr(df.loc[s.index,'deg'])),
      cov2_mean=('cov2','mean'), cov2_var=('cov2','var'), N_mean=('N','mean')).round(4).to_string())

print("\n## 2.2 OLS: prop ~ deg + cov2 + deg:cov2  (standardized; hypothesis => NEGATIVE interaction) ##  [FINDING]")
X, zd, zc = design(df); beta = ols(y, X)
names = ["intercept","deg","cov2","deg:cov2"]
bs = np.zeros((NB, 4))
for i in range(NB):
    idx = RNG.integers(0, len(df), len(df))
    bs[i] = ols(y[idx], X[idx])
for j,nm in enumerate(names):
    lo,hi = np.percentile(bs[:,j],[2.5,97.5])
    print(f"  {nm:10s} = {beta[j]:+.4f}  CI95 [{lo:+.4f},{hi:+.4f}]" + ("   <- INTERACTION" if nm=="deg:cov2" else ""))
sign = "NEGATIVE (as hypothesis predicts)" if beta[3]<0 else "POSITIVE (opposite to hypothesis)"
int_lo,int_hi = np.percentile(bs[:,3],[2.5,97.5])
null = int_lo<=0<=int_hi
print(f"  -> interaction is {sign}; CI {'INCLUDES 0 (null)' if null else 'excludes 0'}")

print("\n## 2.3 corr(prop, deg) WITHIN coverage terciles (pooled across bands) ##  [FINDING]")
df["covbin"] = pd.qcut(df["cov2"], 3, labels=["low-cov","mid-cov","high-cov"])
for b in ["low-cov","mid-cov","high-cov"]:
    d = df[df.covbin==b]
    comp = d.groupby('band').size().to_dict()
    print(f"  {b:9s} (cov2 {d.cov2.min():.3f}-{d.cov2.max():.3f}): n={len(d)} corr(prop,deg)={d.prop.corr(d.deg):+.3f}  band-composition={comp}")
print("  hypothesis: corr strong at low-cov, weak at high-cov, regardless of band.")

print("\n## 2.4 controlling for N (is coverage just standing in for scale?) ##  [FINDING]")
# partial corr(prop, deg | N) and (prop, deg | cov2); and does deg:cov2 survive adding deg:N ?
def pcorr(a, b, c):  # partial corr of a,b controlling for c
    ra = a - np.polyval(np.polyfit(c, a, 1), c); rb = b - np.polyval(np.polyfit(c, b, 1), c)
    return np.corrcoef(ra, rb)[0,1]
dd = df.copy()
print(f"  partial corr(prop, deg | N)    = {pcorr(dd.prop.values, dd.deg.values, dd.N.values):+.3f}")
print(f"  partial corr(prop, deg | cov2) = {pcorr(dd.prop.values, dd.deg.values, dd.cov2.values):+.3f}")
print(f"  partial corr(cov2, deg | N)    = {pcorr(dd.cov2.values, dd.deg.values, dd.N.values):+.3f}  (deg vs coverage once scale removed)")
# augmented model with N and deg:N; does deg:cov2 survive?
zN = z(dd["N"].values); zd2 = z(dd["deg"].values); zc2 = z(dd["cov2"].values)
Xa = np.column_stack([np.ones(len(dd)), zd2, zc2, zN, zd2*zc2, zd2*zN])
ba = ols(y, Xa)
bsa = np.zeros((NB,6))
for i in range(NB):
    idx = RNG.integers(0,len(dd),len(dd)); bsa[i]=ols(y[idx],Xa[idx])
na = ["intercept","deg","cov2","N","deg:cov2","deg:N"]
print("  augmented OLS prop ~ deg+cov2+N+deg:cov2+deg:N:")
for j,nm in enumerate(na):
    lo,hi=np.percentile(bsa[:,j],[2.5,97.5]); print(f"    {nm:9s}={ba[j]:+.4f} CI[{lo:+.4f},{hi:+.4f}]")
dcov_lo,dcov_hi = np.percentile(bsa[:,4],[2.5,97.5])
print(f"  -> deg:cov2 after controlling for N/deg:N: {ba[4]:+.4f} CI[{dcov_lo:+.4f},{dcov_hi:+.4f}] "
      f"({'still nonzero' if not (dcov_lo<=0<=dcov_hi) else 'now NULL -> coverage stood in for scale'})")

print("\n## 2.5 trials per coverage bin + band dominance ##  [ARTIFACT]")
ct = pd.crosstab(df.covbin, df.band)
print(ct.to_string())
print("  band-overlap in coverage (min/max cov2 per band):")
for band in ["10-15","30-40","80-100"]:
    d=df[df.band==band]; print(f"    {band:7s}: cov2 [{d.cov2.min():.3f}, {d.cov2.max():.3f}]")
print("="*92)
