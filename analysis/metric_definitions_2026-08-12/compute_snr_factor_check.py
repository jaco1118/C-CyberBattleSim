"""Task METRIC-DEFINITIONS, Q2 (quantitative part): how far does norm(h1)/norm(h2) depart
from 1 on the population that feeds the published SNR figures (0.492 level, log-log slope)?

Source of the SNR column (compute_attenuation_analysis.py:572-576):
    attenuation_df['snr'] = np.where(
        attenuation_df['zero_noise_floor'], np.nan,
        attenuation_df['change_drift_full'] / attenuation_df['agent_drift_full']
    )
where agent_drift_full = _rel_drift(h1.combined, h2.combined) = norm(h2-h1)/norm(h1), and
change_drift_full = _rel_drift(h2.combined, h3.combined) = norm(h3-h2)/norm(h2)
(cyberbattle_env_compressed.py:1056-1058, 1121-1122). So the code's SNR is
    [norm(h3-h2)/norm(h2)] / [norm(h2-h1)/norm(h1)] = [norm(h3-h2)/norm(h2-h1)] * [norm(h1)/norm(h2)]
-- the quotient of RELATIVE quantities, carrying an extra norm(h1)/norm(h2) factor beyond the
"ratio of the two movements" reading. This script quantifies that factor directly from norm_h1/
norm_h2, both stored verbatim in the drift CSV (cyberbattle_env_compressed.py:1096,1125-1126),
on the same population and filters used to build the published SNR figures:
  - dataset: attenuation_drift_logs (confirmed the "live 5-seed grid" -- its raw row counts
    187,647/640,230/775,122 match evidence_taskT.md's Gate row counts table exactly)
  - change_type == 'membership_leave' (the type the published 0.492 SNR figure is reported for;
    SNR is structurally undefined for membership_join)
  - touched_node_visible != False (load_and_filter, compute_attenuation_analysis.py:508)
  - event_phase in {'immediate', 'attributed'} (compute_attenuation_analysis.py:514)
  - zero_noise_floor excluded: agent_drift_full >= ZERO_NOISE_FLOOR_THRESHOLD (1e-9)
    (compute_attenuation_analysis.py:564, 571-576)

No experiment re-run. Reads only the already-on-disk attenuation_drift_logs/drift_<band>.csv.
"""
import os

import numpy as np
import pandas as pd

DATA_DIR = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents/attenuation_drift_logs"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BANDS = ["10-15", "30-40", "80-100"]
ZERO_NOISE_FLOOR_THRESHOLD = 1e-9

COLS = ["change_type", "touched_node_visible", "event_phase", "agent_drift_full",
        "change_drift_full", "norm_h1", "norm_h2"]


def main():
    print("=== Q2: norm(h1)/norm(h2) on the SNR-eligible membership_leave population ===\n")
    rows = []
    all_ratio = []
    all_snr_code = []
    all_snr_intended = []
    for band in BANDS:
        df = pd.read_csv(os.path.join(DATA_DIR, f"drift_{band}.csv"), usecols=COLS, low_memory=False)
        df = df[df["touched_node_visible"] != False]  # noqa: E712
        df = df[df["event_phase"].isin(["immediate", "attributed"])]
        leaves = df[df["change_type"] == "membership_leave"].copy()
        leaves = leaves[leaves["agent_drift_full"] >= ZERO_NOISE_FLOOR_THRESHOLD]
        leaves = leaves.dropna(subset=["norm_h1", "norm_h2", "agent_drift_full", "change_drift_full"])

        ratio = leaves["norm_h1"] / leaves["norm_h2"]
        snr_code = leaves["change_drift_full"] / leaves["agent_drift_full"]  # what the code actually computes
        snr_intended = snr_code / ratio  # = norm(h3-h2)/norm(h2-h1), the "movement ratio" reading with no extra factor

        med, lo, hi = ratio.median(), ratio.min(), ratio.max()
        p5, p95 = ratio.quantile(0.05), ratio.quantile(0.95)
        print(f"--- band {band} (n={len(leaves)}) ---")
        print(f"  norm_h1/norm_h2: median={med:.4f}  [min={lo:.4f}, max={hi:.4f}]  [p5={p5:.4f}, p95={p95:.4f}]")
        print(f"  snr as coded (change_drift_full/agent_drift_full): median={snr_code.median():.4f}")
        print(f"  snr if intended as norm(h3-h2)/norm(h2-h1) (no h1/h2-norm factor): median={snr_intended.median():.4f}")
        print(f"  ratio of the two SNR readings (median): {(snr_code.median()/snr_intended.median()):.4f}\n")

        rows.append(dict(band=band, n=len(leaves), ratio_median=med, ratio_min=lo, ratio_max=hi,
                          ratio_p5=p5, ratio_p95=p95,
                          snr_code_median=snr_code.median(), snr_intended_median=snr_intended.median()))
        all_ratio.append(ratio)
        all_snr_code.append(snr_code)
        all_snr_intended.append(snr_intended)

    pooled_ratio = pd.concat(all_ratio)
    pooled_code = pd.concat(all_snr_code)
    pooled_intended = pd.concat(all_snr_intended)
    print("=== pooled across all 3 bands ===")
    print(f"  norm_h1/norm_h2: median={pooled_ratio.median():.4f}  "
          f"[min={pooled_ratio.min():.4f}, max={pooled_ratio.max():.4f}]  "
          f"[p5={pooled_ratio.quantile(0.05):.4f}, p95={pooled_ratio.quantile(0.95):.4f}]")
    print(f"  snr as coded: median={pooled_code.median():.4f}")
    print(f"  snr if intended (movement ratio only): median={pooled_intended.median():.4f}")

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "snr_factor_check.csv"), index=False)
    print("\nDone.")


if __name__ == "__main__":
    main()
