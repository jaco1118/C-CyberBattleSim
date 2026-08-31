"""Task REPLAY-FLAG-CLOSEOUT addendum: behavioural-signature check for graphdepth_sweep/ (the
narrow, 2-hop-restricted predecessor of graphdepth_sweep_wide/), surfaced during STEP 0.6's
unrestricted flag search in TASK FLAG-GROUND-TRUTH but never itself directly checked there.

Launch command, recovered the same way as graphdepth_sweep_wide's (raw session transcript,
grep -o '"command":"[^"]*mkdir -p graphdepth_sweep\\nexport[^"]*"' over ~/.claude/projects/*/*.jsonl):
    mkdir -p graphdepth_sweep
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
    export RQ2C=1 LEG=1 YEG_DRIFT_DIR=graphdepth_sweep
    nohup $PY compute_attenuation_analysis.py --manifest attenuation_manifest.yaml --collect ...
Same RQ2C=1-only, no-CX_DIAG pattern as graphdepth_sweep_wide -- included here for a complete
STEP 2 inventory rather than left as an inferred-but-unchecked gap. Identical logic to
compute_flag_ground_truth.py, applied to this one directory. No experiment re-run.
"""
import os

import numpy as np
import pandas as pd

AGENTS_DIR = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BANDS = ["10-15", "30-40", "80-100"]
DATASET = "graphdepth_sweep"

COLS = ["change_type", "n_scenario", "n_discovered_h2", "touched_node_visible"]


def main():
    print(f"=== REPLAY-FLAG-CLOSEOUT addendum: behavioural signature for {DATASET} ===\n")
    rows = []
    pooled_used = 0
    pooled_visible = 0
    pooled_not_visible = 0
    for band in BANDS:
        path = os.path.join(AGENTS_DIR, DATASET, f"drift_{band}.csv")
        df = pd.read_csv(path, usecols=COLS, low_memory=False)
        leaves = df[df["change_type"] == "membership_leave"].copy()
        n_read = len(leaves)
        if n_read == 0:
            print(f"--- band {band}: n_rows_read=0 -> no events, undefined ---\n")
            rows.append(dict(dataset=DATASET, band=band, n_rows_read=0, n_rows_used=0,
                              n_rows_dropped_missing_col=0, n_visible=0, n_not_visible=0,
                              unknown_fraction="no events, undefined"))
            continue

        missing_mask = leaves["touched_node_visible"].isna()
        n_dropped_missing = int(missing_mask.sum())
        used = leaves[~missing_mask]
        n_used = len(used)
        n_visible = int((used["touched_node_visible"] == True).sum())  # noqa: E712
        n_not_visible = int((used["touched_node_visible"] == False).sum())  # noqa: E712
        assert n_visible + n_not_visible == n_used

        unknown_fraction = (n_not_visible / n_used) if n_used else "no events, undefined"
        discfrac = (used["n_discovered_h2"] / used["n_scenario"]).replace([np.inf, -np.inf], np.nan).dropna()
        discfrac_median = float(discfrac.median()) if len(discfrac) else None

        uf_str = f"{unknown_fraction:.4f}" if isinstance(unknown_fraction, float) else unknown_fraction
        print(f"--- band {band} ---")
        print(f"  n_rows_read={n_read}  n_rows_dropped_missing_col={n_dropped_missing}  n_rows_used={n_used}")
        print(f"  n_visible={n_visible}  n_not_visible={n_not_visible}  unknown_fraction={uf_str}")
        print(f"  n_discovered_h2/n_scenario median at leave-time: {discfrac_median}\n")

        rows.append(dict(dataset=DATASET, band=band, n_rows_read=n_read, n_rows_used=n_used,
                          n_rows_dropped_missing_col=n_dropped_missing,
                          n_visible=n_visible, n_not_visible=n_not_visible,
                          unknown_fraction=unknown_fraction, discfrac_median=discfrac_median))
        pooled_used += n_used
        pooled_visible += n_visible
        pooled_not_visible += n_not_visible

    pooled_uf = (pooled_not_visible / pooled_used) if pooled_used else "no events, undefined"
    pooled_uf_str = f"{pooled_uf:.4f}" if isinstance(pooled_uf, float) else pooled_uf
    print(f"=== {DATASET} POOLED: n_used={pooled_used}  n_visible={pooled_visible}  "
          f"n_not_visible={pooled_not_visible}  unknown_fraction={pooled_uf_str} ===")
    rows.append(dict(dataset=DATASET, band="POOLED", n_rows_read=pooled_used, n_rows_used=pooled_used,
                      n_visible=pooled_visible, n_not_visible=pooled_not_visible, unknown_fraction=pooled_uf))

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "graphdepth_sweep_narrow_signature.csv"), index=False)
    print("\nDone.")


if __name__ == "__main__":
    main()
