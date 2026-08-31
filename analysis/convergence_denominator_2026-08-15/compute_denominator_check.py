"""Task CONVERGENCE DENOMINATOR CHECK, Q2-Q6: read-only inspection of the absolute values behind
Table IV.1's relative-change figures, for the same 15 manifest checkpoints (2026-07-26) and the
same tensorboard scalar ("train/Root owned nodes") the accepted run used (commit 2108f6d/780b65a).

Imports compute_attenuation_analysis.py's sibling compute_convergence_check.py DIRECTLY (unmodified,
not reimplemented) so the window/mean/denominator computation is guaranteed identical to the already-
accepted result -- this script only adds reporting of the intermediate absolute values that script's
own CLI output does not print (it only prints pre/fin to 3dp and Delta% to 2dp; this script reports
pre/fin to a higher precision, the raw absolute node difference, and per-band aggregates).

No training, no resumption, no new evaluation -- pure re-read of the same already-logged tfevents
scalars via the same already-committed tool.
"""
import glob
import os
import statistics
import sys

REPO = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim"
AGENTS = os.path.join(REPO, "cyberbattle/agents")
sys.path.insert(0, AGENTS)

import compute_convergence_check as ccc  # noqa: E402

BANDS = ["10-15", "30-40", "80-100"]
SEEDS = [42, 100, 123, 200, 300]
WINDOW = 50000
STOP = 250000
METRIC = "train/Root owned nodes"
THRESHOLD = 5.0


def main():
    print(f"=== Q2: absolute values, metric={METRIC!r}, window={WINDOW}, stop={STOP} ===\n")
    rows = []
    n_readable = 0
    n_unreadable = 0
    unreadable_detail = []
    n_zero_pre = 0

    for band in BANDS:
        for seed in SEEDS:
            hits = glob.glob(os.path.join(AGENTS, "logs",
                                           f"trpo_250k_tuned_compressed_band{band}_seed{seed}_2026-07-26_*"))
            if not hits:
                n_unreadable += 1
                unreadable_detail.append((band, seed, "no run folder found"))
                continue
            run_dir = os.path.join(sorted(hits)[-1], "TRPO_x_control_SecureBERT", "TRPO_1")
            try:
                r = ccc.delta_pct(run_dir, METRIC, WINDOW, STOP)
            except (FileNotFoundError, KeyError) as e:
                n_unreadable += 1
                unreadable_detail.append((band, seed, str(e)))
                continue

            if r["npre"] == 0 or r["nfin"] == 0:
                n_unreadable += 1
                unreadable_detail.append((band, seed,
                                           f"empty window: npre={r['npre']} nfin={r['nfin']}"))
                continue

            n_readable += 1
            pre, fin = r["pre"], r["fin"]
            abs_diff = fin - pre
            if pre == 0.0:
                n_zero_pre += 1
                delta_pct_str = "undefined (pre=0)"
            else:
                delta_pct_str = f"{r['delta_pct']:+.2f}%"

            rows.append(dict(band=band, seed=seed, pre=pre, fin=fin, abs_diff=abs_diff,
                              delta_pct=r["delta_pct"], delta_pct_str=delta_pct_str,
                              npre=r["npre"], nfin=r["nfin"]))
            print(f"  band {band:>7}  seed {seed:>4}  pre={pre:10.4f}  fin={fin:10.4f}  "
                  f"abs_diff={abs_diff:+9.4f}  delta%={delta_pct_str:>18}  (npre={r['npre']}, nfin={r['nfin']})")

    print(f"\n=== Q6: readability ===")
    print(f"  readable (both windows non-empty): {n_readable}/15")
    print(f"  unreadable: {n_unreadable}/15")
    for band, seed, reason in unreadable_detail:
        print(f"    band {band} seed {seed}: {reason}")
    print(f"  seeds with pre==0 (undefined ratio, counted not epsilon-floored): {n_zero_pre}")

    print(f"\n=== Q3: per-band aggregates (absolute values) ===")
    for band in BANDS:
        band_rows = [r for r in rows if r["band"] == band]
        pres = [r["pre"] for r in band_rows]
        abs_diffs = [r["abs_diff"] for r in band_rows]
        min_pre = min(pres)
        median_abs_diff = statistics.median(abs_diffs)
        median_pre = statistics.median(pres)
        one_node_pct_at_median = (1.0 / median_pre * 100.0) if median_pre else float("nan")
        one_node_pct_at_min = (1.0 / min_pre * 100.0) if min_pre else float("nan")
        print(f"  band {band}:")
        print(f"    smallest earlier-window mean across 5 seeds: {min_pre:.4f} nodes")
        print(f"    median earlier-window mean across 5 seeds:   {median_pre:.4f} nodes")
        print(f"    median absolute difference across 5 seeds:   {median_abs_diff:+.4f} nodes")
        print(f"    1 node / median earlier-window mean = {one_node_pct_at_median:.2f}%")
        print(f"    1 node / smallest earlier-window mean = {one_node_pct_at_min:.2f}%")

    print(f"\n=== Q4: node-count needed for exactly 5% tolerance, band 10-15 ===")
    band1015 = [r for r in rows if r["band"] == "10-15"]
    pres_1015 = [r["pre"] for r in band1015]
    median_pre_1015 = statistics.median(pres_1015)
    min_pre_1015 = min(pres_1015)
    nodes_at_5pct_median = 0.05 * median_pre_1015
    nodes_at_5pct_min = 0.05 * min_pre_1015
    print(f"  median earlier-window mean, band 10-15: {median_pre_1015:.4f} nodes")
    print(f"  nodes needed to hit exactly 5% at the median earlier-window mean: {nodes_at_5pct_median:.4f}")
    print(f"  smallest earlier-window mean, band 10-15 (worst case): {min_pre_1015:.4f} nodes")
    print(f"  nodes needed to hit exactly 5% at the smallest earlier-window mean: {nodes_at_5pct_min:.4f}")
    for r in band1015:
        nodes_at_5pct_this_seed = 0.05 * r["pre"]
        print(f"    seed {r['seed']}: pre={r['pre']:.4f}  nodes-for-5%={nodes_at_5pct_this_seed:.4f}")


if __name__ == "__main__":
    main()
