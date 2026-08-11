"""Task PROVENANCE-COMMIT, Item 3: probe_p.py's per-hop mean shift-norm, recomputed on the real,
unbiased (GRAPH-DEPTH-WIDE) population instead of the synthetic BFS/DFS-tree proxy.

Not a share (STEP 0, Q3.1, accepted in full): probe_p.py's own construction is a two-level,
unweighted MEAN OF MEANS of survivor shift-norms `norm(hp[n]-h[n])`, grouped by hop distance to
the departing node -- never a proportion of a total, never a shared denominator. Reproduced exactly
here:
  - per event, per hop bin: mean of that event's own survivor shift-norms in that bin
  - across events: mean of those per-event means (unweighted -- an event with 1 survivor at a hop
    counts the same as an event with 40)
  - the departing node itself is never in the survivor set
  - unreachable survivors (no path from v in the pre-change graph) go in their own bin, excluded
    from every finite-hop aggregate -- probe_p.py's own convention (its bin 99)
  - events with ZERO reachable survivors at any hop contribute to no finite bin at all; they are
    counted and reported separately, never folded in as zeros

Also reports the one-level pooled alternative (mean over all survivors from all events, ignoring
event boundaries) as an explicitly labelled second figure, since the two can diverge when event
sizes vary sharply -- exactly the situation at the largest band, where most events have very few
survivors at any finite hop.

Source: cyberbattle/agents/graphdepth_sweep_wide/leaveembed_<band>/*/*.jsonl (widened logger,
commit 1522b71; decomposition precedent, commit da5c539) -- every present node's embedding at
each single-node membership_leave event, plus hop_distance (full, uncapped BFS) per survivor.
Restricted to n_touched_nodes==1, matching every other analysis in this project on this data.
"""
import json
import glob
import os
from collections import defaultdict
import numpy as np
import pandas as pd

SWEEP_DIR = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents/graphdepth_sweep_wide"
OUT_DIR = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/analysis/provenance_commit_2026-08-11"
BANDS = ["10-15", "30-40", "80-100"]
HOP_BINS = list(range(1, 11))  # 1 through 10, shown even where exactly zero

# probe_p.py's own reported figures (synthetic proxy), for side-by-side comparison only -- not a target to match
PROXY_1HOP = {"10-15": 0.52, "30-40": 0.45, "80-100": 0.50}
PROXY_2HOP = {"10-15": 0.32, "30-40": 0.26, "80-100": 0.31}


def load_single_node_events(band):
    recs = []
    for f in sorted(glob.glob(os.path.join(SWEEP_DIR, f"leaveembed_{band}", "*", "*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r["n_touched_nodes"] == 1:
                    recs.append(r)
    return recs


def main():
    print("=== ITEM 3: per-hop mean shift-norm, wide population ===\n")
    summary_rows = []
    pooled_rows = []

    for band in BANDS:
        recs = load_single_node_events(band)
        n_events_total = len(recs)

        # two-level (event-then-across-events) accumulator: bin -> list of per-event means
        two_level = defaultdict(list)
        # one-level pooled accumulator: bin -> list of every individual survivor shift-norm
        pooled = defaultdict(list)
        # per-bin: how many events contributed at all, how many survivor-instances summed
        n_events_per_bin = defaultdict(int)
        n_survivors_per_bin = defaultdict(int)

        n_zero_reachable_survivors = 0

        for r in recs:
            pre = r["pre_embeddings"]
            post = r["post_embeddings"]
            hop_distance = r["hop_distance"]
            surv = [k for k in post if k in pre]

            # bucket this event's own survivors by hop, computing shift-norm per survivor
            event_bins = defaultdict(list)
            for n in surv:
                shift = float(np.linalg.norm(np.array(post[n], dtype=np.float64) - np.array(pre[n], dtype=np.float64)))
                d = hop_distance.get(n)
                bin_label = "unreachable" if d is None else int(d)
                event_bins[bin_label].append(shift)
                pooled[bin_label].append(shift)

            has_finite_hop_survivor = any(isinstance(b, int) for b in event_bins.keys())
            if not has_finite_hop_survivor:
                n_zero_reachable_survivors += 1

            for bin_label, vals in event_bins.items():
                two_level[bin_label].append(float(np.mean(vals)))  # this event's own mean in this bin
                n_events_per_bin[bin_label] += 1
                n_survivors_per_bin[bin_label] += len(vals)

        print(f"--- band {band} ---")
        print(f"  single-node events: {n_events_total}")
        print(f"  events with ZERO reachable survivors at any finite hop (excluded from every hop mean, not a zero): "
              f"{n_zero_reachable_survivors} ({n_zero_reachable_survivors/n_events_total*100:.2f}%)")

        for bin_label in HOP_BINS + ["unreachable"]:
            n_ev = n_events_per_bin.get(bin_label, 0)
            n_sv = n_survivors_per_bin.get(bin_label, 0)
            two_lvl_mean = float(np.mean(two_level[bin_label])) if two_level[bin_label] else float("nan")
            pooled_mean = float(np.mean(pooled[bin_label])) if pooled[bin_label] else float("nan")
            diverge = (abs(two_lvl_mean - pooled_mean) if (not np.isnan(two_lvl_mean) and not np.isnan(pooled_mean)) else float("nan"))
            label = f"{bin_label}hop" if isinstance(bin_label, int) else bin_label
            print(f"  {label:12s}: n_events={n_ev:5d}  n_survivors={n_sv:6d}  "
                  f"two-level mean={two_lvl_mean:.4f}  pooled mean={pooled_mean:.4f}  |diff|={diverge:.4f}")
            summary_rows.append(dict(
                band=band, hop_bin=label, n_events=n_ev, n_survivors=n_sv,
                two_level_mean_shift_norm=two_lvl_mean, pooled_mean_shift_norm=pooled_mean,
                abs_diff_two_level_vs_pooled=diverge,
            ))

        p1 = next((row["two_level_mean_shift_norm"] for row in summary_rows if row["band"] == band and row["hop_bin"] == "1hop"), float("nan"))
        p2 = next((row["two_level_mean_shift_norm"] for row in summary_rows if row["band"] == band and row["hop_bin"] == "2hop"), float("nan"))
        print(f"  proxy (synthetic, probe_p.py) comparison, NOT a target: "
              f"1hop proxy={PROXY_1HOP[band]} wide={p1:.4f} (diff {p1-PROXY_1HOP[band]:+.4f})  "
              f"2hop proxy={PROXY_2HOP[band]} wide={p2:.4f} (diff {p2-PROXY_2HOP[band]:+.4f})")
        print()

        pooled_rows.append(dict(band=band, n_events_total=n_events_total,
                                 n_zero_reachable_survivors=n_zero_reachable_survivors))

    pd.DataFrame(summary_rows).to_csv(os.path.join(OUT_DIR, "item3_hop_shares_wide.csv"), index=False)
    pd.DataFrame(pooled_rows).to_csv(os.path.join(OUT_DIR, "item3_zero_reachable_counts.csv"), index=False)
    print("Done.")


if __name__ == "__main__":
    main()
