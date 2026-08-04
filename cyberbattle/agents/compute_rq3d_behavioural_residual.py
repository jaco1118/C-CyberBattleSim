"""RQ3D Addendum 7: does a behavioural-residual ranking change the departures-vs-loss finding?

Pure analysis on already-collected data (rq3d_data/, from rq3d_rollout.py) -- no new episodes, no
new rollout, no training, no compute beyond this script.

BACKGROUND: this rollout's PRIMARY ranking metric (compute_rq3d_renormalize.py) is
  loss(episode) = static_root_owned_count(seed,topology,band) - final_root_owned_count(episode)
Under that ranking, top-decile-loss episodes have MORE root_owned_departures than rest (4.16 vs
2.58 pooled) -- the OPPOSITE direction from Task CX PART 3's original figures (0.70 vs 1.23). One
flagged explanation: `loss` is built from final_root_owned_count, which departures mechanically
deplete, so `loss` and `root_owned_departures` are entangled by construction.

STEP 2 finding (re-checked against the rescued text at commit 1d6aaab, section 3.8): the ORIGINAL
analysis defines, per-episode, `residual = loss - gross mechanical`, stated in that exact form
("the per-episode residual (loss - gross mechanical) is dominated by noise"). Section 3.5's
band-level table uses "mechanical" and "root departures/ep" interchangeably ("mechanical
(root departures/ep, seed sd)"), so "gross mechanical" for a single episode is read literally as
root_owned_departures itself (same root-owned-count units as loss, no regression scaling implied).
IMPORTANT: section 3.9 explicitly ranks on "positive-LOSS episodes", NOT "positive-residual" --
so the rescued text does NOT indicate the original TOP-DECILE RANKING itself was residualized; the
residual formula in 3.8 is used for a SEPARATE regression (behavioural residual ~ churn/type), not
for 3.9's ranking. This script's residualized ranking is therefore an EXPLORATORY check on this
rollout's own data, not a reproduction of a residualized original ranking (none is evidenced).

THIS SCRIPT computes: behavioural_residual(episode) = loss(episode) - root_owned_departures(episode)
(the literal, unit-matched subtraction per the formula above), re-ranks the same pooled positive-
residual episodes into a new top-decile, and reports the departures comparison under this ranking
alongside the two already-reported results.

CIRCULARITY CAVEAT (symmetric to the original one, now in the OPPOSITE direction, stated up front
rather than after the fact): subtracting departures directly out of the ranking variable makes
`behavioural_residual` NEGATIVELY entangled with departures by construction -- an episode with more
departures gets a mechanically LOWER residual, all else equal. So if the top-residual group shows
FEWER departures than the rest, part of that is expected from the subtraction itself, the same way
part of the original raw-loss result's MORE-departures direction was expected from loss being built
on final_root_owned_count. This exercise is reported as a sensitivity check on how much the ranking
metric's construction drives the direction, not as an unbiased arbiter of the "true" relationship.

Usage: python compute_rq3d_behavioural_residual.py [--data-dir rq3d_data]
"""
import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np

BANDS = ["10-15", "30-40", "80-100"]
N_BOOT = 10000


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_all(data_dir):
    change, static = [], []
    for band in BANDS:
        for rec in load_jsonl(os.path.join(data_dir, "change", f"eventgraph_{band}", "event_episode.jsonl")):
            rec["band"] = band
            change.append(rec)
        for rec in load_jsonl(os.path.join(data_dir, "static", f"eventgraph_{band}", "event_episode.jsonl")):
            rec["band"] = band
            static.append(rec)
    return change, static


def build_static_lookup(static_eps):
    by_key = defaultdict(list)
    for r in static_eps:
        by_key[(r["band"], r["seed"], r["scenario_id"])].append(r["final_root_owned_count"])
    return {k: float(np.mean(v)) for k, v in by_key.items()}


def unpaired_bootstrap_diff(a, b, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan"), (float("nan"), float("nan"))
    idx_a = rng.integers(0, len(a), size=(n, len(a)))
    idx_b = rng.integers(0, len(b), size=(n, len(b)))
    diffs = a[idx_a].mean(axis=1) - b[idx_b].mean(axis=1)
    return float(np.mean(a) - np.mean(b)), (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def split_top_decile(sorted_eps):
    n_top = max(1, math.ceil(0.10 * len(sorted_eps))) if sorted_eps else 0
    return sorted_eps[:n_top], sorted_eps[n_top:]


def report(label, top, rest):
    top_dep = [r["root_owned_departures"] for r in top]
    rest_dep = [r["root_owned_departures"] for r in rest]
    diff, ci = unpaired_bootstrap_diff(top_dep, rest_dep)
    resolved = not (math.isnan(ci[0]) or math.isnan(ci[1])) and (ci[0] > 0 or ci[1] < 0)
    print(f"{label}: top n={len(top)} mean_dep={np.mean(top_dep) if top_dep else float('nan'):.3f} | "
          f"rest n={len(rest)} mean_dep={np.mean(rest_dep) if rest_dep else float('nan'):.3f} | "
          f"diff={diff:+.3f} 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}] "
          f"({'resolved' if resolved else 'NOT resolved'})")
    return {"label": label, "n_top": len(top), "n_rest": len(rest),
            "top_mean": np.mean(top_dep) if top_dep else float("nan"),
            "rest_mean": np.mean(rest_dep) if rest_dep else float("nan"),
            "diff": diff, "ci": ci, "resolved": resolved}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="rq3d_data")
    a = ap.parse_args()

    change, static = load_all(a.data_dir)
    static_lookup = build_static_lookup(static)
    print(f"Loaded {len(change)} change-arm episodes, {len(static)} static-arm episodes.\n")

    for r in change:
        key = (r["band"], r["seed"], r["scenario_id"])
        srac = static_lookup.get(key)
        r["static_root_owned_count"] = srac
        r["loss"] = (srac - r["final_root_owned_count"]) if srac is not None else None
        r["departures_per_static_root"] = (
            r["root_owned_departures"] / srac if (srac is not None and srac > 0) else None)
        r["behavioural_residual"] = (
            r["loss"] - r["root_owned_departures"] if r["loss"] is not None else None)

    scored = [r for r in change if r["loss"] is not None]

    print("=" * 78)
    print("(a) RAW-LOSS ranking (already reported, primary metric: loss = static_root_owned - final)")
    print("=" * 78)
    positive_loss = sorted([r for r in scored if r["loss"] > 0], key=lambda r: r["loss"], reverse=True)
    top_a, rest_a = split_top_decile(positive_loss)
    res_a = report("  raw-loss-ranked, raw departures", top_a, rest_a)

    top_a_ratio = [r["departures_per_static_root"] for r in top_a if r["departures_per_static_root"] is not None]
    rest_a_ratio = [r["departures_per_static_root"] for r in rest_a if r["departures_per_static_root"] is not None]
    diff_ar, ci_ar = unpaired_bootstrap_diff(top_a_ratio, rest_a_ratio)
    print(f"  raw-loss-ranked, RENORMALIZED (dep/static-root): top mean={np.mean(top_a_ratio):.4f} "
          f"rest mean={np.mean(rest_a_ratio):.4f} diff={diff_ar:+.4f} 95% CI [{ci_ar[0]:+.4f}, {ci_ar[1]:+.4f}]")

    print()
    print("=" * 78)
    print("(b) BEHAVIOURAL-RESIDUAL ranking (new, this addendum): residual = loss - root_owned_departures")
    print("    CIRCULARITY CAVEAT: subtracting departures out of the ranking variable makes this ")
    print("    NEGATIVELY entangled with departures by construction -- see script header.")
    print("=" * 78)
    positive_resid = sorted([r for r in scored if r["behavioural_residual"] is not None and r["behavioural_residual"] > 0],
                             key=lambda r: r["behavioural_residual"], reverse=True)
    top_b, rest_b = split_top_decile(positive_resid)
    res_b = report("  residual-ranked, raw departures", top_b, rest_b)

    print()
    print("=" * 78)
    print("SUMMARY: does 'top group has MORE departures' survive under each ranking?")
    print("=" * 78)
    print(f"  Original CX PART 3 (different population, ranking metric unconfirmed but text says "
          f"'loss', not residual): top-decile-loss=0.70 < rest=1.23 -- FEWER departures in top group.")
    print(f"  (a) This rollout, raw-loss ranking:        top={res_a['top_mean']:.3f} "
          f"{'>' if res_a['top_mean'] > res_a['rest_mean'] else '<'} rest={res_a['rest_mean']:.3f} "
          f"-- {'MORE' if res_a['top_mean'] > res_a['rest_mean'] else 'FEWER'} departures in top group "
          f"({'resolved' if res_a['resolved'] else 'not resolved'}).")
    print(f"  (b) This rollout, behavioural-residual ranking: top={res_b['top_mean']:.3f} "
          f"{'>' if res_b['top_mean'] > res_b['rest_mean'] else '<'} rest={res_b['rest_mean']:.3f} "
          f"-- {'MORE' if res_b['top_mean'] > res_b['rest_mean'] else 'FEWER'} departures in top group "
          f"({'resolved' if res_b['resolved'] else 'not resolved'}) "
          f"-- NOTE: this direction is partly expected BY CONSTRUCTION (see circularity caveat).")


if __name__ == "__main__":
    main()
