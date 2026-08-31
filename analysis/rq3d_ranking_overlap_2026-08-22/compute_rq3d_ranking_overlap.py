"""RQ3d clean same-episode ranking-overlap analysis.

User's explicit brief: use the EXISTING RQ3D 3000-episode dataset only (no new rollout, no
retraining). Retrieve cyberbattle/agents/rq3d_data/{change,static}/eventgraph_<band>/
event_episode.jsonl from commit 3d8c9aa on branch attenuation-pooling-scale (the commit that
added them -- "RQ3D: full renormalized top-decile-loss analysis (3000 episodes, all 3 bands)").
Fetched via `git show <commit>:<path>` at run time rather than re-committed here, since the data
already has a durable home in git history; re-copying it into this branch would just duplicate
~3000 rows of already-safely-committed data.

DEFINITIONS (unchanged, per the user's explicit instruction -- verified against
analysis/rq1b_mech_split_scale_2026-08-08/compute_mechanical_share_scale.py:19-26, the project's
own canonical "FORMULA B", and matching compute_rq3d_renormalize.py /
compute_rq3d_behavioural_residual.py's own conventions on this exact dataset):

  static_root_owned_count(band, seed, scenario_id)
      = mean(final_root_owned_count) over STATIC-arm episodes sharing that (band, seed, scenario_id)
  gross_root_loss(episode)      = static_root_owned_count(...) - final_root_owned_count(episode)
  mechanical_root_loss(episode) = root_owned_departures(episode)          [a logged per-episode count]
  behavioural_residual(episode) = gross_root_loss(episode) - mechanical_root_loss(episode)

EPISODE IDENTITY: (band, seed, scenario_id, episode) -- "episode" alone is only a per-(seed,topology)
sub-run index (rq3d_rollout.py starts a fresh env per (band,seed,topology) block), so the full tuple
is the unique key, matching this project's established (seed, scenario_id, episode [,band]) convention
used throughout (e.g. compute_attenuation_analysis.py's own group_cols).

USABLE POPULATION: a change episode is "usable" iff its (band,seed,scenario_id) key has at least one
matching static-arm episode (so static_root_owned_count, and therefore gross_root_loss and
behavioural_residual, are both defined). No epsilon substitution, no exclusion of static==0 unless it
makes departures_per_static_root undefined elsewhere (not used in this script).

RANKING (worst decile), for EACH of the two metrics independently, matching established convention
(compute_rq3d_renormalize.py's positive-loss filter / compute_rq3d_behavioural_residual.py's
positive-residual filter, both reused unchanged here -- not a new convention):
  - restrict to usable episodes with metric > 0 (episodes with metric <= 0 cannot be a "loss" episode
    under either metric and are excluded from that metric's own ranking -- stated explicitly per the
    user's request to state exactly how zero/negative values are handled);
  - sort descending; worst decile = ceil(0.10 * n_positive), minimum 1.
This is applied SEPARATELY to gross_root_loss and to behavioural_residual, over the SAME usable
episode population (the "positive" sub-filter differs per metric, but the universe it's drawn from,
and the episode-identity keys used for the overlap comparison in step 4, are identical).

POOLING: computed for transparency, but flagged, not treated as a valid primary result, because
evidence_taskRQ3D.md already established (and this script independently re-confirms) that pooling an
ABSOLUTE-count metric across bands of very different scale (static_root_owned_count means ~7.8/23/30)
structurally starves the smallest band of pooled top-decile representation. Per-band is the primary,
trustworthy view; pooled is reported alongside with that caveat attached, not silently used to
replace per-band results per the user's own "pooled only if statistically appropriate" instruction.

Safety: reads only already-committed git history (git show, read-only). No training, no environment
reset, no checkpoint/encoder touched, no new episodes generated. Nothing beyond git show + pandas/
numpy/scipy statistics is run.
"""
import json
import math
import subprocess
from collections import defaultdict

import numpy as np
from scipy import stats

COMMIT = "3d8c9aa"  # attenuation-pooling-scale: adds cyberbattle/agents/rq3d_data/**/event_episode.jsonl
BANDS = ["10-15", "30-40", "80-100"]
N_BOOT = 10000
BOOT_SEED = 0

ORIGINAL_LOSS_SHARE = {"10-15": 18, "30-40": 32, "80-100": 34}  # CX PART 3 3.9, rescued text (1d6aaab)
ORIGINAL_RAW_DEPARTURES = {"top_decile": 0.70, "rest": 1.23}    # same source


def git_show(path):
    out = subprocess.run(["git", "show", f"{COMMIT}:{path}"], capture_output=True, text=True, check=True)
    return out.stdout


def load_jsonl_str(s):
    return [json.loads(line) for line in s.splitlines() if line.strip()]


def load_all():
    change, static = [], []
    for band in BANDS:
        cp = f"cyberbattle/agents/rq3d_data/change/eventgraph_{band}/event_episode.jsonl"
        sp = f"cyberbattle/agents/rq3d_data/static/eventgraph_{band}/event_episode.jsonl"
        for r in load_jsonl_str(git_show(cp)):
            r["band"] = band
            change.append(r)
        for r in load_jsonl_str(git_show(sp)):
            r["band"] = band
            static.append(r)
    return change, static


def build_static_lookup(static_eps):
    by_key = defaultdict(list)
    for r in static_eps:
        by_key[(r["band"], r["seed"], r["scenario_id"])].append(r["final_root_owned_count"])
    return {k: float(np.mean(v)) for k, v in by_key.items()}


def episode_key(r):
    return (r["band"], r["seed"], r["scenario_id"], r["episode"])


def split_worst_decile(rows_sorted_desc):
    n_top = max(1, math.ceil(0.10 * len(rows_sorted_desc))) if rows_sorted_desc else 0
    return rows_sorted_desc[:n_top], rows_sorted_desc[n_top:]


def loss_share_pct(top, positive_all, col):
    denom = sum(r[col] for r in positive_all)
    return (100.0 * sum(r[col] for r in top) / denom) if denom else float("nan")


def band_analysis(usable, band_filter, label):
    rows = [r for r in usable if band_filter(r)]
    n_usable = len(rows)

    # --- gross-loss ranking ---
    pos_gross = [r for r in rows if r["gross_root_loss"] > 0]
    n_zero_gross = sum(1 for r in rows if r["gross_root_loss"] == 0)
    n_neg_gross = sum(1 for r in rows if r["gross_root_loss"] < 0)
    gross_sorted = sorted(pos_gross, key=lambda r: r["gross_root_loss"], reverse=True)
    top_g, rest_g = split_worst_decile(gross_sorted)
    gross_share = loss_share_pct(top_g, pos_gross, "gross_root_loss")

    # --- behavioural-residual ranking ---
    pos_behav = [r for r in rows if r["behavioural_residual"] > 0]
    n_zero_behav = sum(1 for r in rows if r["behavioural_residual"] == 0)
    n_neg_behav = sum(1 for r in rows if r["behavioural_residual"] < 0)
    behav_sorted = sorted(pos_behav, key=lambda r: r["behavioural_residual"], reverse=True)
    top_b, rest_b = split_worst_decile(behav_sorted)
    behav_share = loss_share_pct(top_b, pos_behav, "behavioural_residual")

    # --- overlap (episode-identity sets) ---
    keys_g = {episode_key(r) for r in top_g}
    keys_b = {episode_key(r) for r in top_b}
    inter = keys_g & keys_b
    union = keys_g | keys_b
    jaccard = (len(inter) / len(union)) if union else float("nan")
    pct_exit = (100.0 * (len(keys_g) - len(inter)) / len(keys_g)) if keys_g else float("nan")
    pct_new = (100.0 * (len(keys_b) - len(inter)) / len(keys_b)) if keys_b else float("nan")

    # --- Spearman, full usable population (both quantities always defined together) ---
    if n_usable >= 3:
        rho, pval = stats.spearmanr([r["gross_root_loss"] for r in rows],
                                     [r["behavioural_residual"] for r in rows])
    else:
        rho, pval = float("nan"), float("nan")

    # --- mechanical_root_loss diagnostic: gross-loss worst decile vs rest (descriptive only) ---
    # "rest" = remaining 90% of the POSITIVE-gross-loss population (rest_g), matching the
    # established convention in compute_rq3d_renormalize.py ("top 10% of that positive-loss
    # subset ... remaining 90% = rest") -- NOT the full remaining dataset including zero/negative
    # -loss episodes. Verified against that convention: this exact definition reproduces RQ3D's
    # own previously-reported pooled 4.16/2.58 figure (see run_output.log).
    mech_top = [r["mechanical_root_loss"] for r in top_g]
    mech_rest_pool = [r["mechanical_root_loss"] for r in rest_g]
    mech_diag = {
        "top_mean": float(np.mean(mech_top)) if mech_top else float("nan"),
        "top_median": float(np.median(mech_top)) if mech_top else float("nan"),
        "top_n": len(mech_top),
        "rest_mean": float(np.mean(mech_rest_pool)) if mech_rest_pool else float("nan"),
        "rest_median": float(np.median(mech_rest_pool)) if mech_rest_pool else float("nan"),
        "rest_n": len(mech_rest_pool),
    }

    return {
        "label": label, "n_usable": n_usable,
        "n_zero_gross": n_zero_gross, "n_neg_gross": n_neg_gross, "n_pos_gross": len(pos_gross),
        "n_zero_behav": n_zero_behav, "n_neg_behav": n_neg_behav, "n_pos_behav": len(pos_behav),
        "n_top_g": len(top_g), "n_top_b": len(top_b),
        "gross_share_pct": gross_share, "behav_share_pct": behav_share,
        "n_intersection": len(inter), "n_union": len(union), "jaccard": jaccard,
        "pct_gross_exits": pct_exit, "pct_behav_new": pct_new,
        "spearman_rho": rho, "spearman_p": pval,
        "mech_diag": mech_diag,
        "band_of_top_g": defaultdict(int, {b: sum(1 for r in top_g if r["band"] == b) for b in BANDS}),
        "band_of_top_b": defaultdict(int, {b: sum(1 for r in top_b if r["band"] == b) for b in BANDS}),
    }


def unpaired_bootstrap_diff(a, b, n=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan"), (float("nan"), float("nan"))
    idx_a = rng.integers(0, len(a), size=(n, len(a)))
    idx_b = rng.integers(0, len(b), size=(n, len(b)))
    diffs = a[idx_a].mean(axis=1) - b[idx_b].mean(axis=1)
    return float(np.mean(a) - np.mean(b)), (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def main():
    change, static = load_all()
    print(f"Loaded {len(change)} change-arm episodes, {len(static)} static-arm episodes, "
          f"from commit {COMMIT} (attenuation-pooling-scale), read-only via git show.\n")

    static_lookup = build_static_lookup(static)
    usable, excluded = [], []
    for r in change:
        key = (r["band"], r["seed"], r["scenario_id"])
        srac = static_lookup.get(key)
        if srac is None:
            excluded.append(r)
            continue
        r["static_root_owned_count"] = srac
        r["gross_root_loss"] = srac - r["final_root_owned_count"]
        r["mechanical_root_loss"] = r["root_owned_departures"]
        r["behavioural_residual"] = r["gross_root_loss"] - r["mechanical_root_loss"]
        usable.append(r)

    print(f"Usable change episodes (valid static pairing): {len(usable)} / {len(change)} "
          f"(excluded, no matching static-arm episode: {len(excluded)})")
    if excluded:
        for band in BANDS:
            n = sum(1 for r in excluded if r["band"] == band)
            if n:
                print(f"  band {band}: {n} excluded")
    print()

    print("=" * 100)
    print("PER-BAND RESULTS")
    print("=" * 100)
    band_results = {}
    for band in BANDS:
        res = band_analysis(usable, lambda r, b=band: r["band"] == b, band)
        band_results[band] = res
        print(f"\n--- band {band} ---")
        print(f"n_usable={res['n_usable']}")
        print(f"gross_root_loss: n_positive={res['n_pos_gross']} n_zero={res['n_zero_gross']} "
              f"n_negative={res['n_neg_gross']} -> worst decile n={res['n_top_g']}, "
              f"loss-share={res['gross_share_pct']:.1f}%  (historical: {ORIGINAL_LOSS_SHARE[band]}%, "
              f"diff={res['gross_share_pct'] - ORIGINAL_LOSS_SHARE[band]:+.1f}pp)")
        print(f"behavioural_residual: n_positive={res['n_pos_behav']} n_zero={res['n_zero_behav']} "
              f"n_negative={res['n_neg_behav']} -> worst decile n={res['n_top_b']}, "
              f"residual-share={res['behav_share_pct']:.1f}%")
        print(f"overlap: |gross|={res['n_top_g']} |behav|={res['n_top_b']} "
              f"|intersection|={res['n_intersection']} |union|={res['n_union']} "
              f"Jaccard={res['jaccard']:.3f}")
        print(f"  {res['pct_gross_exits']:.1f}% of gross-loss worst decile EXITS under behavioural ranking")
        print(f"  {res['pct_behav_new']:.1f}% of behavioural worst decile was NOT in gross-loss worst decile")
        print(f"Spearman(gross_root_loss, behavioural_residual): rho={res['spearman_rho']:.4f} "
              f"p={res['spearman_p']:.4g} (n={res['n_usable']})")
        md = res["mech_diag"]
        print(f"mechanical_root_loss diagnostic (descriptive only): "
              f"gross-worst-decile mean={md['top_mean']:.3f} median={md['top_median']:.3f} (n={md['top_n']}) | "
              f"rest mean={md['rest_mean']:.3f} median={md['rest_median']:.3f} (n={md['rest_n']})")

    print()
    print("=" * 100)
    print("POOLED (all 3 bands) -- reported for transparency, flagged per the known scale confound")
    print("=" * 100)
    res_p = band_analysis(usable, lambda r: True, "POOLED")
    print(f"n_usable={res_p['n_usable']}")
    print(f"gross_root_loss worst decile (n={res_p['n_top_g']}) band composition: "
          f"{dict(res_p['band_of_top_g'])}")
    print(f"behavioural_residual worst decile (n={res_p['n_top_b']}) band composition: "
          f"{dict(res_p['band_of_top_b'])}")
    pooled_appropriate = all(res_p['band_of_top_g'][b] > 0 for b in BANDS) and \
        all(res_p['band_of_top_b'][b] > 0 for b in BANDS)
    print(f"All 3 bands represented in BOTH pooled worst deciles? {pooled_appropriate}")
    if pooled_appropriate:
        print(f"gross-loss-share={res_p['gross_share_pct']:.1f}%  "
              f"behav-residual-share={res_p['behav_share_pct']:.1f}%  "
              f"Jaccard={res_p['jaccard']:.3f}  "
              f"exits={res_p['pct_gross_exits']:.1f}%  new={res_p['pct_behav_new']:.1f}%  "
              f"Spearman rho={res_p['spearman_rho']:.4f}")
        print("-> Pooled figures reported as a valid supplementary view.")
    else:
        print("-> POOLED RANKING IS NOT STATISTICALLY APPROPRIATE: at least one band is unrepresented "
              "(or near-zero) in a pooled worst-decile set, reproducing the exact scale confound "
              "evidence_taskRQ3D.md already found for the pooled raw-loss ranking. Per-band results "
              "above are the primary, trustworthy view; the pooled numbers below are shown ONLY to "
              "make the confound visible, not as a usable additional result.")
    md_p = res_p["mech_diag"]
    print(f"[shown regardless, for completeness] pooled gross-loss-share={res_p['gross_share_pct']:.1f}%  "
          f"behav-residual-share={res_p['behav_share_pct']:.1f}%  Jaccard={res_p['jaccard']:.3f}  "
          f"exits={res_p['pct_gross_exits']:.1f}%  new={res_p['pct_behav_new']:.1f}%  "
          f"Spearman rho={res_p['spearman_rho']:.4f} p={res_p['spearman_p']:.4g}  "
          f"mech diag top mean={md_p['top_mean']:.3f} median={md_p['top_median']:.3f} (n={md_p['top_n']}) | "
          f"rest mean={md_p['rest_mean']:.3f} median={md_p['rest_median']:.3f} (n={md_p['rest_n']})")

    print()
    print("=" * 100)
    print("STEP 7: reproduce/check the existing 2.19-vs-2.45 result (Addendum 7, "
          "compute_rq3d_behavioural_residual.py)")
    print("=" * 100)
    print("Exact original construction: POOLED across all 3 bands, rank usable positive-residual "
          "episodes by behavioural_residual descending, worst decile vs rest, compare mean "
          "root_owned_departures (== mechanical_root_loss) between the two groups.")
    top_b_pool = [r["mechanical_root_loss"] for r in
                  sorted([r for r in usable if r["behavioural_residual"] > 0],
                         key=lambda r: r["behavioural_residual"], reverse=True)]
    n_top_pool = max(1, math.ceil(0.10 * len(top_b_pool)))
    top_dep = top_b_pool[:n_top_pool]
    rest_dep = top_b_pool[n_top_pool:]
    diff, ci = unpaired_bootstrap_diff(top_dep, rest_dep)
    print(f"Reproduced: top-decile (n={len(top_dep)}) mean departures={np.mean(top_dep):.3f}  "
          f"rest (n={len(rest_dep)}) mean departures={np.mean(rest_dep):.3f}  "
          f"diff={diff:+.3f} 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    print("Originally reported: top=2.19, rest=2.45 (CI [-0.549,+0.041], NOT resolved).")

    print("\nDone.")


if __name__ == "__main__":
    main()
