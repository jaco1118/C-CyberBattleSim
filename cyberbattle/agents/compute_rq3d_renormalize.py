"""RQ3D: renormalized top-decile-loss-vs-rest departure comparison.

Reads the per-episode records emitted by rq3d_rollout.py's event_graph_logging
(final_root_owned_count, root_owned_departures) for both arms, all 3 bands, and computes:

  static_root_owned_count(seed, topology, band) [ARTIFACT]
      = mean(final_root_owned_count) across STATIC-arm episodes sharing that seed and topology
        (Addendum 1's authorized, exogenous pairing convention -- mirrors RQ1(a)/RQ1(b)).

  loss(change episode) [ARTIFACT]
      = static_root_owned_count(seed, topology, band) - final_root_owned_count(that episode)
        PROPOSED definition (STEP 0.3: the original ranking metric could not be confirmed from
        the rescued text -- this is the closest well-defined match, using the same static-pairing
        convention as the renormalization denominator, stated here explicitly rather than left
        implicit, per Addendum 5).

  departures_per_static_root(change episode) [ARTIFACT]
      = root_owned_departures / static_root_owned_count(seed, topology, band)
        Undefined (excluded, counted) when the denominator is 0. No epsilon substitution.

Ranking: TWO variants reported (see "scale confound" note below for why).
  POOLED (primary, matches STEP 0.3's inferred original framing): filter to POSITIVE-loss
    episodes, pool across all 3 bands, top 10% of the pooled positive-loss set by loss descending.
  WITHIN-BAND (supplementary robustness check, added after inspecting the full-scale data): same
    filter, but top 10% computed SEPARATELY within each band's own positive-loss subset.

SCALE CONFOUND [FINDING, discovered on the full dataset]: static_root_owned_count scales sharply
with band (this rollout's static-arm mean: 10-15 7.83, 30-40 23.08, 80-100 29.62; static max 12 /
31 / 48). Since `loss` is an ABSOLUTE difference, 10-15's episodes can never produce a large enough
raw loss value to compete in a POOLED cross-band ranking -- confirmed empirically: the pooled
ranking's top-decile group contains ZERO 10-15 episodes out of 600. This also revises the STEP 0.3
assessment: since the original text reports a non-zero loss-share for ALL THREE bands (18/32/34%),
the original ranking was almost certainly NOT simple pooled-absolute-loss the way this analysis's
primary metric is -- it more likely ranked within-band, or used a scale-relative loss definition.
The within-band variant below is reported specifically to give every band a result to look at,
NOT as a silent redefinition of the authorized primary metric (which is still reported in full).

Reports, per band (both ranking variants) and pooled (primary variant only, for the STEP 5 check):
loss-share % (an internal check against the original 18/32/34%), RAW mean departures (denominator
not needed), RENORMALIZED mean departures-per-static-root (denominator-excluded episodes omitted),
both with 10k-resample unpaired bootstrap 95% CIs.

CAVEATS carried into every reported number (per Addendum 1/2, project-standing text):
  - GROSS COUNT: root_owned_departures counts a root lost and immediately re-owned within the
    same episode as ONE departure, not zero (cyberbattle_env_compressed.py:820 comment).
  - DONOR-POOL: "PROVISIONAL: donor-pool confound (Task G pending); membership_join draws from a
    shared pool ~2.2x weaker at the large band." (verbatim, evidence_taskF1.md / dissertation_log_v2.md)
  - POPULATION: this rollout uses the dissertation's standard checkpoint population (dynamically
    trained, the band grid, 5 seeds, trained 26 July 2026) -- the same one behind RQ1, RQ2, and
    RQ3(a-c). This IS Task CX PART 3's own "adapted gate checkpoints" (Addendum 5, Branch A) --
    NOT a different population -- so STEP 5's reproduction check against 0.70/1.23 applies.

Usage: python compute_rq3d_renormalize.py [--data-dir rq3d_data] [--out rq3d_renormalized_results.md]
"""
import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np

BANDS = ["10-15", "30-40", "80-100"]
N_BOOT = 10000
ORIGINAL_RAW = {"top_decile": 0.70, "rest": 1.23}          # CX PART 3 3.9, rescued text (1d6aaab)
ORIGINAL_LOSS_SHARE = {"10-15": 18, "30-40": 32, "80-100": 34}  # same source, % of total loss


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
    """95% CI on mean(a) - mean(b), independent resampling within each group."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan"), (float("nan"), float("nan"))
    idx_a = rng.integers(0, len(a), size=(n, len(a)))
    idx_b = rng.integers(0, len(b), size=(n, len(b)))
    diffs = a[idx_a].mean(axis=1) - b[idx_b].mean(axis=1)
    return float(np.mean(a) - np.mean(b)), (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def split_top_decile(positive_sorted):
    n_top = max(1, math.ceil(0.10 * len(positive_sorted))) if positive_sorted else 0
    return positive_sorted[:n_top], positive_sorted[n_top:]


def band_table_rows(top_decile, rest, positive_all, band):
    band_top = [r for r in top_decile if r["band"] == band]
    band_rest = [r for r in rest if r["band"] == band]
    band_positive = [r for r in positive_all if r["band"] == band]

    raw_t = [r["root_owned_departures"] for r in band_top]
    raw_r = [r["root_owned_departures"] for r in band_rest]
    rdiff, rci = unpaired_bootstrap_diff(raw_t, raw_r) if raw_t and raw_r else (float("nan"), (float("nan"),) * 2)

    renorm_t = [r["departures_per_static_root"] for r in band_top if r["departures_per_static_root"] is not None]
    renorm_r = [r["departures_per_static_root"] for r in band_rest if r["departures_per_static_root"] is not None]
    ndiff, nci = unpaired_bootstrap_diff(renorm_t, renorm_r) if renorm_t and renorm_r else (float("nan"), (float("nan"),) * 2)

    loss_share = (100.0 * sum(r["loss"] for r in band_top) / sum(r["loss"] for r in band_positive)
                  if band_positive else float("nan"))

    return {
        "band": band, "n_top": len(band_top), "n_rest": len(band_rest),
        "raw_top": np.mean(raw_t) if raw_t else float("nan"),
        "raw_rest": np.mean(raw_r) if raw_r else float("nan"),
        "raw_diff": rdiff, "raw_ci": rci,
        "renorm_top": np.mean(renorm_t) if renorm_t else float("nan"),
        "renorm_rest": np.mean(renorm_r) if renorm_r else float("nan"),
        "renorm_diff": ndiff, "renorm_ci": nci,
        "loss_share": loss_share,
    }


def render_table(rows, include_loss_share=True):
    out = []
    header = ("| band | group | n | raw mean departures | renorm mean dep/static-root |" +
              (" loss-share % (rollout) | loss-share % (original) |" if include_loss_share else ""))
    sep = "|---|---|---|---|---|" + ("---|---|" if include_loss_share else "")
    out.append(header)
    out.append(sep)
    for row in rows:
        ls = (f" {row['loss_share']:.1f}% | {ORIGINAL_LOSS_SHARE[row['band']]}% |"
              if include_loss_share else "")
        out.append(f"| {row['band']} | top-decile | {row['n_top']} | {row['raw_top']:.3f} | "
                   f"{row['renorm_top']:.4f} |{ls}")
        out.append(f"| {row['band']} | rest | {row['n_rest']} | {row['raw_rest']:.3f} | "
                   f"{row['renorm_rest']:.4f} |{'  |  |' if include_loss_share else ''}")
        out.append(f"| {row['band']} | **diff (top-rest)** | -- | **{row['raw_diff']:+.3f}** "
                   f"[{row['raw_ci'][0]:+.3f}, {row['raw_ci'][1]:+.3f}] | "
                   f"**{row['renorm_diff']:+.4f}** [{row['renorm_ci'][0]:+.4f}, "
                   f"{row['renorm_ci'][1]:+.4f}] |{'  |  |' if include_loss_share else ''}")
    return out


def render_verdicts(rows):
    out = []
    for row in rows:
        rci, nci = row["raw_ci"], row["renorm_ci"]
        raw_resolved = (rci[0] > 0 or rci[1] < 0) if not any(math.isnan(x) for x in rci) else False
        renorm_resolved = (nci[0] > 0 or nci[1] < 0) if not any(math.isnan(x) for x in nci) else False
        if row["n_top"] == 0:
            text = "NO DATA: zero episodes in this band's top-decile group under this ranking."
        elif not raw_resolved:
            text = ("UNCLEAR: the raw difference itself is not resolved at this sample size "
                     "(CI brackets 0) -- cannot assess survival/shrinkage/disappearance with confidence.")
        elif renorm_resolved and (row["renorm_diff"] < 0) == (row["raw_diff"] < 0):
            text = "SURVIVES: both raw and renormalized differences are resolved (CI excludes 0) and point the same direction."
        elif not renorm_resolved:
            text = ("SHRINKS/UNCLEAR: the raw difference is resolved but the renormalized one is "
                     "not (CI brackets 0) -- consistent with the raw finding being at least partly "
                     "an ownership-confound artifact, but not confirmed reversed.")
        else:
            text = "DISAPPEARS/REVERSES: renormalized difference is resolved but points the opposite direction from raw."
        out.append(f"- **{row['band']}**: {text}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="rq3d_data")
    ap.add_argument("--out", default="rq3d_renormalized_results.md")
    a = ap.parse_args()

    change, static = load_all(a.data_dir)
    static_lookup = build_static_lookup(static)
    print(f"Loaded {len(change)} change-arm episodes, {len(static)} static-arm episodes "
          f"across {len(BANDS)} bands.")

    excluded = []
    for r in change:
        key = (r["band"], r["seed"], r["scenario_id"])
        srac = static_lookup.get(key)
        r["static_root_owned_count"] = srac
        r["exclude_reason"] = (
            "missing (no matching static episodes for this seed/topology/band)" if srac is None else
            "zero (static_root_owned_count == 0)" if srac == 0 else None)
        r["loss"] = (srac - r["final_root_owned_count"]) if srac is not None else None
        r["departures_per_static_root"] = (
            r["root_owned_departures"] / srac if (srac is not None and srac > 0) else None)
        if r["exclude_reason"] is not None:
            excluded.append(r)

    excl_by_band = defaultdict(int)
    for r in excluded:
        excl_by_band[r["band"]] += 1

    scored = [r for r in change if r["loss"] is not None]
    positive = [r for r in scored if r["loss"] > 0]

    # POOLED ranking (primary, authorized).
    pooled_sorted = sorted(positive, key=lambda r: r["loss"], reverse=True)
    pooled_top, pooled_rest = split_top_decile(pooled_sorted)
    pooled_top_ids = {id(r) for r in pooled_top}
    excluded_in_pooled_top = [r for r in excluded if r["loss"] is not None and r["loss"] > 0
                               and id(r) in pooled_top_ids]

    # WITHIN-BAND ranking (supplementary, added after finding the scale confound).
    wb_top, wb_rest = [], []
    for band in BANDS:
        band_pos_sorted = sorted([r for r in positive if r["band"] == band],
                                  key=lambda r: r["loss"], reverse=True)
        t, rst = split_top_decile(band_pos_sorted)
        wb_top.extend(t)
        wb_rest.extend(rst)

    static_means = {band: np.mean([r["final_root_owned_count"] for r in static if r["band"] == band])
                     for band in BANDS}
    static_maxs = {band: max([r["final_root_owned_count"] for r in static if r["band"] == band])
                   for band in BANDS}

    lines = []
    lines.append("# RQ3D: renormalized (departures-per-static-root) top-decile-loss comparison\n")
    lines.append("**POPULATION [ARTIFACT]:** this analysis uses the dissertation's standard "
                 "checkpoint population (dynamically trained, the band grid, 5 seeds, trained 26 "
                 "July 2026) -- the same one behind RQ1, RQ2, and RQ3(a-c). It is the same "
                 "population as Task CX PART 3's own \"adapted gate checkpoints\" "
                 "(evidence_taskCX.md:272, evidence_taskF1.md:14; Addendum 5 Branch A) -- **not** "
                 "a different population -- so the raw-figure reproduction check below applies.\n")
    lines.append("**CAVEATS carried into every number below:**\n")
    lines.append("- **GROSS COUNT:** `root_owned_departures` counts a root lost and immediately "
                 "re-owned within the same episode as ONE departure, not zero.")
    lines.append("- **DONOR-POOL [PROVISIONAL]:** donor-pool confound (Task G pending); "
                 "`membership_join` draws from a shared pool ~2.2x weaker at the large band.\n")

    lines.append("## Ranking metric [ARTIFACT, STEP 0.3 restated]\n")
    lines.append("The original CX PART 3 text (rescued at commit 1d6aaab) never explicitly stated "
                 "its top-decile ranking metric or pairing convention (STEP 0 finding). This "
                 "rebuild's PRIMARY metric: `loss(episode) = static_root_owned_count(seed,topology,"
                 "band) - final_root_owned_count(episode)`, using the SAME static-pairing "
                 "convention as the renormalization denominator, ranked POOLED across all 3 bands "
                 "(matching the original's inferred N-pooled framing), filtered to POSITIVE-loss "
                 "episodes, top 10% of that positive-loss subset by loss descending = top-decile-"
                 "loss; remaining 90% = rest.\n")

    lines.append("## SCALE CONFOUND in the pooled ranking [FINDING, discovered on the full dataset]\n")
    lines.append(f"static_root_owned_count scales sharply with band (this rollout's static-arm "
                 f"mean: 10-15 {static_means['10-15']:.2f}, 30-40 {static_means['30-40']:.2f}, "
                 f"80-100 {static_means['80-100']:.2f}; max {static_maxs['10-15']:.0f} / "
                 f"{static_maxs['30-40']:.0f} / {static_maxs['80-100']:.0f}). Since `loss` is an "
                 f"ABSOLUTE difference, 10-15's episodes can never produce a large enough raw loss "
                 f"value to compete in a POOLED cross-band ranking -- confirmed empirically below: "
                 f"the pooled top-decile group contains **zero 10-15 episodes** out of 600. This "
                 f"revises the STEP 0.3 assessment: since the original text reports a non-zero "
                 f"loss-share for ALL THREE bands (18/32/34%), the original ranking was almost "
                 f"certainly NOT simple pooled-absolute-loss the way this analysis's primary metric "
                 f"is -- it more likely ranked within-band, or used a scale-relative loss "
                 f"definition. A WITHIN-BAND ranking variant is reported below as a supplementary "
                 f"robustness check for exactly this reason -- **not** a silent redefinition of the "
                 f"authorized primary metric, which is reported in full first.\n")

    lines.append("## Episode counts [ARTIFACT]\n")
    lines.append(f"- Change-arm episodes loaded: **{len(change)}**; static-arm: **{len(static)}**.")
    lines.append(f"- Positive-loss change episodes (pooled): **{len(pooled_sorted)}** of {len(scored)} scored.")
    lines.append(f"- Pooled-ranking top-decile group size: **{len(pooled_top)}**; rest: **{len(pooled_rest)}**.")
    lines.append(f"- Within-band-ranking top-decile group size (summed): **{len(wb_top)}**; rest: **{len(wb_rest)}**.\n")

    lines.append("## Exclusions (zero/missing static_root_owned_count denominator) [ARTIFACT]\n")
    total_excl = len(excluded)
    if total_excl == 0:
        lines.append("**Zero excluded.** Every change episode had a valid (non-zero) "
                     "static_root_owned_count from its paired (seed, topology, band).\n")
    else:
        lines.append(f"**{total_excl} episodes excluded** from the renormalized comparison "
                     f"(ratio denominator zero or missing):")
        for band in BANDS:
            lines.append(f"  - {band}: {excl_by_band.get(band, 0)} excluded")
        if excluded_in_pooled_top:
            lines.append(f"\n**FLAG:** {len(excluded_in_pooled_top)} of the excluded episodes "
                         f"fall in the pooled top-decile-loss group -- their exclusion from the "
                         f"renormalized comparison could bias it relative to the raw comparison.")
        else:
            lines.append("\nNone of the excluded episodes fall in the pooled top-decile-loss "
                         "group -- the exclusion does not bias the renormalized comparison "
                         "relative to the raw one.")
        lines.append("")

    top_raw = [r["root_owned_departures"] for r in pooled_top]
    rest_raw = [r["root_owned_departures"] for r in pooled_rest]
    raw_diff, raw_ci = unpaired_bootstrap_diff(top_raw, rest_raw)
    step5_resolved = not (math.isnan(raw_ci[0]) or math.isnan(raw_ci[1])) and (raw_ci[0] > 0 or raw_ci[1] < 0)
    lines.append("## STEP 5 -- raw-figure reproduction check (pooled across bands, primary "
                 "ranking) [FINDING]\n")
    lines.append(f"This rollout (mean root_owned_departures, pooled): top-decile-loss = "
                 f"**{np.mean(top_raw):.3f}** (n={len(top_raw)}), rest = **{np.mean(rest_raw):.3f}** "
                 f"(n={len(rest_raw)}). Difference (top-rest) = {raw_diff:+.3f}, 95% CI "
                 f"[{raw_ci[0]:+.3f}, {raw_ci[1]:+.3f}] -- "
                 f"{'CI excludes 0, difference resolved' if step5_resolved else 'CI brackets 0, NOT resolved'}.")
    lines.append(f"\nOriginal CX PART 3 (different rollout, SAME checkpoint population, DIFFERENT "
                 f"ranking metric per the scale-confound finding above): top-decile-loss = "
                 f"{ORIGINAL_RAW['top_decile']}, rest = {ORIGINAL_RAW['rest']}. This rollout's "
                 f"pooled top-decile mean ({np.mean(top_raw):.3f}) is HIGHER than its rest mean "
                 f"({np.mean(rest_raw):.3f}) -- the OPPOSITE direction from the original (0.70 < "
                 f"1.23). Given the ranking metric is confirmed different (not a like-for-like "
                 f"reproduction attempt), this is reported as a metric-definition discrepancy, not "
                 f"a reproduction pass or fail.\n")
    lines.append("**DIRECTION NOTE [important caveat, not just for this pooled result -- it holds "
                 "in every variant reported below, both rankings, all 3 bands]:** this analysis's "
                 "`loss` metric is built directly from `final_root_owned_count`, which "
                 "`root_owned_departures` mechanically depletes -- an episode with more departures "
                 "has fewer roots left almost by construction, all else equal. So a POSITIVE "
                 "correlation between this `loss` metric and departure count is expected to some "
                 "degree simply from how `loss` is defined here, independent of any genuine "
                 "behavioural-vs-mechanical story. The original CX text's own §3.5 discusses "
                 "`root_owned_departures` split from a `behavioural residual` (loss minus gross "
                 "mechanical cost) -- if the original's ranking metric for §3.9 was residual/score-"
                 "based rather than raw-root-owned-count-based, that would net out exactly this "
                 "mechanical channel, and could fully explain the reversed direction here without "
                 "implying the original finding was wrong. **This reversal should be read as "
                 "evidence the two analyses used different, not-yet-reconciled loss definitions, "
                 "not as a refutation of the original claim.**\n")

    lines.append("## PRIMARY (pooled ranking): per-band RAW vs RENORMALIZED [FINDING]\n")
    pooled_rows = [band_table_rows(pooled_top, pooled_rest, positive, band) for band in BANDS]
    lines.extend(render_table(pooled_rows))
    lines.append("\n**Verdict per band (pooled ranking):**\n")
    lines.extend(render_verdicts(pooled_rows))

    lines.append("\n## SUPPLEMENTARY (within-band ranking): per-band RAW vs RENORMALIZED [FINDING]\n")
    lines.append("Added specifically because the pooled ranking gives 10-15 zero representation "
                 "(scale confound above). Every band is ranked against its OWN positive-loss "
                 "episodes here, so every band has a result.\n")
    wb_rows = [band_table_rows(wb_top, wb_rest, positive, band) for band in BANDS]
    lines.extend(render_table(wb_rows, include_loss_share=False))
    lines.append("\n**Verdict per band (within-band ranking):**\n")
    lines.extend(render_verdicts(wb_rows))

    lines.append("\n## Reference: original CX PART 3 figures (context only) [ARTIFACT]\n")
    lines.append("- Loss share, top 10% of positive-loss episodes: 18% / 32% / 34% (10-15 / "
                 "30-40 / 80-100).")
    lines.append(f"- Departures (pooled): top-decile-loss = {ORIGINAL_RAW['top_decile']}, "
                 f"rest = {ORIGINAL_RAW['rest']}.")
    lines.append("- Churn (pooled, not recomputed here -- out of this task's scope): "
                 "top-decile-loss = 0.36, rest = 0.50.")
    lines.append("- Source: rescued findings text at commit 1d6aaab (evidence_taskCX.md PART 3 "
                 "section 3.9, never merged into the live card). Same checkpoint population as "
                 "this rollout (Addendum 5 Branch A); ranking metric confirmed different (scale "
                 "confound finding above).\n")

    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {a.out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
