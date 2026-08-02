"""Task RQ2C-1 analysis: does the chosen action follow the VIEW, or only the action SET?

Pure post-hoc aggregation over the per-event records written by the RQ2C env diagnostic
(cyberbattle_env_compressed.py `_rq2c_probe`, gated on RQ2C=1) during a forced-action replay
driven by compute_attenuation_analysis.py --collect (CX_REPLAY=1). This script writes NO new
simulation data and re-runs nothing; it only reads the rq2c_*.jsonl logs and tallies.

Per single-node membership_leave event the diagnostic already computed, LIVE against the real
pre/post candidate sets, using candidate IDENTITY (source,target,vuln,type(outcome).__name__):
  GROUP (i)  = the policy's pre-change preferred action is NO LONGER in the post candidate set.
               (No choice-changed metric by construction -- the original choice cannot be reselected.)
  GROUP (ii) = it IS still in the post set -> changed = 1 if the actual post choice differs, else 0.
The group-(ii) divergence rate is the load-bearing RQ2(c) number: near-zero => behaviour follows
the action set (the view alone does not move the choice); materially >0 => the view itself moves it.

Every emitted number is labelled ARTIFACT (a raw count/logged quantity) or FINDING (an
interpreted result). CI convention matches this project's existing scripts: a 0.95 bootstrap with
the EPISODE as the resampling unit (events within an episode are autocorrelated), reported per band
AND per seed (the seed is this project's unit for across-condition claims). A Wilson event-level
interval is reported alongside as a naive reference only.

Usage:
  python compute_rq2c_action_divergence.py --input-dir <dir with rq2c_*.jsonl> --out <table.md>
"""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

BANNER = ("RQ2C: single-node membership-leave events only; batch events excluded; "
          "80-100 band agent is not confirmed converged (see Task F4); group (i) events "
          "have no choice-changed metric by construction.")

# Scope note (explicit, per the RQ2C-1 acceptance): what trajectory this is measured on.
SCOPE_NOTE = (
    "SCOPE: measured on FRESH episodes from a per-seed-seeded STOCHASTIC rollout of the trained "
    "policy -- the exploratory regime that actually produces membership_leave events (the "
    "DETERMINISTIC policy discovers ~nothing and fires 0 leaves, so it cannot measure RQ2(c)). "
    "Same checkpoints / seeds / bands as the headline attenuation sweep, and the same standard "
    "attenuation config (patch_service OFF, no CX_DIAG constraint relaxation). These are NOT the "
    "literal reported headline episodes: the stored attenuation_step3_logs actions are not "
    "faithfully replayable (that sweep lacked per-seed seeding; only a distributional <1pp "
    "reproduction was ever checked, never a trajectory-identical one -- see the replay-fidelity "
    "scope correction). Nor is this the stochastic-action-selection headline NUMBERS themselves. "
    "The COUNTERFACTUAL pre/post policy predicts are deterministic=True (noise-free before/after "
    "comparison -- that is where 'no stochastic sampling' matters). Reproducible: torch/np/random "
    "seeded per seed + set_num_threads(1) + PYTHONHASHSEED=0.")

# run_id is "<band>_seed<seed>", e.g. "10-15_seed42"
_RUNID_RE = re.compile(r"^(?P<band>.+)_seed(?P<seed>\d+)$")
BAND_ORDER = ["10-15", "30-40", "80-100"]


def load_records(input_dir):
    """Load every rq2c_*.jsonl under input_dir (recursively). Returns a list of dicts, each
    augmented with parsed band/seed from its run_id. No filtering here."""
    paths = sorted(glob.glob(os.path.join(input_dir, "**", "rq2c_*.jsonl"), recursive=True))
    records = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                m = _RUNID_RE.match(str(r.get("run_id", "")))
                if not m:
                    continue
                r["_band"] = m.group("band")
                r["_seed"] = int(m.group("seed"))
                records.append(r)
    return records, paths


def wilson_interval(k, n, z=1.959963984540054):
    """Naive event-level Wilson 0.95 interval for a proportion (reference only; ignores the
    episode clustering the bootstrap accounts for)."""
    if n == 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def episode_cluster_bootstrap(episode_changed, n_iter=10000, seed=0):
    """0.95 CI for the group-(ii) divergence rate, resampling the EPISODE (cluster) with
    replacement to respect within-episode autocorrelation. `episode_changed` maps an episode key
    to a list of 0/1 `changed` indicators for its group-(ii) events. Rate = pooled sum/count over
    the resampled episodes. Matches this project's episode-as-unit bootstrap convention."""
    keys = list(episode_changed.keys())
    if not keys:
        return float("nan"), float("nan")
    ks = np.array([sum(episode_changed[k]) for k in keys], dtype=float)      # changed per episode
    ns = np.array([len(episode_changed[k]) for k in keys], dtype=float)      # group-ii events per episode
    total_n = ns.sum()
    if total_n == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    n_ep = len(keys)
    rates = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.randint(0, n_ep, size=n_ep)
        denom = ns[idx].sum()
        rates[i] = (ks[idx].sum() / denom) if denom > 0 else np.nan
    rates = rates[~np.isnan(rates)]
    if rates.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


def summarize(records, clean_only=True):
    """Aggregate a flat record list into a stats dict. When clean_only, restrict the group/changed
    tally to steps whose single-node-leave count is exactly 1 (n_qualifying_events_this_step == 1),
    so each before/after choice pair maps to exactly one departing node (co-firing leaves in one
    step share one step-level choice and would double-count). Exclusion and total counts are
    reported over ALL records regardless, so nothing is silently dropped."""
    s = {
        "n_records_total": len(records),
        "n_clean_step_records": 0,
        "n_excluded_no_candidate_pre": 0,
        "n_excluded_no_candidate_post": 0,
        "n_excluded_no_obs_pre_available": 0,  # structurally 0 in live replay; reported explicitly
        "n_group_i": 0,
        "n_group_ii": 0,
        "n_changed_group_ii": 0,
        "emb_dist_group_ii": [],
        "episode_changed": defaultdict(list),  # (seed,episode) -> [changed 0/1] for group ii
    }
    for r in records:
        excl = r.get("excluded")
        if excl == "no_candidate_pre":
            s["n_excluded_no_candidate_pre"] += 1
        elif excl == "no_candidate_post":
            s["n_excluded_no_candidate_post"] += 1
        if clean_only and r.get("n_qualifying_events_this_step") != 1:
            continue
        s["n_clean_step_records"] += 1
        if excl is not None:
            continue
        grp = r.get("group")
        if grp == "i":
            s["n_group_i"] += 1
        elif grp == "ii":
            s["n_group_ii"] += 1
            ch = int(r.get("changed"))
            s["n_changed_group_ii"] += ch
            s["episode_changed"][(r["_seed"], r["episode"])].append(ch)
            if r.get("emb_dist") is not None:
                s["emb_dist_group_ii"].append(float(r["emb_dist"]))
    return s


def rate_block(s, boot_seed=0):
    """Derive the group-(ii) divergence rate + CIs + secondary emb_dist from a summary dict."""
    n_ii = s["n_group_ii"]
    n_ch = s["n_changed_group_ii"]
    rate = (n_ch / n_ii) if n_ii > 0 else float("nan")
    w_lo, w_hi = wilson_interval(n_ch, n_ii)
    b_lo, b_hi = episode_cluster_bootstrap(s["episode_changed"], seed=boot_seed)
    ed = np.array(s["emb_dist_group_ii"], dtype=float) if s["emb_dist_group_ii"] else np.array([])
    return {
        "n_group_ii": n_ii, "n_changed": n_ch, "rate": rate,
        "wilson_lo": w_lo, "wilson_hi": w_hi, "boot_lo": b_lo, "boot_hi": b_hi,
        "emb_n": ed.size,
        "emb_mean": float(ed.mean()) if ed.size else float("nan"),
        "emb_median": float(np.median(ed)) if ed.size else float("nan"),
        "emb_all_zero": bool(ed.size > 0 and np.all(ed == 0.0)),
    }


def fmt(x, nd=3):
    return "nan" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="dir containing rq2c_*.jsonl (searched recursively)")
    ap.add_argument("--out", required=True, help="output markdown table path")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    records, paths = load_records(args.input_dir)
    by_band = defaultdict(list)
    for r in records:
        by_band[r["_band"]].append(r)
    bands = [b for b in BAND_ORDER if b in by_band] + [b for b in sorted(by_band) if b not in BAND_ORDER]

    lines = []
    def emit(x=""):
        lines.append(x)

    emit(f"# Task RQ2C-1 -- action-divergence (does the choice follow the view or only the action set)")
    emit()
    emit(f"> {BANNER}")
    emit()
    emit(f"> {SCOPE_NOTE}")
    emit()
    emit(f"Source files ({len(paths)}) under `{args.input_dir}`; {len(records)} total single-node "
         f"membership_leave event records. Bootstrap: {args.n_boot} iters, episode-clustered, 0.95.")
    emit()

    # ---- Per-band table (clean single-leave steps) ----
    emit("## Per band (ARTIFACT counts; FINDING = group-(ii) divergence rate)")
    emit()
    emit("| band | total records (ARTIFACT) | clean-step records (ARTIFACT) | excl no_cand_pre (ARTIFACT) | excl no_cand_post (ARTIFACT) | excl no_obs_pre (ARTIFACT) | n_group_i (ARTIFACT) | n_group_ii (ARTIFACT) | n_changed (ARTIFACT) | rate group-ii (FINDING) | boot 0.95 CI | wilson 0.95 CI | emb_dist mean/median (ARTIFACT) |")
    emit("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    pooled_rates = {}
    band_summaries = {}
    for b in bands:
        s = summarize(by_band[b], clean_only=True)
        band_summaries[b] = s
        rb = rate_block(s, boot_seed=abs(hash(b)) % (2**31))
        pooled_rates[b] = rb["rate"]
        emit(f"| {b} | {s['n_records_total']} | {s['n_clean_step_records']} | "
             f"{s['n_excluded_no_candidate_pre']} | {s['n_excluded_no_candidate_post']} | "
             f"{s['n_excluded_no_obs_pre_available']} | {s['n_group_i']} | {rb['n_group_ii']} | "
             f"{rb['n_changed']} | **{fmt(rb['rate'])}** | [{fmt(rb['boot_lo'])}, {fmt(rb['boot_hi'])}] | "
             f"[{fmt(rb['wilson_lo'])}, {fmt(rb['wilson_hi'])}] | "
             f"{fmt(rb['emb_mean'])} / {fmt(rb['emb_median'])}"
             f"{'  **ALL-ZERO(!)**' if rb['emb_all_zero'] else ''} |")
    emit()

    # ---- Per seed within band ----
    emit("## Per seed within band (the across-condition significance unit)")
    emit()
    emit("| band | seed | clean-step records | n_group_i | n_group_ii | n_changed | rate group-ii (FINDING) | boot 0.95 CI |")
    emit("|---|---|---|---|---|---|---|---|")
    for b in bands:
        seeds = sorted({r["_seed"] for r in by_band[b]})
        for sd in seeds:
            recs = [r for r in by_band[b] if r["_seed"] == sd]
            s = summarize(recs, clean_only=True)
            rb = rate_block(s, boot_seed=(abs(hash((b, sd))) % (2**31)))
            emit(f"| {b} | {sd} | {s['n_clean_step_records']} | {s['n_group_i']} | {rb['n_group_ii']} | "
                 f"{rb['n_changed']} | {fmt(rb['rate'])} | [{fmt(rb['boot_lo'])}, {fmt(rb['boot_hi'])}] |")
    emit()

    # ---- Implausible-value investigation (mandatory before presenting a finding) ----
    emit("## Implausible-value check (mandatory, OUTPUT AND REPORTING)")
    emit()
    finite = {b: r for b, r in pooled_rates.items() if not (isinstance(r, float) and math.isnan(r))}
    all_zero = bool(finite) and all(r == 0.0 for r in finite.values())
    all_one = bool(finite) and all(r == 1.0 for r in finite.values())
    if all_zero:
        emit("- **ALL bands rate == 0.000.** Per the spec this must be investigated before reporting: "
             "it could mean the candidate-identity comparison always finds a 'match' (identity bug) "
             "rather than a genuine finding. Inspect: (a) chosen_pre vs chosen_post keys are ever "
             "distinct in the raw log; (b) group-i is non-empty (proves membership testing works). "
             "**Do NOT present 0.000 as the RQ2(c) answer until this is resolved.**")
    elif all_one:
        emit("- **ALL bands rate == 1.000.** Investigate before reporting: could mean obs_pre and "
             "obs_post are accidentally identical / a logging offset feeds the same array, or the "
             "snap is degenerate. Inspect emb_dist (should be >0 for genuinely moved views) and that "
             "chosen_pre != chosen_post is real. **Do NOT present 1.000 as the answer until resolved.**")
    else:
        emit(f"- Pooled per-band rates {{ {', '.join(f'{b}: {fmt(r)}' for b, r in pooled_rates.items())} }} "
             f"are neither all-0 nor all-1 -> not in the degenerate regime the spec flags. "
             f"Still cross-check: group-i non-empty (membership test lives) and some emb_dist>0.")
    # supporting diagnostics for the check
    for b in bands:
        s = band_summaries[b]
        ed = np.array(s["emb_dist_group_ii"], dtype=float) if s["emb_dist_group_ii"] else np.array([])
        n_pos = int((ed > 0).sum()) if ed.size else 0
        emit(f"  - {b}: n_group_i={s['n_group_i']}, n_group_ii={s['n_group_ii']}, "
             f"emb_dist>0 in {n_pos}/{ed.size} group-ii events.")
    emit()

    emit("## Interpretation key")
    emit("- rate group-ii near 0 => FINDING: behaviour follows the ACTION SET, not the view "
         "(the preferred action, when it survives, is reselected).")
    emit("- rate group-ii materially > 0 => FINDING: the VIEW itself changes the choice even when "
         "the preferred action is still available.")
    emit("- group (i) (preferred action removed) carries no changed metric by construction; its size "
         "is the share of leaves that act purely through candidate-set membership.")

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[rq2c] wrote {args.out} ({len(records)} records across {len(bands)} band(s))")
    for b in bands:
        print(f"[rq2c] {b}: pooled group-ii rate = {fmt(pooled_rates[b])}")


if __name__ == "__main__":
    main()
