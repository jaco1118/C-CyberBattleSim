"""Task Y-REWARD-STAGE: earliest per-stage-local point at which the reward-based convergence rule
was already satisfied, for each of the three crossed-design cells (N=30, N=60, N=90).

METHOD (per-stage-local, confirmed by the user -- NOT cumulative-global):
For every seed, for every stage in that seed's own resume chain, call compute_convergence_check.py's
own delta_pct() on THAT stage's tfevents file ALONE (default stop = that file's own max logged step).
No step offsets across stages, no concatenation. This is the exact same call already used to produce
the reported final-stage numbers, just repeated at every earlier stage too. Metric = rollout/ep_rew_mean
(the project's current reward-based rule), window = 50000, threshold = 5.0, min-frac = 0.8 (need 4/5).

N=30 / N=60: seeds share a uniform stage count (1 and 7 respectively) -- align by stage NUMBER.
N=90: seeds have different stage counts and different per-stage step sizes -- align by each seed's own
CUMULATIVE step total (nominal sum of train_iterations across its stages, matching the STEP 0
stage-chain table) at which its most recent stage ended, using carry-forward (a seed's status at a
given cumulative marker = its most-recently-COMPLETED stage's own local result, even if that seed
stopped training before this marker and is being carried forward, not actively training at that point).

DEGENERATE CASES: pre-window mean <=0 or non-finite -> "undefined", counted separately, never floored.
Fewer than 100,000 steps of local history before stop -> skip, counted separately.
DURABLE CONVERGENCE: earliest point reported must stay within tolerance at every later tested point.
"""
import sys
import math
import numpy as np

sys.path.insert(0, "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents")
import compute_convergence_check as ccc  # noqa: E402

AG = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents"
METRIC = "rollout/ep_rew_mean"
WINDOW = 50000
THRESHOLD = 5.0
MIN_FRAC = 0.8
SEEDS = [42, 100, 123, 200, 300]

# (stage_label, folder, nominal train_iterations) per seed, in chain order
N30_CHAIN = {
    s: [("stg1", f"{AG}/logs/yN30_s{s}_stg1_2026-08-03_21-28-47/TRPO_x_control_SecureBERT", 250000)]
    for s in SEEDS
}

N60_STAGE_DATES = {
    1: "2026-08-03_21-28-47", 2: "2026-08-03_22-24-07", 3: "2026-08-03_23-00-46",
    4: "2026-08-03_23-36-32", 5: "2026-08-04_00-13-41", 6: "2026-08-05_22-55-50",
    7: "2026-08-05_23-47-20",
}
N60_CHAIN = {
    s: [(f"stg{k}", f"{AG}/logs/yN60_s{s}_stg{k}_{N60_STAGE_DATES[k]}/TRPO_x_control_SecureBERT", 250000)
        for k in range(1, 8)]
    for s in SEEDS
}

N90_CHAIN = {
    42: [("static500k", f"{AG}/logs/yprobe_n90_static500k_2026-08-03_02-20-13/TRPO_x_control_SecureBERT", 500000),
         ("resume750k", f"{AG}/logs/yprobe_n90_resume750k_2026-08-03_03-28-23/TRPO_x_control_SecureBERT", 250000),
         ("ext125M", f"{AG}/logs/yprobe_n90_s42_ext125M_2026-08-03_13-42-07/TRPO_x_control_SecureBERT", 500000)],
    100: [("static500k", f"{AG}/logs/yprobe_n90_s100_static500k_2026-08-03_08-43-27/TRPO_x_control_SecureBERT", 500000),
          ("resume750k", f"{AG}/logs/yprobe_n90_s100_resume750k_2026-08-03_10-08-38/TRPO_x_control_SecureBERT", 250000),
          ("ext125M", f"{AG}/logs/yprobe_n90_s100_ext125M_2026-08-03_13-42-07/TRPO_x_control_SecureBERT", 500000),
          ("ext150M(stg1)", f"{AG}/logs/yN90ext_s100_stg1_2026-08-05_22-55-50/TRPO_x_control_SecureBERT", 250000)],
    123: [("static500k", f"{AG}/logs/yprobe_n90_s123_static500k_2026-08-03_08-43-27/TRPO_x_control_SecureBERT", 500000)],
    200: [("static500k", f"{AG}/logs/yprobe_n90_s200_static500k_2026-08-03_08-43-27/TRPO_x_control_SecureBERT", 500000)],
    300: [("static500k", f"{AG}/logs/yprobe_n90_s300_static500k_2026-08-03_08-43-27/TRPO_x_control_SecureBERT", 500000),
          ("resume750k", f"{AG}/logs/yprobe_n90_s300_resume750k_2026-08-03_10-08-38/TRPO_x_control_SecureBERT", 250000),
          ("ext100M(stg1)", f"{AG}/logs/yN90ext_s300_stg1_2026-08-05_22-55-50/TRPO_x_control_SecureBERT", 250000)],
}


def stage_result(folder):
    """Per-stage-local delta_pct at that stage's own default stop (its own max logged step)."""
    steps, vals = ccc.series(folder, METRIC)
    stop = int(steps.max())
    if stop - 2 * WINDOW < steps.min() - 1e-9 and (steps < (stop - 2 * WINDOW)).sum() == 0:
        pass  # history check done properly below via npre
    r = ccc.delta_pct(folder, METRIC, WINDOW, stop)
    skip = r["npre"] == 0 or (stop - 2 * WINDOW) < 0
    undefined = (not skip) and (not np.isfinite(r["pre"]) or r["pre"] <= 0)
    within = (not skip) and (not undefined) and abs(r["delta_pct"]) < THRESHOLD
    return {"stop": r["stop"], "pre": r["pre"], "fin": r["fin"], "delta_pct": r["delta_pct"],
            "skip": skip, "undefined": undefined, "within": within}


def cell_verdict(per_seed_results):
    """per_seed_results: list of stage_result dicts (one per seed) at one alignment point."""
    usable = [r for r in per_seed_results if not r["skip"] and not r["undefined"]]
    n = len(per_seed_results)
    need = math.ceil(MIN_FRAC * n)
    n_within = sum(1 for r in per_seed_results if r["within"])
    mean_abs = float(np.mean([abs(r["delta_pct"]) for r in usable])) if usable else float("nan")
    converged = (len(usable) == n) and (mean_abs < THRESHOLD) and (n_within >= need)
    return {"mean_abs": mean_abs, "n_within": n_within, "need": need, "n": n, "converged": converged,
            "n_skip": sum(1 for r in per_seed_results if r["skip"]),
            "n_undefined": sum(1 for r in per_seed_results if r["undefined"])}


def run_n30_n60(name, chain):
    print(f"\n{'='*90}\n{name}\n{'='*90}")
    n_stages = len(next(iter(chain.values())))
    per_stage_seed_results = []  # [stage_idx] -> {seed: result}
    for k in range(n_stages):
        row = {}
        for s in SEEDS:
            label, folder, _ = chain[s][k]
            row[s] = stage_result(folder)
        per_stage_seed_results.append(row)

    verdicts = []
    for k in range(n_stages):
        v = cell_verdict(list(per_stage_seed_results[k].values()))
        verdicts.append(v)
        print(f"  stage {k+1}: mean|d%|={v['mean_abs']:.3f}%  within={v['n_within']}/{v['n']} "
              f"(need>={v['need']})  skip={v['n_skip']} undefined={v['n_undefined']}  "
              f"-> {'CONVERGED' if v['converged'] else 'not converged'}")
        for s in SEEDS:
            r = per_stage_seed_results[k][s]
            flag = "SKIP" if r["skip"] else ("UNDEF" if r["undefined"] else ("YES" if r["within"] else "no"))
            print(f"      seed {s}: stop={r['stop']:>7} pre={r['pre']:>8.3f} fin={r['fin']:>8.3f} "
                  f"d%={r['delta_pct']:>+7.2f}%  within={flag}")

    # earliest DURABLE convergence: converged at k and every later stage
    earliest = None
    for k in range(n_stages):
        if all(verdicts[j]["converged"] for j in range(k, n_stages)):
            earliest = k
            break
    return per_stage_seed_results, verdicts, earliest


def run_n90():
    name = "N=90"
    print(f"\n{'='*90}\n{name}\n{'='*90}")
    # per-seed list of (cumulative_end, stage_result)
    per_seed_points = {}
    for s in SEEDS:
        cum = 0
        pts = []
        for label, folder, ti in N90_CHAIN[s]:
            cum += ti
            pts.append((cum, label, stage_result(folder)))
        per_seed_points[s] = pts
        print(f"  seed {s} stage chain (nominal cumulative -> per-stage-local result):")
        for cum_end, label, r in pts:
            flag = "SKIP" if r["skip"] else ("UNDEF" if r["undefined"] else ("YES" if r["within"] else "no"))
            print(f"    cum={cum_end:>8} [{label:>14}] stop={r['stop']:>7} d%={r['delta_pct']:>+7.2f}% within={flag}")

    markers = sorted(set(cum for s in SEEDS for cum, _, _ in per_seed_points[s]))
    print(f"\n  alignment markers (union of all seeds' stage-end cumulative totals): {markers}")

    verdicts = []
    per_marker_seed = []
    for m in markers:
        row = {}
        for s in SEEDS:
            # most recently completed stage as of m: last point with cum_end <= m
            candidates = [(cum_end, r) for cum_end, label, r in per_seed_points[s] if cum_end <= m]
            cum_end, r = candidates[-1]  # latest
            row[s] = r
        per_marker_seed.append(row)
        v = cell_verdict(list(row.values()))
        verdicts.append(v)
        print(f"  cum={m:>8}: mean|d%|={v['mean_abs']:.3f}%  within={v['n_within']}/{v['n']} "
              f"(need>={v['need']})  skip={v['n_skip']} undefined={v['n_undefined']}  "
              f"-> {'CONVERGED' if v['converged'] else 'not converged'}")

    earliest = None
    for i, m in enumerate(markers):
        if all(verdicts[j]["converged"] for j in range(i, len(markers))):
            earliest = i
            break
    return markers, per_marker_seed, verdicts, earliest


if __name__ == "__main__":
    n30_stages, n30_verdicts, n30_earliest = run_n30_n60("N=30", N30_CHAIN)
    n60_stages, n60_verdicts, n60_earliest = run_n30_n60("N=60", N60_CHAIN)
    n90_markers, n90_stages, n90_verdicts, n90_earliest = run_n90()

    print(f"\n{'='*90}\nSUMMARY\n{'='*90}")
    print(f"N=30: earliest durable-converged stage = "
          f"{'stage ' + str(n30_earliest+1) if n30_earliest is not None else 'NEVER (within tested range)'} "
          f"of {len(n30_verdicts)} tested")
    print(f"N=60: earliest durable-converged stage = "
          f"{'stage ' + str(n60_earliest+1) if n60_earliest is not None else 'NEVER (within tested range)'} "
          f"of {len(n60_verdicts)} tested")
    print(f"N=90: earliest durable-converged cumulative marker = "
          f"{n90_markers[n90_earliest] if n90_earliest is not None else 'NEVER (within tested range)'} "
          f"of {n90_markers}")
