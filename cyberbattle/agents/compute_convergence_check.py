"""F4 convergence-rule check (reusable).

Applies the pre-registered F4 band rule (evidence_taskF4.md) to one or more training runs:

  metric  : `train/Root owned nodes` (the control-goal count; same metric as the headline)
  window  : 50,000 timesteps
  per-seed: Delta% = (mean over final window - mean over preceding window) / mean over preceding * 100
            a seed is WITHIN-BAND iff |Delta%| < threshold (default 5%)
  band    : CONVERGED iff mean|Delta%| < threshold AND at least ceil(0.8*N) of N seeds are within-band
            (i.e. >=4/5 for the standard 5-seed check)

The windows are anchored at the run's own final logged step (`--stop` to override), i.e.
[stop-2*window, stop-window) vs [stop-window, stop). For a resumed run whose tensorboard restarts
its local step counter at 0, pass the run's local steps (this is exactly how the 500k->750k and
750k->1.25M extensions are checked: local final-50k vs preceding-50k = the corresponding absolute
window). Same method used for the real 80-100 band and every N=90 cell check.

Usage:
  python compute_convergence_check.py --run seed42=<run_folder_or_event_file> \
                                      --run seed100=<...> [--window 50000] \
                                      [--metric "train/Root owned nodes"] [--threshold 5.0] \
                                      [--stop STEP] [--min-frac 0.8]
A <run> value may be a run folder (the .tfevents file is found recursively) or a direct event file.
"""
import argparse
import glob
import math
import os

import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def _event_file(path):
    if os.path.isfile(path) and ".tfevents." in os.path.basename(path):
        return path
    hits = glob.glob(os.path.join(path, "**", "*.tfevents.*"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"no .tfevents file under {path}")
    return sorted(hits)[-1]


def series(path, metric):
    ea = event_accumulator.EventAccumulator(_event_file(path), size_guidance={"scalars": 0})
    ea.Reload()
    if metric not in ea.Tags()["scalars"]:
        raise KeyError(f"metric {metric!r} not in {path}; available: {ea.Tags()['scalars']}")
    ev = ea.Scalars(metric)
    return np.array([e.step for e in ev]), np.array([e.value for e in ev])


def delta_pct(path, metric, window, stop=None):
    steps, vals = series(path, metric)
    stop = int(stop) if stop is not None else int(steps.max())

    def wmean(lo, hi):
        m = (steps >= lo) & (steps < hi)
        return (float(vals[m].mean()) if m.sum() else float("nan")), int(m.sum())

    pre, npre = wmean(stop - 2 * window, stop - window)
    fin, nfin = wmean(stop - window, stop)
    d = (fin - pre) / pre * 100.0 if pre else float("nan")
    return {"stop": stop, "pre": pre, "fin": fin, "npre": npre, "nfin": nfin, "delta_pct": d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="label=path (path = run folder or .tfevents file); repeatable")
    ap.add_argument("--window", type=int, default=50000)
    ap.add_argument("--metric", default="train/Root owned nodes")
    ap.add_argument("--threshold", type=float, default=5.0, help="within-band |Delta%%| threshold")
    ap.add_argument("--min-frac", type=float, default=0.8, help="fraction of seeds required within band")
    ap.add_argument("--stop", type=int, default=None, help="anchor step (default: each run's max step)")
    args = ap.parse_args()

    rows = []
    for spec in args.run:
        label, path = spec.split("=", 1)
        r = delta_pct(path, args.metric, args.window, args.stop)
        within = abs(r["delta_pct"]) < args.threshold
        rows.append((label, r, within))

    print(f"metric={args.metric!r}  window={args.window}  threshold={args.threshold}%")
    print(f"{'run':>10} {'stop':>10} {'pre':>9} {'fin':>9} {'Delta%':>9} {'within?':>8}")
    for label, r, within in rows:
        print(f"{label:>10} {r['stop']:>10} {r['pre']:>9.3f} {r['fin']:>9.3f} "
              f"{r['delta_pct']:>+8.2f}% {'YES' if within else 'no':>8}")

    n = len(rows)
    absd = [abs(r["delta_pct"]) for _, r, _ in rows]
    n_within = sum(w for _, _, w in rows)
    need = math.ceil(args.min_frac * n)
    mean_abs = float(np.mean(absd)) if absd else float("nan")
    converged = (mean_abs < args.threshold) and (n_within >= need)
    print(f"\nmean|Delta%| = {mean_abs:.2f}%  ({'<' if mean_abs < args.threshold else '>='}{args.threshold}%)")
    print(f"within-band  = {n_within}/{n}  (need >= {need})")
    print(f"BAND VERDICT (F4 rule): {'CONVERGED' if converged else 'NOT CONVERGED'}")


if __name__ == "__main__":
    main()
