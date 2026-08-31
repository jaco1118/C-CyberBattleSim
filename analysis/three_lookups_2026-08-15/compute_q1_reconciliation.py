"""Task THREE-LOOKUPS, Question Set 1.6: does the row-writing mechanism (1.1-1.3) reconcile the
four Section IV.1 numbers (4,660 episodes; 1,355,136 steps; 1,602,999 rows; 290,799 attributable-
change rows)?

4,660 / 1,602,999 / 290,799 are all already on record (evidence_cards/evidence_taskT.md's "Stopping
target and resulting episode counts" and "Gate row counts" tables, read directly, no computation
needed). 1,355,136 (total environment steps) does NOT appear anywhere in this project's evidence
cards, dissertation log, or git history (searched: grep -rn "1355136\|1,355,136" . --include="*.txt"
--include="*.md" --include="*.log", excluding .git; git log --all -S"1355136" --oneline -- both
empty). This script computes it the one natural way available -- summing, per episode, the number
of distinct `step` values present in that episode's drift-CSV rows (every step gets at least one
row via _log_drift_rows, so distinct-step-count per episode IS the episode's step length) -- and
checks whether the mechanism read in source (1.1-1.3) explains the two numbers Section IV.1's own
text flags as needing explanation: the 247,863-row excess over one-per-step, and the 62.4
attributable-change rows per episode.

No experiment re-run. Reads only the already-on-disk attenuation_drift_logs/drift_<band>.csv
(the "live 5-seed grid", confirmed the Section IV.1 source dataset by its exact row-count match to
evidence_taskT.md's own tables in an earlier task this session).
"""
import os

import pandas as pd

AGENTS_DIR = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BANDS = ["10-15", "30-40", "80-100"]

COLS = ["seed", "scenario_id", "episode", "step", "change_type", "n_touched_nodes",
        "event_phase", "relevant"]


def main():
    print("=== Q1.6: reconciliation check ===\n")
    total_rows = 0
    total_episodes = 0
    total_steps = 0
    total_attributed_rows = 0
    total_retained_rows = 0  # event_phase in {immediate, attributed}, per load_and_filter's own gate
    per_band_episodes = {}
    per_band_rows = {}

    all_ep_keys = set()
    steps_per_episode_key = {}

    for band in BANDS:
        df = pd.read_csv(os.path.join(AGENTS_DIR, "attenuation_drift_logs", f"drift_{band}.csv"),
                          usecols=COLS, low_memory=False)
        n_rows = len(df)
        total_rows += n_rows
        per_band_rows[band] = n_rows

        ep_keys = df[["seed", "scenario_id", "episode"]].drop_duplicates()
        n_episodes = len(ep_keys)
        per_band_episodes[band] = n_episodes
        total_episodes += n_episodes

        # steps per episode = distinct `step` values within that (seed, scenario_id, episode) group
        steps_by_ep = df.groupby(["seed", "scenario_id", "episode"])["step"].nunique()
        total_steps += int(steps_by_ep.sum())

        n_attributed = int((df["event_phase"] == "attributed").sum())
        total_attributed_rows += n_attributed

        n_retained = int(df["event_phase"].isin(["immediate", "attributed"]).sum())
        total_retained_rows += n_retained

        print(f"--- band {band} ---")
        print(f"  rows={n_rows}  episodes={n_episodes}  steps={int(steps_by_ep.sum())}  "
              f"attributed_rows={n_attributed}  retained(immediate+attributed)={n_retained}")

    print(f"\n=== totals ===")
    print(f"  episodes={total_episodes} (thesis: 4,660)")
    print(f"  rows={total_rows} (thesis: 1,602,999)")
    print(f"  steps={total_steps} (thesis: 1,355,136)")
    print(f"  retained (event_phase in immediate/attributed)={total_retained_rows} (thesis: 290,799)")
    print(f"  attributed-only rows={total_attributed_rows}")

    excess = total_rows - total_steps
    print(f"\n  rows - steps = {total_rows} - {total_steps} = {excess}  (thesis: 247,863)")
    print(f"  attributed rows alone = {total_attributed_rows}  "
          f"(does attributed_rows == rows-steps excess? {total_attributed_rows == excess})")

    per_ep_attributable = total_retained_rows / total_episodes
    print(f"\n  retained/episodes = {total_retained_rows}/{total_episodes} = {per_ep_attributable:.4f}  "
          f"(thesis: 290,799/4,660 = 62.4)")

    print(f"\n=== per-band episode split (1.4) ===")
    for band in BANDS:
        print(f"  {band}: {per_band_episodes[band]}")
    print(f"  80-100 == 5 seeds x 400 episode budget? {per_band_episodes['80-100']} == 2000: "
          f"{per_band_episodes['80-100'] == 2000}")


if __name__ == "__main__":
    main()
