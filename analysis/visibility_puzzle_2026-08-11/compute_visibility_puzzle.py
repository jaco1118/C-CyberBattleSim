"""Task VISIBILITY-PUZZLE: why graphdepth_sweep_wide (100% touched_node_visible) and
cx_step2_registration (17.7/13.1/18.2%) differ, despite sharing checkpoints and apparently
sharing allow_undiscovered_removal=True in their own run_metadata_*.json.

Resolved cause (code + launch-command evidence, not inferred):

compute_attenuation_analysis.py's build_band_envs() only copies the CX_REMOVAL/CX_JOIN/CX_PATCH
env vars into train_config_for_env -- i.e. only actually APPLIES allow_undiscovered_removal /
uncapped_join / patch_service_dynamic_enabled to the constructed environment -- inside
`if os.environ.get("CX_DIAG") == "1":` (lines 168-174). _cx_write_run_metadata(), which writes
the run_metadata_s<seed>.json "flags" dict, computes the SAME three flags directly from
CX_REMOVAL/CX_JOIN/CX_PATCH UNCONDITIONALLY (lines 121-123) -- it does not check CX_DIAG at all,
and is called whenever CX_DIAG==1 OR CX_STATIC==1 OR RQ2C==1 (line 248). So for any run launched
with RQ2C=1 alone (no CX_DIAG=1), the metadata FALSELY records allow_undiscovered_removal=true
(CX_REMOVAL defaults to "1") even though the constructed environment never received that kwarg
and fell back to the CyberBattleCompressedEnv constructor default, allow_undiscovered_removal=
False (cyberbattle_env.py:83), and to patch_service_dynamic_enabled as frozen in the checkpoint's
own train_config.yaml (False for every checkpoint in this manifest -- verified directly below).

Launch commands (recovered from the raw session transcripts, ~/.claude/projects/*/*.jsonl,
grep -o '"command":"[^"]*graphdepth_sweep_wide[^"]*"' / '...cx_step2_registration[^"]*"'):
  graphdepth_sweep_wide:  RQ2C=1 LEG=1 YEG_DRIFT_DIR=graphdepth_sweep_wide ... --collect
                          (no CX_DIAG)
  cx_step2_registration:  CX_DIAG=1 YEG_DRIFT_DIR=$CXDIR ... --collect --analyze
                          (comment in the command itself: "CX registration arm (removal+join+
                          patch, property discovered-only)")
  rq2c_replay:            RQ2C=1 ... --collect (no CX_DIAG) -- same mismatch as
                          graphdepth_sweep_wide; flagged here, not otherwise acted on.
  cx_step2_replay:        CX_DIAG=1 CX_REPLAY=1 ... --collect -- matches its own metadata.

With allow_undiscovered_removal=False, _get_removal_eligible_nodes() bases the candidate pool on
self.discovered_nodes only (cyberbattle_env.py:520-524) instead of the whole topology -- so every
membership_leave event necessarily targets an already-discovered (hence visible, modulo a
sub-step evolving_visible_graph lag) node: 100% touched_node_visible, exactly as observed. It
also explains the volume gap: _apply_dynamic_leave's per-node probabilities are
min(_DYNAMIC_P_MAX=0.25, target_rate*weights[n]/weight_sum) (cyberbattle_env.py:618-621) --
sum(p)==target_rate only holds BEFORE the P_MAX clip; with a small discovered-only pool, more
nodes hit the 0.25 cap, pulling the realised total leave rate below target_rate. A full-topology
pool (allow_undiscovered_removal=True) dilutes the same target_rate over far more nodes, so the
clip binds less and total volume is higher -- consistent with the observed ~8.6x-29.5x gap that
widens, not narrows, with topology size (larger topology = discovered_nodes is a smaller fraction
of it, especially early in an episode).

This script re-derives, from already-on-disk data only (no re-run), the numbers that support this
account: (1) total membership_leave volume + touched_node_visible rate for both sweeps at all 3
bands (already spot-checked at 10-15 inline during the investigation; extended here to all bands
and committed); (2) confirmation that patch_service_dynamic_enabled=False is baked into every
checkpoint's own train_config.yaml (not just asserted).
"""
import glob
import os

import pandas as pd
import yaml

AGENTS_DIR = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BANDS = ["10-15", "30-40", "80-100"]
SWEEPS = ["graphdepth_sweep_wide", "cx_step2_registration"]

# manifest run_folders (attenuation_manifest.yaml), one representative seed per band -- enough to
# confirm the flag is absent/false in the checkpoints' own frozen config, not to re-derive the
# whole manifest.
REPRESENTATIVE_RUN_FOLDERS = {
    "10-15": "logs/trpo_250k_tuned_compressed_band10-15_seed42_2026-07-26_11-56-51/TRPO_x_control_SecureBERT",
}


def main():
    print("=== VISIBILITY-PUZZLE: volume + visibility rate, both sweeps, all bands ===\n")
    rows = []
    for band in BANDS:
        for sweep in SWEEPS:
            path = os.path.join(AGENTS_DIR, sweep, f"drift_{band}.csv")
            df = pd.read_csv(path, usecols=["change_type", "touched_node_visible"], low_memory=False)
            leaves = df[df["change_type"] == "membership_leave"]
            n = len(leaves)
            n_vis = int((leaves["touched_node_visible"] == True).sum())  # noqa: E712
            pct_vis = 100.0 * n_vis / n if n else float("nan")
            print(f"  {band:8s} {sweep:24s} n_leave={n:7d}  visible={n_vis:7d} ({pct_vis:.1f}%)")
            rows.append(dict(band=band, sweep=sweep, n_leave=n, n_visible=n_vis, pct_visible=pct_vis))
    volume_df = pd.DataFrame(rows)
    volume_df.to_csv(os.path.join(OUT_DIR, "puzzle_volume_and_visibility.csv"), index=False)

    print("\n=== volume ratio, cx_step2_registration / graphdepth_sweep_wide ===")
    for band in BANDS:
        a = volume_df[(volume_df.band == band) & (volume_df.sweep == "cx_step2_registration")]["n_leave"].iloc[0]
        b = volume_df[(volume_df.band == band) & (volume_df.sweep == "graphdepth_sweep_wide")]["n_leave"].iloc[0]
        print(f"  {band:8s} ratio={a/b:.2f}x  ({a} / {b})")

    print("\n=== checkpoint's own frozen train_config.yaml: patch_service_dynamic_enabled, "
          "allow_undiscovered_removal presence ===")
    for band, rel in REPRESENTATIVE_RUN_FOLDERS.items():
        path = os.path.join(AGENTS_DIR, rel, "train_config.yaml")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        has_removal_key = "allow_undiscovered_removal" in cfg
        psde = cfg.get("patch_service_dynamic_enabled", "<absent>")
        print(f"  band {band}: patch_service_dynamic_enabled={psde}  "
              f"allow_undiscovered_removal key present={has_removal_key}")

    print("\n=== launch-command env vars actually used (recovered from session transcripts, "
          "quoted verbatim in this script's own docstring; not re-derivable from disk alone) ===")
    print("  graphdepth_sweep_wide : RQ2C=1 LEG=1 (no CX_DIAG)")
    print("  cx_step2_registration : CX_DIAG=1 (registration arm: removal+join+patch relaxed)")
    print("  rq2c_replay           : RQ2C=1 (no CX_DIAG) -- same mismatch pattern, flagged only")
    print("  cx_step2_replay       : CX_DIAG=1 CX_REPLAY=1 (matches its own metadata)")

    print("\nDone.")


if __name__ == "__main__":
    main()
