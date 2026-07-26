# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    compute_attenuation_analysis.py
    Phase 0 gate run: runs the three existing Compressed scale checkpoints (10-15, 30-40,
    80-100 nodes) with drift_logging=True against attenuation_manifest.yaml, filters and
    aggregates the resulting per-step drift log into per-episode statistics, fits the
    attenuation-ratio-vs-n_discovered log-log slope and the SNR-vs-n_discovered trend, and
    prints/writes the RQ3 gate decision.

    PROVISIONAL: all three checkpoints share the confounded cross-band join_donor_pool_20_topologies
    donor pool (431 vs 192 novel donor/topology pairs across bands) -- every output of this script
    carries a banner to that effect. These numbers are a directional gate read only and must be
    regenerated on same-band donor pools before appearing in the thesis.

    Usage:
        python compute_attenuation_analysis.py --manifest attenuation_manifest.yaml [--collect] [--analyze]
        (default: both --collect and --analyze; pass one explicitly to skip the other, e.g. to
        re-run only the analysis on an already-collected CSV)
"""

import argparse
import copy
import os
import pickle
import sys
import textwrap
import warnings
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)

from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

from cyberbattle.utils.train_utils import algorithm_models  # noqa: E402
from cyberbattle._env.cyberbattle_env_switch import RandomSwitchEnv  # noqa: E402
from cyberbattle.gae.model import GAEEncoder  # noqa: E402
from cyberbattle.utils.math_utils import bootstrap_ci  # noqa: E402

PROVISIONAL_BANNER = (
    "PROVISIONAL -- all three checkpoints share the confounded cross-band "
    "join_donor_pool_20_topologies donor pool (431 vs 192 novel donor/topology pairs across "
    "bands). These numbers are a directional gate read only and must be regenerated on "
    "same-band donor pools before appearing in the thesis."
)

DRIFT_LOG_DIR = os.path.join(script_dir, "attenuation_drift_logs")
OUTPUT_DIR = os.path.join(script_dir, "attenuation_analysis_output")


# =====================================================================================
# STEP 1/2: manifest-driven env/model construction and data collection
# =====================================================================================

def load_band_model_and_encoder(run_folder):
    with open(os.path.join(run_folder, "train_config.yaml")) as f:
        train_config = yaml.safe_load(f)
    with open(train_config['graph_encoder_config_path']) as f:
        config_encoder = yaml.safe_load(f)
    with open(train_config['graph_encoder_spec_path']) as f:
        spec_encoder = yaml.safe_load(f)
    config_encoder.update(spec_encoder)
    graph_encoder = GAEEncoder(config_encoder['node_feature_vector_size'],
                                config_encoder['model_config']['layers'],
                                config_encoder['edge_feature_vector_size'])
    graph_encoder.load_state_dict(torch.load(train_config['graph_encoder_path']))
    graph_encoder.eval()
    train_config['node_embeddings_dimensions'] = config_encoder['model_config']['layers'][-1]['out_channels']

    checkpoints_dir = os.path.join(run_folder, "checkpoints", "1")
    checkpoint_files = sorted(
        (f for f in os.listdir(checkpoints_dir) if f.startswith("checkpoint_") and "vecnormalize" not in f),
        key=lambda x: int(x.split("checkpoint_")[1].split("_steps")[0])
    )
    checkpoint_path = os.path.join(checkpoints_dir, checkpoint_files[-1])  # last (final) checkpoint
    model = algorithm_models[train_config.get('algorithm', 'ppo')].load(checkpoint_path)
    vecnormalize_path = checkpoint_path.replace("checkpoint_", "checkpoint_vecnormalize_", 1).rsplit(".", 1)[0] + ".pkl"
    return train_config, graph_encoder, model, checkpoint_path, vecnormalize_path


def build_band_envs(train_config, graph_encoder, topology_source_folder, n_topologies, logger,
                     drift_log_path, band_label, seed):
    from cyberbattle._env.cyberbattle_env_compressed import CyberBattleCompressedEnv

    topo_folder = os.path.join(project_root, "cyberbattle", "data", "env_samples", topology_source_folder)
    with open(os.path.join(topo_folder, "split.yaml")) as f:
        split_info = yaml.safe_load(f)
    topology_ids = [str(x['id']) for x in split_info.get('training_set', [])][:n_topologies]

    # Donor pool: mirrors test_agent.py's load_test_envs exactly (same file, same exclusion rule)
    donor_pool_by_source = {}
    if train_config.get('dynamic_mode') in ('join', 'both'):
        donor_envs_path = train_config.get('dynamic_join_donor_envs_path')
        if donor_envs_path:
            donor_source_folder = os.path.join(project_root, "cyberbattle", "data", "env_samples", donor_envs_path)
            for sub in os.listdir(donor_source_folder):
                if os.path.isdir(os.path.join(donor_source_folder, sub)) and sub.isdigit():
                    donor_network_file = os.path.join(donor_source_folder, sub, f"network_{train_config['nlp_extractor']}.pkl")
                    with open(donor_network_file, 'rb') as f:
                        donor_network = pickle.load(f)
                    donor_pool_by_source[sub] = [
                        (sub, node_id, copy.deepcopy(node_data["data"]))
                        for node_id, node_data in donor_network.network.nodes(data=True)
                    ]

    envs = []
    train_config_for_env = {k: v for k, v in train_config.items() if k != 'verbose'}
    for idx, folder in enumerate(topology_ids):
        network_file = os.path.join(topo_folder, folder, f"network_{train_config['nlp_extractor']}.pkl")
        with open(network_file, 'rb') as f:
            network = pickle.load(f)
        env = CyberBattleCompressedEnv(
            initial_environment=network, logger=logger, verbose=0,
            drift_logging=True, drift_log_path=drift_log_path, drift_sample_rate=1,
            drift_run_id=f"{band_label}_seed{seed}", drift_seed=seed, drift_scenario_id=folder,
            **{k: v for k, v in train_config_for_env.items()
               if k not in ('drift_logging', 'drift_log_path', 'drift_sample_rate',
                             'drift_run_id', 'drift_seed', 'drift_scenario_id')}
        )
        env.set_graph_encoder(graph_encoder)
        if donor_pool_by_source:
            env.dynamic_join_donor_pool = [
                entry for other_id, pool in donor_pool_by_source.items() if other_id != folder for entry in pool
            ]
        envs.append(env)
    return envs, topology_ids


def collect_band_data(band_label, band_config, manifest, logger):
    """AMENDMENT (task T v2 + amendment 1): loops over band_config['run_folders'] (one per
    seed, 5 for this grid) rather than a single run_folder, writing all seeds' episodes into
    the SAME per-band drift CSV (reset once, before the loop, not per seed) so downstream
    analysis sees one band-level file with a populated 'seed' column per row -- exactly what
    compute_episode_aggregates/compute_response_rates now group on (in addition to
    scenario_id/episode) to avoid silently merging same-numbered episodes from different
    seeds' independent runs."""
    drift_log_path = os.path.join(DRIFT_LOG_DIR, f"drift_{band_label}.csv")
    os.makedirs(DRIFT_LOG_DIR, exist_ok=True)
    if os.path.exists(drift_log_path):
        os.remove(drift_log_path)

    run_folders = band_config['run_folders']
    combined_final_counts = Counter()
    combined_shortfall = {}
    combined_skip_info = {
        "n_episodes_completed": 0, "n_episodes_skipped": 0,
        "n_episodes_attempted": 0, "skip_reason_counts": Counter(),
    }
    per_seed_skip_info = {}
    per_seed_shortfall = {}
    n_seeds_run = 0

    for run_folder_rel in run_folders:
        run_folder = os.path.join(script_dir, run_folder_rel)
        train_config, graph_encoder, model, checkpoint_path, vecnormalize_path = load_band_model_and_encoder(run_folder)
        seed = train_config['seeds_runs'][0]
        final_counts, shortfall, n_episodes_completed, skip_info = _collect_one_seed(
            band_label, band_config, manifest, logger, train_config, graph_encoder, model,
            checkpoint_path, vecnormalize_path, drift_log_path, seed,
        )
        n_seeds_run += 1
        per_seed_skip_info[seed] = skip_info
        per_seed_shortfall[seed] = shortfall  # a seed-level shortfall must stay visible per seed,
        # not just diluted into a pooled total -- one struggling seed among five should not be
        # masked by the other four's surplus.
        for ct, c in final_counts.items():
            combined_final_counts[ct] += c
        for ct, s in shortfall.items():
            existing = combined_shortfall.setdefault(ct, {"count": 0, "target": s["target"], "shortfall": 0, "reason": s["reason"]})
            existing["count"] += s["count"]
        combined_skip_info["n_episodes_completed"] += skip_info["n_episodes_completed"]
        combined_skip_info["n_episodes_skipped"] += skip_info["n_episodes_skipped"]
        combined_skip_info["n_episodes_attempted"] += skip_info["n_episodes_attempted"]
        combined_skip_info["skip_reason_counts"].update(skip_info["skip_reason_counts"])

    combined_skip_info["skip_reason_counts"] = dict(combined_skip_info["skip_reason_counts"])
    combined_skip_info["per_seed"] = per_seed_skip_info
    # Recompute shortfall's "shortfall" field against n_seeds_run * target (the pooled total
    # expected if every seed alone had hit the per-seed stopping target) -- comparing the pooled
    # count against a single seed's target would trivially show "no shortfall" once 5 seeds are
    # summed, even if one seed individually fell far short.
    for ct, s in combined_shortfall.items():
        s["target"] = s["target"] * n_seeds_run
        s["shortfall"] = max(0, s["target"] - s["count"])
    combined_shortfall = {ct: s for ct, s in combined_shortfall.items() if s["shortfall"] > 0}
    combined_shortfall["_per_seed"] = per_seed_shortfall

    return drift_log_path, dict(combined_final_counts), combined_shortfall, combined_skip_info["n_episodes_completed"], combined_skip_info


def _collect_one_seed(band_label, band_config, manifest, logger, train_config, graph_encoder,
                       model, checkpoint_path, vecnormalize_path, drift_log_path, seed):
    envs, topology_ids = build_band_envs(
        train_config, graph_encoder, band_config['topology_source_folder'],
        manifest['n_topologies_per_band'], logger, drift_log_path, band_label, seed
    )
    print(f"[{band_label} seed={seed}] loaded checkpoint {os.path.basename(checkpoint_path)}, "
          f"{len(envs)} topologies from {band_config['topology_source_folder']}")

    switch_env = RandomSwitchEnv(
        list(range(len(envs))), manifest['switch_interval_episodes'], envs_list=envs,
        save_to_csv=False, verbose=0,
    )
    vec_env = DummyVecEnv([lambda: Monitor(switch_env)])
    if os.path.exists(vecnormalize_path):
        loaded_vecnormalize = VecNormalize.load(vecnormalize_path, vec_env)
        vec_env = loaded_vecnormalize
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        vec_env = VecNormalize(vec_env, norm_obs=train_config.get('norm_obs', True), norm_reward=False)
        vec_env.training = False
        warnings.warn(f"[{band_label}] no VecNormalize stats found at {vecnormalize_path}; "
                       f"evaluating with fresh normalization statistics")

    target = manifest['target_relevant_events_per_change_type']
    max_episodes = manifest['max_episodes_per_band']
    change_types_of_interest = ["property", "membership_leave", "membership_join"]
    # "property" fires only from the legacy patch_service_dynamic_enabled mechanism -- if the
    # checkpoint's own frozen train_config has it off (true for all three real scale checkpoints),
    # no amount of episodes will ever produce a property event. Track this separately from an
    # ordinary "ran out of episode budget" shortfall so the two are never conflated in reporting.
    structurally_impossible = set()
    if not train_config.get('patch_service_dynamic_enabled', False):
        structurally_impossible.add("property")
    stopping_change_types = [ct for ct in change_types_of_interest if ct not in structurally_impossible]
    if structurally_impossible:
        print(f"[{band_label}] change type(s) {structurally_impossible} are structurally impossible "
              f"under this checkpoint's frozen config (patch_service_dynamic_enabled=False) -- "
              f"excluded from the stopping condition, reported as a structural (not budget) shortfall")
    n_crashes_skipped = 0
    n_episodes = 0

    def relevant_counts():
        # Cheap incremental check: count relevant, filtered (touched_node_visible + phase) rows
        # per change_type directly from whichever underlying envs' in-memory logger, since we
        # write straight to disk (drift_log_path shared ACROSS ALL 5 SEEDS in this band now --
        # see collect_band_data amendment) -- read back the file itself, which is flushed per
        # DriftLogger.flush_every (200) rows. For a responsive stopping check we flush all envs
        # explicitly here (cheap, I/O only). MUST filter to this seed's own rows: since the file
        # now accumulates across seeds sequentially, an unfiltered count would already exceed
        # target from a PRIOR seed's rows the moment seed 2+ starts, stopping it almost
        # immediately after its first checked episode -- confirmed as the actual failure mode
        # before this filter was added.
        for env in envs:
            env._drift_logger.flush()
        if not os.path.exists(drift_log_path):
            return {ct: 0 for ct in change_types_of_interest}
        df = pd.read_csv(drift_log_path)
        df = df[df['seed'] == seed]
        counts = {}
        for ct in change_types_of_interest:
            mask = (
                (df['change_type'] == ct)
                & (df['touched_node_visible'] == True)  # noqa: E712
                & (df['event_phase'].isin(['immediate', 'attributed']))
                & (df['relevant'] == True)  # noqa: E712
            )
            counts[ct] = int(mask.sum())
        return counts

    # B1: skipped episodes are tracked separately from completed ones -- not just logged, but
    # surfaced through to the gate summary. Skipped episodes are systematically biased (they are
    # cut short by a crash mid-episode, which correlates with longer/larger-discovered-set
    # episodes if the crash's trigger correlates with time-in-episode at all), which would bias
    # the sample against exactly the large-n_discovered region the slope estimate depends on --
    # a nonzero skip count must be visible, not silently absorbed into "n_episodes".
    n_episodes_completed = 0
    n_episodes_skipped = 0
    skip_reason_counts = Counter()

    state = vec_env.reset()
    while (n_episodes_completed + n_episodes_skipped) < max_episodes:
        try:
            with torch.no_grad():
                action, _ = model.predict(state)
            state, reward, done, info = vec_env.step(action)
        except ValueError as e:
            # Pre-existing, unrelated bug in _synthesize_recon_vulnerability's embedding
            # fallback (see Step 0 triage) -- confirmed structurally absent given
            # patch_service_dynamic_enabled=False in all three checkpoints' own frozen config,
            # kept here only as defensive insurance for a long unattended batch run.
            if "zero-dimensional arrays cannot be concatenated" in str(e):
                n_crashes_skipped += 1
                n_episodes_skipped += 1
                skip_reason_counts["recon_synthesis_bug"] += 1
                state = vec_env.reset()
                continue
            raise
        if done[0]:
            n_episodes_completed += 1
            if n_episodes_completed % 10 == 0:
                counts = relevant_counts()
                print(f"[{band_label} seed={seed}] episode {n_episodes_completed}: relevant-event counts so far {counts}")
                if all(counts.get(ct, 0) >= target for ct in stopping_change_types):
                    print(f"[{band_label} seed={seed}] target of {target} relevant events per change type reached "
                          f"after {n_episodes_completed} completed episodes "
                          f"(excluding structurally impossible {structurally_impossible or 'none'})")
                    break

    for env in envs:
        env._drift_logger.close()

    final_counts = relevant_counts()
    shortfall = {
        ct: {
            "count": final_counts.get(ct, 0),
            "target": target,
            "shortfall": max(0, target - final_counts.get(ct, 0)),
            "reason": "structurally_impossible" if ct in structurally_impossible else "budget_exhausted",
        }
        for ct in change_types_of_interest if max(0, target - final_counts.get(ct, 0)) > 0
    }
    print(f"[{band_label} seed={seed}] data collection finished: {n_episodes_completed} completed episodes, "
          f"{n_episodes_skipped} SKIPPED episodes ({dict(skip_reason_counts)}), "
          f"final relevant-event counts: {final_counts}")
    if n_episodes_skipped:
        print(f"[{band_label} seed={seed}] *** {n_episodes_skipped} episode(s) skipped -- see B1 caveat: skipped "
              f"episodes may be systematically biased toward longer/larger-discovered-set episodes, "
              f"which would bias the sample against the large-n_discovered region the slope depends on ***")
    if shortfall:
        print(f"[{band_label} seed={seed}] SHORTFALL -- did not reach the {target}-event target for: {shortfall}")
    skip_info = {
        "n_episodes_completed": n_episodes_completed,
        "n_episodes_skipped": n_episodes_skipped,
        "n_episodes_attempted": n_episodes_completed + n_episodes_skipped,
        "skip_reason_counts": dict(skip_reason_counts),
    }
    return final_counts, shortfall, n_episodes_completed, skip_info


# =====================================================================================
# STEP 3: filtering
# =====================================================================================

def load_and_filter(drift_log_path, band_label):
    df = pd.read_csv(drift_log_path)
    n_total = len(df)

    # Stage 4 (reported first, since it's a precondition sanity check, not a filter that removes
    # rows from the attenuation analysis): encoder determinism check on no_change rows.
    determinism_rows = df[df['event_phase'] == 'no_change']
    max_determinism_drift = float(determinism_rows['change_drift_full'].max()) if len(determinism_rows) else float('nan')
    if len(determinism_rows) and max_determinism_drift > 1e-3:
        print(f"[{band_label}] *** ENCODER DETERMINISM CHECK FAILED *** "
              f"max change_drift_full on no_change rows = {max_determinism_drift} (expected ~0). "
              f"encode() may not be deterministic -- every drift number from this band is suspect.")
    else:
        print(f"[{band_label}] encoder determinism check OK: max no_change drift = {max_determinism_drift:.2e} "
              f"over {len(determinism_rows)} rows")

    # Stage 1: drop touched_node_visible=False (visibility artefacts), reported/counted separately.
    visibility_lag_rows = df[df['touched_node_visible'] == False]  # noqa: E712
    n_visibility_dropped = len(visibility_lag_rows)
    df = df[df['touched_node_visible'] != False]  # noqa: E712 (keeps True and NaN/no_change rows)

    # Stage 2: keep event_phase in {immediate, attributed} for attenuation (drop no_change and any
    # remaining 'fired' rows not yet resolved -- 'fired' rows are pending-only and have no h3/valid
    # attenuation_ratio computed against a discovery transition anyway).
    n_before_phase_filter = len(df)
    attenuation_df = df[df['event_phase'].isin(['immediate', 'attributed'])].copy()
    n_dropped_phase = n_before_phase_filter - len(attenuation_df)

    print(f"[{band_label}] filtering: {n_total} total rows -> "
          f"{n_visibility_dropped} dropped (touched_node_visible=False, visibility-lag artefacts) -> "
          f"{n_dropped_phase} further dropped (event_phase not in immediate/attributed) -> "
          f"{len(attenuation_df)} rows retained for attenuation analysis")

    # visibility-lag statistics (Step 6 also wants median visibility_lag_steps and unattributed
    # fraction per change type -- computed here once, reused later)
    visibility_stats = {}
    for ct in df['change_type'].dropna().unique():
        ct_rows = df[(df['change_type'] == ct)]
        attributed = ct_rows[ct_rows['event_phase'] == 'attributed']
        fired_never_attributed = ct_rows[(ct_rows['event_phase'] == 'fired') & (ct_rows['attributed'] == False)]  # noqa: E712
        fired_total_events = ct_rows[ct_rows['event_phase'].isin(['fired'])]['event_id'].nunique()
        visibility_stats[ct] = {
            'median_visibility_lag_steps': float(attributed['visibility_lag_steps'].median()) if len(attributed) else None,
            'n_attributed': len(attributed),
            'n_never_attributed': fired_never_attributed['event_id'].nunique(),
            'n_fired_total_distinct_events': fired_total_events,
        }

    return attenuation_df, visibility_stats, n_visibility_dropped, max_determinism_drift


# =====================================================================================
# STEP 4/5: the two metrics, per-episode aggregation, regression and statistics
# =====================================================================================

AGG_SLICES = ["mean", "max", "min"]


# A step where agent_drift_full is (numerically) exactly zero means the agent's own action that
# step was a true no-op on the visible graph -- no competing noise at all, not "noise so small
# division blows up." An epsilon-clip division (the first attempt here) turns these into
# astronomically large SNR values (observed: 1.86e9) that silently dominate the per-episode
# median once they make up the majority of rows (confirmed empirically: 73.6% of real
# membership_leave rows have agent_drift_full == 0) -- reporting that as "SNR" would be reporting
# a numerical artifact, not a signal-to-noise measurement. These rows are excluded from the SNR
# statistic (set to NaN, matching pandas' skipna aggregation) rather than approximated, and their
# prevalence is reported explicitly rather than absorbed silently.
ZERO_NOISE_FLOOR_THRESHOLD = 1e-9


def compute_episode_aggregates(attenuation_df):
    """Per (band, change_type, episode): median n_discovered, median attenuation_ratio (full +
    per-slice), median SNR, median agent_drift_full (noise floor), and absolute norms."""
    attenuation_df = attenuation_df.copy()
    attenuation_df['zero_noise_floor'] = attenuation_df['agent_drift_full'] < ZERO_NOISE_FLOOR_THRESHOLD
    attenuation_df['snr'] = np.where(
        attenuation_df['zero_noise_floor'],
        np.nan,
        attenuation_df['change_drift_full'] / attenuation_df['agent_drift_full']
    )
    # 'seed' added to group_cols (AMENDMENT, task T v2 + amendment 1): scenario_id/episode
    # numbering restarts independently within each seed's own run, so grouping without 'seed'
    # would silently merge same-numbered episodes from different seeds' checkpoints as if they
    # were one episode -- the unit of analysis is (seed, scenario_id, episode), not just the
    # latter two.
    group_cols = ['seed', 'scenario_id', 'episode', 'change_type']
    agg_dict = {
        'n_discovered': 'median',
        'n_scenario': 'median',
        'attenuation_ratio_full': 'median',
        'snr': 'median',
        'agent_drift_full': 'median',
        'change_drift_full': 'median',
        'norm_h1': 'median',
        'norm_h2': 'median',
        'norm_h3': 'median',
        'zero_noise_floor': 'mean',  # fraction of this episode's rows with a zero noise floor
    }
    for agg in AGG_SLICES:
        agg_dict[f'attenuation_ratio_{agg}'] = 'median'
        agg_dict[f'change_drift_{agg}'] = 'median'  # relative drift per slice, episode-level
        agg_dict[f'agent_drift_{agg}'] = 'median'
        # Per-slice absolute norms (STEP 2 columns), carried through so absolute drift per slice
        # can be derived downstream as rel_drift(s) * norm_h1_s -- not stored as its own column.
        for snapshot in ('h1', 'h2', 'h3'):
            col = f'norm_{snapshot}_{agg}'
            if col in attenuation_df.columns:
                agg_dict[col] = 'median'
    episode_df = attenuation_df.groupby(group_cols, as_index=False).agg(agg_dict)
    episode_df = episode_df.rename(columns={'zero_noise_floor': 'zero_noise_floor_fraction'})
    episode_df['n_rows'] = attenuation_df.groupby(group_cols).size().values
    return episode_df


def bootstrap_series_ci(series):
    series = series.dropna().values
    if len(series) == 0:
        return float('nan'), float('nan'), float('nan')
    if len(series) == 1:
        return float(series[0]), float(series[0]), float(series[0])
    return bootstrap_ci(series, confidence=0.95)


# Task A (2026-07-25): the per-slice attenuation_ratio_{mean,max,min} slopes (~9-10 for
# max/min) are an ARTIFACT of dividing by norm(delta_h_G_slice), which is exactly zero whenever
# a non-extreme node leaves and the elementwise max/min over the remaining running nodes is
# bit-identical before and after (confirmed empirically: change_drift_max/min are exactly 0.0 for
# 39.0%/43.2% of real membership_leave events; change_drift_mean/full are never exactly zero).
# The attenuation_ratio_* columns and the slope-fitting code above are left in place (offline
# re-analysis only, per Task A's explicit scope -- no env/logging changes), just relabeled
# ARTIFACT wherever reported. This response-rate metric replaces it as the per-slice FINDING:
# a rate whose denominator is a positive integer event count, never a near-zero vector norm.
RESPONSE_RATE_SLICES = ["mean", "max", "min", "full"]
N_DISCOVERED_BIN_EDGES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def compute_response_rates(raw_events_df, change_type="membership_leave", taus=(0.0, 1e-9)):
    """Per-slice response-rate metric: moved_s(event) = 1 if change_drift_s > tau else 0;
    response_rate_s(cell) = sum(moved_s) / count(events in cell) -- denominator is an event
    count, never norm(delta_h_G_slice) (that is the deprecated attenuation_ratio_* metric).
    Uses the persisted RELATIVE change_drift_{slice} columns directly (Decision 2: equivalent to
    the absolute-norm test whenever the before-norm is positive, which _rel_drift's own 1e-12
    floor guarantees at write time). Bootstrap resampling unit is the EPISODE, matching the rest
    of this script -- not the event, since events within an episode are autocorrelated.

    Returns (results_df, guard_report). results_df has one row per (slice, tau,
    n_discovered_bin): n_events, n_episodes, response_rate, ci_lo, ci_hi (response_rate=NaN and
    n_episodes=0 for an empty cell -- never a fabricated rate). guard_report records every
    exclusion and why, expected to be entirely zero per Step 0's empirical pre-check, verified
    here rather than assumed.
    """
    df = raw_events_df[raw_events_df['change_type'] == change_type].copy()
    guard_report = {"change_type": change_type, "input_rows": len(df)}

    non_immediate = df[df['event_phase'] != 'immediate']
    guard_report['excluded_non_immediate_phase'] = len(non_immediate)
    if len(non_immediate):
        guard_report['excluded_non_immediate_phase_breakdown'] = non_immediate['event_phase'].value_counts().to_dict()
    df = df[df['event_phase'] == 'immediate']
    guard_report['events_after_phase_filter'] = len(df)

    results_rows = []
    for slice_name in RESPONSE_RATE_SLICES:
        col = 'change_drift_full' if slice_name == 'full' else f'change_drift_{slice_name}'
        slice_df = df.copy()

        nan_mask = slice_df[col].isna()
        guard_report[f'{slice_name}_excluded_nan_or_missing'] = int(nan_mask.sum())
        slice_df = slice_df[~nan_mask]

        # "non-positive before-norm -> undefined ratio" (Decision 2 guard): _rel_drift's own
        # 1e-12 floor at write time means this should be structurally impossible to observe as
        # NaN/inf here -- checked directly (negative or non-finite), not assumed absent.
        invalid_mask = (slice_df[col] < 0) | (~np.isfinite(slice_df[col]))
        guard_report[f'{slice_name}_excluded_negative_or_nonfinite'] = int(invalid_mask.sum())
        slice_df = slice_df[~invalid_mask]

        guard_report[f'{slice_name}_valid_events'] = len(slice_df)
        guard_report[f'{slice_name}_exact_zero_events'] = int((slice_df[col] == 0.0).sum())
        # headline (all n_discovered pooled): fraction of valid events where this slice moved at
        # all (tau=0.0) -- the single number Step 4's "nearly every event" / "minority" claim
        # is checked against, before looking at the binned-by-N breakdown.
        guard_report[f'{slice_name}_overall_response_rate_tau0'] = (
            float((slice_df[col] > 0.0).mean()) if len(slice_df) else float('nan')
        )

        slice_df = slice_df.copy()
        slice_df['n_discovered_bin'] = pd.cut(slice_df['n_discovered'], bins=N_DISCOVERED_BIN_EDGES, right=False)

        for tau in taus:
            moved_col = f'_moved_tau_{tau}'
            slice_df[moved_col] = (slice_df[col] > tau).astype(float)
            # observed=True: a categorical groupby (n_discovered_bin comes from pd.cut) with
            # observed=False produces the FULL CROSS PRODUCT of every (scenario_id, episode) with
            # every possible bin category, including bins that episode has zero events in (as
            # phantom NaN rows) -- inflating n_episodes to a constant across every bin (caught
            # empirically: every bin reported n_episodes=1584, implausible against 830 real
            # episodes). observed=True restricts to combinations that actually occurred.
            # 'seed' added (AMENDMENT, task T v2 + amendment 1): without it, same-numbered
            # episodes from different seeds' independent runs would be merged as if they were
            # one episode -- see the identical note in compute_episode_aggregates.
            episode_bin_rates = (
                slice_df.groupby(['seed', 'scenario_id', 'episode', 'n_discovered_bin'], observed=True)[moved_col]
                .mean().reset_index()
            )
            for bin_val in slice_df['n_discovered_bin'].cat.categories:
                bin_events = slice_df[slice_df['n_discovered_bin'] == bin_val]
                bin_episode_rates = episode_bin_rates.loc[
                    episode_bin_rates['n_discovered_bin'] == bin_val, moved_col
                ]
                n_events_in_bin = len(bin_events)
                n_episodes_in_bin = len(bin_episode_rates)
                if n_episodes_in_bin == 0:
                    rate, lo, hi = float('nan'), float('nan'), float('nan')
                else:
                    rate, lo, hi = bootstrap_series_ci(bin_episode_rates)
                results_rows.append({
                    'change_type': change_type, 'slice': slice_name, 'tau': tau,
                    'n_discovered_bin': str(bin_val), 'n_events': n_events_in_bin,
                    'n_episodes': n_episodes_in_bin, 'response_rate': rate,
                    'ci_lo': lo, 'ci_hi': hi,
                })

    return pd.DataFrame(results_rows), guard_report


def fit_loglog_slope(x, y, n_bootstrap=2000):
    """Log-log slope of y vs x with a bootstrap CI (resampling rows with replacement)."""
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return float('nan'), float('nan'), float('nan'), len(x)
    log_x, log_y = np.log(x), np.log(y)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    rng = np.random.default_rng(42)
    slopes = []
    n = len(x)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            s, _ = np.polyfit(log_x[idx], log_y[idx], 1)
            slopes.append(s)
        except np.linalg.LinAlgError:
            continue
    lower, upper = np.percentile(slopes, [2.5, 97.5]) if slopes else (float('nan'), float('nan'))
    return float(slope), float(lower), float(upper), n


# =====================================================================================
# STEP 6/7: gate decision, plots, CSV, text summary
# =====================================================================================

def analyze_band(band_label, drift_log_path):
    attenuation_df, visibility_stats, n_visibility_dropped, max_determinism_drift = load_and_filter(drift_log_path, band_label)
    episode_df = compute_episode_aggregates(attenuation_df)
    episode_df['band'] = band_label
    attenuation_df = attenuation_df.copy()
    attenuation_df['band'] = band_label
    return episode_df, visibility_stats, n_visibility_dropped, max_determinism_drift, attenuation_df


def _safe_tight_layout(fig):
    """tight_layout() is purely cosmetic spacing -- with very thin/degenerate data (e.g. a
    single distinct x value on a log-scaled axis) matplotlib's tick locator can fail to compute
    a layout at all. Skip it rather than losing the whole figure over spacing."""
    try:
        fig.tight_layout()
    except ValueError as e:
        warnings.warn(f"tight_layout() failed on degenerate axis data, skipping (figure content is unaffected): {e}")


def _safe_savefig(fig, path):
    """The PNG plots are a nicety on top of the CSV/gate-summary text, which are the analytically
    load-bearing outputs. With enough thin/degenerate per-subplot data, matplotlib's log-scale
    tick locator can still raise at draw time even when every axis was individually guarded
    beforehand (a mixed-subplot figure's shared draw pass can hit this in a subplot-specific way
    that isn't always visible from the per-axis data alone) -- skip that one PNG rather than
    losing the whole run's CSV/summary over a plot rendering quirk."""
    try:
        fig.savefig(path, dpi=120)
        return True
    except ValueError as e:
        warnings.warn(f"Failed to render {path} due to a matplotlib log-scale/degenerate-data "
                       f"issue (CSV and gate summary are unaffected): {e}")
        return False


def _has_positive_finite(*arrays):
    """Whether at least one value across the given arrays is positive and finite -- guards
    ax.set_xscale/set_yscale('log'), which raises if the axis has no positive data to place
    ticks on (can happen for a thin-data change type, e.g. all-NaN SNR in a sparse smoke test)."""
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        if np.any(np.isfinite(arr) & (arr > 0)):
            return True
    return False


def run_gate_and_outputs(all_episode_df, all_visibility_stats, manifest_bands, shortfalls, skip_infos=None, all_raw_event_df=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_lines = []

    def emit(line=""):
        print(line)
        summary_lines.append(line)

    emit("=" * 100)
    emit(PROVISIONAL_BANNER)
    emit("=" * 100)
    emit("")

    # B1: skipped-episode accounting, surfaced explicitly rather than only logged during
    # collection. Skipped episodes are not a neutral absence -- they are systematically likely to
    # be longer/larger-discovered-set episodes (cut short mid-episode by a crash), which biases
    # the sample against exactly the large-n_discovered region the slope estimate depends on.
    skip_infos = skip_infos or {}
    total_attempted = total_completed = total_skipped = 0
    all_skip_reasons = Counter()
    any_skip_info_known = False
    for band, info in skip_infos.items():
        if info is None:
            continue
        any_skip_info_known = True
        total_attempted += info.get("n_episodes_attempted", 0)
        total_completed += info.get("n_episodes_completed", 0)
        total_skipped += info.get("n_episodes_skipped", 0)
        all_skip_reasons.update(info.get("skip_reason_counts", {}))
    emit("--- Episode accounting (B1) ---")
    if not any_skip_info_known:
        emit("Skip accounting unavailable for this data (collected before this amendment, or "
             "--analyze run without a matching --collect sidecar file). Treat any slope estimate "
             "below as unable to rule out skip-related bias until re-collected.")
    else:
        emit(f"Total episodes attempted: {total_attempted}; completed: {total_completed}; "
             f"skipped: {total_skipped} ({100 * total_skipped / max(1, total_attempted):.1f}%)")
        emit(f"Skip reason counts (all bands): {dict(all_skip_reasons)}")
        for band, info in skip_infos.items():
            if info is not None:
                emit(f"  band {band}: attempted={info.get('n_episodes_attempted')}, "
                     f"completed={info.get('n_episodes_completed')}, skipped={info.get('n_episodes_skipped')}, "
                     f"reasons={info.get('skip_reason_counts')}")
        if total_skipped > 0:
            emit(f"*** {total_skipped} episode(s) were skipped. Skipped episodes are systematically "
                 f"likely to be longer/larger-discovered-set episodes, biasing the sample AGAINST "
                 f"the large-n_discovered region the slope estimate depends on. Interpret the SLOPE "
                 f"and any n_discovered~100+ LEVEL read below with this caveat explicitly in mind. ***")
        else:
            emit("No episodes were skipped -- the slope/level estimates below are not subject to this caveat.")
    emit("")

    # B2: honest coverage statement. All four DynPen change types are named explicitly, whether
    # present or absent -- an absent type must read as "untested", never silently as "tested and
    # found unremarkable" (an empty change type would otherwise just never appear in any table).
    ALL_CHANGE_TYPES = ["property", "membership_join", "membership_leave", "connectivity"]
    change_types_present = sorted(all_episode_df['change_type'].dropna().unique().tolist())
    change_types_absent = [ct for ct in ALL_CHANGE_TYPES if ct not in change_types_present]
    emit("--- Change-type coverage (B2) ---")
    emit(f"PRESENT (analysed below): {change_types_present or '(none -- see note below)'}")
    emit(f"ABSENT (NOT analysed -- untested, not 'tested and unremarkable'): {change_types_absent}")
    if "property" in change_types_absent:
        emit("  'property': confirmed absent because all three checkpoints' frozen train_config.yaml "
             "have patch_service_dynamic_enabled=False (see Part A1 pre-flight confirmation) -- the "
             "mechanism cannot fire, not merely under-sampled. Property-change coverage is deferred "
             "to Phase 2, which requires it enabled (and is currently blocked by the "
             "_synthesize_recon_vulnerability crash -- see dissertation_log_v2.md Phase 2 blocker).")
    if "connectivity" in change_types_absent:
        emit("  'connectivity': no independent connectivity-change mechanism exists in this codebase "
             "yet (only the agent's own action revealing an edge, which is not an external change "
             "event -- see dissertation_log_v2.md). Connectivity coverage awaits that mechanism.")
    emit("")
    emit(f"*** THE GATE DECISION BELOW SPEAKS ONLY TO: {change_types_present}. It is NOT a "
         f"finding about property or connectivity, which remain untested. ***")
    emit("")

    if not change_types_present:
        emit("No change types were observed at all -- nothing to analyze. Skipping plots/gate.")
        summary_path = os.path.join(OUTPUT_DIR, "gate_summary.txt")
        with open(summary_path, "w") as f:
            f.write("\n".join(summary_lines))
        print(f"\nSummary (no data) written to {summary_path}")
        return

    if any(shortfalls.values()):
        emit("SHORTFALL WARNING: at least one band did not reach the target relevant-event count "
             "for at least one change type. Two distinct reasons are tracked and must not be "
             "conflated: 'structurally_impossible' (the mechanism cannot fire under this "
             "checkpoint's frozen config, no amount of episodes would help) vs 'budget_exhausted' "
             "(an ordinary under-sampling shortfall):")
        for band, s in shortfalls.items():
            if s:
                emit(f"  band {band}:")
                for ct, detail in s.items():
                    if ct == "_per_seed":
                        continue
                    emit(f"    {ct}: {detail['count']}/{detail['target']} (pooled across seeds, {detail['reason']})")
        emit("Analysis below proceeds on whatever data was collected; treat thin-data bands with "
             "extra caution, not silently as equivalent to a full sample.")
        emit("")

    # Per-seed shortfall detail: a pooled total can hide one struggling seed among five.
    emit("--- Per-seed relevant-event shortfall detail ---")
    for band, s in shortfalls.items():
        per_seed = s.get("_per_seed", {}) if isinstance(s, dict) else {}
        for seed, seed_shortfall in per_seed.items():
            if seed_shortfall:
                emit(f"  band {band} seed {seed}: {seed_shortfall}")
    emit("")

    # --- Metric 1: attenuation ratio vs n_discovered, log-log, per change type + per slice ---
    fig1, axes1 = plt.subplots(1, len(change_types_present), figsize=(6 * len(change_types_present), 5), squeeze=False)
    slope_results = {}
    for i, ct in enumerate(change_types_present):
        ax = axes1[0][i]
        ct_df = all_episode_df[all_episode_df['change_type'] == ct]
        all_x, all_y = [], []
        for agg, marker in zip(['full'] + AGG_SLICES, ['o', '^', 's', 'v']):
            col = 'attenuation_ratio_full' if agg == 'full' else f'attenuation_ratio_{agg}'
            x = ct_df['n_discovered'].values.astype(float)
            y = ct_df[col].values.astype(float)
            all_x.append(x)
            all_y.append(y)
            ax.scatter(x, y, marker=marker, alpha=0.5, label=f"{agg} slice" if agg != 'full' else "combined")
            slope, lo, hi, n = fit_loglog_slope(x, y)
            slope_results[(ct, agg)] = (slope, lo, hi, n)
        # y = x reference line (NOT a 1/N decay line -- under mean pooling, delta_h_G ~ delta_h_v/N,
        # so attenuation_ratio ~ N and the reference is slope +1, i.e. "signal dilution factor")
        valid_x = ct_df['n_discovered'].values.astype(float)
        valid_x = valid_x[valid_x > 0]
        if len(valid_x):
            xs = np.array([max(1, valid_x.min()), valid_x.max()])
            ax.plot(xs, xs, 'k--', alpha=0.6, label="y = x (signal dilution, not decay)")
        if _has_positive_finite(*all_x) and _has_positive_finite(*all_y):
            ax.set_xscale('log')
            ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, "insufficient positive data to log-scale this axis",
                    ha='center', va='center', transform=ax.transAxes, fontsize=9, color='red')
        ax.set_xlabel("n_discovered (log)")
        ax.set_ylabel("attenuation ratio ||delta_h_v|| / ||delta_h_G|| (log)")
        ax.set_title(f"Attenuation ratio (signal dilution) -- {ct}")
        ax.legend(fontsize=8)
    fig1.suptitle("PROVISIONAL -- confounded cross-band donor pool, directional gate read only")
    _safe_tight_layout(fig1)
    fig1_path = os.path.join(OUTPUT_DIR, "attenuation_ratio_vs_n_discovered.png")
    fig1_ok = _safe_savefig(fig1, fig1_path)
    plt.close(fig1)

    # --- Metric 2: SNR vs n_discovered, per change type, SNR=1 reference ---
    fig2, axes2 = plt.subplots(1, len(change_types_present), figsize=(6 * len(change_types_present), 5), squeeze=False)
    snr_slope_results = {}
    for i, ct in enumerate(change_types_present):
        ax = axes2[0][i]
        ct_df = all_episode_df[all_episode_df['change_type'] == ct]
        x = ct_df['n_discovered'].values.astype(float)
        y = ct_df['snr'].values.astype(float)
        if not np.isfinite(y).any():
            # SNR is structurally undefined for this change type (e.g. membership_join: every
            # retained row is an "attributed" phase with no h3/change_drift_full pair by
            # construction -- see the gate summary's LEVEL/SLOPE explanation). An all-NaN y-axis
            # combined with a log-scaled x-axis in the same multi-subplot figure trips
            # matplotlib's tick locator at draw time, so skip the scatter/reference-line and
            # annotate plainly instead of leaving a permanently-failing plot.
            ax.text(0.5, 0.5, f"SNR structurally undefined for {ct}\n(see gate summary text)",
                    ha='center', va='center', transform=ax.transAxes, fontsize=9, color='red')
            slope, lo, hi, n = float('nan'), float('nan'), float('nan'), 0
        else:
            ax.scatter(x, y, alpha=0.5)
            slope, lo, hi, n = fit_loglog_slope(x, y)
            ax.axhline(1.0, color='r', linestyle='--', alpha=0.7, label="SNR = 1")
            if _has_positive_finite(x):
                ax.set_xscale('log')
            else:
                ax.text(0.5, 0.5, "insufficient positive data to log-scale this axis",
                        ha='center', va='center', transform=ax.transAxes, fontsize=9, color='red')
            ax.legend(fontsize=8)
        snr_slope_results[ct] = (slope, lo, hi, n)
        ax.set_xlabel("n_discovered (log)")
        ax.set_ylabel("SNR = change_drift_full / agent_drift_full")
        ax.set_title(f"SNR (practical detectability) -- {ct}")
    fig2.suptitle("PROVISIONAL -- confounded cross-band donor pool, directional gate read only")
    _safe_tight_layout(fig2)
    fig2_path = os.path.join(OUTPUT_DIR, "snr_vs_n_discovered.png")
    fig2_ok = _safe_savefig(fig2, fig2_path)
    plt.close(fig2)

    # --- agent_drift (noise floor) vs n_discovered -- does the floor itself dilute with N? ---
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    fig3_x_values = []
    for ct in change_types_present:
        ct_df = all_episode_df[all_episode_df['change_type'] == ct]
        fig3_x_values.append(ct_df['n_discovered'].values.astype(float))
        ax3.scatter(ct_df['n_discovered'], ct_df['agent_drift_full'], alpha=0.5, label=ct)
    if _has_positive_finite(*fig3_x_values):
        ax3.set_xscale('log')
    else:
        ax3.text(0.5, 0.5, "insufficient positive data to log-scale this axis",
                 ha='center', va='center', transform=ax3.transAxes, fontsize=9, color='red')
    ax3.set_xlabel("n_discovered (log)")
    ax3.set_ylabel("agent_drift_full (noise floor)")
    ax3.set_title("Noise floor vs n_discovered (PROVISIONAL)")
    ax3.legend(fontsize=8)
    _safe_tight_layout(fig3)
    fig3_path = os.path.join(OUTPUT_DIR, "agent_drift_noise_floor_vs_n_discovered.png")
    fig3_ok = _safe_savefig(fig3, fig3_path)
    plt.close(fig3)

    # --- tidy per-episode CSV backing every figure ---
    csv_path = os.path.join(OUTPUT_DIR, "attenuation_episode_aggregates.csv")
    all_episode_df.to_csv(csv_path, index=False)

    # --- absolute-drift version of the same analysis (Step 5: does the trend survive both
    # normalizations, since ||h_G|| is itself likely N-dependent -- max slice grows, min shrinks) ---
    emit("")
    emit("--- Absolute-norm check (does the trend survive both normalizations?) ---")
    abs_slope_results = {}
    for ct in change_types_present:
        ct_df = all_episode_df[all_episode_df['change_type'] == ct]
        x = ct_df['n_discovered'].values.astype(float)
        for norm_col in ['norm_h1', 'norm_h2', 'norm_h3']:
            y = ct_df[norm_col].values.astype(float)
            slope, lo, hi, n = fit_loglog_slope(x, y)
            abs_slope_results[(ct, norm_col)] = (slope, lo, hi, n)
            emit(f"  {ct}: log-log slope of {norm_col} vs n_discovered = {slope:.3f} "
                 f"[{lo:.3f}, {hi:.3f}] (n={n})")

    # --- STEP 6: THE GATE ---
    emit("")
    emit("#" * 100)
    emit("# GATE DECISION")
    emit("#" * 100)

    response_rate_df = None  # Task A: populated below for membership_leave, written to its own CSV
    for ct in change_types_present:
        ct_df = all_episode_df[all_episode_df['change_type'] == ct]
        n_episodes_ct = len(ct_df)
        emit("")
        emit(f"--- change_type = {ct} (n episodes = {n_episodes_ct}) ---")

        # Zero-noise-floor prevalence: steps where the agent's own action was a true no-op this
        # step (agent_drift_full == 0) are excluded from SNR (see ZERO_NOISE_FLOOR_THRESHOLD) --
        # report how much of the data that affects rather than let it disappear silently. These
        # are, if anything, evidence AGAINST the attenuation story on their own: a change event
        # co-occurring with zero agent-driven noise is trivially detectable, not merely low-SNR.
        mean_zero_frac = ct_df['zero_noise_floor_fraction'].mean() if 'zero_noise_floor_fraction' in ct_df else float('nan')
        emit(f"  DATA QUALITY: mean fraction of steps per episode with a zero agent-driven noise "
             f"floor (excluded from SNR, see ZERO_NOISE_FLOOR_THRESHOLD) = {mean_zero_frac:.1%}. "
             f"On these steps the change event faced literally no competing noise -- trivially "
             f"detectable, which if anything argues against attenuation, not for thin data.")

        n_valid_snr = ct_df['snr'].notna().sum()
        if n_valid_snr == 0:
            emit(f"  LEVEL/SLOPE: SNR is STRUCTURALLY undefined for {ct} in this dataset (0/{n_episodes_ct} "
                 f"episode-rows have a valid change_drift_full/agent_drift_full pair) -- this is not a "
                 f"data-volume shortfall. For membership_join specifically: every retained row comes from "
                 f"the 'attributed' event_phase (a join's own 'fired' rows are always excluded per Step 3, "
                 f"since a join never fires on an already-visible node), and attributed rows measure the "
                 f"h1->h2 discovery-transition of a LATER step, which has no h3/change_drift_full pair by "
                 f"construction. SNR as defined (change_drift_full/agent_drift_full, the h2->h3 pair) does "
                 f"not apply to join; only the attenuation_ratio MECHANISM read below is available for it.")
            mean_snr, lo_snr, hi_snr = float('nan'), float('nan'), float('nan')
            snr_slope, snr_lo, snr_hi, n_slope = float('nan'), float('nan'), float('nan'), 0
        else:
            # (a) LEVEL: SNR at approximately 100 discovered nodes
            near_100 = ct_df[(ct_df['n_discovered'] >= 80) & (ct_df['n_discovered'] <= 120)]
            if near_100['snr'].notna().sum() >= 3:
                mean_snr, lo_snr, hi_snr = bootstrap_series_ci(near_100['snr'])
                emit(f"  LEVEL: SNR at ~100 discovered nodes (n={near_100['snr'].notna().sum()} valid "
                     f"episode-rows in [80,120]): {mean_snr:.3f} [{lo_snr:.3f}, {hi_snr:.3f}]")
            else:
                mean_snr, lo_snr, hi_snr = float('nan'), float('nan'), float('nan')
                emit(f"  LEVEL: insufficient valid data near n_discovered~100 (only "
                     f"{near_100['snr'].notna().sum()} valid episode-rows in [80,120]) -- cannot report "
                     f"a reliable LEVEL read for this change type")

            # (b) SLOPE: is SNR declining in n_discovered?
            x = ct_df['n_discovered'].values.astype(float)
            y = ct_df['snr'].values.astype(float)
            snr_slope, snr_lo, snr_hi, n_slope = fit_loglog_slope(x, y)
            emit(f"  SLOPE: log-log SNR-vs-n_discovered slope = {snr_slope:.3f} [{snr_lo:.3f}, {snr_hi:.3f}] (n={n_slope})")

        # Interpretation
        if np.isnan(mean_snr):
            emit("  INTERPRETATION: cannot be determined -- insufficient/no valid SNR data at the reference scale.")
        elif mean_snr > 1 and (np.isnan(snr_slope) or snr_slope >= -0.05):
            emit("  INTERPRETATION: SNR > 1 and flat/rising -> attenuation story is WEAK for this "
                 "change type. RQ3 should pivot to the frozen-GAE-out-of-distribution explanation.")
        elif mean_snr > 1 and snr_slope < -0.05:
            emit("  INTERPRETATION: SNR > 1 but clearly declining -> hypothesis ALIVE as a prediction; "
                 "test at 150 and 250 nodes.")
        else:
            emit("  INTERPRETATION: SNR near or below 1 -> hypothesis SUPPORTED at current scales.")

        # Attenuation-ratio slope for this change type (mechanism test).
        # Task A (2026-07-25): the max/min slice slopes are labeled ARTIFACT, not a finding --
        # they divide by norm(delta_h_G_slice), which is exactly zero whenever a non-extreme node
        # leaves (confirmed: change_drift_max/min are exactly 0.0 for 39.0%/43.2% of real
        # membership_leave events), manufacturing a slope via an epsilon-floored near-zero
        # denominator rather than measuring a real signal. full/mean never hit this (mean shifts
        # by ~1/N on every event; full contains mean), so their slopes remain a FINDING.
        for agg in ['full'] + AGG_SLICES:
            slope, lo, hi, n = slope_results.get((ct, agg), (float('nan'),) * 3 + (0,))
            label = "ARTIFACT" if agg in ("max", "min") else "FINDING"
            emit(f"  MECHANISM [{label}]: attenuation_ratio ({agg} slice) log-log slope vs n_discovered = "
                 f"{slope:.3f} [{lo:.3f}, {hi:.3f}] (n={n}); reference (pure mean-pool dilution) = +1.0"
                 + ("  <- divides by norm(delta_h_G_slice), exactly 0 for 39-43% of events; "
                    "superseded by the response-rate metric below" if agg in ("max", "min") else ""))

        # Task A: response-rate metric supersedes the max/min attenuation_ratio slopes above for
        # membership_leave -- denominator is an event count, never a near-zero vector norm.
        if ct == "membership_leave" and all_raw_event_df is not None and len(all_raw_event_df):
            emit("")
            emit("  --- Response-rate metric (Task A, FINDING, supersedes the ARTIFACT slopes above) ---")
            emit("  PROVISIONAL: gate data uses the shared confounded donor pool (novelty confound, "
                 "perturbation about 2.2x weaker at the large band); membership change only; property "
                 "change produced zero events; connectivity not implemented.")
            rr_df, guard_report = compute_response_rates(all_raw_event_df, change_type=ct, taus=(0.0, 1e-9))
            response_rate_df = rr_df
            emit(f"  Input rows for {ct}: {guard_report['input_rows']}; "
                 f"excluded (event_phase != 'immediate'): {guard_report['excluded_non_immediate_phase']}"
                 + (f" {guard_report.get('excluded_non_immediate_phase_breakdown', {})}"
                    if guard_report['excluded_non_immediate_phase'] else " -- zero skipped, as expected")
                 + f"; events after phase filter: {guard_report['events_after_phase_filter']}")
            for slice_name in RESPONSE_RATE_SLICES:
                n_nan = guard_report[f'{slice_name}_excluded_nan_or_missing']
                n_invalid = guard_report[f'{slice_name}_excluded_negative_or_nonfinite']
                n_valid = guard_report[f'{slice_name}_valid_events']
                n_exact_zero = guard_report[f'{slice_name}_exact_zero_events']
                overall_rate = guard_report[f'{slice_name}_overall_response_rate_tau0']
                emit(f"  slice={slice_name}: valid_events={n_valid} "
                     f"(excluded NaN/missing={n_nan}{'--zero skipped' if n_nan == 0 else ''}, "
                     f"excluded negative/non-finite={n_invalid}{'--zero skipped' if n_invalid == 0 else ''}); "
                     f"exact-zero (non-response) count={n_exact_zero} ({n_exact_zero / max(1, n_valid):.1%}); "
                     f"OVERALL response_rate (tau=0.0, all n_discovered pooled) = {overall_rate:.1%}")
            emit("  Per n_discovered bin (tau=0.0 primary; response_rate=NaN/n_episodes=0 means an empty cell, not a rate of 0):")
            for slice_name in RESPONSE_RATE_SLICES:
                emit(f"    slice={slice_name}:")
                bin_rows = rr_df[(rr_df['slice'] == slice_name) & (rr_df['tau'] == 0.0)].sort_values('n_discovered_bin')
                for _, row in bin_rows.iterrows():
                    if row['n_episodes'] == 0:
                        emit(f"      n_discovered {row['n_discovered_bin']}: EMPTY CELL (n_events=0, n_episodes=0)")
                    else:
                        emit(f"      n_discovered {row['n_discovered_bin']}: response_rate={row['response_rate']:.1%} "
                             f"[{row['ci_lo']:.1%}, {row['ci_hi']:.1%}] (n_events={row['n_events']}, n_episodes={row['n_episodes']})")
            # tau sensitivity check
            tau0_rates = rr_df[rr_df['tau'] == 0.0].set_index(['slice', 'n_discovered_bin'])['response_rate']
            tau_eps_rates = rr_df[rr_df['tau'] == 1e-9].set_index(['slice', 'n_discovered_bin'])['response_rate']
            max_abs_diff = float((tau0_rates - tau_eps_rates).abs().max()) if len(tau0_rates) else float('nan')
            emit(f"  Sensitivity check: max |response_rate(tau=0.0) - response_rate(tau=1e-9)| across all "
                 f"slice/bin cells = {max_abs_diff:.6f} ({'agree -- not a float-noise artifact' if (np.isnan(max_abs_diff) or max_abs_diff < 1e-6) else 'DISAGREE -- investigate before trusting either'})")
            # Step 4: state the corrected picture, confirmed or refuted from the numbers
            mean_rate = guard_report['mean_overall_response_rate_tau0']
            max_rate = guard_report['max_overall_response_rate_tau0']
            min_rate = guard_report['min_overall_response_rate_tau0']
            mean_agg = slope_results.get((ct, 'mean'), (float('nan'),) * 3 + (0,))[0]
            claim_holds = (mean_rate > 0.95) and (max_rate < 0.75) and (min_rate < 0.75)
            emit(f"  CORRECTED PICTURE: mean slice responds on {mean_rate:.1%} of events (near-universal) "
                 f"with a modest ~1/N-consistent slope ({mean_agg:.3f}); max responds on {max_rate:.1%}, "
                 f"min on {min_rate:.1%} of events (minority) -- "
                 + ("CONFIRMED: 'weak-always' (mean) plus 'strong-rarely' (max/min) does not compose into "
                    "reliable perception; this is a genuinely different, more defensible story than the "
                    "ARTIFACT 9-10 slopes." if claim_holds else
                    "NOT confirmed as stated -- the numbers above do not cleanly match the 'weak-always vs "
                    "minority-strong' pattern; read the per-slice rates directly rather than this summary claim."))

        # Visibility lag / unattributed fraction (only meaningful for membership_join)
        vs = None
        for band_stats in all_visibility_stats.values():
            if ct in band_stats:
                vs = band_stats[ct]
                break
        if vs and ct == "membership_join":
            n_fired = vs['n_fired_total_distinct_events']
            n_never = vs['n_never_attributed']
            frac_never = (n_never / n_fired) if n_fired else float('nan')
            emit(f"  VISIBILITY LAG: median visibility_lag_steps = {vs['median_visibility_lag_steps']}; "
                 f"never-attributed fraction = {n_never}/{n_fired} = {frac_never:.1%}"
                 if n_fired else "  VISIBILITY LAG: no fired membership_join events recorded")
            if n_fired and frac_never > 0.3:
                emit("  *** HEADLINE CANDIDATE: a large fraction of join events are never perceived "
                     "by the agent within an episode (not merely perceived late). Step 0 confirmed "
                     "this is not the pre-existing recon-synthesis bug (structurally inert given "
                     "patch_service_dynamic_enabled=False on all three checkpoints), so this reading "
                     "is not a bug artefact. ***")

    # --- Between-seed variance (AMENDMENT, task T v2 + amendment 1): the single-seed gate
    # could not report this at all; its episode-level bootstrap treated episodes from one
    # trained agent as if they sampled agent variability. With 5 seeds per band now trained,
    # report the variance of the headline response-rate numbers ACROSS seeds (not just the
    # within-seed bootstrap CI, which says nothing about seed-to-seed training variance).
    if all_raw_event_df is not None and len(all_raw_event_df) and 'seed' in all_raw_event_df.columns:
        emit("")
        emit("--- Between-seed variance (response rate, tau=0.0, all n_discovered pooled) ---")
        emit("Each seed's OWN overall response rate (5 independently trained checkpoints per band); "
             "mean and std are taken ACROSS these 5 per-seed numbers, not across episodes within one "
             "seed (that within-seed uncertainty is the bootstrap CI reported above/per-bin).")
        for band in sorted(all_raw_event_df['band'].dropna().unique().tolist()):
            band_df = all_raw_event_df[all_raw_event_df['band'] == band]
            for ct in change_types_present:
                ct_band_df = band_df[band_df['change_type'] == ct]
                if not len(ct_band_df):
                    continue
                seeds_present = sorted(ct_band_df['seed'].dropna().unique().tolist())
                per_seed_rates = {slice_name: [] for slice_name in RESPONSE_RATE_SLICES}
                for seed in seeds_present:
                    seed_df = ct_band_df[ct_band_df['seed'] == seed]
                    if not len(seed_df):
                        continue
                    _, seed_guard = compute_response_rates(seed_df, change_type=ct, taus=(0.0,))
                    for slice_name in RESPONSE_RATE_SLICES:
                        rate = seed_guard.get(f'{slice_name}_overall_response_rate_tau0')
                        if rate is not None and not np.isnan(rate):
                            per_seed_rates[slice_name].append(rate)
                emit(f"  band {band}, change_type {ct} ({len(seeds_present)} seeds: {seeds_present}):")
                for slice_name in RESPONSE_RATE_SLICES:
                    rates = per_seed_rates[slice_name]
                    if len(rates) < 2:
                        rates_str = [f'{r:.1%}' for r in rates] if rates else []
                        emit(f"    slice={slice_name}: undefined (n_seeds_with_data={len(rates)}, need >=2 for variance) -- rates={rates_str}")
                        continue
                    mean_rate = float(np.mean(rates))
                    std_rate = float(np.std(rates, ddof=1))
                    emit(f"    slice={slice_name}: mean={mean_rate:.1%}, std={std_rate:.1%} across "
                         f"{len(rates)} seeds, per-seed values=[{', '.join(f'{r:.1%}' for r in rates)}]")

    # --- Per-slice absolute and relative drift, episode-level, with bootstrap CI and
    # between-seed variance (AMENDMENT, task T v2 + amendment 1). Relative drift is the
    # persisted change_drift_{slice} column directly (median across an episode's events).
    # Absolute drift is NOT stored as its own column (per STEP 2's explicit scope) -- derived
    # here as rel_drift(s) * norm_h1_s, using the per-slice norm_h1_{slice} columns added in
    # STEP 2. n_events_pre_at_floor (norm_h1_s <= 1e-9) is computed offline here, from the raw
    # event data, per the numerics convention -- not added as env-side code.
    emit("")
    emit("--- Per-slice absolute and relative drift (episode-level median, bootstrap 0.95 CI) ---")
    emit("Absolute drift = rel_drift(s) * norm_h1_s (never norm_h2_s - norm_h1_s -- that is the "
         "difference of two magnitudes, not the magnitude of their difference).")
    for band in sorted(all_episode_df['band'].dropna().unique().tolist()):
        band_episode_df = all_episode_df[all_episode_df['band'] == band]
        for ct in change_types_present:
            ct_df = band_episode_df[band_episode_df['change_type'] == ct]
            if not len(ct_df):
                continue
            emit(f"  band {band}, change_type {ct}:")
            for slice_name in RESPONSE_RATE_SLICES:
                rel_col = 'change_drift_full' if slice_name == 'full' else f'change_drift_{slice_name}'
                norm_col = 'norm_h1' if slice_name == 'full' else f'norm_h1_{slice_name}'
                if rel_col not in ct_df.columns or norm_col not in ct_df.columns:
                    emit(f"    slice={slice_name}: column missing ({rel_col} or {norm_col}), skipped")
                    continue
                valid = ct_df[[rel_col, norm_col, 'seed']].dropna()
                n_events_pre_at_floor = int((valid[norm_col] <= 1e-9).sum())
                rel_rate, rel_lo, rel_hi = bootstrap_series_ci(valid[rel_col])
                abs_series = valid[rel_col] * valid[norm_col]
                abs_rate, abs_lo, abs_hi = bootstrap_series_ci(abs_series)
                emit(f"    slice={slice_name}: n_episodes={len(valid)}, n_events_pre_at_floor "
                     f"(norm_h1_{slice_name}<=1e-9)={n_events_pre_at_floor}; "
                     f"relative drift median={rel_rate:.4f} [{rel_lo:.4f}, {rel_hi:.4f}]; "
                     f"absolute drift median={abs_rate:.4f} [{abs_lo:.4f}, {abs_hi:.4f}]")
                # Between-seed variance: each seed's OWN median, then variance across those medians.
                per_seed_rel = valid.groupby('seed')[rel_col].median()
                per_seed_abs = (valid[rel_col] * valid[norm_col]).groupby(valid['seed']).median()
                if len(per_seed_rel) >= 2:
                    emit(f"      between-seed (n={len(per_seed_rel)}): relative drift mean of "
                         f"per-seed medians={per_seed_rel.mean():.4f}, std={per_seed_rel.std(ddof=1):.4f}; "
                         f"absolute drift mean of per-seed medians={per_seed_abs.mean():.4f}, "
                         f"std={per_seed_abs.std(ddof=1):.4f}")
                else:
                    emit(f"      between-seed: undefined (only {len(per_seed_rel)} seed(s) with data, need >=2)")

    emit("")
    emit("#" * 100)
    emit(PROVISIONAL_BANNER)
    emit("#" * 100)

    summary_path = os.path.join(OUTPUT_DIR, "gate_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    response_rate_csv_path = None
    if response_rate_df is not None:
        response_rate_csv_path = os.path.join(OUTPUT_DIR, "response_rate_by_bin.csv")
        # Task A provisional banner recorded as a metadata row so the CSV is self-describing
        # even if extracted from this directory on its own.
        banner_row = pd.DataFrame([{
            "change_type": "PROVISIONAL_BANNER",
            "slice": ("gate data uses the shared confounded donor pool (novelty confound, "
                      "perturbation about 2.2x weaker at the large band); membership change only; "
                      "property change produced zero events; connectivity not implemented."),
        }])
        pd.concat([banner_row, response_rate_df], ignore_index=True).to_csv(response_rate_csv_path, index=False)

    print(f"\nOutputs written to {OUTPUT_DIR}:")
    print(f"  {fig1_path}" if fig1_ok else f"  [FAILED to render] {fig1_path}")
    print(f"  {fig2_path}" if fig2_ok else f"  [FAILED to render] {fig2_path}")
    print(f"  {fig3_path}" if fig3_ok else f"  [FAILED to render] {fig3_path}")
    print(f"  {csv_path}")
    if response_rate_csv_path:
        print(f"  {response_rate_csv_path}")
    print(f"  {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="attenuation_manifest.yaml")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if not args.collect and not args.analyze:
        args.collect = args.analyze = True

    with open(args.manifest) as f:
        manifest = yaml.safe_load(f)

    import logging
    logger = logging.getLogger("compute_attenuation_analysis")
    logger.addHandler(logging.NullHandler())

    import json
    all_episode_dfs = []
    all_raw_event_dfs = []
    all_visibility_stats = {}
    shortfalls = {}
    skip_infos = {}
    for band_label, band_config in manifest['bands'].items():
        drift_log_path = os.path.join(DRIFT_LOG_DIR, f"drift_{band_label}.csv")
        skip_info_path = os.path.join(DRIFT_LOG_DIR, f"skip_info_{band_label}.json")
        shortfall_path = os.path.join(DRIFT_LOG_DIR, f"shortfall_{band_label}.json")
        if args.collect:
            drift_log_path, final_counts, shortfall, n_episodes, skip_info = collect_band_data(band_label, band_config, manifest, logger)
            shortfalls[band_label] = shortfall
            skip_infos[band_label] = skip_info
            with open(skip_info_path, "w") as f:
                json.dump(skip_info, f)
            with open(shortfall_path, "w") as f:
                json.dump(shortfall, f)
        if args.analyze:
            if not os.path.exists(drift_log_path):
                print(f"[{band_label}] no drift log found at {drift_log_path}, skipping analysis "
                      f"(run with --collect first)")
                continue
            episode_df, visibility_stats, n_visibility_dropped, max_determinism_drift, attenuation_df = analyze_band(band_label, drift_log_path)
            all_episode_dfs.append(episode_df)
            all_raw_event_dfs.append(attenuation_df)
            all_visibility_stats[band_label] = visibility_stats
            if band_label not in shortfalls:
                # --analyze-only re-run: recover the shortfall record from the sidecar file
                # written by the collection run, if available, rather than silently reporting
                # "no shortfall" for a band that in fact under-shot its target.
                if os.path.exists(shortfall_path):
                    with open(shortfall_path) as f:
                        shortfalls[band_label] = json.load(f)
                else:
                    shortfalls[band_label] = {}
            if band_label not in skip_infos:
                # --analyze-only re-run (no fresh --collect this invocation): recover skip
                # accounting from the sidecar file written by the collection run, if available,
                # rather than silently reporting nothing.
                if os.path.exists(skip_info_path):
                    with open(skip_info_path) as f:
                        skip_infos[band_label] = json.load(f)
                else:
                    skip_infos[band_label] = None  # genuinely unknown -- pre-dates this amendment

    if args.analyze and all_episode_dfs:
        all_episode_df = pd.concat(all_episode_dfs, ignore_index=True)
        all_raw_event_df = pd.concat(all_raw_event_dfs, ignore_index=True) if all_raw_event_dfs else pd.DataFrame()
        run_gate_and_outputs(all_episode_df, all_visibility_stats, manifest['bands'], shortfalls, skip_infos, all_raw_event_df)


if __name__ == "__main__":
    main()
