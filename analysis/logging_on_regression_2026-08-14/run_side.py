"""Task LOGGING-ON-REGRESSION, STEP 1/2: one side of the paired comparison.

Reuses the technique of the existing, already-validated 2000-step/2-seed regression
(analysis/recovered_scripts_2026-08-04/drift_regression_check_v2.py): `repo_root` is a
parameter so the identical harness logic can import EITHER the pre-instrumentation
`cyberbattle` package (commit 7cdfb2b, extracted via `git archive` into a scratch tree since
the instrumentation was never on a separate branch) OR the current real repo, from a fresh
Python process each time (no import caching risk across repo_root switches).

Departure from that existing harness, per this task's explicit STEP 2 requirement: this script
drives the rollout with a REAL, REPORTED TRPO checkpoint via `model.predict()` (stochastic,
matching the regime that actually produced Chapter IV's figures -- deterministic policies barely
explore, per this project's own established finding), not a synthetic fixed random-action array.
Checkpoint: band 10-15, seed 42 (logs/trpo_250k_tuned_compressed_band10-15_seed42_2026-07-26_
11-56-51/TRPO_x_control_SecureBERT/checkpoints/1/checkpoint_250000_steps.zip) -- one of the 15
manifest checkpoints behind every reported attenuation figure in this project. Topology:
cyberbattle/data/env_samples/scalability_10_15/1 (a real manifest-band topology, not a synthetic
stand-in). Encoder + VecNormalize loaded exactly as compute_attenuation_analysis.py's own
production pipeline does (train_config keys -> env kwargs, VecNormalize.load with training=False,
norm_reward=False).

Side-A-specific kwargs filtering: commit 7cdfb2b's CyberBattleEnv/CyberBattleCompressedEnv
constructors (read directly via `git show 7cdfb2b:...`) accept every dynamic_* / change_* /
patch_service_dynamic_enabled key that appears in this checkpoint's own train_config.yaml -- this
checkpoint's config was captured at/around the SAME commit, so no key mismatch is expected -- but
NOT allow_undiscovered_removal/uncapped_join (added later, confirmed absent from 7cdfb2b) and NOT
any drift_logging/event_graph_logging/leave_embedding_logging kwarg (the instrumentation itself).
Side A therefore never receives ANY instrumentation-related kwarg -- the concept did not exist yet.
Side B receives the checkpoint's own train_config keys (unchanged -- allow_undiscovered_removal
etc. are absent from train_config.yaml for this checkpoint, so both sides run the SAME dynamic-
change configuration by construction, not just by coincidence) plus drift_logging=True,
drift_sample_rate=1 (matching every reported run, per Task METRIC-DEFINITIONS), drift_log_path=None
(in-memory only -- the log content itself is not the comparison artifact; only the agent's own
trajectory is).

Records go to a pickle with one entry per step: episode index, step-within-episode, full action
identity (source_node, target_node, vulnerability, outcome, min_distance_action, plus the raw
continuous action vector fed to the policy), reward, done, cumulative (running) reward, discovered-
node count, root-owned-node count, and the full float64 observation vector (graph_embeddings +
discrete_features concatenated).

Usage: python run_side.py <repo_root> <side_tag> <seed> <n_steps> <out.pkl>
"""
import os
import pickle
import random
import sys

REPO_ROOT_REAL = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim"
RUN_FOLDER = os.path.join(
    REPO_ROOT_REAL, "cyberbattle/agents",
    "logs/trpo_250k_tuned_compressed_band10-15_seed42_2026-07-26_11-56-51/TRPO_x_control_SecureBERT")
TOPOLOGY_PKL = os.path.join(REPO_ROOT_REAL, "cyberbattle/data/env_samples/scalability_10_15/1/network_SecureBERT.pkl")

# Constructor kwarg allow-list for the pre-instrumentation side (7cdfb2b), read directly from
# that commit's own CyberBattleEnv.__init__ / CyberBattleCompressedEnv.__init__ signatures.
SIDE_A_ALLOWED_TRAIN_CONFIG_KEYS = {
    "goal", "rewards_dict", "penalties_dict", "episode_iterations",
    "change_interval", "change_type", "patch_service_dynamic_enabled", "dynamic_mode",
    "dynamic_min_alive_nodes", "dynamic_min_alive_fraction", "dynamic_batch_interval",
    "dynamic_batch_size_mean", "dynamic_batch_max_fraction", "dynamic_degree_weighting",
    "dynamic_max_alive_nodes", "dynamic_max_alive_fraction", "dynamic_max_joins_per_episode",
    "dynamic_join_rate_interval", "dynamic_join_batch_interval", "dynamic_join_batch_size_mean",
    "dynamic_join_batch_max_fraction", "dynamic_join_value_weighting",
    "graph_embeddings_aggregations", "node_embeddings_dimensions", "outcome_dimensions",
    "discrete_features", "pca_components", "distance_metric", "sample_subset_samples",
    "remove_all_obstacles", "remove_main_obstacles", "precise_action_space_positions",
    "precise_graph_encoding",
}


def main():
    repo_root, side_tag, seed, n_steps, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    sys.path.insert(0, repo_root)

    import numpy as np
    import torch
    import yaml
    from sb3_contrib import TRPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

    from cyberbattle.gae.model import GAEEncoder
    from cyberbattle._env.cyberbattle_env_switch import RandomSwitchEnv
    import cyberbattle._env.cyberbattle_env_compressed as ccenv
    from cyberbattle.simulation import model as cb_model

    torch.set_num_threads(1)

    with open(os.path.join(RUN_FOLDER, "train_config.yaml")) as f:
        train_config = yaml.safe_load(f)
    with open(train_config["graph_encoder_config_path"]) as f:
        config_encoder = yaml.safe_load(f)
    with open(train_config["graph_encoder_spec_path"]) as f:
        spec_encoder = yaml.safe_load(f)
    config_encoder.update(spec_encoder)
    graph_encoder = GAEEncoder(config_encoder["node_feature_vector_size"],
                                config_encoder["model_config"]["layers"],
                                config_encoder["edge_feature_vector_size"])
    graph_encoder.load_state_dict(torch.load(train_config["graph_encoder_path"]))
    graph_encoder.eval()
    train_config["node_embeddings_dimensions"] = config_encoder["model_config"]["layers"][-1]["out_channels"]

    import pickle as pkl
    with open(TOPOLOGY_PKL, "rb") as f:
        network = pkl.load(f)

    import logging
    logger = logging.getLogger("logging_on_regression")
    logger.addHandler(logging.NullHandler())

    if side_tag == "A":
        kwargs = {k: v for k, v in train_config.items() if k in SIDE_A_ALLOWED_TRAIN_CONFIG_KEYS}
    elif side_tag == "B":
        kwargs = {k: v for k, v in train_config.items() if k != "verbose"}
        kwargs["drift_logging"] = True
        kwargs["drift_sample_rate"] = 1
        kwargs["drift_log_path"] = None
        kwargs["drift_run_id"] = f"logging_on_regression_seed{seed}"
        kwargs["drift_seed"] = seed
        kwargs["drift_scenario_id"] = "scalability_10_15_1"
    elif side_tag == "B_off":
        # Diagnostic control (added after the first full run diverged): current HEAD, but with
        # drift_logging left at its default (False) -- isolates whether a divergence from Side A
        # is caused by the instrumentation itself, or by unrelated code changes accumulated in
        # the current-HEAD tree since commit 7cdfb2b (three-plus weeks of unrelated development).
        kwargs = {k: v for k, v in train_config.items() if k != "verbose"}
    else:
        raise ValueError(f"unknown side_tag {side_tag!r}")

    env = ccenv.CyberBattleCompressedEnv(initial_environment=network, logger=logger, verbose=0, **kwargs)
    env.set_graph_encoder(graph_encoder)
    env.set_pca_components(train_config.get("pca_components"))

    switch_env = RandomSwitchEnv([0], switch_interval=10 ** 9, envs_list=[env], save_to_csv=False, verbose=0)
    vec_env = DummyVecEnv([lambda: switch_env])
    vecnormalize_path = os.path.join(RUN_FOLDER, "checkpoints", "1", "checkpoint_vecnormalize_250000_steps.pkl")
    vec_env = VecNormalize.load(vecnormalize_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    checkpoint_path = os.path.join(RUN_FOLDER, "checkpoints", "1", "checkpoint_250000_steps.zip")
    model = TRPO.load(checkpoint_path)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    rows = []
    obs = vec_env.reset()
    cumulative_reward = 0.0
    episode_idx = 0
    step_in_episode = 0
    for i in range(n_steps):
        action, _ = model.predict(obs, deterministic=False)
        next_obs, reward, done, infos = vec_env.step(action)
        info = infos[0]
        r = float(reward[0])
        d = bool(done[0])
        cumulative_reward += r
        underlying_env = vec_env.venv.envs[0].current_env
        rows.append(dict(
            episode=episode_idx, step_in_episode=step_in_episode,
            action_vector=np.asarray(action[0], dtype=np.float64).copy(),
            source_node=info.get("source_node"), target_node=info.get("target_node"),
            vulnerability=info.get("vulnerability"), outcome=info.get("outcome"),
            min_distance_action=info.get("min_distance_action"),
            reward=r, done=d, cumulative_reward=cumulative_reward,
            n_discovered=len(underlying_env.discovered_nodes),
            n_root_owned=len([n for n in underlying_env.owned_nodes
                               if n in underlying_env.environment.nodes
                               and underlying_env.get_node(n).privilege_level == cb_model.PrivilegeLevel.ROOT]),
            obs_pre_step=np.asarray(obs["graph_embeddings"], dtype=np.float64).ravel().copy().tolist()
                         + np.asarray(obs["discrete_features"], dtype=np.float64).ravel().copy().tolist(),
        ))
        obs = next_obs
        step_in_episode += 1
        if d:
            episode_idx += 1
            step_in_episode = 0
            cumulative_reward = 0.0

    with open(out_path, "wb") as f:
        pickle.dump(rows, f)
    print(f"[{side_tag} seed={seed}] wrote {len(rows)} rows to {out_path} (repo_root={repo_root})")


if __name__ == "__main__":
    main()
