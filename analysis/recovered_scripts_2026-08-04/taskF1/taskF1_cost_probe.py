"""
Task F Pass 1, STEP 0.5: measured wall-clock for a 5000-step TRPO training run on the chosen
single topology (scalability_30_40/44, one of the gate's 8 official 30-40-band topologies),
under the ADAPTED-like config (dynamic_mode=both, patch_service_dynamic_enabled=True,
drift_logging=True) -- the pricier of the two STEP 1 conditions, so this is a conservative
(upper-bound) per-step cost estimate. Faithfully replicates train_agent.py's train_model() setup
(DummyVecEnv -> Monitor -> VecNormalize -> TRPO with algo_config.yaml's tuned hyperparameters),
minus checkpointing/tensorboard callbacks (irrelevant to wall-clock/step).
"""
import sys, os, pickle, logging, time
sys.path.insert(0, "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim")
os.chdir("/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents")

import torch, yaml
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import TRPO
from cyberbattle.utils.envs_utils import wrap_graphs_to_compressed_envs
from cyberbattle.utils.file_utils import load_yaml
from cyberbattle.utils.train_utils import replace_with_classes
from cyberbattle.gae.model import GAEEncoder
from cyberbattle._env.cyberbattle_env_switch import RandomSwitchEnv

torch.set_num_threads(4)

ENCODER_DIR = "../gae/logs/default/SecureBERT"
with open(os.path.join(ENCODER_DIR, "train_config_encoder.yaml")) as f:
    config_encoder = yaml.safe_load(f)
with open(os.path.join(ENCODER_DIR, "model_spec.yaml")) as f:
    spec_encoder = yaml.safe_load(f)
config_encoder.update(spec_encoder)
graph_encoder = GAEEncoder(config_encoder["node_feature_vector_size"], config_encoder["model_config"]["layers"], config_encoder["edge_feature_vector_size"])
graph_encoder.load_state_dict(torch.load(os.path.join(ENCODER_DIR, "encoder.pth")))
graph_encoder.eval()

rewards_config = load_yaml("config/rewards_config.yaml")
train_config = load_yaml("config/train_config.yaml")
algo_config = load_yaml("config/algo_config.yaml")
logger = logging.getLogger("taskF1_cost_probe"); logger.addHandler(logging.NullHandler())

common_kwargs = dict(
    rewards_dict=rewards_config["rewards_dict"]["control"], penalties_dict=rewards_config["penalties_dict"]["control"],
    goal="control", episode_iterations=train_config["episode_iterations"],
    proportional_cutoff_coefficient=train_config["proportional_cutoff_coefficient"],
    winning_reward=train_config["winning_reward"], losing_reward=train_config["losing_reward"],
    random_starter_node=train_config["random_starter_node"], stop_at_goal_reached=train_config["stop_at_goal_reached"],
    isolation_filter_threshold=train_config["isolation_filter_threshold"],
    remove_main_obstacles=train_config["remove_main_obstacles"], remove_all_obstacles=train_config["remove_all_obstacles"],
    dynamic_mode="both", change_interval=train_config["change_interval"],
    patch_service_dynamic_enabled=True,  # ADAPTED-like: pricier of the two STEP 1 conditions
    static_defender_eviction_goal=train_config["static_defender_eviction_goal"],
    outcome_dimensions=train_config["outcome_dimensions"], discrete_features=train_config["discrete_features"],
    graph_embeddings_aggregations=train_config["graph_embeddings_aggregations"],
    node_embeddings_dimensions=config_encoder["model_config"]["layers"][-1]["out_channels"],
    sample_subset_samples=train_config["sample_subset_samples"],
    verbose=0,
    drift_logging=True, drift_log_path="/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1/cost_probe_drift.csv",
    drift_sample_rate=1, drift_run_id="cost_probe", drift_seed=42, drift_scenario_id="scalability_30_40_44",
)

TOPOLOGY_PKL = "../data/env_samples/scalability_30_40/44/network_SecureBERT.pkl"
with open(TOPOLOGY_PKL, "rb") as f:
    network = pickle.load(f)

env = wrap_graphs_to_compressed_envs(network, logger, **common_kwargs)
env.set_graph_encoder(graph_encoder)
env.set_pca_components(768)

# Real training never wraps a bare CyberBattleCompressedEnv (old gym.Env) directly in
# DummyVecEnv -- it always goes through RandomSwitchEnv (gymnasium.Env) first, which is what
# actually produces gymnasium-compatible spaces for SB3. Single-element list, switch_interval
# larger than this probe's step budget so it never switches away from topology 44.
switch_env = RandomSwitchEnv([0], switch_interval=10**9, envs_list=[env], save_to_csv=False, verbose=0)
train_envs = DummyVecEnv([lambda: Monitor(switch_env)])
algorithm_config = dict(algo_config["trpo"])
vecnormalize_gamma = algorithm_config.get("gamma", 0.99)
train_envs = VecNormalize(train_envs, norm_obs=train_config["norm_obs"], norm_reward=train_config["norm_reward"], gamma=vecnormalize_gamma)

learning_rate = algorithm_config.pop("learning_rate")
algorithm_config.pop("learning_rate_type", None)
algorithm_config.pop("learning_rate_final", None)

policy_kwargs = replace_with_classes(dict(algo_config["policy_kwargs"]))
for key in ["lstm_hidden_size", "n_lstm_layers"]:
    policy_kwargs.pop(key, None)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TRPO("MultiInputPolicy", train_envs, policy_kwargs=policy_kwargs, learning_rate=learning_rate,
             **algorithm_config, verbose=0, device=device)

N_STEPS = 5000
t0 = time.perf_counter()
model.learn(total_timesteps=N_STEPS)
elapsed = time.perf_counter() - t0

print(f"device={device}")
print(f"5000-step TRPO training run (topology scalability_30_40/44, ADAPTED-like config): {elapsed:.2f}s")
print(f"per-step: {elapsed/N_STEPS*1000:.3f} ms")
print(f"extrapolated to 250000 steps (1 run): {elapsed/N_STEPS*250000/60:.2f} min = {elapsed/N_STEPS*250000/3600:.3f} h")
print(f"extrapolated to 250000 steps x 10 runs (STEP 1 total): {elapsed/N_STEPS*250000*10/3600:.3f} h = {elapsed/N_STEPS*250000*10/3600/24:.3f} days")
