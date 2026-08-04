"""
Task F Pass 1, STEP 1: single-topology TRPO training driver, run in-process (no env pickling, so
drift_logging works cleanly). Mirrors train_agent.py's train_model() setup exactly
(RandomSwitchEnv -> DummyVecEnv -> Monitor -> VecNormalize -> TRPO with algo_config.yaml's tuned
hyperparameters, CheckpointCallback save_freq/save_vecnormalize), producing the same
run_folder/checkpoints/1/checkpoint_*_steps.zip + checkpoint_vecnormalize_*_steps.pkl layout that
STEP 2's load path (compute_attenuation_analysis.load_band_model_and_encoder) expects. The tuned
env+training config is taken verbatim from the completed gate's own 30-40 seed42 train_config.yaml
(the exact tuned config used throughout), with ONLY these overridden per condition:
  STATIC : dynamic_mode=none,  patch_service_dynamic_enabled=false
  ADAPTED: dynamic_mode=both,  patch_service_dynamic_enabled=true   (membership + property active)
Plus name/seed/drift_* per run. Nothing else differs between the two conditions.

Usage: python taskF1_train.py <seed> <static|adapted> <run_folder>
"""
import sys, os, copy, logging, pickle, random
sys.path.insert(0, "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim")
os.chdir("/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents")

import numpy as np
import torch
import yaml
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import TRPO

from cyberbattle.utils.envs_utils import wrap_graphs_to_compressed_envs
from cyberbattle.utils.file_utils import load_yaml
from cyberbattle.utils.train_utils import replace_with_classes
from cyberbattle.utils.math_utils import set_seeds
from cyberbattle.gae.model import GAEEncoder
from cyberbattle._env.cyberbattle_env_switch import RandomSwitchEnv

SEED = int(sys.argv[1])
CONDITION = sys.argv[2]
RUN_FOLDER = os.path.abspath(sys.argv[3])  # resolve BEFORE the chdir below, or it lands under agents/
STEPS_OVERRIDE = int(sys.argv[4]) if len(sys.argv) > 4 else None  # smoke-test only
assert CONDITION in ("static", "adapted")
assert os.path.isabs(RUN_FOLDER)

torch.set_num_threads(4)

REPO = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim"
# Tuned config template: the gate's own 30-40 seed42 train_config.yaml (identical tuned config).
GATE_TRAIN_CONFIG = os.path.join(
    REPO, "cyberbattle/agents/logs",
    "trpo_250k_tuned_compressed_band30-40_seed42_2026-07-26_12-47-19",
    "TRPO_x_control_SecureBERT", "train_config.yaml")
TOPOLOGY_PKL = os.path.join(REPO, "cyberbattle/data/env_samples/scalability_30_40/44/network_SecureBERT.pkl")
DONOR_FOLDER = os.path.join(REPO, "cyberbattle/data/env_samples/join_donor_pool_20_topologies")

os.makedirs(RUN_FOLDER, exist_ok=True)
os.makedirs(os.path.join(RUN_FOLDER, "checkpoints", "1"), exist_ok=True)

with open(GATE_TRAIN_CONFIG) as f:
    cfg = yaml.safe_load(f)

if STEPS_OVERRIDE is not None:
    cfg["train_iterations"] = STEPS_OVERRIDE
    cfg["checkpoints_save_freq"] = max(50, STEPS_OVERRIDE // 2)

# --- per-condition overrides (the ONLY differences between static and adapted) ---
if CONDITION == "static":
    cfg["dynamic_mode"] = "none"
    cfg["patch_service_dynamic_enabled"] = False
else:  # adapted: membership + property active
    cfg["dynamic_mode"] = "both"
    cfg["patch_service_dynamic_enabled"] = True

cfg["name"] = f"trpo_250k_F1_{CONDITION}_seed{SEED}"
cfg["seeds_runs"] = [SEED]
# drift logging ON for both (STEP 1.4)
DRIFT_LOG_PATH = os.path.join(RUN_FOLDER, "train_drift.csv")
cfg["drift_logging"] = True
cfg["drift_log_path"] = DRIFT_LOG_PATH
cfg["drift_sample_rate"] = 1
cfg["drift_run_id"] = cfg["name"]
cfg["drift_seed"] = SEED
cfg["drift_scenario_id"] = "scalability_30_40_44"

logger = logging.getLogger("taskF1_train"); logger.addHandler(logging.NullHandler())

# --- encoder ---
with open(cfg["graph_encoder_config_path"]) as f:
    config_encoder = yaml.safe_load(f)
with open(cfg["graph_encoder_spec_path"]) as f:
    spec_encoder = yaml.safe_load(f)
config_encoder.update(spec_encoder)
graph_encoder = GAEEncoder(config_encoder["node_feature_vector_size"], config_encoder["model_config"]["layers"],
                           config_encoder["edge_feature_vector_size"])
graph_encoder.load_state_dict(torch.load(cfg["graph_encoder_path"]))
graph_encoder.eval()
cfg["node_embeddings_dimensions"] = config_encoder["model_config"]["layers"][-1]["out_channels"]

set_seeds(SEED)

# --- env construction (mirrors train_agent.py env-processing + build_band_envs donor attach) ---
with open(TOPOLOGY_PKL, "rb") as f:
    network = pickle.load(f)

# env kwargs = full cfg minus keys the constructor's super() chain doesn't want; the constructor
# swallows the rest via **kwargs (same as train_agent.py passing **config through wrap_graphs).
env = wrap_graphs_to_compressed_envs(network, logger, **cfg)
env.set_graph_encoder(graph_encoder)
env.set_pca_components(cfg["pca_components"])

# Donor pool for dynamic-join (adapted only; static has dynamic_mode=none so join never fires,
# but attaching a pool is harmless and mirrors the exact build path). element id "44" is never in
# the 1..20 donor folder, so no self-exclusion loss.
if cfg.get("dynamic_mode") in ("join", "both"):
    donor_pool = []
    for sub in sorted(os.listdir(DONOR_FOLDER)):
        subpath = os.path.join(DONOR_FOLDER, sub)
        if os.path.isdir(subpath) and sub.isdigit() and sub != "44":
            with open(os.path.join(subpath, f"network_{cfg['nlp_extractor']}.pkl"), "rb") as f:
                donor_network = pickle.load(f)
            donor_pool += [(sub, node_id, copy.deepcopy(node_data["data"]))
                           for node_id, node_data in donor_network.network.nodes(data=True)]
    env.dynamic_join_donor_pool = donor_pool
    print(f"[{cfg['name']}] donor pool size: {len(donor_pool)} nodes")

# --- wrap: RandomSwitchEnv (single env, never switches) -> DummyVecEnv(Monitor) -> VecNormalize ---
switch_env = RandomSwitchEnv([0], switch_interval=10**9, envs_list=[env], save_to_csv=False, verbose=0)
train_envs = DummyVecEnv([lambda: Monitor(switch_env)])
algorithm_config = copy.deepcopy(cfg["algorithm_hyperparams"])
vecnormalize_gamma = algorithm_config.get("gamma", 0.99)
train_envs = VecNormalize(train_envs, norm_obs=cfg["norm_obs"], norm_reward=cfg["norm_reward"], gamma=vecnormalize_gamma)

# --- learning rate (constant for trpo tuned config) ---
if algorithm_config.get("learning_rate_type") == "linear":
    from cyberbattle.utils.math_utils import linear_schedule
    learning_rate = linear_schedule(algorithm_config["learning_rate"], algorithm_config["learning_rate_final"])
else:
    learning_rate = algorithm_config["learning_rate"]
algorithm_config.pop("learning_rate_type", None)
algorithm_config.pop("learning_rate", None)
algorithm_config.pop("learning_rate_final", None)

policy_kwargs = replace_with_classes(copy.deepcopy(cfg["policy_kwargs"]))
for key in ["lstm_hidden_size", "n_lstm_layers"]:
    policy_kwargs.pop(key, None)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TRPO("MultiInputPolicy", train_envs, policy_kwargs=policy_kwargs, learning_rate=learning_rate,
             **algorithm_config, verbose=0, device=device)

checkpoint_callback = CheckpointCallback(
    save_freq=cfg["checkpoints_save_freq"],
    save_path=os.path.join(RUN_FOLDER, "checkpoints", "1"),
    name_prefix="checkpoint", save_vecnormalize=True,
)

# --- dump a complete train_config.yaml for STEP 2's loader + reproducibility ---
# (store the drift logger config too; static string values only, no live objects)
cfg_to_save = copy.deepcopy(cfg)
with open(os.path.join(RUN_FOLDER, "train_config.yaml"), "w") as f:
    yaml.safe_dump(cfg_to_save, f)

print(f"[{cfg['name']}] device={device}, starting 250000-step TRPO training on topology 44 "
      f"(dynamic_mode={cfg['dynamic_mode']}, patch_service_dynamic_enabled={cfg['patch_service_dynamic_enabled']})")
model.learn(total_timesteps=cfg["train_iterations"], callback=checkpoint_callback)

# ensure a final checkpoint at exactly train_iterations even if save_freq didn't land on it
final_model_path = os.path.join(RUN_FOLDER, "checkpoints", "1", f"checkpoint_{cfg['train_iterations']}_steps.zip")
if not os.path.exists(final_model_path):
    model.save(final_model_path)
    train_envs.save(os.path.join(RUN_FOLDER, "checkpoints", "1", f"checkpoint_vecnormalize_{cfg['train_iterations']}_steps.pkl"))

# flush drift log
try:
    switch_env.envs_list[0]._drift_logger.close()
except Exception:
    pass

print(f"[{cfg['name']}] TRAINING COMPLETE. checkpoints in {os.path.join(RUN_FOLDER, 'checkpoints', '1')}")
