"""
Task F4 training driver: extend static single-topology agents to a larger budget.
Two modes:
  RESUME (30-40, 80-100): load the 250k checkpoint + its VecNormalize, continue static training to
    <target> steps on the SAME topology (reset_num_timesteps=False, so num_timesteps continues
    250k->target and the vecnorm stats keep accumulating). Faithful single resume: dynamic_mode=none
    (no dynamic-change RNG), constant LR, optimiser+vecnorm restored.
  FRESH (10-15): train a new static agent from scratch to <target> steps.
Static-only (dynamic_mode=none, patch off), tuned config from the gate template, drift logging on.
Keeps the 250k checkpoints untouched (writes to a new run folder).

Usage:
  fresh:  python taskF4_train.py <seed> <topology_subpath> <run_folder> <target_steps>
  resume: python taskF4_train.py <seed> <topology_subpath> <run_folder> <target_steps> <resume_ckpt> <resume_vecnorm>
"""
import sys, os, copy, logging, pickle
sys.path.insert(0, "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim")
os.chdir("/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents")
import numpy as np, torch, yaml
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import TRPO
from cyberbattle.utils.envs_utils import wrap_graphs_to_compressed_envs
from cyberbattle.utils.train_utils import replace_with_classes
from cyberbattle.utils.math_utils import set_seeds
from cyberbattle.gae.model import GAEEncoder
from cyberbattle._env.cyberbattle_env_switch import RandomSwitchEnv

SEED = int(sys.argv[1]); TOPO = sys.argv[2]; RUN_FOLDER = os.path.abspath(sys.argv[3]); TARGET = int(sys.argv[4])
RESUME_CKPT = os.path.abspath(sys.argv[5]) if len(sys.argv) > 5 else None
RESUME_VN = os.path.abspath(sys.argv[6]) if len(sys.argv) > 6 else None
assert os.path.isabs(RUN_FOLDER)
torch.set_num_threads(4)
REPO = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim"
GATE_TRAIN_CONFIG = os.path.join(REPO, "cyberbattle/agents/logs",
    "trpo_250k_tuned_compressed_band30-40_seed42_2026-07-26_12-47-19", "TRPO_x_control_SecureBERT", "train_config.yaml")
os.makedirs(os.path.join(RUN_FOLDER, "checkpoints", "1"), exist_ok=True)
cfg = yaml.safe_load(open(GATE_TRAIN_CONFIG))
cfg["dynamic_mode"] = "none"; cfg["patch_service_dynamic_enabled"] = False   # STATIC only
cfg["train_iterations"] = TARGET
cfg["name"] = f"trpo_F4_static_{TOPO.replace('/', '_')}_seed{SEED}_{TARGET}"
cfg["seeds_runs"] = [SEED]
DRIFT = os.path.join(RUN_FOLDER, "train_drift.csv")
cfg["drift_logging"] = True; cfg["drift_log_path"] = DRIFT; cfg["drift_sample_rate"] = 1
cfg["drift_run_id"] = cfg["name"]; cfg["drift_seed"] = SEED; cfg["drift_scenario_id"] = TOPO.replace("/", "_")
logger = logging.getLogger("f4"); logger.addHandler(logging.NullHandler())

ce = yaml.safe_load(open(cfg["graph_encoder_config_path"])); ce.update(yaml.safe_load(open(cfg["graph_encoder_spec_path"])))
ge = GAEEncoder(ce["node_feature_vector_size"], ce["model_config"]["layers"], ce["edge_feature_vector_size"])
ge.load_state_dict(torch.load(cfg["graph_encoder_path"])); ge.eval()
cfg["node_embeddings_dimensions"] = ce["model_config"]["layers"][-1]["out_channels"]
set_seeds(SEED)

net = pickle.load(open(os.path.join(REPO, "cyberbattle/data/env_samples", TOPO, "network_SecureBERT.pkl"), "rb"))
env = wrap_graphs_to_compressed_envs(net, logger, **cfg)
env.set_graph_encoder(ge); env.set_pca_components(cfg["pca_components"])
switch_env = RandomSwitchEnv([0], switch_interval=10**9, envs_list=[env], save_to_csv=False, verbose=0)
base_vec = DummyVecEnv([lambda: Monitor(switch_env)])
gamma = cfg["algorithm_hyperparams"].get("gamma", 0.99)
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_cb = CheckpointCallback(save_freq=cfg["checkpoints_save_freq"],
    save_path=os.path.join(RUN_FOLDER, "checkpoints", "1"), name_prefix="checkpoint", save_vecnormalize=True)

if RESUME_CKPT:
    # continue an existing run on the SAME topology to TARGET
    train_envs = VecNormalize.load(RESUME_VN, base_vec)
    train_envs.training = True; train_envs.norm_reward = cfg["norm_reward"]
    model = TRPO.load(RESUME_CKPT, device=device)
    model.set_env(train_envs)
    remaining = TARGET - model.num_timesteps
    print(f"[{cfg['name']}] RESUME on {TOPO} from num_timesteps={model.num_timesteps} -> {TARGET} (+{remaining}); device={device}")
    model.learn(total_timesteps=remaining, callback=ckpt_cb, reset_num_timesteps=False)
else:
    train_envs = VecNormalize(base_vec, norm_obs=cfg["norm_obs"], norm_reward=cfg["norm_reward"], gamma=gamma)
    algo = copy.deepcopy(cfg["algorithm_hyperparams"])
    lr = algo["learning_rate"]; [algo.pop(k, None) for k in ("learning_rate", "learning_rate_type", "learning_rate_final")]
    pk = replace_with_classes(copy.deepcopy(cfg["policy_kwargs"]))
    for k in ("lstm_hidden_size", "n_lstm_layers"): pk.pop(k, None)
    model = TRPO("MultiInputPolicy", train_envs, policy_kwargs=pk, learning_rate=lr, **algo, verbose=0, device=device)
    print(f"[{cfg['name']}] FRESH on {TOPO} -> {TARGET}; device={device}")
    model.learn(total_timesteps=TARGET, callback=ckpt_cb)

with open(os.path.join(RUN_FOLDER, "train_config.yaml"), "w") as f:
    yaml.safe_dump(copy.deepcopy(cfg), f)
final = os.path.join(RUN_FOLDER, "checkpoints", "1", f"checkpoint_{TARGET}_steps.zip")
if not os.path.exists(final):
    model.save(final); train_envs.save(os.path.join(RUN_FOLDER, "checkpoints", "1", f"checkpoint_vecnormalize_{TARGET}_steps.pkl"))
try: switch_env.envs_list[0]._drift_logger.close()
except Exception: pass
print(f"[{cfg['name']}] COMPLETE. final num_timesteps={model.num_timesteps}. checkpoints in {RUN_FOLDER}/checkpoints/1")
