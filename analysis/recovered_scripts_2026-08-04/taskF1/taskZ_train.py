"""
Task Z STEP 1 training driver: the three-arm pooling ablation.
Mirrors taskF1_train.py / taskF2_train.py STATIC single-topology TRPO 250k EXACTLY (same tuned config
template, same encoder, same RandomSwitchEnv->DummyVecEnv(Monitor)->VecNormalize->TRPO wrapping, same
set_seeds, dynamic_mode=none, patch off). The ONLY difference between the three arms:

  ARM 1  full observation: graph_embeddings_aggregations = [mean, max, min]  -> 192 pooled + 64 next_esc = 256
  ARM 2  mean channel only: graph_embeddings_aggregations = [mean]           -> 64 pooled + 64 next_esc = 128
  ARM 3  full 256, but the 128 extremal dims (graph_embeddings[64:192] = max+min slots) are set to a fixed
         constant 0.0 AFTER VecNormalize (the Task-N/0.3 fix: substitute post-normalisation so the channel
         is BIT-EXACTLY constant and a valid control, not eps=1e-8-amplified noise).

Arm 1 is RETRAINED FRESH here (not reused from the reported F1/F2 runs) so STEP 1.2's reproduction check
tests THIS harness rather than comparing a fresh run against an old one.

Usage: python taskZ_train.py <seed> <topology_subpath> <arm 1|2|3> <run_folder> [steps_override]
  e.g. python taskZ_train.py 42 scalability_30_40/44 1 /path/z_arm1_30-40_seed42
"""
import sys, os, copy, logging, pickle
sys.path.insert(0, "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim")
os.chdir("/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents")
import numpy as np, torch, yaml
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecEnvWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import TRPO
from cyberbattle.utils.envs_utils import wrap_graphs_to_compressed_envs
from cyberbattle.utils.train_utils import replace_with_classes
from cyberbattle.utils.math_utils import set_seeds
from cyberbattle.gae.model import GAEEncoder
from cyberbattle._env.cyberbattle_env_switch import RandomSwitchEnv

EXTREMAL = slice(64, 192)   # max+min pooled slots; mean = 0:64, next_escalation = 192:256

class ExtremalMask(VecEnvWrapper):
    """Zero graph_embeddings[64:192] AFTER VecNormalize (arm 3). Same class must be used at eval."""
    def __init__(self, venv, sl=EXTREMAL, const=0.0):
        super().__init__(venv); self.sl = sl; self.const = const
    def reset(self):
        obs = self.venv.reset(); obs["graph_embeddings"][:, self.sl] = self.const; return obs
    def step_wait(self):
        obs, r, d, info = self.venv.step_wait(); obs["graph_embeddings"][:, self.sl] = self.const
        return obs, r, d, info

SEED = int(sys.argv[1]); TOPO = sys.argv[2]; ARM = int(sys.argv[3])
RUN_FOLDER = os.path.abspath(sys.argv[4])
STEPS_OVERRIDE = int(sys.argv[5]) if len(sys.argv) > 5 else None
assert ARM in (1, 2, 3); assert os.path.isabs(RUN_FOLDER)
torch.set_num_threads(4)
REPO = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim"
GATE_TRAIN_CONFIG = os.path.join(REPO, "cyberbattle/agents/logs",
    "trpo_250k_tuned_compressed_band30-40_seed42_2026-07-26_12-47-19", "TRPO_x_control_SecureBERT", "train_config.yaml")
TOPOLOGY_PKL = os.path.join(REPO, "cyberbattle/data/env_samples", TOPO, "network_SecureBERT.pkl")
os.makedirs(os.path.join(RUN_FOLDER, "checkpoints", "1"), exist_ok=True)

cfg = yaml.safe_load(open(GATE_TRAIN_CONFIG))
if STEPS_OVERRIDE is not None:
    cfg["train_iterations"] = STEPS_OVERRIDE; cfg["checkpoints_save_freq"] = max(50, STEPS_OVERRIDE // 2)
cfg["dynamic_mode"] = "none"; cfg["patch_service_dynamic_enabled"] = False   # STATIC only, exactly as F1/F2
cfg["name"] = f"trpo_Z_arm{ARM}_{TOPO.replace('/', '_')}_seed{SEED}"
cfg["seeds_runs"] = [SEED]
cfg["drift_logging"] = False   # pure instrumentation; OFF here (does not affect the training RNG stream)
if ARM == 2:
    cfg["graph_embeddings_aggregations"] = ["mean"]   # the one config change for arm 2

logger = logging.getLogger("taskZ_train"); logger.addHandler(logging.NullHandler())
ce = yaml.safe_load(open(cfg["graph_encoder_config_path"])); ce.update(yaml.safe_load(open(cfg["graph_encoder_spec_path"])))
ge = GAEEncoder(ce["node_feature_vector_size"], ce["model_config"]["layers"], ce["edge_feature_vector_size"])
ge.load_state_dict(torch.load(cfg["graph_encoder_path"])); ge.eval()
cfg["node_embeddings_dimensions"] = ce["model_config"]["layers"][-1]["out_channels"]
set_seeds(SEED)

# optional resume (argv6=ckpt, argv7=vecnorm): continue the SAME arm on the SAME topology to TARGET.
RESUME_CKPT = os.path.abspath(sys.argv[6]) if len(sys.argv) > 6 else None
RESUME_VN = os.path.abspath(sys.argv[7]) if len(sys.argv) > 7 else None
net = pickle.load(open(TOPOLOGY_PKL, "rb"))
env = wrap_graphs_to_compressed_envs(net, logger, **cfg)
env.set_graph_encoder(ge); env.set_pca_components(cfg["pca_components"])
switch_env = RandomSwitchEnv([0], switch_interval=10**9, envs_list=[env], save_to_csv=False, verbose=0)
base_vec = DummyVecEnv([lambda: Monitor(switch_env)])
algo = copy.deepcopy(cfg["algorithm_hyperparams"]); gamma = algo.get("gamma", 0.99)
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_cb = CheckpointCallback(save_freq=cfg["checkpoints_save_freq"],
    save_path=os.path.join(RUN_FOLDER, "checkpoints", "1"), name_prefix="checkpoint", save_vecnormalize=True)
with open(os.path.join(RUN_FOLDER, "train_config.yaml"), "w") as f:
    yaml.safe_dump(copy.deepcopy(cfg), f)

if RESUME_CKPT:
    train_envs = VecNormalize.load(RESUME_VN, base_vec)
    train_envs.training = True; train_envs.norm_reward = cfg["norm_reward"]
    if ARM == 3:
        train_envs = ExtremalMask(train_envs)   # re-wrap AFTER the loaded VecNormalize, same as training
    model = TRPO.load(RESUME_CKPT, device=device); model.set_env(train_envs)
    remaining = cfg["train_iterations"] - model.num_timesteps
    print(f"[{cfg['name']}] ARM {ARM} RESUME topo={TOPO} from {model.num_timesteps} -> {cfg['train_iterations']} (+{remaining}); mask={'yes' if ARM==3 else 'no'}")
    model.learn(total_timesteps=remaining, callback=ckpt_cb, reset_num_timesteps=False)
else:
    train_envs = VecNormalize(base_vec, norm_obs=cfg["norm_obs"], norm_reward=cfg["norm_reward"], gamma=gamma)
    if ARM == 3:
        train_envs = ExtremalMask(train_envs)   # AFTER VecNormalize
    if algo.get("learning_rate_type") == "linear":
        from cyberbattle.utils.math_utils import linear_schedule
        lr = linear_schedule(algo["learning_rate"], algo["learning_rate_final"])
    else:
        lr = algo["learning_rate"]
    [algo.pop(k, None) for k in ("learning_rate_type", "learning_rate", "learning_rate_final")]
    pk = replace_with_classes(copy.deepcopy(cfg["policy_kwargs"]))
    for k in ("lstm_hidden_size", "n_lstm_layers"): pk.pop(k, None)
    model = TRPO("MultiInputPolicy", train_envs, policy_kwargs=pk, learning_rate=lr, **algo, verbose=0, device=device)
    print(f"[{cfg['name']}] ARM {ARM} device={device} topo={TOPO} -> {cfg['train_iterations']} steps "
          f"(aggregations={cfg['graph_embeddings_aggregations']}, mask={'yes' if ARM==3 else 'no'})")
    model.learn(total_timesteps=cfg["train_iterations"], callback=ckpt_cb)

final = os.path.join(RUN_FOLDER, "checkpoints", "1", f"checkpoint_{cfg['train_iterations']}_steps.zip")
if not os.path.exists(final):
    model.save(final); train_envs.save(os.path.join(RUN_FOLDER, "checkpoints", "1", f"checkpoint_vecnormalize_{cfg['train_iterations']}_steps.pkl"))
print(f"[{cfg['name']}] COMPLETE. final num_timesteps={model.num_timesteps}. checkpoints in {RUN_FOLDER}/checkpoints/1")
