"""Task H 0.4: measure training steps/sec at a given scenario size (fresh TRPO, same hyperparams as
F4). Reports fps over a short run after warmup. Calibrate the method against the known 30-40=249 /
80-100=89 (tensorboard time/fps) by also running those.
Usage: python throughput_test.py <scenario_pkl> <label> <n_steps>"""
import sys, os, time, logging, pickle, copy
sys.path.insert(0, "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim")
os.chdir("/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents")
import numpy as np, torch, yaml
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import TRPO
from cyberbattle.utils.envs_utils import wrap_graphs_to_compressed_envs
from cyberbattle.utils.train_utils import replace_with_classes
from cyberbattle.utils.math_utils import set_seeds
from cyberbattle.gae.model import GAEEncoder
from cyberbattle._env.cyberbattle_env_switch import RandomSwitchEnv

PKL = sys.argv[1]; LABEL = sys.argv[2]; NSTEPS = int(sys.argv[3])
torch.set_num_threads(4)
REF = "/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1/runs/trpo_250k_F1_static_seed42/train_config.yaml"
cfg = yaml.safe_load(open(REF))
cfg["dynamic_mode"] = "none"; cfg["patch_service_dynamic_enabled"] = False
ce = yaml.safe_load(open(cfg["graph_encoder_config_path"])); ce.update(yaml.safe_load(open(cfg["graph_encoder_spec_path"])))
ge = GAEEncoder(ce["node_feature_vector_size"], ce["model_config"]["layers"], ce["edge_feature_vector_size"])
ge.load_state_dict(torch.load(cfg["graph_encoder_path"])); ge.eval()
cfg["node_embeddings_dimensions"] = ce["model_config"]["layers"][-1]["out_channels"]
logger = logging.getLogger("thr"); logger.addHandler(logging.NullHandler())
set_seeds(42)
net = pickle.load(open(PKL, "rb"))
env = wrap_graphs_to_compressed_envs(net, logger, **cfg)
env.set_graph_encoder(ge); env.set_pca_components(cfg["pca_components"])
switch = RandomSwitchEnv([0], switch_interval=10**9, envs_list=[env], save_to_csv=False, verbose=0)
base = DummyVecEnv([lambda: Monitor(switch)])
gamma = cfg["algorithm_hyperparams"].get("gamma", 0.99)
train_envs = VecNormalize(base, norm_obs=cfg["norm_obs"], norm_reward=cfg["norm_reward"], gamma=gamma)
algo = copy.deepcopy(cfg["algorithm_hyperparams"])
lr = algo["learning_rate"]; [algo.pop(k, None) for k in ("learning_rate", "learning_rate_type", "learning_rate_final")]
pk = replace_with_classes(copy.deepcopy(cfg["policy_kwargs"]))
for k in ("lstm_hidden_size", "n_lstm_layers"): pk.pop(k, None)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = TRPO("MultiInputPolicy", train_envs, policy_kwargs=pk, learning_rate=lr, **algo, verbose=0, device=dev)
n_nodes = net.network.number_of_nodes()
# warmup (~one rollout) then timed run
model.learn(total_timesteps=max(1024, NSTEPS // 4))
t0 = time.time(); model.learn(total_timesteps=NSTEPS, reset_num_timesteps=False); dt = time.time() - t0
fps = NSTEPS / dt
print(f"[{LABEL}] n_nodes={n_nodes} device={dev} timed_steps={NSTEPS} wall={dt:.1f}s -> fps={fps:.1f}")
