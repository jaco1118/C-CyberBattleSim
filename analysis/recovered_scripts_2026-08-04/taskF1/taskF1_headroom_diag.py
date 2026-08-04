"""
Task F Pass 1, STEP 2 item-2 headroom diagnostic. For a given checkpoint, run N episodes under
STATIC eval and report the full terminal-state distribution (CHECK 1 capture method), not just
root_owned_percentage: root_owned count, owned-at-any-privilege count, discovered count, won.
This separates three cases: (i) capture bug returned (everything exactly 0, including discovered);
(ii) real floor -- the agent does things (discovers/owns) but rarely reaches ROOT, so the
root% metric has no headroom; (iii) healthy range in root%.

get_statistics() index map (cyberbattle_env.py:1053):
  0 owned_nodes(any priv, incl starter) 1 discovered 2 not_discovered 3 disrupted 4 num_nodes
  5 ownable(reachable) 6 discoverable 7 disruptable 8 net_avail 9 reimaged 10 events
  11 discovered_amount 12 discoverable_amount 13 goal_reached 14 root_owned
Read via switch_env cached terminal attributes (set in step() before reset).

Usage: python taskF1_headroom_diag.py <run_folder_abs> <label> <seed> <n_episodes>
"""
import sys, os, logging, pickle
sys.path.insert(0, "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim")
os.chdir("/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents")

import numpy as np, torch, yaml
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import TRPO
from cyberbattle.utils.envs_utils import wrap_graphs_to_compressed_envs
from cyberbattle.gae.model import GAEEncoder
from cyberbattle._env.cyberbattle_env_switch import RandomSwitchEnv

RUN_FOLDER = os.path.abspath(sys.argv[1]); LABEL = sys.argv[2]; SEED = int(sys.argv[3]); N = int(sys.argv[4])
DETERMINISTIC = (len(sys.argv) <= 5) or (sys.argv[5] != "stochastic")
torch.set_num_threads(4)
REPO = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim"
TOPOLOGY_PKL = os.path.join(REPO, "cyberbattle/data/env_samples/scalability_30_40/44/network_SecureBERT.pkl")
CKPT = os.path.join(RUN_FOLDER, "checkpoints/1/checkpoint_250000_steps.zip")
VECNORM = os.path.join(RUN_FOLDER, "checkpoints/1/checkpoint_vecnormalize_250000_steps.pkl")

with open(os.path.join(RUN_FOLDER, "train_config.yaml")) as f: cfg = yaml.safe_load(f)
with open(cfg["graph_encoder_config_path"]) as f: config_encoder = yaml.safe_load(f)
with open(cfg["graph_encoder_spec_path"]) as f: spec_encoder = yaml.safe_load(f)
config_encoder.update(spec_encoder)
graph_encoder = GAEEncoder(config_encoder["node_feature_vector_size"], config_encoder["model_config"]["layers"],
                           config_encoder["edge_feature_vector_size"])
graph_encoder.load_state_dict(torch.load(cfg["graph_encoder_path"])); graph_encoder.eval()
cfg["node_embeddings_dimensions"] = config_encoder["model_config"]["layers"][-1]["out_channels"]
logger = logging.getLogger("diag"); logger.addHandler(logging.NullHandler())

c = dict(cfg); c["dynamic_mode"] = "none"; c["patch_service_dynamic_enabled"] = False; c["drift_logging"] = False
with open(TOPOLOGY_PKL, "rb") as f: network = pickle.load(f)
env = wrap_graphs_to_compressed_envs(network, logger, **c)
env.set_graph_encoder(graph_encoder); env.set_pca_components(c["pca_components"])
switch_env = RandomSwitchEnv([0], switch_interval=10**9, envs_list=[env], save_to_csv=False, verbose=0)
tmp = DummyVecEnv([lambda: Monitor(RandomSwitchEnv([0], switch_interval=10**9, envs_list=[env], save_to_csv=False, verbose=0))])
vecnorm = VecNormalize.load(VECNORM, tmp); vecnorm.training = False; vecnorm.norm_reward = False
model = TRPO.load(CKPT, device="cpu")
np.random.seed(SEED); torch.manual_seed(SEED)
import random as _r; _r.seed(SEED)

def norm(o):
    b = {k: np.asarray(v, np.float32)[None, ...] for k, v in o.items()}
    n = vecnorm.normalize_obs(b); return {k: v[0] for k, v in n.items()}

obs, _ = switch_env.reset()
root, owned, disc, won = [], [], [], []
ep = 0; guard = 0
while ep < N and guard < N * 5000 + 100000:
    guard += 1
    a, _ = model.predict(norm(obs), deterministic=DETERMINISTIC)
    obs, r, done, trunc, info = switch_env.step(a)
    if done or trunc:
        st = switch_env.current_env.get_statistics() if False else None  # not used post-reset
        root.append(int(switch_env.root_owned_nodes))
        owned.append(int(switch_env.owned_nodes))          # any-privilege owned (incl starter)
        disc.append(int(switch_env.discovered_nodes))
        won.append(int(switch_env.episode_won))
        obs, _ = switch_env.reset(); ep += 1

root = np.array(root); owned = np.array(owned); disc = np.array(disc); won = np.array(won)
reach = int(switch_env.reachable_count)
def d(x): return f"min={x.min()} med={int(np.median(x))} mean={x.mean():.3f} max={x.max()}"
print(f"[{LABEL} seed{SEED} N={N} static-eval] reachable(ownable)={reach}")
print(f"  root_owned:      {d(root)}  frac_zero={(root==0).mean():.3f}")
print(f"  owned_any_priv:  {d(owned)}  frac_le1(only starter/none)={(owned<=1).mean():.3f}")
print(f"  discovered:      {d(disc)}  frac_zero={(disc==0).mean():.3f}")
print(f"  won_frac={won.mean():.3f}")
print(f"  root%_score: min={ (root/max(reach,1)).min():.4f} mean={(root/max(reach,1)).mean():.4f} max={(root/max(reach,1)).max():.4f}")
