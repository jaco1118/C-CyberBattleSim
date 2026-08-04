"""F4 convergence eval: load one run's checkpoint at <ckpt_step>, run N static episodes, measure mean
root_owned. Reconstructs the 'train/Root owned nodes' convergence signal by evaluating the actual
policy (the F4 resume runs emit no tensorboard). Appends one CSV row to <out_csv>.
Usage: python conv_eval.py <run_folder> <topo_subpath> <ckpt_step> <n_episodes> <out_csv> <band> <seed>
"""
import sys, os, logging, pickle
sys.path.insert(0, "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim")
os.chdir("/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents")
import numpy as np, torch, yaml, csv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import TRPO
from cyberbattle.utils.envs_utils import wrap_graphs_to_compressed_envs
from cyberbattle.gae.model import GAEEncoder
from cyberbattle._env.cyberbattle_env_switch import RandomSwitchEnv

RUN = os.path.abspath(sys.argv[1]); TOPO = sys.argv[2]; STEP = sys.argv[3]
N = int(sys.argv[4]); OUT = os.path.abspath(sys.argv[5]); BAND = sys.argv[6]; SEED = int(sys.argv[7])
torch.set_num_threads(4)
REPO = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim"
CKPT = os.path.join(RUN, "checkpoints", "1", f"checkpoint_{STEP}_steps.zip")
VECN = os.path.join(RUN, "checkpoints", "1", f"checkpoint_vecnormalize_{STEP}_steps.pkl")
cfg = yaml.safe_load(open(os.path.join(RUN, "train_config.yaml")))
ce = yaml.safe_load(open(cfg["graph_encoder_config_path"])); ce.update(yaml.safe_load(open(cfg["graph_encoder_spec_path"])))
ge = GAEEncoder(ce["node_feature_vector_size"], ce["model_config"]["layers"], ce["edge_feature_vector_size"])
ge.load_state_dict(torch.load(cfg["graph_encoder_path"])); ge.eval()
cfg["node_embeddings_dimensions"] = ce["model_config"]["layers"][-1]["out_channels"]
logger = logging.getLogger("conv"); logger.addHandler(logging.NullHandler())
ec = dict(cfg); ec["dynamic_mode"] = "none"; ec["patch_service_dynamic_enabled"] = False
ec["drift_logging"] = False
net = pickle.load(open(os.path.join(REPO, "cyberbattle/data/env_samples", TOPO, "network_SecureBERT.pkl"), "rb"))
env = wrap_graphs_to_compressed_envs(net, logger, **ec)
env.set_graph_encoder(ge); env.set_pca_components(ec["pca_components"])
switch = RandomSwitchEnv([0], switch_interval=10**9, envs_list=[env], save_to_csv=False, verbose=0)
tmp = DummyVecEnv([lambda: Monitor(RandomSwitchEnv([0], switch_interval=10**9, envs_list=[env], save_to_csv=False, verbose=0))])
vecn = VecNormalize.load(VECN, tmp); vecn.training = False; vecn.norm_reward = False
model = TRPO.load(CKPT, device="cpu")
np.random.seed(SEED); torch.manual_seed(SEED)
import random as _r; _r.seed(SEED)
def norm(o):
    b = {k: np.asarray(v, np.float32)[None, ...] for k, v in o.items()}
    n = vecn.normalize_obs(b); return {k: v[0] for k, v in n.items()}
obs, _ = switch.reset(); ep = 0; guard = 0; roots = []
while ep < N and guard < N * 6000 + 200000:
    guard += 1
    a, _ = model.predict(norm(obs), deterministic=False)
    obs, r, done, trunc, info = switch.step(a)
    if done or trunc:
        roots.append(int(switch.root_owned_nodes)); obs, _ = switch.reset(); ep += 1
roots = np.array(roots, float)
row = dict(band=BAND, seed=SEED, ckpt=int(STEP), n=len(roots), mean_root=roots.mean(), sd_root=roots.std(ddof=1))
newf = not os.path.exists(OUT)
with open(OUT, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys()))
    if newf: w.writeheader()
    w.writerow(row)
print(f"[{BAND} seed{SEED} ckpt{STEP}] n={len(roots)} mean_root_owned={roots.mean():.3f} sd={roots.std(ddof=1):.3f}")
