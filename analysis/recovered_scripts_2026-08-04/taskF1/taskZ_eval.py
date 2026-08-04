"""
Task Z eval: frozen-policy eval of one trained arm on its own topology, recording the root-owned COUNT
(STEP 2.1 primary metric) via the CHECK-1 terminal-capture method (read switch_env.root_owned_nodes /
reachable_count / discovered_nodes BEFORE our own reset). Stochastic (deterministic=False), matching the
reported F1/F2 eval convention exactly. Reads the run's own train_config.yaml, so arm 2's [mean] aggregation
(128-dim obs) is picked up automatically and is self-consistent with its saved VecNormalize + policy.

ARM 3: the 128 extremal dims graph_embeddings[64:192] are zeroed AFTER vecnorm.normalize_obs (the same
post-normalisation mask used at training), so the frozen policy sees the same masked input distribution.

Usage: python taskZ_eval.py <run_folder> <topology_subpath> <arm 1|2|3> <seed> <eval_cond> <n_episodes> <out_dir> [change_interval]
  eval_cond: static | membership
"""
import sys, os, copy, logging, pickle
sys.path.insert(0, "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim")
os.chdir("/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents")
import numpy as np, torch, yaml, pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import TRPO
from cyberbattle.utils.envs_utils import wrap_graphs_to_compressed_envs
from cyberbattle.gae.model import GAEEncoder
from cyberbattle._env.cyberbattle_env_switch import RandomSwitchEnv

EXTREMAL = slice(64, 192)
RUN = os.path.abspath(sys.argv[1]); TOPO = sys.argv[2]; ARM = int(sys.argv[3]); SEED = int(sys.argv[4])
EVAL_COND = sys.argv[5]; N_EP = int(sys.argv[6]); OUT = os.path.abspath(sys.argv[7])
CI = int(sys.argv[8]) if len(sys.argv) > 8 else None
assert ARM in (1, 2, 3); assert EVAL_COND in ("static", "membership")
os.makedirs(OUT, exist_ok=True); torch.set_num_threads(4)
REPO = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim"
DONOR = os.path.join(REPO, "cyberbattle/data/env_samples/join_donor_pool_20_topologies")
_CS = os.environ.get("CKPT_STEP", "250000")
cfg = yaml.safe_load(open(os.path.join(RUN, "train_config.yaml")))
ce = yaml.safe_load(open(cfg["graph_encoder_config_path"])); ce.update(yaml.safe_load(open(cfg["graph_encoder_spec_path"])))
ge = GAEEncoder(ce["node_feature_vector_size"], ce["model_config"]["layers"], ce["edge_feature_vector_size"])
ge.load_state_dict(torch.load(cfg["graph_encoder_path"])); ge.eval()
cfg["node_embeddings_dimensions"] = ce["model_config"]["layers"][-1]["out_channels"]
lg = logging.getLogger("zeval"); lg.addHandler(logging.NullHandler())
DYN = {"static": ("none", False), "membership": ("both", False)}
dyn_mode, patch_on = DYN[EVAL_COND]
c = dict(cfg); c["dynamic_mode"] = dyn_mode; c["patch_service_dynamic_enabled"] = patch_on; c["drift_logging"] = False
if CI is not None: c["change_interval"] = CI
net = pickle.load(open(os.path.join(REPO, "cyberbattle/data/env_samples", TOPO, "network_SecureBERT.pkl"), "rb"))
env = wrap_graphs_to_compressed_envs(net, lg, **c); env.set_graph_encoder(ge); env.set_pca_components(c["pca_components"])
if dyn_mode in ("join", "both"):
    pool = []
    for sub in sorted(os.listdir(DONOR)):
        sp = os.path.join(DONOR, sub)
        if os.path.isdir(sp) and sub.isdigit():
            dn = pickle.load(open(os.path.join(sp, f"network_{c['nlp_extractor']}.pkl"), "rb"))
            pool += [(sub, nid, copy.deepcopy(nd["data"])) for nid, nd in dn.network.nodes(data=True)]
    env.dynamic_join_donor_pool = pool
switch = RandomSwitchEnv([0], switch_interval=10**9, envs_list=[env], save_to_csv=False, verbose=0)
tmp = DummyVecEnv([lambda: Monitor(RandomSwitchEnv([0], switch_interval=10**9, envs_list=[env], save_to_csv=False, verbose=0))])
vn = VecNormalize.load(os.path.join(RUN, f"checkpoints/1/checkpoint_vecnormalize_{_CS}_steps.pkl"), tmp)
vn.training = False; vn.norm_reward = False
mdl = TRPO.load(os.path.join(RUN, f"checkpoints/1/checkpoint_{_CS}_steps.zip"), device="cpu")
np.random.seed(SEED); torch.manual_seed(SEED)
import random as _r; _r.seed(SEED)
def norm(o):
    b = {k: np.asarray(v, np.float32)[None, ...] for k, v in o.items()}
    n = vn.normalize_obs(b); out = {k: v[0] for k, v in n.items()}
    if ARM == 3: out["graph_embeddings"][EXTREMAL] = 0.0
    return out
obs, _ = switch.reset(); rows = []; ep = 0; guard = 0
while ep < N_EP and guard < N_EP * 5000 + 100000:
    guard += 1
    a, _ = mdl.predict(norm(obs), deterministic=False)
    obs, r, done, trunc, info = switch.step(a)
    if done or trunc:
        rows.append(dict(arm=ARM, band_topo=TOPO.replace("/", "_"), seed=SEED, eval_cond=EVAL_COND,
                         episode=int(switch.current_env._episode_count),
                         root_owned=int(switch.root_owned_nodes), reachable=int(switch.reachable_count),
                         n_discovered=int(switch.discovered_nodes), won=int(bool(switch.episode_won)),
                         score=int(switch.root_owned_nodes) / max(int(switch.reachable_count), 1)))
        obs, _ = switch.reset(); ep += 1
df = pd.DataFrame(rows)
out_csv = os.path.join(OUT, f"zscore_arm{ARM}_{TOPO.replace('/', '_')}_seed{SEED}_{EVAL_COND}.csv")
df.to_csv(out_csv, index=False)
print(f"[arm{ARM} {TOPO} seed{SEED} {EVAL_COND}] episodes={len(df)} root_owned mean={df.root_owned.mean():.3f} "
      f"median={df.root_owned.median():.1f} won={df.won.mean():.3f} -> {out_csv}")
