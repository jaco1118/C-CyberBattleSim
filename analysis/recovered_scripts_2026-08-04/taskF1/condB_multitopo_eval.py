"""
JOB 1: Condition B (ExternalRandomEvents probabilistic defender) on the EXISTING multi-topology
gate checkpoints, band 30-40. Evaluation only, no training. Each gate agent trained across 8
topologies (grid_topologies_30-40/{1..8}); here it is evaluated across those same 8 (RandomSwitchEnv,
switch every 5 episodes, per the gate manifest), stochastically, with the defender at per-node
intervention probability pn. pn=0 attaches NO defender (undisturbed baseline).

Capture: F1-validated terminal method — read switch_env.root_owned_nodes / .reachable_count /
current_env._episode_count at the terminal step BEFORE reset. Never get_statistics() after done;
never join on the post-reset counter. Stochastic eval (deterministic=False).

Firewall defect (static_defender.py:136, outgoing branch tests .incoming) retained for fidelity.

Usage: python condB_multitopo_eval.py <gate_run_folder_abs> <seed> <pn> <n_episodes> <out_dir>
   pn=0.0 -> no defender (baseline).
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
from cyberbattle._env.static_defender import ExternalRandomEvents

RUN_FOLDER = os.path.abspath(sys.argv[1]); SEED = int(sys.argv[2]); PN = float(sys.argv[3])
N_EPISODES = int(sys.argv[4]); OUT_DIR = os.path.abspath(sys.argv[5])
os.makedirs(OUT_DIR, exist_ok=True)
torch.set_num_threads(3)
REPO = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim"
TOPO_DIR = os.path.join(REPO, "cyberbattle/data/env_samples/grid_topologies_30-40")
GRID_SLOTS = ["1","2","3","4","5","6","7","8"]  # the 8 topologies the gate trained on
SWITCH_EVERY = 5  # gate manifest switch_interval_episodes

CKPT = os.path.join(RUN_FOLDER, "checkpoints", "1", "checkpoint_250000_steps.zip")
VECNORM = os.path.join(RUN_FOLDER, "checkpoints", "1", "checkpoint_vecnormalize_250000_steps.pkl")
with open(os.path.join(RUN_FOLDER, "train_config.yaml")) as f: cfg = yaml.safe_load(f)
ce = yaml.safe_load(open(cfg["graph_encoder_config_path"])); ce.update(yaml.safe_load(open(cfg["graph_encoder_spec_path"])))
graph_encoder = GAEEncoder(ce["node_feature_vector_size"], ce["model_config"]["layers"], ce["edge_feature_vector_size"])
graph_encoder.load_state_dict(torch.load(cfg["graph_encoder_path"])); graph_encoder.eval()
cfg["node_embeddings_dimensions"] = ce["model_config"]["layers"][-1]["out_channels"]
logger = logging.getLogger("condB"); logger.addHandler(logging.NullHandler())

# Condition B: only the defender changes state -> dynamic_mode=none, patch off.
env_cfg = dict(cfg); env_cfg["dynamic_mode"] = "none"; env_cfg["patch_service_dynamic_enabled"] = False
env_cfg["drift_logging"] = False  # Condition B is score-vs-pn; defender changes aren't drift-instrumented

envs = []
for slot in GRID_SLOTS:
    net = pickle.load(open(os.path.join(TOPO_DIR, slot, f"network_{env_cfg['nlp_extractor']}.pkl"), "rb"))
    e = wrap_graphs_to_compressed_envs(net, logger, **env_cfg)
    e.set_graph_encoder(graph_encoder); e.set_pca_components(env_cfg["pca_components"])
    if PN > 0:
        e.static_defender_agent = ExternalRandomEvents(PN, logger=logger, verbose=0)
        e.static_defender_eviction_goal = False
    envs.append(e)

switch_env = RandomSwitchEnv(list(range(len(envs))), switch_interval=SWITCH_EVERY, envs_list=envs, save_to_csv=False, verbose=0)
tmp = DummyVecEnv([lambda: Monitor(RandomSwitchEnv(list(range(len(envs))), switch_interval=SWITCH_EVERY, envs_list=envs, save_to_csv=False, verbose=0))])
vecnorm = VecNormalize.load(VECNORM, tmp); vecnorm.training = False; vecnorm.norm_reward = False
model = TRPO.load(CKPT, device="cpu")

np.random.seed(SEED); torch.manual_seed(SEED)
import random as _r; _r.seed(SEED)
def norm(o):
    b = {k: np.asarray(v, np.float32)[None, ...] for k, v in o.items()}
    n = vecnorm.normalize_obs(b); return {k: v[0] for k, v in n.items()}

obs, _ = switch_env.reset()
rows = []; ep = 0; guard = 0
while ep < N_EPISODES and guard < N_EPISODES * 6000 + 100000:
    guard += 1
    a, _ = model.predict(norm(obs), deterministic=False)
    obs, r, done, trunc, info = switch_env.step(a)
    if done or trunc:
        root = int(switch_env.root_owned_nodes); reach = int(switch_env.reachable_count)
        disc = int(switch_env.discovered_nodes); won = int(bool(switch_env.episode_won))
        epn = int(switch_env.current_env._episode_count)
        slot = GRID_SLOTS[switch_env.current_env_index]
        rows.append(dict(seed=SEED, pn=PN, grid_slot=slot, episode=epn,
                         root_owned=root, reachable=reach, n_discovered=disc, won=won,
                         score=root / max(reach, 1)))
        obs, _ = switch_env.reset(); ep += 1

df = pd.DataFrame(rows)
out = os.path.join(OUT_DIR, f"condB_multitopo_seed{SEED}_pn{PN}.csv")
df.to_csv(out, index=False)
s = df["score"].to_numpy()
print(f"[gate seed{SEED} pn={PN}] episodes={len(df)} mean={s.mean():.4f} median={np.median(s):.4f} "
      f"min={s.min():.4f} max={s.max():.4f} frac0={(s==0).mean():.3f} -> {out}")
