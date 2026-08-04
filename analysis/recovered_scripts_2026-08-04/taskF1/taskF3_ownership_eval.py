"""
Task F2 STEP 2 evaluation: frozen-policy stochastic eval of one F2 static agent on ITS training
topology (scalability_80_100/<topology_id>), one eval condition. F1-validated terminal capture.

Eval conditions:
  static             dynamic_mode=none, patch off  (baseline)
  membership         dynamic_mode=both, patch off, change_interval=natural (20)
  membership_matched dynamic_mode=both, patch off, change_interval=<override> (calibrated to ~11.23 leave/ep)
  property           dynamic_mode=none, patch on

For 3.6 (mechanical vs behavioural loss): for each membership_leave event, record whether the
departing node was ALREADY OWNED at the moment the change fired. Captured WITHOUT touching the
drift schema: snapshot current_env.owned_nodes BEFORE the step, then after the step check each
leave node_id against that snapshot. Written to a separate side CSV.

Usage:
 python taskF2_eval.py <run_folder> <seed> <topology_id> <eval_cond> <n_episodes> <out_dir> [change_interval]
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

RUN_FOLDER = os.path.abspath(sys.argv[1]); SEED = int(sys.argv[2]); TOPOLOGY_ID = sys.argv[3]
EVAL_COND = sys.argv[4]; N_EPISODES = int(sys.argv[5]); OUT_DIR = os.path.abspath(sys.argv[6])
CHANGE_INTERVAL_OVERRIDE = int(sys.argv[7]) if len(sys.argv) > 7 else None
assert EVAL_COND in ("static", "membership", "membership_matched", "property")
os.makedirs(OUT_DIR, exist_ok=True)
torch.set_num_threads(4)
REPO = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim"
_topo_sub = TOPOLOGY_ID if "/" in TOPOLOGY_ID else f"scalability_80_100/{TOPOLOGY_ID}"
TOPOLOGY_PKL = os.path.join(REPO, f"cyberbattle/data/env_samples/{_topo_sub}/network_SecureBERT.pkl")
DONOR_FOLDER = os.path.join(REPO, "cyberbattle/data/env_samples/join_donor_pool_20_topologies")
_CS = os.environ.get("CKPT_STEP", "250000")
CKPT = os.path.join(RUN_FOLDER, "checkpoints", "1", f"checkpoint_{_CS}_steps.zip")
VECNORM = os.path.join(RUN_FOLDER, "checkpoints", "1", f"checkpoint_vecnormalize_{_CS}_steps.pkl")

with open(os.path.join(RUN_FOLDER, "train_config.yaml")) as f: cfg = yaml.safe_load(f)
ce = yaml.safe_load(open(cfg["graph_encoder_config_path"])); ce.update(yaml.safe_load(open(cfg["graph_encoder_spec_path"])))
graph_encoder = GAEEncoder(ce["node_feature_vector_size"], ce["model_config"]["layers"], ce["edge_feature_vector_size"])
graph_encoder.load_state_dict(torch.load(cfg["graph_encoder_path"])); graph_encoder.eval()
cfg["node_embeddings_dimensions"] = ce["model_config"]["layers"][-1]["out_channels"]
logger = logging.getLogger("taskF2_eval"); logger.addHandler(logging.NullHandler())

DYN = {"static": ("none", False), "membership": ("both", False),
       "membership_matched": ("both", False), "property": ("none", True)}
dyn_mode, patch_on = DYN[EVAL_COND]

_suffix = EVAL_COND
DRIFT_CSV = os.path.join(OUT_DIR, f"drift_static_seed{SEED}_eval{_suffix}.csv")
SCORE_CSV = os.path.join(OUT_DIR, f"score_static_seed{SEED}_eval{_suffix}.csv")
LEAVE_CSV = os.path.join(OUT_DIR, f"leaveown_static_seed{SEED}_eval{_suffix}.csv")
if os.path.exists(DRIFT_CSV): os.remove(DRIFT_CSV)

env_cfg = dict(cfg)
env_cfg["dynamic_mode"] = dyn_mode
env_cfg["patch_service_dynamic_enabled"] = patch_on
if CHANGE_INTERVAL_OVERRIDE is not None:
    env_cfg["change_interval"] = CHANGE_INTERVAL_OVERRIDE
env_cfg["drift_logging"] = True
env_cfg["drift_log_path"] = DRIFT_CSV
env_cfg["drift_sample_rate"] = 1
env_cfg["drift_run_id"] = f"F2_static_seed{SEED}_eval{EVAL_COND}"
env_cfg["drift_seed"] = SEED
env_cfg["drift_scenario_id"] = _topo_sub.replace("/", "_")

net = pickle.load(open(TOPOLOGY_PKL, "rb"))
env = wrap_graphs_to_compressed_envs(net, logger, **env_cfg)
env.set_graph_encoder(graph_encoder); env.set_pca_components(env_cfg["pca_components"])
if dyn_mode in ("join", "both"):
    # donor folder (join_donor_pool_20_topologies, ids 1..20) is a DISJOINT topology set from the
    # scalability_80_100 training topology, so no self-donation exclusion is needed -- include all 20.
    pool = []
    for sub in sorted(os.listdir(DONOR_FOLDER)):
        sp = os.path.join(DONOR_FOLDER, sub)
        if os.path.isdir(sp) and sub.isdigit():
            dn = pickle.load(open(os.path.join(sp, f"network_{env_cfg['nlp_extractor']}.pkl"), "rb"))
            pool += [(sub, nid, copy.deepcopy(nd["data"])) for nid, nd in dn.network.nodes(data=True)]
    env.dynamic_join_donor_pool = pool

switch_env = RandomSwitchEnv([0], switch_interval=10**9, envs_list=[env], save_to_csv=False, verbose=0)
tmp = DummyVecEnv([lambda: Monitor(RandomSwitchEnv([0], switch_interval=10**9, envs_list=[env], save_to_csv=False, verbose=0))])
vecnorm = VecNormalize.load(VECNORM, tmp); vecnorm.training = False; vecnorm.norm_reward = False
model = TRPO.load(CKPT, device="cpu")
np.random.seed(SEED); torch.manual_seed(SEED)
import random as _r; _r.seed(SEED)
def norm(o):
    b = {k: np.asarray(v, np.float32)[None, ...] for k, v in o.items()}
    n = vecnorm.normalize_obs(b); return {k: v[0] for k, v in n.items()}

is_membership = EVAL_COND in ("membership", "membership_matched")
obs, _ = switch_env.reset()
rows = []; leave_rows = []; ep = 0; guard = 0
while ep < N_EPISODES and guard < N_EPISODES * 6000 + 200000:
    guard += 1
    owned_before = set(switch_env.current_env.owned_nodes) if is_membership else None
    ep_num_now = int(switch_env.current_env._episode_count)
    a, _ = model.predict(norm(obs), deterministic=False)
    obs, r, done, trunc, info = switch_env.step(a)
    if is_membership:
        for evn in switch_env.current_env._last_dynamic_events:
            if evn["change_type"] == "membership_leave":
                for nid in evn["node_ids"]:
                    leave_rows.append(dict(seed=SEED, episode=ep_num_now, node_id=str(nid),
                                           was_owned=int(nid in owned_before)))
    if done or trunc:
        root = int(switch_env.root_owned_nodes); reach = int(switch_env.reachable_count)
        disc = int(switch_env.discovered_nodes); won = int(bool(switch_env.episode_won))
        epn = int(switch_env.current_env._episode_count)
        rows.append(dict(agent_cond="static", seed=SEED, eval_cond=EVAL_COND,
                         scenario_id=_topo_sub.replace("/", "_"), topology_id=TOPOLOGY_ID,
                         episode=epn, root_owned=root, reachable=reach, n_discovered=disc,
                         won=won, score=root / max(reach, 1)))
        obs, _ = switch_env.reset(); ep += 1

switch_env.current_env._drift_logger.close()
pd.DataFrame(rows).to_csv(SCORE_CSV, index=False)
if is_membership:
    pd.DataFrame(leave_rows).to_csv(LEAVE_CSV, index=False)
s = np.array([x["score"] for x in rows])
extra = ""
if is_membership and leave_rows:
    lo = pd.DataFrame(leave_rows); extra = f" leave_events={len(lo)} was_owned_frac={lo.was_owned.mean():.3f} leave/ep={len(lo)/len(rows):.2f}"
print(f"[F2 static seed{SEED} topo{TOPOLOGY_ID} eval={EVAL_COND}"
      f"{'' if CHANGE_INTERVAL_OVERRIDE is None else ' ci='+str(CHANGE_INTERVAL_OVERRIDE)}] "
      f"episodes={len(rows)} mean={s.mean():.4f} median={np.median(s):.4f} frac0={(s==0).mean():.3f}"
      f" max={s.max():.4f}{extra} -> {SCORE_CSV}")
