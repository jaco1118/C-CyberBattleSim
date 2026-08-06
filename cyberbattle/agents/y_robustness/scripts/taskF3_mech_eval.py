"""
Task F3 mechanical channel — measure the INSTANTANEOUS arithmetic score impact of each departure,
directly, resolving the reverse-causation confound that made the cross-episode OLS invalid.

Monkeypatches the env's remove_node_dynamic (instance attribute; no schema/behaviour change) to
record, per removed node, root_owned% just BEFORE and just AFTER that single removal:
  score = root_owned_count / ownable_count   (the control metric)
Also flags was_owned (agent_installed) and was_root (root-privilege, non-starter). Key arithmetic:
only a ROOT-owned departure lowers the ratio; un-owned or owned-non-root departures shrink the
denominator and RAISE it. Summing these per-episode gives the direct arithmetic (mechanical)
displacement, cleanly separated from the agent's behaviour.

Usage: python taskF3_mech_eval.py <run_folder> <seed> <topology_subpath> <n_episodes> <out_dir> <band>
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
from cyberbattle.simulation import model as M

RUN = os.path.abspath(sys.argv[1]); SEED = int(sys.argv[2]); TOPO = sys.argv[3]
N_EP = int(sys.argv[4]); OUT = os.path.abspath(sys.argv[5]); BAND = sys.argv[6]
CI = int(sys.argv[7]) if len(sys.argv) > 7 else None
os.makedirs(OUT, exist_ok=True); torch.set_num_threads(4)
REPO = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim"
DONOR = os.path.join(REPO, "cyberbattle/data/env_samples/join_donor_pool_20_topologies")
cfg = yaml.safe_load(open(os.path.join(RUN, "train_config.yaml")))
ce = yaml.safe_load(open(cfg["graph_encoder_config_path"])); ce.update(yaml.safe_load(open(cfg["graph_encoder_spec_path"])))
ge = GAEEncoder(ce["node_feature_vector_size"], ce["model_config"]["layers"], ce["edge_feature_vector_size"])
ge.load_state_dict(torch.load(cfg["graph_encoder_path"])); ge.eval()
cfg["node_embeddings_dimensions"] = ce["model_config"]["layers"][-1]["out_channels"]
lg = logging.getLogger("f3m"); lg.addHandler(logging.NullHandler())
c = dict(cfg); c["dynamic_mode"] = "both"; c["patch_service_dynamic_enabled"] = False; c["drift_logging"] = False
if CI is not None: c["change_interval"] = CI
net = pickle.load(open(os.path.join(REPO, "cyberbattle/data/env_samples", TOPO, "network_SecureBERT.pkl"), "rb"))
env = wrap_graphs_to_compressed_envs(net, lg, **c); env.set_graph_encoder(ge); env.set_pca_components(c["pca_components"])
pool = []
for sub in sorted(os.listdir(DONOR)):
    sp = os.path.join(DONOR, sub)
    if os.path.isdir(sp) and sub.isdigit():
        dn = pickle.load(open(os.path.join(sp, f"network_{c['nlp_extractor']}.pkl"), "rb"))
        pool += [(sub, nid, copy.deepcopy(nd["data"])) for nid, nd in dn.network.nodes(data=True)]
env.dynamic_join_donor_pool = pool

# --- monkeypatch remove_node_dynamic to record per-removal instantaneous score deltas ---
recs = []
cur_ep = [0]
_orig_remove = env.remove_node_dynamic
def _root_and_reach(e):
    root = sum(1 for n in e.owned_nodes if e.get_node(n).privilege_level == M.PrivilegeLevel.ROOT and n != e.starter_node)
    return root, e.ownable_count
def _wrapped_remove(node_id):
    e = env
    was_owned = int(node_id in e.owned_nodes)
    was_root = int(was_owned and e.get_node(node_id).privilege_level == M.PrivilegeLevel.ROOT and node_id != e.starter_node)
    rb, ab = _root_and_reach(e); sb = rb / max(ab, 1)
    _orig_remove(node_id)
    ra, aa = _root_and_reach(e); sa = ra / max(aa, 1)
    recs.append(dict(band=BAND, seed=SEED, episode=cur_ep[0], was_owned=was_owned, was_root=was_root,
                     score_before=sb, score_after=sa, delta=sa - sb))
env.remove_node_dynamic = _wrapped_remove

switch = RandomSwitchEnv([0], switch_interval=10**9, envs_list=[env], save_to_csv=False, verbose=0)
tmp = DummyVecEnv([lambda: Monitor(RandomSwitchEnv([0], switch_interval=10**9, envs_list=[env], save_to_csv=False, verbose=0))])
_CS = os.environ.get("CKPT_STEP", "250000")
vn = VecNormalize.load(os.path.join(RUN, f"checkpoints/1/checkpoint_vecnormalize_{_CS}_steps.pkl"), tmp)
vn.training = False; vn.norm_reward = False
mdl = TRPO.load(os.path.join(RUN, f"checkpoints/1/checkpoint_{_CS}_steps.zip"), device="cpu")
np.random.seed(SEED); torch.manual_seed(SEED)
import random as _r; _r.seed(SEED)
def norm(o):
    b = {k: np.asarray(v, np.float32)[None, ...] for k, v in o.items()}
    n = vn.normalize_obs(b); return {k: v[0] for k, v in n.items()}

obs, _ = switch.reset(); ep = 0; guard = 0; epscores = []
while ep < N_EP and guard < N_EP*8000 + 200000:
    guard += 1
    cur_ep[0] = int(switch.current_env._episode_count)
    a, _ = mdl.predict(norm(obs), deterministic=False)
    obs, r, done, trunc, info = switch.step(a)
    if done or trunc:
        epscores.append(dict(band=BAND, seed=SEED, episode=int(switch.current_env._episode_count),
                             score=int(switch.root_owned_nodes)/max(int(switch.reachable_count),1)))
        obs, _ = switch.reset(); ep += 1

pd.DataFrame(recs).to_csv(os.path.join(OUT, f"mech_{BAND}_seed{SEED}.csv"), index=False)
pd.DataFrame(epscores).to_csv(os.path.join(OUT, f"mechscore_{BAND}_seed{SEED}.csv"), index=False)
dd = pd.DataFrame(recs)
print(f"[{BAND} seed{SEED}] episodes={ep} removals={len(dd)} "
      f"owned_root={dd.was_root.mean():.3f} owned_any={dd.was_owned.mean():.3f} | "
      f"mean_delta owned_root={dd[dd.was_root==1].delta.mean():.5f} "
      f"unowned={dd[dd.was_owned==0].delta.mean():.5f} | "
      f"per-ep sum_delta(mech displacement)={dd.groupby('episode').delta.sum().mean():.5f}")
