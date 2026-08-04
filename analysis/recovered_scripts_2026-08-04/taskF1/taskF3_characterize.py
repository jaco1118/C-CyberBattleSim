"""
Task F3 STEP 0 (0.2, 0.3): characterise WHO actually leaves, per band. Instrumented stochastic
membership eval of an existing static agent (no training). For each membership_leave event logs the
departed node's degree (access_graph), value, was_owned, was_discovered (all departures are
discovered by the eligibility filter, so was_discovered is structurally 1). Also snapshots, at the
first leave of each episode, the degrees of the WHOLE eligible pool and of the whole network (0.3).

Capture uses a pre-step snapshot of owned/discovered/degree/value (nodes are pruned on leave), so
attributes are read from before removal. Frozen policy, stochastic, F1 terminal method for episode
bookkeeping (not needed for score here — this run is for departure characterisation only).

Usage: python taskF3_characterize.py <run_folder> <seed> <topology_subpath> <n_episodes> <out_dir> <band_label>
   topology_subpath e.g. scalability_30_40/44  or  scalability_80_100/5
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

RUN = os.path.abspath(sys.argv[1]); SEED = int(sys.argv[2]); TOPO = sys.argv[3]
N_EP = int(sys.argv[4]); OUT = os.path.abspath(sys.argv[5]); BAND = sys.argv[6]
CI = int(sys.argv[7]) if len(sys.argv) > 7 else None
os.makedirs(OUT, exist_ok=True); torch.set_num_threads(3)
REPO = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim"
DONOR = os.path.join(REPO, "cyberbattle/data/env_samples/join_donor_pool_20_topologies")
cfg = yaml.safe_load(open(os.path.join(RUN, "train_config.yaml")))
ce = yaml.safe_load(open(cfg["graph_encoder_config_path"])); ce.update(yaml.safe_load(open(cfg["graph_encoder_spec_path"])))
ge = GAEEncoder(ce["node_feature_vector_size"], ce["model_config"]["layers"], ce["edge_feature_vector_size"])
ge.load_state_dict(torch.load(cfg["graph_encoder_path"])); ge.eval()
cfg["node_embeddings_dimensions"] = ce["model_config"]["layers"][-1]["out_channels"]
lg = logging.getLogger("f3c"); lg.addHandler(logging.NullHandler())
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
switch = RandomSwitchEnv([0], switch_interval=10**9, envs_list=[env], save_to_csv=False, verbose=0)
tmp = DummyVecEnv([lambda: Monitor(RandomSwitchEnv([0], switch_interval=10**9, envs_list=[env], save_to_csv=False, verbose=0))])
vn = VecNormalize.load(os.path.join(RUN, "checkpoints/1/checkpoint_vecnormalize_250000_steps.pkl"), tmp)
vn.training = False; vn.norm_reward = False
model = TRPO.load(os.path.join(RUN, "checkpoints/1/checkpoint_250000_steps.zip"), device="cpu")
np.random.seed(SEED); torch.manual_seed(SEED)
import random as _r; _r.seed(SEED)
def norm(o):
    b = {k: np.asarray(v, np.float32)[None, ...] for k, v in o.items()}
    n = vn.normalize_obs(b); return {k: v[0] for k, v in n.items()}

# whole-network degree distribution (static; 0.3)
ag = env.access_graph
wholedeg = [ag.degree(n) if n in ag else 0 for n in env.environment.nodes()]
pd.DataFrame({"degree": wholedeg}).to_csv(os.path.join(OUT, f"wholenet_deg_{BAND}_seed{SEED}.csv"), index=False)

departed = []; eligpool = []
obs, _ = switch.reset(); ep = 0; guard = 0
while ep < N_EP and guard < N_EP*8000 + 200000:
    guard += 1
    e = switch.current_env
    owned_before = set(e.owned_nodes); disc_before = set(e.discovered_nodes)
    # snapshot degree/value for eligible nodes (pre-step)
    elig = e._get_removal_eligible_nodes()
    attr = {n: (ag.degree(n) if n in ag else 0, float(e.get_node(n).value)) for n in elig}
    first_leave_this_ep = [False]
    a, _ = model.predict(norm(obs), deterministic=False)
    obs, r, done, trunc, info = switch.step(a)
    for evn in e._last_dynamic_events:
        if evn["change_type"] == "membership_leave":
            if not first_leave_this_ep[0]:
                first_leave_this_ep[0] = True
                for n in elig:  # eligible-pool degree snapshot at first leave (0.3)
                    eligpool.append(dict(band=BAND, seed=SEED, degree=attr[n][0]))
            for nid in evn["node_ids"]:
                deg, val = attr.get(nid, (ag.degree(nid) if nid in ag else 0, float("nan")))
                departed.append(dict(band=BAND, seed=SEED, node_id=str(nid), degree=deg, value=val,
                                     was_owned=int(nid in owned_before), was_discovered=int(nid in disc_before)))
    if done or trunc:
        obs, _ = switch.reset(); ep += 1

dep = pd.DataFrame(departed); dep.to_csv(os.path.join(OUT, f"departed_{BAND}_seed{SEED}.csv"), index=False)
pd.DataFrame(eligpool).to_csv(os.path.join(OUT, f"eligpool_deg_{BAND}_seed{SEED}.csv"), index=False)
print(f"[{BAND} seed{SEED} topo={TOPO} ci={CI or 'nat'}] episodes={ep} leave_events={len(dep)} "
      f"was_owned_frac={dep.was_owned.mean():.3f} was_disc_frac={dep.was_discovered.mean():.3f} "
      f"departed_deg mean={dep.degree.mean():.2f} median={dep.degree.median():.1f} | "
      f"wholenet_deg mean={np.mean(wholedeg):.2f} median={np.median(wholedeg):.1f} | leave/ep={len(dep)/max(ep,1):.2f}")
