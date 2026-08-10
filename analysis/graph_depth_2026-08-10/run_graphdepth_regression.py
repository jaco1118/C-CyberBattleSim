"""Task GRAPH-DEPTH STEP 2 regression: run the drift_logging=True path with a FIXED action
sequence and seeded env RNG, so any divergence between OLD (pre-GRAPH-DEPTH) and NEW code is a
real behavioural/RNG change -- not a stochastic action-selection artefact. Writes the drift CSV +
a trajectory npz (per-step returned obs + reward + done). tag in {old, new_off, new_on}.

Modelled directly on the already-proven Task L STEP 2 regression
(analysis/recovered_scripts_2026-08-04/taskF1/run_L_regression.py, commit c05a16a on
attenuation-pooling-scale) -- same fixed action sequence convention, same seeded env RNG, same
drift_logging=True baseline for every tag. The only substantive difference: tag 'new_on' sets
leave_embedding_logging=True (this task's new flag) instead of event_graph_logging=True.
"""
import sys, os, logging, pickle, yaml, random
sys.path.insert(0, "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim")
os.chdir("/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents")
import numpy as np, torch
from cyberbattle.utils.envs_utils import wrap_graphs_to_compressed_envs
from cyberbattle.gae.model import GAEEncoder
from cyberbattle._env.cyberbattle_env_switch import RandomSwitchEnv

TAG = sys.argv[1]; SEED = int(sys.argv[2]); STEPS = int(sys.argv[3]); BAND = sys.argv[4]; OUT = os.path.abspath(sys.argv[5])
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))

REPO = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim"
REF = "/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1/runs/trpo_250k_F1_static_seed42"
TOPO = {"30-40": "scalability_30_40/44", "80-100": "scalability_80_100/5"}[BAND]

cfg = yaml.safe_load(open(os.path.join(REF, "train_config.yaml")))
ce = yaml.safe_load(open(cfg["graph_encoder_config_path"])); ce.update(yaml.safe_load(open(cfg["graph_encoder_spec_path"])))
ge = GAEEncoder(ce["node_feature_vector_size"], ce["model_config"]["layers"], ce["edge_feature_vector_size"])
ge.load_state_dict(torch.load(cfg["graph_encoder_path"])); ge.eval()
cfg["node_embeddings_dimensions"] = ce["model_config"]["layers"][-1]["out_channels"]
lg = logging.getLogger("graphdepth_reg"); lg.addHandler(logging.NullHandler())

c = dict(cfg)
c["dynamic_mode"] = "both"; c["patch_service_dynamic_enabled"] = True; c["change_type"] = "mixed"; c["change_interval"] = 20
c["drift_logging"] = True; c["drift_log_path"] = os.path.join(OUT, f"drift_{TAG}.csv"); c["drift_sample_rate"] = 1
c["drift_run_id"] = TAG; c["drift_seed"] = SEED; c["drift_scenario_id"] = BAND
if TAG == "new_on":
    c["leave_embedding_logging"] = True
    c["leave_embedding_log_dir"] = os.path.join(OUT, f"le_{TAG}")
# NOTE: for tag 'old' the compressed env predates this task's kwargs entirely; never set them there.

net = pickle.load(open(os.path.join(REPO, "cyberbattle/data/env_samples", TOPO, "network_SecureBERT.pkl"), "rb"))
env = wrap_graphs_to_compressed_envs(net, lg, **c)
env.set_graph_encoder(ge); env.set_pca_components(c["pca_components"])

import copy as _cp
DONOR = os.path.join(REPO, "cyberbattle/data/env_samples/join_donor_pool_20_topologies")
pool = []
for sub in sorted(os.listdir(DONOR)):
    sp = os.path.join(DONOR, sub)
    if os.path.isdir(sp) and sub.isdigit():
        dn = pickle.load(open(os.path.join(sp, f"network_{c['nlp_extractor']}.pkl"), "rb"))
        pool += [(sub, nid, _cp.deepcopy(nd["data"])) for nid, nd in dn.network.nodes(data=True)]
env.dynamic_join_donor_pool = pool

sw = RandomSwitchEnv([0], switch_interval=10**9, envs_list=[env], save_to_csv=False, verbose=0)
np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)
ag = np.random.RandomState(12345)
adim = env.action_space.shape[0]
actions = ag.uniform(-4, 4, size=(STEPS, adim)).astype(np.float32)
obs, _ = sw.reset()
rew = []; obs_g = []; dones = []
for i in range(STEPS):
    o, r, d, t, info = sw.step(actions[i])
    rew.append(float(r)); dones.append(int(bool(d or t)))
    obs_g.append(np.asarray(o["graph_embeddings"], np.float64).copy())
    if d or t:
        obs, _ = sw.reset()
env._drift_logger.close()
if getattr(env, "_leave_embed_f", None) is not None:
    env._leave_embed_f.close()
np.savez(os.path.join(OUT, f"traj_{TAG}.npz"), reward=np.array(rew), done=np.array(dones), obs=np.array(obs_g))
print(f"[{TAG}] steps={STEPS} band={BAND} wrote drift_{TAG}.csv + traj_{TAG}.npz")
