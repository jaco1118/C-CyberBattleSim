"""Reproduce the D3 substitution empty-action-space crash with diagnostics."""
import sys, os, copy, logging, pickle
sys.path.insert(0, "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim")
os.chdir("/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents")
import numpy as np, torch, yaml
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import TRPO
from cyberbattle.utils.envs_utils import wrap_graphs_to_compressed_envs
from cyberbattle.gae.model import GAEEncoder
from cyberbattle._env.cyberbattle_env_switch import RandomSwitchEnv

REPO = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim"
RUN = "/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1/runs/trpo_250k_F1_static_seed42"
SEED = 42; TOPO = "scalability_30_40/44"
torch.set_num_threads(4)
CKPT = os.path.join(RUN, "checkpoints", "1", "checkpoint_250000_steps.zip")
VECN = os.path.join(RUN, "checkpoints", "1", "checkpoint_vecnormalize_250000_steps.pkl")
cfg = yaml.safe_load(open(os.path.join(RUN, "train_config.yaml")))
ce = yaml.safe_load(open(cfg["graph_encoder_config_path"])); ce.update(yaml.safe_load(open(cfg["graph_encoder_spec_path"])))
ge = GAEEncoder(ce["node_feature_vector_size"], ce["model_config"]["layers"], ce["edge_feature_vector_size"])
ge.load_state_dict(torch.load(cfg["graph_encoder_path"])); ge.eval()
cfg["node_embeddings_dimensions"] = ce["model_config"]["layers"][-1]["out_channels"]
logger = logging.getLogger("repro"); logger.addHandler(logging.NullHandler())
ec = dict(cfg); ec["dynamic_mode"] = "none"; ec["patch_service_dynamic_enabled"] = True
ec["change_type"] = "substitute"; ec["change_interval"] = 20
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

obs, _ = switch.reset(); ep = 0; guard = 0; last_sub = None
while ep < 200 and guard < 2_000_000:
    guard += 1
    cur = switch.current_env
    na = len(cur.action_embeddings)
    if na == 0:
        print(f"EMPTY action_embeddings at guard={guard} ep={ep}")
        print(f"  owned={len(cur.owned_nodes)} discovered={len(cur.discovered_nodes)} "
              f"running_owned={sum(1 for n in cur.owned_nodes if cur.get_node(n).status.name=='Running')} "
              f"running_disc={sum(1 for n in cur.discovered_nodes if cur.get_node(n).status.name=='Running')}")
        print(f"  processed_pairs={len(cur.processed_pairs)}")
        print(f"  last_substitution={last_sub}")
        # per-node actionable catalogue sizes for owned & discovered
        for n in list(cur.owned_nodes)[:8]:
            pt = cur.vulnerabilities_embeddings_per_node_type.get(n, {"local":[],"remote":[]})
            print(f"    owned {n}: vulns={len(cur.get_node(n).vulnerabilities)} actionable_local={len(pt['local'])} actionable_remote={len(pt['remote'])} running={cur.get_node(n).status.name}")
        break
    a, _ = model.predict(norm(obs), deterministic=False)
    obs, r, done, trunc, info = switch.step(a)
    for evn in cur._last_dynamic_events:
        if evn["change_type"] == "property_substitution":
            last_sub = (int(cur._episode_count), evn["node_ids"][0], evn["removed_vuln"], evn["added_vuln"],
                        evn["node_ids"][0] in cur.owned_nodes)
    if done or trunc:
        obs, _ = switch.reset(); ep += 1
print("done guard", guard, "ep", ep)
