"""Quantify substitution stranding: per episode, count empty-action-space steps and final score,
to separate the stranding artifact from genuine disruption. seed42 / topo44, 200 eps."""
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
import cyberbattle._env.cyberbattle_env_compressed as CEC

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
logger = logging.getLogger("strand"); logger.addHandler(logging.NullHandler())

# Monkeypatch the guard sentinel path to count empty-action-space hits.
_orig = CEC.CyberBattleCompressedEnv.find_closest_action_embedding
def _patched(self, action_vector, no_output=False):
    if not self.action_embeddings:
        self._strand_hits = getattr(self, "_strand_hits", 0) + 1
    return _orig(self, action_vector, no_output=no_output)
CEC.CyberBattleCompressedEnv.find_closest_action_embedding = _patched

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

obs, _ = switch.reset(); ep = 0; guard = 0; rows = []
env._strand_hits = 0
while ep < 200 and guard < 2_000_000:
    guard += 1
    a, _ = model.predict(norm(obs), deterministic=False)
    obs, r, done, trunc, info = switch.step(a)
    if done or trunc:
        root = int(switch.root_owned_nodes); reach = int(switch.reachable_count)
        rows.append(dict(episode=ep, root_owned=root, reachable=reach, score=root/max(reach,1),
                         strand_steps=env._strand_hits))
        env._strand_hits = 0
        obs, _ = switch.reset(); ep += 1

d = pd.DataFrame(rows)
d["stranded"] = d.strand_steps > 0
print(f"episodes={len(d)}")
print(f"mean_score={d.score.mean():.4f}  frac_zero={ (d.score==0).mean():.3f}")
print(f"episodes with >=1 empty-action step (stranded): {d.stranded.mean():.3f} ({d.stranded.sum()}/{len(d)})")
print(f"of the zero-score episodes, frac stranded: {d[d.score==0].stranded.mean():.3f}")
print(f"of the stranded episodes: mean_score={d[d.stranded].score.mean():.4f} mean_strand_steps={d[d.stranded].strand_steps.mean():.1f}")
print(f"NON-stranded episodes: mean_score={d[~d.stranded].score.mean():.4f} frac_zero={(d[~d.stranded].score==0).mean():.3f}")
print(f"total empty-action steps across all eps: {d.strand_steps.sum()}")
