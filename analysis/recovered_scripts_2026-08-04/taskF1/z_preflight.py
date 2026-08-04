"""Task Z STEP 0.3 pre-flight assertion: arm-3 substitution applied AFTER normalize_obs must produce
BIT-EXACTLY constant substituted dims (the fix Task N identified for the VecNormalize eps=1e-8 silent
failure). graph_embeddings layout = [mean 0:64, max 64:128, min 128:192, next_escalation 192:256].
Arm 3 substitutes dims 64:192 (the 128 extremal = max+min) to a constant, keeps mean (0:64) and
next_escalation (192:256). Runs ~1200 steps on the 30-40 F1 static agent. No training."""
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
SEED = 42; TOPO = "scalability_30_40/44"; C = 0.0   # the fixed post-normalisation constant
EXTREMAL = slice(64, 192)   # max+min dims; mean = 0:64, next_escalation = 192:256
torch.set_num_threads(4)
CKPT = os.path.join(RUN, "checkpoints", "1", "checkpoint_250000_steps.zip")
VECN = os.path.join(RUN, "checkpoints", "1", "checkpoint_vecnormalize_250000_steps.pkl")
cfg = yaml.safe_load(open(os.path.join(RUN, "train_config.yaml")))
ce = yaml.safe_load(open(cfg["graph_encoder_config_path"])); ce.update(yaml.safe_load(open(cfg["graph_encoder_spec_path"])))
ge = GAEEncoder(ce["node_feature_vector_size"], ce["model_config"]["layers"], ce["edge_feature_vector_size"])
ge.load_state_dict(torch.load(cfg["graph_encoder_path"])); ge.eval()
cfg["node_embeddings_dimensions"] = ce["model_config"]["layers"][-1]["out_channels"]
logger = logging.getLogger("z"); logger.addHandler(logging.NullHandler())
ec = dict(cfg); ec["dynamic_mode"] = "none"; ec["patch_service_dynamic_enabled"] = False
net = pickle.load(open(os.path.join(REPO, "cyberbattle/data/env_samples", TOPO, "network_SecureBERT.pkl"), "rb"))
env = wrap_graphs_to_compressed_envs(net, logger, **ec); env.set_graph_encoder(ge); env.set_pca_components(ec["pca_components"])
switch = RandomSwitchEnv([0], switch_interval=10**9, envs_list=[env], save_to_csv=False, verbose=0)
tmp = DummyVecEnv([lambda: Monitor(RandomSwitchEnv([0], switch_interval=10**9, envs_list=[env], save_to_csv=False, verbose=0))])
vecn = VecNormalize.load(VECN, tmp); vecn.training = False; vecn.norm_reward = False
model = TRPO.load(CKPT, device="cpu")
np.random.seed(SEED); torch.manual_seed(SEED)
import random as _r; _r.seed(SEED)

def arm1_normed(obs):
    b = {k: np.asarray(v, np.float32)[None, ...] for k, v in obs.items()}
    n = vecn.normalize_obs(b); return {k: v[0].copy() for k, v in n.items()}

def arm3_substitute(normed):  # applied AFTER normalize_obs
    out = {k: v.copy() for k, v in normed.items()}
    out["graph_embeddings"][EXTREMAL] = C
    return out

obs, _ = switch.reset(); steps = 0
sub_vals = []; mean_absdiff = 0.0
NSTEPS = 1200
while steps < NSTEPS:
    n1 = arm1_normed(obs)
    n3 = arm3_substitute(n1)
    sub_vals.append(n3["graph_embeddings"][EXTREMAL].copy())
    mean_absdiff = max(mean_absdiff, float(np.max(np.abs(n1["graph_embeddings"][0:64] - n3["graph_embeddings"][0:64]))))
    a, _ = model.predict(n1, deterministic=False)   # drive env with the real (arm-1) obs
    obs, r, done, trunc, info = switch.step(a); steps += 1
    if done or trunc: obs, _ = switch.reset()

S = np.array(sub_vals)  # (NSTEPS, 128)
distinct = np.unique(S)
print("=== Task Z 0.3 PRE-FLIGHT ASSERTION (arm-3, substitute AFTER normalize_obs) ===")
print(f"(a) steps sampled: {S.shape[0]} ; substituted dims per step: {S.shape[1]} (=128 extremal)")
print(f"(b) distinct values across all {S.size} substituted entries: {distinct.size}  -> {'PASS (==1)' if distinct.size==1 else 'FAIL (>1) -- DO NOT TRAIN ARM 3'}")
print(f"(c) the constant's value: {distinct.tolist() if distinct.size<=3 else distinct[:3].tolist()+['...']}")
print(f"(d) 64 mean dims: max |arm1 - arm3| over all steps = {mean_absdiff:.3e}  -> {'PASS (==0)' if mean_absdiff==0.0 else 'nonzero'}")
