"""
Task F Pass 1, STEP 2 (Condition A) evaluation: frozen-policy eval of one trained agent under one
eval condition, single topology 44. Uses the CHECK 1-validated capture method:
  - manual stepping loop (no SB3 auto-reset), manual obs normalization via the agent's own saved
    VecNormalize stats;
  - at the terminal step, BEFORE our own reset, read switch_env.root_owned_nodes,
    switch_env.reachable_count, switch_env.discovered_nodes (RandomSwitchEnv caches these in step()
    before any reset) and current_env._episode_count. Never get_statistics() after reset; never
    join on the post-reset counter (off by one).
Writes a per-episode score CSV (joinable to the drift CSV on seed/scenario_id/episode) and the
per-event drift CSV (the env writes it internally).

Eval conditions (never two change types together, 2.1):
  static     : dynamic_mode=none, patch_service_dynamic_enabled=False
  membership : dynamic_mode=both, patch_service_dynamic_enabled=False
  property   : dynamic_mode=none, patch_service_dynamic_enabled=True

Evaluation is STOCHASTIC (deterministic=False): this matches (a) TRPO's optimised objective (an
expectation over the policy's action distribution) and (b) the inherited/released evaluation
convention -- SB3 predict()'s default is deterministic=False and every eval predict call in the
released codebase (test_utils.py:60/243/275, callbacks.py:267) uses that default. Deterministic
eval collapses this continuous-action policy (greedy action -> cosine-nearest -> repeated
unproductive action); that collapse is reported as an architecture finding, not silently avoided.

Condition A eval conditions (never two change types together, 2.1):
  static     : dynamic_mode=none, patch=False
  membership : dynamic_mode=both, patch=False
  property   : dynamic_mode=none, patch=True
Condition B (replication, STEP 3): eval_cond="defender", defender_prob=pn -> the released study's
  ExternalRandomEvents probabilistic defender (start/stop service, allow/block firewall rule, per
  node per attacker step at prob pn), dynamic_mode=none, patch=False.

Usage:
  python taskF1_eval.py <run_folder> <agent_cond> <seed> <eval_cond> <n_episodes> <out_dir> [defender_prob]
"""
import sys, os, copy, logging, pickle
sys.path.insert(0, "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim")
os.chdir("/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents")

import numpy as np
import torch
import yaml
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import TRPO
from cyberbattle.utils.envs_utils import wrap_graphs_to_compressed_envs
from cyberbattle.gae.model import GAEEncoder
from cyberbattle._env.cyberbattle_env_switch import RandomSwitchEnv

RUN_FOLDER = os.path.abspath(sys.argv[1])
AGENT_COND = sys.argv[2]      # static | adapted
SEED = int(sys.argv[3])
EVAL_COND = sys.argv[4]       # static | membership | property | defender
N_EPISODES = int(sys.argv[5])
OUT_DIR = os.path.abspath(sys.argv[6])
DEFENDER_PROB = float(sys.argv[7]) if len(sys.argv) > 7 else None
assert AGENT_COND in ("static", "adapted")
assert EVAL_COND in ("static", "membership", "property", "property_substitution", "defender")
if EVAL_COND == "defender":
    assert DEFENDER_PROB is not None
os.makedirs(OUT_DIR, exist_ok=True)

torch.set_num_threads(4)
REPO = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim"
TOPOLOGY_PKL = os.path.join(REPO, "cyberbattle/data/env_samples/scalability_30_40/44/network_SecureBERT.pkl")
DONOR_FOLDER = os.path.join(REPO, "cyberbattle/data/env_samples/join_donor_pool_20_topologies")

CKPT = os.path.join(RUN_FOLDER, "checkpoints", "1", "checkpoint_250000_steps.zip")
VECNORM = os.path.join(RUN_FOLDER, "checkpoints", "1", "checkpoint_vecnormalize_250000_steps.pkl")

with open(os.path.join(RUN_FOLDER, "train_config.yaml")) as f:
    cfg = yaml.safe_load(f)
with open(cfg["graph_encoder_config_path"]) as f:
    config_encoder = yaml.safe_load(f)
with open(cfg["graph_encoder_spec_path"]) as f:
    spec_encoder = yaml.safe_load(f)
config_encoder.update(spec_encoder)
graph_encoder = GAEEncoder(config_encoder["node_feature_vector_size"], config_encoder["model_config"]["layers"],
                           config_encoder["edge_feature_vector_size"])
graph_encoder.load_state_dict(torch.load(cfg["graph_encoder_path"]))
graph_encoder.eval()
cfg["node_embeddings_dimensions"] = config_encoder["model_config"]["layers"][-1]["out_channels"]
logger = logging.getLogger("taskF1_eval"); logger.addHandler(logging.NullHandler())

# --- eval-condition dynamic overrides (never two change types together) ---
DYN = {
    "static":     ("none", False),
    "membership": ("both", False),
    "property":   ("none", True),
    "property_substitution": ("none", True),  # Task D3: legacy path on, but change_type=substitute (set below)
    "defender":   ("none", False),   # Condition B: only the ExternalRandomEvents defender changes state
}
dyn_mode, patch_on = DYN[EVAL_COND]

_suffix = EVAL_COND if EVAL_COND != "defender" else f"defender_p{DEFENDER_PROB}"
DRIFT_CSV = os.path.join(OUT_DIR, f"drift_{AGENT_COND}_seed{SEED}_eval{_suffix}.csv")
SCORE_CSV = os.path.join(OUT_DIR, f"score_{AGENT_COND}_seed{SEED}_eval{_suffix}.csv")
if os.path.exists(DRIFT_CSV):
    os.remove(DRIFT_CSV)

env_cfg = dict(cfg)
env_cfg["dynamic_mode"] = dyn_mode
env_cfg["patch_service_dynamic_enabled"] = patch_on
# Task D3: property removal uses the default change_type ("patch"); substitution flips only the subtype,
# keeping the same one-event-per-change_interval(=20) cadence so removal vs substitution differ in KIND
# not frequency.
if EVAL_COND == "property_substitution":
    env_cfg["change_type"] = "substitute"
env_cfg["drift_logging"] = True
env_cfg["drift_log_path"] = DRIFT_CSV
env_cfg["drift_sample_rate"] = 1
env_cfg["drift_run_id"] = f"F1_{AGENT_COND}_seed{SEED}_eval{EVAL_COND}"
env_cfg["drift_seed"] = SEED
env_cfg["drift_scenario_id"] = "scalability_30_40_44"

with open(TOPOLOGY_PKL, "rb") as f:
    network = pickle.load(f)
env = wrap_graphs_to_compressed_envs(network, logger, **env_cfg)
env.set_graph_encoder(graph_encoder)
env.set_pca_components(env_cfg["pca_components"])
if EVAL_COND == "defender":
    from cyberbattle._env.static_defender import ExternalRandomEvents
    env.static_defender_agent = ExternalRandomEvents(DEFENDER_PROB, logger=logger, verbose=0)
    env.static_defender_eviction_goal = False
if dyn_mode in ("join", "both"):
    pool = []
    for sub in sorted(os.listdir(DONOR_FOLDER)):
        sp = os.path.join(DONOR_FOLDER, sub)
        if os.path.isdir(sp) and sub.isdigit() and sub != "44":
            with open(os.path.join(sp, f"network_{env_cfg['nlp_extractor']}.pkl"), "rb") as f:
                dn = pickle.load(f)
            pool += [(sub, nid, copy.deepcopy(nd["data"])) for nid, nd in dn.network.nodes(data=True)]
    env.dynamic_join_donor_pool = pool

switch_env = RandomSwitchEnv([0], switch_interval=10**9, envs_list=[env], save_to_csv=False, verbose=0)

# load the agent's own VecNormalize stats (frozen)
tmp_vec = DummyVecEnv([lambda: Monitor(RandomSwitchEnv([0], switch_interval=10**9, envs_list=[env], save_to_csv=False, verbose=0))])
vecnorm = VecNormalize.load(VECNORM, tmp_vec)
vecnorm.training = False
vecnorm.norm_reward = False

model = TRPO.load(CKPT, device="cpu")

# deterministic per-cell seeding
np.random.seed(SEED)
torch.manual_seed(SEED)
import random as _random
_random.seed(SEED)

def normalize(obs_dict):
    batched = {k: np.asarray(v, dtype=np.float32)[None, ...] for k, v in obs_dict.items()}
    normed = vecnorm.normalize_obs(batched)
    return {k: v[0] for k, v in normed.items()}

obs, _ = switch_env.reset()
rows = []
ep = 0
guard = 0
while ep < N_EPISODES and guard < N_EPISODES * 5000 + 100000:
    guard += 1
    action, _ = model.predict(normalize(obs), deterministic=False)  # stochastic: matches TRPO objective + inherited eval convention
    obs, reward, done, truncated, info = switch_env.step(action)
    if done or truncated:
        # CHECK 1 method: terminal cached attributes, read BEFORE our reset
        root_owned = int(switch_env.root_owned_nodes)
        reachable = int(switch_env.reachable_count)
        discovered = int(switch_env.discovered_nodes)
        won = bool(switch_env.episode_won)
        episode_num = int(switch_env.current_env._episode_count)  # terminal, before reset
        score = root_owned / max(reachable, 1)
        rows.append(dict(
            agent_cond=AGENT_COND, seed=SEED, eval_cond=EVAL_COND,
            scenario_id="scalability_30_40_44", episode=episode_num,
            root_owned=root_owned, reachable=reachable, n_discovered=discovered,
            won=int(won), score=score,
        ))
        obs, _ = switch_env.reset()
        ep += 1

switch_env.current_env._drift_logger.close()

df = pd.DataFrame(rows)
df.to_csv(SCORE_CSV, index=False)

s = df["score"].to_numpy()
print(f"[{AGENT_COND} seed{SEED} eval={EVAL_COND}] episodes={len(df)} "
      f"score min={s.min():.4f} median={np.median(s):.4f} mean={s.mean():.4f} max={s.max():.4f} "
      f"frac_zero={(s==0).mean():.3f} won_frac={df['won'].mean():.3f}")
print(f"  wrote {SCORE_CSV}")
print(f"  wrote {DRIFT_CSV}")
