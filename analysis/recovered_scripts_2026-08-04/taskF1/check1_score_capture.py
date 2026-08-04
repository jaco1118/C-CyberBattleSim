"""
Task F Pass 1, CHECK 1 (before STEP 2): does the score read attach to the right episode?
SB3 vec envs auto-reset on done, so get_statistics() read AFTER done in an outer loop may read a
freshly-reset episode. Verify empirically over 20 consecutive episodes, two capture methods:
  (a) NAIVE / post-reset: current_env.get_statistics() read AFTER the env has been reset (what a
      vec-auto-reset outer loop would see).
  (b) TERMINAL / correct: RandomSwitchEnv's cached terminal attributes (self.root_owned_nodes,
      self.reachable_count), computed inside step() at the terminal step BEFORE any reset, read
      before we manually reset.
Also: does current_env._episode_count read in the outer loop (post-reset) match the 'episode'
field the drift rows carry for that same episode? (off-by-one join-key check.)

Uses a manual stepping loop (full control over capture timing) with manual obs normalization via
the loaded VecNormalize stats -- so no auto-reset can occur between the terminal step and our
capture. The naive method (a) is simulated by reading get_statistics() only AFTER we reset.
Policy: the completed gate's 30-40 seed42 final checkpoint (a real policy, so terminal
root_owned > 0 and the contrast with a fresh reset's 0 is visible). Eval config: static (no
change), the cleanest contrast.
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

torch.set_num_threads(2)

REPO = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim"
GATE_RUN = os.path.join(REPO, "cyberbattle/agents/logs",
                        "trpo_250k_tuned_compressed_band30-40_seed42_2026-07-26_12-47-19",
                        "TRPO_x_control_SecureBERT")
CKPT = os.path.join(GATE_RUN, "checkpoints/1/checkpoint_250000_steps.zip")
VECNORM = os.path.join(GATE_RUN, "checkpoints/1/checkpoint_vecnormalize_250000_steps.pkl")
TOPOLOGY_PKL = os.path.join(REPO, "cyberbattle/data/env_samples/scalability_30_40/44/network_SecureBERT.pkl")
DRIFT_CSV = "/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1/check1_drift.csv"
if os.path.exists(DRIFT_CSV):
    os.remove(DRIFT_CSV)

with open(os.path.join(GATE_RUN, "train_config.yaml")) as f:
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

logger = logging.getLogger("check1"); logger.addHandler(logging.NullHandler())

# static eval config (cleanest contrast); drift on so we can cross-check episode numbers
env_cfg = dict(cfg)
env_cfg["dynamic_mode"] = "none"
env_cfg["patch_service_dynamic_enabled"] = False
env_cfg["drift_logging"] = True
env_cfg["drift_log_path"] = DRIFT_CSV
env_cfg["drift_sample_rate"] = 1
env_cfg["drift_run_id"] = "check1"
env_cfg["drift_seed"] = 42
env_cfg["drift_scenario_id"] = "scalability_30_40_44"

with open(TOPOLOGY_PKL, "rb") as f:
    network = pickle.load(f)
env = wrap_graphs_to_compressed_envs(network, logger, **env_cfg)
env.set_graph_encoder(graph_encoder)
env.set_pca_components(env_cfg["pca_components"])
switch_env = RandomSwitchEnv([0], switch_interval=10**9, envs_list=[env], save_to_csv=False, verbose=0)

# Load VecNormalize stats through a throwaway DummyVecEnv, then use its normalize_obs manually.
tmp_vec = DummyVecEnv([lambda: Monitor(RandomSwitchEnv([0], switch_interval=10**9, envs_list=[env], save_to_csv=False, verbose=0))])
vecnorm = VecNormalize.load(VECNORM, tmp_vec)
vecnorm.training = False
vecnorm.norm_reward = False

model = TRPO.load(CKPT, device="cpu")

np.random.seed(42)
torch.manual_seed(42)

def normalize(obs_dict):
    batched = {k: np.asarray(v, dtype=np.float32)[None, ...] for k, v in obs_dict.items()}
    normed = vecnorm.normalize_obs(batched)
    return {k: v[0] for k, v in normed.items()}

obs, _ = switch_env.reset()
N_EP = 20
rows = []
ep = 0
guard = 0
while ep < N_EP and guard < 200000:
    guard += 1
    action, _ = model.predict(normalize(obs), deterministic=True)
    obs, reward, done, truncated, info = switch_env.step(action)
    if done or truncated:
        # method (b) TERMINAL: cached attributes computed in step() before any reset
        rob_b = switch_env.root_owned_nodes
        reach_b = switch_env.reachable_count
        score_b = rob_b / max(reach_b, 1)
        ep_count_terminal = switch_env.current_env._episode_count  # before our reset
        # now reset (the manual analogue of SB3's auto-reset)
        obs, _ = switch_env.reset()
        # method (a) NAIVE: get_statistics() read AFTER reset (what a vec-auto-reset loop sees)
        stats_a = switch_env.current_env.get_statistics()
        rob_a = stats_a[14]
        reach_a = stats_a[5]
        score_a = rob_a / max(reach_a, 1)
        ep_count_postreset = switch_env.current_env._episode_count
        rows.append(dict(ep=ep, terminal_episode_count=ep_count_terminal, postreset_episode_count=ep_count_postreset,
                         root_owned_terminal_b=rob_b, reachable_b=reach_b, score_b=round(score_b, 4),
                         root_owned_postreset_a=rob_a, reachable_a=reach_a, score_a=round(score_a, 4)))
        ep += 1

switch_env.current_env._drift_logger.close()

print("=== per-episode capture, method (b) terminal vs method (a) post-reset ===")
print(f"{'ep':>3} {'termEpCnt':>9} {'postEpCnt':>9} {'rootB':>6} {'reachB':>6} {'scoreB':>7} | {'rootA':>6} {'reachA':>6} {'scoreA':>7}")
agree = 0
for r in rows:
    a_eq_b = (r['root_owned_terminal_b'] == r['root_owned_postreset_a'] and r['reachable_b'] == r['reachable_a'])
    agree += int(a_eq_b)
    print(f"{r['ep']:>3} {r['terminal_episode_count']:>9} {r['postreset_episode_count']:>9} "
          f"{r['root_owned_terminal_b']:>6} {r['reachable_b']:>6} {r['score_b']:>7} | "
          f"{r['root_owned_postreset_a']:>6} {r['reachable_a']:>6} {r['score_a']:>7}")

print(f"\nmethod (a) == method (b) on {agree}/{len(rows)} episodes")
print(f"terminal_episode_count == postreset_episode_count on "
      f"{sum(r['terminal_episode_count']==r['postreset_episode_count'] for r in rows)}/{len(rows)} episodes "
      f"(if 0, the outer-loop post-reset counter is off by one)")

# cross-check terminal episode counts against the episode field the drift rows actually carry
df = pd.read_csv(DRIFT_CSV)
drift_eps = sorted(df["episode"].unique().tolist())
term_eps = sorted(r["terminal_episode_count"] for r in rows)
print(f"\ndrift-row episode values (unique, first 25): {drift_eps[:25]}")
print(f"terminal_episode_count series (method b):   {term_eps}")
print(f"every terminal_episode_count present in drift episodes: "
      f"{all(e in set(drift_eps) for e in term_eps)}")
postreset_eps = sorted(r["postreset_episode_count"] for r in rows)
print(f"postreset_episode_count series (method a):  {postreset_eps}")
print(f"postreset counts present in drift episodes: "
      f"{all(e in set(drift_eps) for e in postreset_eps)} "
      f"(if False, joining score on the post-reset counter attaches it to a non-existent/next episode)")
