"""
Re-check requested after Task I-2: Task I's original action_embeddings memory figures
(exponent 2.03, 193MB/1.24GB/134GB at N=100/250/1000/2500) came from the SAME buggy pipeline
that omitted sample_subset_samples (confirmed in Task I-2) -- meaning __balance_action_space_by_
outcome never ran, and what Task I actually measured as "n_action_embeddings_final" was really
the RAW, never-pruned candidate count the whole time.

This script measures, with sample_subset_samples=100 correctly wired (matching real training):
  - n_actions_pre_balance: size of self.action_embeddings the instant before
    __balance_action_space_by_outcome prunes it (the raw/pre-sampling candidate count a real
    construction call actually builds, transiently, before pruning)
  - n_actions_post_balance: size after pruning (the persistent/resting cache size)
for both the full_rebuild scenario (processed_pairs cleared -- worst case, matches Task I's
original per-state condition) and the steady_state scenario (processed_pairs already populated
-- the realistic in-episode condition), at all 7 Task I sizes.

Usage: python taskI_memory_recheck.py <topology_pkl> <n_label> <n_steps> <n_outer> <output_json>
"""
import sys, os, pickle, logging, json
sys.path.insert(0, "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim")
os.chdir("/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents")

import torch, yaml
import numpy as np
from cyberbattle.utils.envs_utils import wrap_graphs_to_compressed_envs
from cyberbattle.utils.file_utils import load_yaml
from cyberbattle.gae.model import GAEEncoder
import cyberbattle._env.cyberbattle_env_compressed as ccenv

topology_pkl = sys.argv[1]
n_label = sys.argv[2]
n_steps = int(sys.argv[3])
n_outer_repeats = int(sys.argv[4])
output_json = sys.argv[5]

torch.set_num_threads(4)

# Instrument the name-mangled balance method to record pre-balance size before it prunes.
_pre_balance_sizes = []
_orig_balance = ccenv.CyberBattleCompressedEnv._CyberBattleCompressedEnv__balance_action_space_by_outcome


def _instrumented_balance(self):
    _pre_balance_sizes.append(len(self.action_embeddings))
    return _orig_balance(self)


ccenv.CyberBattleCompressedEnv._CyberBattleCompressedEnv__balance_action_space_by_outcome = _instrumented_balance

ENCODER_DIR = "../gae/logs/default/SecureBERT"
with open(os.path.join(ENCODER_DIR, "train_config_encoder.yaml")) as f:
    config_encoder = yaml.safe_load(f)
with open(os.path.join(ENCODER_DIR, "model_spec.yaml")) as f:
    spec_encoder = yaml.safe_load(f)
config_encoder.update(spec_encoder)
graph_encoder = GAEEncoder(config_encoder["node_feature_vector_size"], config_encoder["model_config"]["layers"], config_encoder["edge_feature_vector_size"])
graph_encoder.load_state_dict(torch.load(os.path.join(ENCODER_DIR, "encoder.pth")))
graph_encoder.eval()

rewards_config = load_yaml("config/rewards_config.yaml")
train_config = load_yaml("config/train_config.yaml")
logger = logging.getLogger("taskI_mem_recheck"); logger.addHandler(logging.NullHandler())

common_kwargs = dict(
    rewards_dict=rewards_config["rewards_dict"]["control"], penalties_dict=rewards_config["penalties_dict"]["control"],
    goal="control", episode_iterations=train_config["episode_iterations"],
    proportional_cutoff_coefficient=train_config["proportional_cutoff_coefficient"],
    winning_reward=train_config["winning_reward"], losing_reward=train_config["losing_reward"],
    random_starter_node=train_config["random_starter_node"], stop_at_goal_reached=train_config["stop_at_goal_reached"],
    isolation_filter_threshold=train_config["isolation_filter_threshold"],
    remove_main_obstacles=train_config["remove_main_obstacles"], remove_all_obstacles=train_config["remove_all_obstacles"],
    dynamic_mode=train_config["dynamic_mode"], change_interval=train_config["change_interval"],
    patch_service_dynamic_enabled=train_config["patch_service_dynamic_enabled"],
    static_defender_eviction_goal=train_config["static_defender_eviction_goal"],
    outcome_dimensions=train_config["outcome_dimensions"], discrete_features=train_config["discrete_features"],
    graph_embeddings_aggregations=train_config["graph_embeddings_aggregations"],
    node_embeddings_dimensions=config_encoder["model_config"]["layers"][-1]["out_channels"],
    sample_subset_samples=train_config["sample_subset_samples"],  # correctly wired this time
    verbose=0,
)

with open(topology_pkl, "rb") as f:
    network = pickle.load(f)
n_nodes_ground_truth = len(list(network.network.nodes()))

outer_results = []

for outer_idx in range(n_outer_repeats):
    env = wrap_graphs_to_compressed_envs(network, logger, **common_kwargs)
    env.set_graph_encoder(graph_encoder)
    env.set_pca_components(768)

    np.random.seed(2000 + outer_idx)
    torch.manual_seed(2000 + outer_idx)
    obs = env.reset()
    action_width = env.action_space.shape[0]

    step_idx = 0
    while step_idx < n_steps:
        action = np.random.uniform(-1, 1, size=action_width).astype(np.float32)
        try:
            obs, reward, done, info = env.step(action)
        except Exception:
            obs = env.reset()
            step_idx += 1
            continue
        step_idx += 1
        if done:
            obs = env.reset()

    n_discovered_final = len(env.discovered_nodes)
    n_owned_final = len(env.owned_nodes)

    # full_rebuild scenario: processed_pairs cleared, one call
    env.processed_pairs = set()
    env.action_embeddings = {}
    _pre_balance_sizes.clear()
    env.create_continuous_action_space()
    full_rebuild_pre = _pre_balance_sizes[-1]
    full_rebuild_post = len(env.action_embeddings)

    # steady_state scenario: populate fully first (untimed), then one more call on top
    env.processed_pairs = set()
    env.action_embeddings = {}
    env.create_continuous_action_space()  # populate
    _pre_balance_sizes.clear()
    env.create_continuous_action_space()  # the call we measure
    steady_state_pre = _pre_balance_sizes[-1]
    steady_state_post = len(env.action_embeddings)

    outer_results.append(dict(
        outer_repeat=outer_idx,
        n_nodes_ground_truth=n_nodes_ground_truth,
        n_discovered_final=n_discovered_final,
        n_owned_final=n_owned_final,
        full_rebuild_pre_balance=full_rebuild_pre,
        full_rebuild_post_balance=full_rebuild_post,
        steady_state_pre_balance=steady_state_pre,
        steady_state_post_balance=steady_state_post,
    ))

with open(output_json, "w") as f:
    json.dump(dict(n_label=n_label, topology_pkl=topology_pkl, n_nodes_ground_truth=n_nodes_ground_truth,
                    outer_results=outer_results), f, indent=2)

print(f"n_label={n_label} n_nodes_ground_truth={n_nodes_ground_truth}")
for r in outer_results:
    print(f"  outer {r['outer_repeat']}: n_discovered={r['n_discovered_final']} n_owned={r['n_owned_final']} "
          f"full_rebuild(pre/post)={r['full_rebuild_pre_balance']}/{r['full_rebuild_post_balance']} "
          f"steady_state(pre/post)={r['steady_state_pre_balance']}/{r['steady_state_post_balance']}")
