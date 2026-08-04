"""
Task I-2 STEP 1, corrected methodology (also corrects a bug found in Task I's own scripts):
Task I's taskI_profile_one_topology.py / taskI_profile_clean.py never passed
sample_subset_samples through to the env constructor, so both silently ran with action-space
balancing DISABLED (class default sample_subset_samples=False) -- unlike real training
(train_agent.py passes **config, which includes sample_subset_samples=100). Verified directly:
with the flag missing, one topology's action_embeddings reached 43,326 entries; with it
correctly wired to 100 (matching train_config.yaml), the same topology capped at exactly 800
(8 outcome categories x 100). This script fixes that omission and additionally accepts
precise_action_space_positions / precise_graph_encoding as CLI-toggleable, to produce the
Task I-2 comparison under the alternative flag setting.

Usage: python taskI2_profile_clean_v2.py <topology_pkl> <n_label> <n_steps> <n_outer> <n_inner> <precise_flags:0|1> <output_json>
"""
import sys, os, pickle, logging, json, time
sys.path.insert(0, "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim")
os.chdir("/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents")

import torch, yaml
import numpy as np
from cyberbattle.utils.envs_utils import wrap_graphs_to_compressed_envs
from cyberbattle.utils.file_utils import load_yaml
from cyberbattle.gae.model import GAEEncoder

topology_pkl = sys.argv[1]
n_label = sys.argv[2]
n_steps = int(sys.argv[3])
n_outer_repeats = int(sys.argv[4])
n_inner_repeats = int(sys.argv[5])
precise_flags = bool(int(sys.argv[6]))
output_json = sys.argv[7]

torch.set_num_threads(4)

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
logger = logging.getLogger("taskI2_clean"); logger.addHandler(logging.NullHandler())

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
    sample_subset_samples=train_config["sample_subset_samples"],  # THE FIX: was omitted in Task I's scripts
    precise_action_space_positions=precise_flags,
    precise_graph_encoding=precise_flags,
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
    n_visible_graph_nodes = len(env.evolving_visible_graph.nodes())
    n_visible_graph_edges = len(env.evolving_visible_graph.edges())

    # encode() timing
    encode_times = []
    for _ in range(n_inner_repeats):
        t0 = time.perf_counter()
        _ = env.encode(env.evolving_visible_graph)
        encode_times.append(time.perf_counter() - t0)

    # create_continuous_action_space() timing: force a full rebuild each repeat.
    # Under precise_flags=True, also pass nodes_to_recalculate to exercise that code path
    # (matching how step() actually calls it, cyberbattle_env_compressed.py:585-594) -- scoped
    # to the current source/target (here: two arbitrary owned/discovered nodes, since this is an
    # isolated timing call not a real step) to test the SAME call signature production uses.
    action_space_times = []
    for _ in range(n_inner_repeats):
        env.processed_pairs = set()
        env.action_embeddings = {}
        t0 = time.perf_counter()
        if precise_flags and env.owned_nodes and env.discovered_nodes:
            probe_source = sorted(env.owned_nodes)[0]
            probe_target = sorted(env.discovered_nodes)[0]
            env.create_continuous_action_space(nodes_to_recalculate=[probe_source, probe_target])
        else:
            env.create_continuous_action_space()
        action_space_times.append(time.perf_counter() - t0)
    n_action_embeddings_final = len(env.action_embeddings)

    # find_closest_action_embedding() timing
    match_times = []
    for _ in range(n_inner_repeats):
        query = np.random.uniform(-1, 1, size=action_width).astype(np.float32)
        t0 = time.perf_counter()
        _ = env.find_closest_action_embedding(query, no_output=True)
        match_times.append(time.perf_counter() - t0)

    outer_results.append(dict(
        outer_repeat=outer_idx,
        n_nodes_ground_truth=n_nodes_ground_truth,
        n_discovered_final=n_discovered_final,
        n_owned_final=n_owned_final,
        n_visible_graph_nodes=n_visible_graph_nodes,
        n_visible_graph_edges=n_visible_graph_edges,
        n_action_embeddings_final=n_action_embeddings_final,
        encode_times_s=encode_times,
        encode_mean_s=float(np.mean(encode_times)),
        encode_std_s=float(np.std(encode_times, ddof=1)),
        action_space_times_s=action_space_times,
        action_space_mean_s=float(np.mean(action_space_times)),
        action_space_std_s=float(np.std(action_space_times, ddof=1)),
        match_times_s=match_times,
        match_mean_s=float(np.mean(match_times)),
        match_std_s=float(np.std(match_times, ddof=1)),
    ))

with open(output_json, "w") as f:
    json.dump(dict(n_label=n_label, topology_pkl=topology_pkl, n_nodes_ground_truth=n_nodes_ground_truth,
                    precise_flags=precise_flags, sample_subset_samples=train_config["sample_subset_samples"],
                    outer_results=outer_results), f, indent=2)

print(f"n_label={n_label} n_nodes_ground_truth={n_nodes_ground_truth} precise_flags={precise_flags}")
for r in outer_results:
    print(f"  outer {r['outer_repeat']}: n_discovered={r['n_discovered_final']} n_owned={r['n_owned_final']} "
          f"n_edges={r['n_visible_graph_edges']} n_actions={r['n_action_embeddings_final']} "
          f"encode_mean={r['encode_mean_s']:.6f}s action_space_mean={r['action_space_mean_s']:.6f}s match_mean={r['match_mean_s']:.6f}s")
