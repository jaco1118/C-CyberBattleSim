"""
Task I STEP 1, corrected methodology: the first pass (taskI_profile_one_topology.py) averaged
encode()/action-space/matching cost across an ENTIRE 300-step episode, which mixes costs from a
huge range of intermediate discovered-graph sizes (from 1 node at reset up to however much was
discovered by step 300) into one noisy "mean per call" number -- that is why the first pass's
fits had poor R^2 (0.08-0.53): the noise is structural (mixing sizes), not just repeat-to-repeat
jitter. This script isolates each cost component as a clean function of a FIXED graph state:

  1. Runs the same kind of 300-step random rollout to reach a realistic discovered/owned state
     (real edges from real exploit outcomes, not synthetic).
  2. Records the actual discovered-node-count and edge-count of evolving_visible_graph at that
     point (the quantity 0.2 says encode() scales with -- NOT the topology's total N, since not
     all N nodes are necessarily discovered in 300 steps).
  3. Times 10 repeated, side-effect-free encode() calls on that FIXED graph state.
  4. Times 10 repeated create_continuous_action_space() calls, each preceded by clearing
     processed_pairs so every repeat performs a genuine full rebuild (fair, consistent cost,
     representative of a first-call/precise_action_space_positions=False cost) on the SAME fixed
     owned/discovered node sets.
  5. Times 10 repeated find_closest_action_embedding() calls against the resulting fixed
     action_embeddings dict.
  6. Repeats the whole rollout 3 times per topology (3 different random seeds) -- giving 3
     independent (n_discovered, n_edges) states per topology, each with its own low-noise
     per-component cost estimate (mean/std over the 10 inner repeats).
"""
import sys, os, pickle, logging, json, time, copy
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
output_json = sys.argv[6]

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
logger = logging.getLogger("taskI_clean"); logger.addHandler(logging.NullHandler())

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
    n_action_embeddings_before_rebuild = len(env.action_embeddings)

    # --- 1. encode() timing: side-effect-free w.r.t. evolving_visible_graph, repeat directly ---
    encode_times = []
    for _ in range(n_inner_repeats):
        t0 = time.perf_counter()
        _ = env.encode(env.evolving_visible_graph)
        encode_times.append(time.perf_counter() - t0)

    # --- 2. create_continuous_action_space() timing: force a full rebuild each repeat by
    #     clearing processed_pairs (and action_embeddings, to avoid unbounded dict growth across
    #     repeats skewing memory, though timing is what's measured) ---
    action_space_times = []
    for _ in range(n_inner_repeats):
        env.processed_pairs = set()
        env.action_embeddings = {}
        t0 = time.perf_counter()
        env.create_continuous_action_space()
        action_space_times.append(time.perf_counter() - t0)
    n_action_embeddings_final = len(env.action_embeddings)

    # --- 3. find_closest_action_embedding() timing: fixed action_embeddings from the last
    #     rebuild above, fresh random query vector each repeat (query point doesn't change
    #     candidate-set size, only which is nearest) ---
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
                    outer_results=outer_results), f, indent=2)

print(f"n_label={n_label} n_nodes_ground_truth={n_nodes_ground_truth}")
for r in outer_results:
    print(f"  outer {r['outer_repeat']}: n_discovered={r['n_discovered_final']} n_visible_nodes={r['n_visible_graph_nodes']} "
          f"n_edges={r['n_visible_graph_edges']} n_actions={r['n_action_embeddings_final']} "
          f"encode_mean={r['encode_mean_s']:.6f}s (+/-{r['encode_std_s']:.6f}) "
          f"action_space_mean={r['action_space_mean_s']:.6f}s (+/-{r['action_space_std_s']:.6f}) "
          f"match_mean={r['match_mean_s']:.6f}s (+/-{r['match_std_s']:.6f})")
