"""
Task I-2 STEP 1, final methodology. Adds a THIRD, more decisive measurement beyond Task I's
"full rebuild" timing: the STEADY-STATE cost of create_continuous_action_space() when
processed_pairs is already fully populated (the realistic in-episode condition, since
processed_pairs persists across steps in real production -- it is never cleared mid-episode).
This is required to actually test whether precise_action_space_positions changes anything: with
processed_pairs cleared (Task I's original method, and this script's "full_rebuild" scenario),
BOTH flag settings behave identically -- the scoping optimisation only skips ALREADY-processed
pairs, so starting from empty defeats it entirely. This script measures three scenarios per
state:

  A) full_rebuild:  processed_pairs cleared first, then create_continuous_action_space() with
     the call signature the given flag setting actually uses in production (no args if
     precise_action_space_positions=False; nodes_to_recalculate=[probe pair] if True). Same as
     Task I / taskI2_profile_clean_v2.py's method -- kept for continuity.
  B) steady_state:   processed_pairs is FIRST fully populated (one untimed full build), THEN the
     SAME call signature as A is timed again on that already-populated state -- this is the
     realistic per-step cost once an episode has been running a while, which is the condition
     that actually occurs during training (processed_pairs only grows, never resets mid-episode,
     confirmed by grep: it is only ever assigned in reset()).

Also still measures encode() (flag-independent, kept for completeness/consistency-check) and
find_closest_action_embedding() (against the resulting, sample_subset_samples-capped
action_embeddings from scenario B, the realistic candidate set an agent actually searches
against most of the time).

Usage: python taskI2_profile_v3.py <topology_pkl> <n_label> <n_steps> <n_outer> <n_inner> <precise_flags:0|1> <output_json>
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
logger = logging.getLogger("taskI2_v3"); logger.addHandler(logging.NullHandler())

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
    sample_subset_samples=train_config["sample_subset_samples"],
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
    n_visible_graph_edges = len(env.evolving_visible_graph.nodes())

    def call_action_space():
        if precise_flags and env.owned_nodes and env.discovered_nodes:
            probe_source = sorted(env.owned_nodes)[0]
            probe_target = sorted(env.discovered_nodes)[0]
            env.create_continuous_action_space(nodes_to_recalculate=[probe_source, probe_target])
        else:
            env.create_continuous_action_space()

    # --- encode() timing (flag-independent) ---
    encode_times = []
    for _ in range(n_inner_repeats):
        t0 = time.perf_counter()
        _ = env.encode(env.evolving_visible_graph)
        encode_times.append(time.perf_counter() - t0)

    # --- A) full_rebuild: clear processed_pairs first each repeat ---
    full_rebuild_times = []
    for _ in range(n_inner_repeats):
        env.processed_pairs = set()
        env.action_embeddings = {}
        t0 = time.perf_counter()
        call_action_space()
        full_rebuild_times.append(time.perf_counter() - t0)
    n_actions_after_full_rebuild = len(env.action_embeddings)

    # --- B) steady_state: populate processed_pairs fully ONCE (untimed), then time repeated
    #     calls on that already-populated state (the realistic in-episode condition) ---
    env.processed_pairs = set()
    env.action_embeddings = {}
    env.create_continuous_action_space()  # untimed full build to populate processed_pairs
    steady_state_times = []
    for _ in range(n_inner_repeats):
        t0 = time.perf_counter()
        call_action_space()
        steady_state_times.append(time.perf_counter() - t0)
    n_actions_steady_state = len(env.action_embeddings)

    # --- match timing against the steady-state (realistic) action_embeddings ---
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
        n_visible_graph_edges=n_visible_graph_edges,
        n_actions_after_full_rebuild=n_actions_after_full_rebuild,
        n_actions_steady_state=n_actions_steady_state,
        encode_mean_s=float(np.mean(encode_times)),
        encode_std_s=float(np.std(encode_times, ddof=1)),
        full_rebuild_mean_s=float(np.mean(full_rebuild_times)),
        full_rebuild_std_s=float(np.std(full_rebuild_times, ddof=1)),
        steady_state_mean_s=float(np.mean(steady_state_times)),
        steady_state_std_s=float(np.std(steady_state_times, ddof=1)),
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
          f"n_actions(full/steady)={r['n_actions_after_full_rebuild']}/{r['n_actions_steady_state']} "
          f"encode={r['encode_mean_s']:.6f}s full_rebuild={r['full_rebuild_mean_s']:.6f}s "
          f"steady_state={r['steady_state_mean_s']:.6f}s match={r['match_mean_s']:.6f}s")
