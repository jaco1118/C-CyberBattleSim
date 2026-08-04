"""
Task I STEP 1: solo timing profile for ONE topology, run as a standalone process (so each size
gets a clean process / clean peak-RSS baseline, and nothing else runs concurrently). Reuses the
env's own existing instrumentation (graph_encoder_time, action_calculation_time,
action_space_creation_time, balance_action_space_time, inner_step_time,
update_evolving_visible_graph_time) -- the same accumulators already in production code -- rather
than re-deriving a parallel measurement. Adds an external encode() call counter via a thin wrapper
(does not alter the existing timing, which is measured at the call site in step()/reset()).

Usage: python taskI_profile_one_topology.py <topology_pkl_path> <n_label> <n_steps> <n_repeats> <output_json>
"""
import sys, os, pickle, logging, json, time, resource
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
n_repeats = int(sys.argv[4])
output_json = sys.argv[5]

torch.set_num_threads(4)  # single-process solo run; keep consistent with the rest of the session's convention

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
logger = logging.getLogger("taskI"); logger.addHandler(logging.NullHandler())

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

repeats_data = []

for repeat_idx in range(n_repeats):
    env = wrap_graphs_to_compressed_envs(network, logger, **common_kwargs)
    env.set_graph_encoder(graph_encoder)
    env.set_pca_components(config_encoder["edge_feature_vector_size"] if "edge_feature_vector_size" in config_encoder else 768)

    # external encode() call counter -- wraps, does not replace, the bound method
    encode_call_count = [0]
    _orig_encode = env.encode

    def _counting_encode(graph, _orig=_orig_encode, _counter=encode_call_count):
        _counter[0] += 1
        return _orig(graph)

    env.encode = _counting_encode

    np.random.seed(1000 + repeat_idx)
    torch.manual_seed(1000 + repeat_idx)

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB, monotonic peak so far

    t_reset_start = time.perf_counter()
    obs = env.reset()
    t_reset_encode = env.graph_encoder_time  # first encode(), inside reset()
    obs_width = len(obs["graph_embeddings"])
    action_width = env.action_space.shape[0]

    n_episodes = 1
    n_crashes = 0
    step_idx = 0
    while step_idx < n_steps:
        action = np.random.uniform(-1, 1, size=action_width).astype(np.float32)
        try:
            obs, reward, done, info = env.step(action)
        except Exception as e:
            n_crashes += 1
            obs = env.reset()
            n_episodes += 1
            step_idx += 1
            continue
        step_idx += 1
        if done:
            obs = env.reset()
            n_episodes += 1

    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB

    n_encode_calls = encode_call_count[0]
    repeats_data.append(dict(
        repeat=repeat_idx,
        n_nodes_ground_truth=n_nodes_ground_truth,
        obs_width=obs_width,
        action_width=action_width,
        n_steps=n_steps,
        n_episodes=n_episodes,
        n_crashes=n_crashes,
        n_encode_calls_total=n_encode_calls,
        encode_calls_per_episode=n_encode_calls / n_episodes,
        graph_encoder_time_total_s=env.graph_encoder_time,
        mean_time_per_encode_call_s=(env.graph_encoder_time / n_encode_calls) if n_encode_calls else None,
        action_calculation_time_total_s=env.action_calculation_time,
        mean_time_per_action_match_s=env.action_calculation_time / n_steps,
        action_space_creation_time_total_s=env.action_space_creation_time,
        balance_action_space_time_total_s=env.balance_action_space_time,
        inner_step_time_total_s=env.inner_step_time,
        mean_inner_step_time_s=env.inner_step_time / n_steps,
        update_evolving_visible_graph_time_total_s=env.update_evolving_visible_graph_time,
        mean_wallclock_per_step_s=(
            env.graph_encoder_time + env.action_calculation_time + env.action_space_creation_time
            + env.inner_step_time + env.update_evolving_visible_graph_time
        ) / n_steps,
        rss_peak_kb_before=rss_before,
        rss_peak_kb_after=rss_after,
    ))

with open(output_json, "w") as f:
    json.dump(dict(n_label=n_label, topology_pkl=topology_pkl, repeats=repeats_data), f, indent=2)

print(f"n_label={n_label} n_nodes_ground_truth={n_nodes_ground_truth} done, {len(repeats_data)} repeats -> {output_json}")
for r in repeats_data:
    print(f"  repeat {r['repeat']}: obs_width={r['obs_width']} action_width={r['action_width']} "
          f"encode_calls={r['n_encode_calls_total']} episodes={r['n_episodes']} crashes={r['n_crashes']} "
          f"mean_encode_s={r['mean_time_per_encode_call_s']} mean_step_s={r['mean_wallclock_per_step_s']} "
          f"rss_after_kb={r['rss_peak_kb_after']}")
