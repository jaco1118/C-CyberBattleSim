"""
Task F Pass 1, CHECK 2 (before STEP 2): does the topology reset between episodes?
Code (cyberbattle_env.py:169) shows reset_env() does
    self.environment = copy.deepcopy(self.__initial_environment.network)
so every episode rebuilds the live graph from the pristine, never-mutated initial_environment;
dynamic leave/join/patch only ever mutate the live copy. Verify empirically: log node count and
total vulnerability count at the FIRST step of 20 consecutive episodes, under membership and
property conditions. Also log the same at each episode's LAST step, to show the network DID erode
within the episode (so the constant episode-START values prove reset actually restores it, rather
than nothing having changed).
"""
import sys, os, logging, pickle
sys.path.insert(0, "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim")
os.chdir("/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents")

import numpy as np
import torch
import yaml
import copy
from cyberbattle.utils.envs_utils import wrap_graphs_to_compressed_envs
from cyberbattle.gae.model import GAEEncoder

torch.set_num_threads(2)
REPO = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim"
GATE_RUN = os.path.join(REPO, "cyberbattle/agents/logs",
                        "trpo_250k_tuned_compressed_band30-40_seed42_2026-07-26_12-47-19",
                        "TRPO_x_control_SecureBERT")
TOPOLOGY_PKL = os.path.join(REPO, "cyberbattle/data/env_samples/scalability_30_40/44/network_SecureBERT.pkl")
DONOR_FOLDER = os.path.join(REPO, "cyberbattle/data/env_samples/join_donor_pool_20_topologies")

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
logger = logging.getLogger("check2"); logger.addHandler(logging.NullHandler())

with open(TOPOLOGY_PKL, "rb") as f:
    network = pickle.load(f)
PRISTINE_NODES = len(list(network.network.nodes()))
PRISTINE_VULNS = sum(len(network.network.nodes[n]["data"].vulnerabilities) for n in network.network.nodes())
print(f"pristine topology 44: nodes={PRISTINE_NODES}, total_vulnerabilities={PRISTINE_VULNS}\n")


def total_vulns(env):
    return sum(len(env.get_node(n).vulnerabilities) for n in env.environment.nodes())


def run_condition(label, dynamic_mode, patch_enabled):
    c = dict(cfg)
    c["dynamic_mode"] = dynamic_mode
    c["patch_service_dynamic_enabled"] = patch_enabled
    c["drift_logging"] = False
    env = wrap_graphs_to_compressed_envs(network, logger, **c)
    env.set_graph_encoder(graph_encoder)
    env.set_pca_components(c["pca_components"])
    if dynamic_mode in ("join", "both"):
        pool = []
        for sub in sorted(os.listdir(DONOR_FOLDER)):
            sp = os.path.join(DONOR_FOLDER, sub)
            if os.path.isdir(sp) and sub.isdigit() and sub != "44":
                with open(os.path.join(sp, f"network_{c['nlp_extractor']}.pkl"), "rb") as f:
                    dn = pickle.load(f)
                pool += [(sub, nid, copy.deepcopy(nd["data"])) for nid, nd in dn.network.nodes(data=True)]
        env.dynamic_join_donor_pool = pool

    np.random.seed(7)
    torch.manual_seed(7)
    import random as _r; _r.seed(7)
    action_dim = env.action_space.shape[0]

    print(f"=== condition: {label} (dynamic_mode={dynamic_mode}, patch={patch_enabled}) ===")
    print(f"{'ep':>3} {'start_nodes':>11} {'start_vulns':>11} {'end_nodes':>9} {'end_vulns':>9}")
    start_nodes_series, start_vulns_series = [], []
    for ep in range(20):
        env.reset()
        start_nodes = len(list(env.environment.nodes()))
        start_vulns = total_vulns(env)
        start_nodes_series.append(start_nodes)
        start_vulns_series.append(start_vulns)
        last_nodes, last_vulns = start_nodes, start_vulns
        for step in range(300):
            a = np.random.uniform(-1, 1, size=action_dim).astype(np.float32)
            obs, reward, done, info = env.step(a)
            last_nodes = len(list(env.environment.nodes()))
            last_vulns = total_vulns(env)
            if done:
                break
        print(f"{ep:>3} {start_nodes:>11} {start_vulns:>11} {last_nodes:>9} {last_vulns:>9}")
    print(f"\nstart_nodes constant across 20 episodes: {len(set(start_nodes_series))==1} "
          f"(all == pristine {PRISTINE_NODES}: {all(x==PRISTINE_NODES for x in start_nodes_series)})")
    print(f"start_vulns constant across 20 episodes: {len(set(start_vulns_series))==1} "
          f"(all == pristine {PRISTINE_VULNS}: {all(x==PRISTINE_VULNS for x in start_vulns_series)})")
    print()


run_condition("MEMBERSHIP", "both", False)
run_condition("PROPERTY", "none", True)
