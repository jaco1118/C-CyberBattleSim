# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    cyberbattle_env_compressed.py
    Class containing the sub-class of the CyberBattleEnv with the compressed environment.
    This environment is suited for graph and vulnerabilities invariant agents that are independent of the topology of application.
    The graph is compressed into an embedding by the GAE starting from node embeddings and the action space is a concatenation of embeddings:
    - source node embedding: GAE embedding
    - target node embedding: GAE embedding
    - vulnerability embedding: NLP extracted embedding
    - outcome embedding: one-hot encoding of the possible outcomes
    The action space is continuous and the closest action is selected based on a distance metric.
"""

import time
from typing import Dict
import networkx as nx
import numpy as np
import sys
import os
from collections import defaultdict, namedtuple
import copy
from typing import TypedDict
import torch
from gym import spaces
import numpy
from scipy.spatial import distance as distance_cosine
from torch_geometric.utils import from_networkx
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)
from cyberbattle._env.cyberbattle_env import CyberBattleEnv # noqa: E402
from cyberbattle.simulation.model import Collection, CredentialAccess, Discovery, Reconnaissance, DenialOfService, PrivilegeEscalation, Persistence, LateralMove, Exfiltration, \
    DefenseEvasion # noqa: E402
from cyberbattle.simulation import model # noqa: E402
from cyberbattle.utils.encoding_utils import map_outcome_to_string # noqa: E402
from cyberbattle.utils.data_utils import flatten_dict_with_arrays, flatten # noqa: E402
from cyberbattle.utils.drift_logger import DriftLogger # noqa: E402

# Format of the info dict returned by the step function
StepInfo = TypedDict(
    'StepInfo', {
        'description': str,
        'duration_in_ms': float,
        'step_count': int,
        'network_availability': float,
        'source_node': str,
        'target_node': str,
        'source_node_tag': str,
        'target_node_tag': str,
        'vulnerability': str,
        'vulnerability_type': str,
        'outcome': str,
        'outcome_class': model.VulnerabilityOutcome,
        'end_episode_reason': int,
        'min_distance_action': float,
        # Diagnostic-only: comma-joined change_type(s) fired by maybe_apply_dynamic_step this
        # step ("property"/"membership_leave"/"membership_join"), or None if none fired. Always
        # populated (cheap), regardless of drift_logging -- see CyberBattleEnv._last_dynamic_events.
        'change_type': object,
    })

# Diagnostic-only (drift instrumentation): a single encode() snapshot, decomposed into the
# per-node embeddings dict, the combined (concatenated) pooled vector, the same pooled vector
# split per aggregation slice, and the discovered-subgraph size actually passed to pooling.
_DriftSnapshot = namedtuple("_DriftSnapshot", ["node_embeddings", "combined", "slices", "n_discovered"])


class CyberBattleCompressedEnv(CyberBattleEnv):
    """OpenAI Gym environment interface to the CyberBattle simulation.

    # Observation
        Graph embedding (and eventually node embedding of the node of interest if goal is node-specific) + number of discovered nodes + number of owned nodes

    # Actions
        Source node embedding x target node embedding x vulnerability language embedding x outcome one-hot embedding
    """

    @property
    def name(self) -> str:
        return "CyberBattleCompressedEnv"

    def __init__(self,
                 edge_feature_aggregations = None, # which aggregation(s) to use for the edge embeddings in the graph, determine also the dimension of the edge embeddings
                 graph_embeddings_aggregations = None, # which aggregation(s) to use for the node embeddings in the graph, determine also the dimension of the node embeddings
                 node_embeddings_dimensions=64, # dimension of the node embeddings, determined by the GAE Encoder output size
                 outcome_dimensions=10,  # number of possible outcomes
                 discrete_features=None, # additional features to be added to the observation space
                 # whether PCA reduction has been performed on the vulnerability embeddings during graph generation
                 pca_components=768,
                 distance_metric='cosine', # in the action space to determine the closest action
                 sample_subset_samples=False, # use a sample of actions in the action space at every timestep to reduce the number of points and hence the distance calculation time load
                 remove_all_obstacles=False, # flag that removes the "obstacles" for a given goal, e.g. all disruption actions for control games
                 remove_main_obstacles=False, # flag that removes the "main obstacles" for a given goal, e.g. all disruption actions that kill the starter node for control games
                 precise_action_space_positions=False, # flag to set whether each time node embeddings change, the action space should be updated (precise but time intensive)
                 precise_graph_encoding=False, # flag to set whether each time node embeddings change, the graph should be re-encoded (precise but time intensive)
                 # --- Diagnostic-only drift instrumentation (Phase 1, thesis pivot) ---
                 # Off by default and must not alter training behaviour, reward, or agent
                 # decisions when off: step() takes the exact same code path (one encode() call
                 # at most on any given step, as today) when drift_logging=False.
                 drift_logging=False, # enable the 3-snapshot (h1/h2/h3) drift-logging protocol in step()
                 drift_log_path=None, # CSV path to flush logged rows to; None -> in-memory only (self._drift_logger.all_rows), no file written
                 drift_sample_rate=1, # log every step if 1, every Nth step if >1 (both the extra encode() calls and the CSV row are skipped on non-sampled steps)
                 drift_run_id=None, # opaque identifiers threaded into every logged row for downstream analysis; this env has no
                 drift_seed=None,   # inherent notion of "seed" or "run_id" (those are set by the caller, e.g. an evaluation script),
                 drift_scenario_id=None, # so these are just passed through as-is
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.env_type = "compressed"
        edge_feature_aggregations = edge_feature_aggregations or ["mean"]
        graph_embeddings_aggregations = graph_embeddings_aggregations or ["mean", "max", "min"]
        discrete_features = discrete_features or ["owned_nodes", "discovered_nodes"]
        # aggregations to be used to put embeddings on the edges
        self.edge_feature_aggregations = edge_feature_aggregations
        # aggregations to be used to put embeddings on the nodes
        self.graph_embeddings_aggregations = graph_embeddings_aggregations
        self.node_embeddings_dimensions = node_embeddings_dimensions  # default choice for GAE
        self.distance_metric = distance_metric
        self.outcome_dimensions = outcome_dimensions
        self.discrete_features = discrete_features
        # determine whether PCA was used in order to calculate the right dimensions
        self.vulnerability_embeddings_dimensions = pca_components
        self.sample_subset_samples = sample_subset_samples
        self.remove_all_obstacles = remove_all_obstacles
        self.remove_main_obstacles = remove_main_obstacles
        self.precise_action_space_positions = precise_action_space_positions
        self.precise_graph_encoding = precise_graph_encoding

        # action space is continuous within the cartesian product of the embeddings of the source node, target node, vulnerability, and outcome (one-hot embedding of 9 discrete outcomes)
        self.action_space = spaces.Box(low=-4, high=4, # using -4, +4 assuming the last layer ensure normalization and most points lie in the normal space
                                           shape=(self.node_embeddings_dimensions * 2 + self.vulnerability_embeddings_dimensions + self.outcome_dimensions,),
                                           dtype=numpy.float32)

        if self.verbose > 1:
            self.logger.info("Action space: " + str(self.action_space))

        if self.goal.endswith("node"):
            graph_box_space = spaces.Box(
                    low=-16, high=16,
                    shape=(self.node_embeddings_dimensions * len(self.graph_embeddings_aggregations) 
                        + self.node_embeddings_dimensions  # owned_not_root_mean
                        + self.node_embeddings_dimensions,), # interest node
                    dtype=numpy.float64
                )
        else:
            graph_box_space = spaces.Box(
                    low=-16, high=16,
                    shape=(self.node_embeddings_dimensions * len(self.graph_embeddings_aggregations)
                        + self.node_embeddings_dimensions,), # owned_not_root_mean
                    dtype=numpy.float64
                )

        # observation space is a dictionary with the graph embeddings and the discrete features desired among options available
        self.observation_space = spaces.Dict({
            "graph_embeddings": graph_box_space,
            "discrete_features": spaces.Box(
                low=0, high=300,
                shape=(len(self.discrete_features),),
                dtype=numpy.float64
            ),
        })

        if self.verbose > 1:
            self.logger.info("Observation space: " + str(self.observation_space))

        self.graph_encoder_time = 0
        self.action_calculation_time = 0
        self.action_space_creation_time = 0
        self.update_evolving_visible_graph_time = 0
        self.inner_step_time = 0
        self.balance_action_space_time = 0
        # embeddings created just once and re-use in order to avoid recalculation
        self.create_vulnerabilities_embeddings()
        self.create_vulnerabilities_embeddings_per_node_type()

        # --- Diagnostic-only drift instrumentation state ---
        self.drift_logging = drift_logging
        self.drift_sample_rate = max(1, drift_sample_rate)
        self.drift_run_id = drift_run_id
        self.drift_seed = drift_seed
        self.drift_scenario_id = drift_scenario_id
        self._drift_logger = DriftLogger(drift_log_path, self.graph_embeddings_aggregations) if self.drift_logging else None
        self._episode_count = -1  # incremented to 0 on the first reset()
        self._last_connectivity_event = None  # set by add_edge_evolving_visible_graph, consumed/reset once per step
        self._drift_acted_on_nodes = set()  # nodes acted on (as source or target) so far this episode
        self._drift_prev_step_removed_ids = []  # membership_leave node IDs from the previous logged step, for action-validity checking
        # Deferred change attribution (Phase 1 follow-up): per-episode registry of touched nodes
        # not yet visible (not in h2, i.e. not yet discovered before this step's dynamic mutation)
        # at the moment their event fired. Each entry: {"event_id", "change_type", "node_id",
        # "step_fired"}. Resolved (popped) the step the node enters self.discovered_nodes; any
        # left over at episode end are flushed as permanently unattributed -- see reset() below.
        self._drift_pending_nodes = []
        self._drift_event_counter = 0  # unique event_id source, shared by a "fired" row and its later "attributed"/flush row

    # Reset function calling the original reset and preparing the overlay graph to encode and continuous observation
    def reset(self, **kwargs):
        if self.verbose > 1:
            if self.graph_encoder_time != 0:
                self.logger.info(f"Graph embedding time in the episode: {self.graph_encoder_time}")
                self.logger.info(f"Action calculation time in the episode: {self.action_calculation_time}")
                self.logger.info(f"Action space creation time in the episode: {self.action_space_creation_time}")
                self.logger.info(f"Update evolving visible graph time in the episode: {self.update_evolving_visible_graph_time}")
                self.logger.info(f"Inner step time in the episode: {self.inner_step_time}")
        super().reset_env()
        # keep track of the processed pairs (source node, target node) for which actions have been added to the continuous action space to avoid to re-add them
        self.processed_pairs = set()
        self.reset_evolving_visible_graph()
        self.action_embeddings = {}
        self.exploited_vulnerabilities_per_node_pairs = {}
        # Diagnostic-only drift instrumentation: flush any events from the episode that just
        # ended whose touched node was never discovered (permanently unattributed -- e.g. a
        # joined node the agent never happened to reconnaissance the right parent for), BEFORE
        # clearing the registry for the new episode. A large unattributed fraction is itself a
        # headline finding (the agent doesn't just perceive joins late, it may never perceive
        # them at all), so this is surfaced via both the flushed rows and a summary log line.
        if self.drift_logging and self._drift_pending_nodes:
            for pending in self._drift_pending_nodes:
                self._drift_logger.log(self._build_drift_row(
                    event_phase="fired", event_id=pending["event_id"], step_fired=pending["step_fired"],
                    visibility_lag_steps=None, touched_node_visible=False, attributed=False,
                    node_origin_is_join=(pending["change_type"] == "membership_join"),
                    change_type=pending["change_type"], change_fired=True, n_touched_nodes=1,
                    relevant=None, episode_override=self._episode_count, step_override=pending["step_fired"],
                ))
            self.logger.info(
                "[DriftLogging] Episode %d ended with %d permanently unattributed change event(s) "
                "(touched node never entered the discovered set): %s",
                self._episode_count, len(self._drift_pending_nodes),
                [(p["change_type"], p["node_id"], p["step_fired"]) for p in self._drift_pending_nodes]
            )
        self._episode_count += 1
        self._last_connectivity_event = None
        self._drift_acted_on_nodes = set()
        self._drift_prev_step_removed_ids = []
        self._drift_pending_nodes = []
        self.graph_encoder_time = 0
        self.action_calculation_time = 0
        self.action_space_creation_time = 0
        self.update_evolving_visible_graph_time = 0
        self.balance_action_space_time = 0
        self.inner_step_time = 0
        start_time = time.time()
        self.node_embeddings, self.observation = self.encode(self.evolving_visible_graph) # at the beginning only source node
        self.graph_encoder_time += time.time() - start_time
        start_time = time.time()
        self.create_continuous_action_space() # at the beginning only local vulnerabilities of source node
        self.action_space_creation_time += time.time() - start_time
        self.edges = []
        self.observation = {
            "graph_embeddings": self.observation,
            "discrete_features": self.create_discrete_features()
        }
        return self.observation

    # Reset the evolving visible graph to the initial state with only the starter node
    def reset_evolving_visible_graph(self):
        self.evolving_visible_graph = nx.DiGraph()
        self.evolving_visible_graph.clear()
        self.add_node_evolving_visible_graph(self.starter_node) # initial node

    # Function to get the feature vector of a node flattened as an array
    def get_node_feature_vector(self, node_id):
        node_features_dict = self.convert_node_info_to_observation(self.get_node(node_id))
        flattened_node_features_dict = flatten_dict_with_arrays(node_features_dict)
        node_features_array = numpy.array(
            flatten([flattened_node_features_dict[key] for key in flattened_node_features_dict]), dtype=numpy.float32)
        return node_features_array

    # Function to add a node to the evolving visible graph with its feature vector
    def add_node_evolving_visible_graph(self, node_id):
        self.evolving_visible_graph.add_node(node_id, x=self.get_node_feature_vector(node_id))

    def remove_node_evolving_visible_graph(self, node_id):
        if node_id in self.evolving_visible_graph.nodes():
            self.evolving_visible_graph.remove_node(node_id)  # networkx auto-removes its edges
            
    # Function to update the node in the evolving visible graph with its feature vector
    def update_node_evolving_visible_graph(self, node_id):
        self.evolving_visible_graph.nodes[node_id].update({'x': self.get_node_feature_vector(node_id)})

    # Function to add an edge to the evolving visible graph with the vulnerabilities embeddings
    def add_edge_evolving_visible_graph(self, source_node, target_node, vuln_key):
        aggregation_functions = {
            "mean": np.mean,
            "sum": np.sum
        }
        if source_node not in self.evolving_visible_graph.nodes():
            self.add_node_evolving_visible_graph(source_node)
        if target_node not in self.evolving_visible_graph.nodes():
            self.add_node_evolving_visible_graph(target_node)
        self.edges.append((source_node, target_node, vuln_key))
        if self.evolving_visible_graph.has_edge(source_node, target_node):
            # re-merge vulnerabilities with aggregators if a vulnerability is already present such that edge feature vector is aggregation of vulnerabilities exploited
            if not self.exploited_vulnerabilities_per_node_pairs.get(source_node).get(target_node):
                self.exploited_vulnerabilities_per_node_pairs[source_node][target_node] = []
            self.exploited_vulnerabilities_per_node_pairs[source_node][target_node].append(self.vulnerabilities_embeddings[vuln_key])
            edge_embedding = []
            for edge_aggregation in self.edge_feature_aggregations:
                edge_embedding.append(aggregation_functions[edge_aggregation](self.exploited_vulnerabilities_per_node_pairs[source_node][target_node], axis=0))
            self.evolving_visible_graph[source_node][target_node]["vulnerabilities_embeddings"] = np.concatenate(edge_embedding)
            # Diagnostic-only (drift instrumentation): change_type="connectivity" tag. Note this
            # fires as a consequence of the AGENT's own successful action (self.reward > 0, the
            # caller's guard), i.e. within the same step's agent-footprint window (h1->h2), not
            # the dynamic-mutation window (h2->h3) that maybe_apply_dynamic_step's events occupy.
            # There is currently no independent, agent-action-decoupled connectivity-change
            # mechanism in this codebase; this is the only existing "connectivity" mutation site.
            self._last_connectivity_event = {"change_type": "connectivity", "node_ids": [source_node, target_node]}
            return True
        else:
            # no edge already exists between two nodes hence create first edge
            self.evolving_visible_graph.add_edge(source_node, target_node)
            self.exploited_vulnerabilities_per_node_pairs[source_node] = {}
            self.exploited_vulnerabilities_per_node_pairs[source_node][target_node] = [self.vulnerabilities_embeddings[vuln_key]]
            edge_embedding = []
            for edge_aggregation in self.edge_feature_aggregations:
                edge_embedding.append(
                    aggregation_functions[edge_aggregation](self.exploited_vulnerabilities_per_node_pairs[source_node][target_node],
                                                            axis=0))
            self.evolving_visible_graph[source_node][target_node]["vulnerabilities_embeddings"] = np.concatenate(
                edge_embedding)
            self._last_connectivity_event = {"change_type": "connectivity", "node_ids": [source_node, target_node]}
            return True

    def remove_node_dynamic(self, node_id):
        # shared purge: ground-truth graph, discovered/owned tracking, win-condition denominators
        if not self.remove_node_common(node_id):
            return
        # the agent's visible/encoded graph (feeds the observation)
        self.remove_node_evolving_visible_graph(node_id)
        # purge every stale reference to the removed node so it cannot resurface as a "ghost"
        # node embedding or action key on a later step (which would crash get_node with a KeyError)
        self.node_embeddings.pop(node_id, None)
        self.action_embeddings = {
            k: v for k, v in self.action_embeddings.items()
            if k[0] != node_id and k[1] != node_id
        }
        self.processed_pairs = {p for p in self.processed_pairs if node_id not in p}
        self.edges = [e for e in self.edges if e[0] != node_id and e[1] != node_id]
        self.exploited_vulnerabilities_per_node_pairs.pop(node_id, None)
        for source in list(self.exploited_vulnerabilities_per_node_pairs.keys()):
            self.exploited_vulnerabilities_per_node_pairs[source].pop(node_id, None)
        # NOTE: the re-encode + action-space rebuild happen in step() right after the dynamic change,
        #       so the observation returned this step reflects the post-removal graph

    # Dynamically add a node (called by the base class's dynamic-join mechanism). vulnerabilities_embeddings
    # and vulnerabilities_embeddings_per_node_type are each built ONCE, at construction, from this
    # instance's own nodes (create_vulnerabilities_embeddings/_per_node_type) -- a joined node's
    # vulnerability IDs are near-certainly not already present, so refresh_vulnerabilities_embeddings_for_node
    # must run before the node can ever be added to evolving_visible_graph (whenever that later
    # happens, via normal discovery), since that path looks embeddings up with no fallback
    # (convert_node_info_to_observation's mean_vulnerabilities_embedding pooling).
    def add_node_dynamic(self, node_id, node_info):
        if not self.add_node_common(node_id, node_info):
            return
        # Top up the embeddings dict now (cheap, no graph-visibility side effect), but deliberately
        # do NOT add the node to evolving_visible_graph here -- it starts undiscovered by design
        # (add_node_common intentionally skips discovered_nodes too), and update_evolving_visible_graph_after_step
        # already adds any newly-discovered node to evolving_visible_graph generically once the
        # agent actually finds it via the injected Reconnaissance outcome. Adding it here early
        # would leak an undiscovered node into the whole-graph GAE embedding.
        self.refresh_vulnerabilities_embeddings_for_node(node_id)
        # re-encode + action-space rebuild happen in step() right after the dynamic change, same as removal

    # Function leveraging the graph parameter and the GAE encoder to encode the graph and gather node embeddings
    def encode(self, graph):
        # Use the GAE Encoder to encode the graph
        node_embeddings = {}
        if self.goal.endswith("node"): # node-specific goal game
            graph = copy.deepcopy(graph) # if goal node is not in the graph, add it in order to have a pure node embedding without mixing with the graph neighbors
            if self.interest_node not in graph.nodes():
                self.add_node_evolving_visible_graph(self.interest_node)
        data = from_networkx(graph)
        if 'vulnerabilities_embeddings' not in data: # case where the graph has no edges
            data.vulnerabilities_embeddings = torch.zeros(self.vulnerability_embeddings_dimensions, dtype=torch.float32)
        data.x = data.x.float()
        data.vulnerabilities_embeddings = data.vulnerabilities_embeddings.float()

        z = self.graph_encoder(data.x, data.edge_index, data.vulnerabilities_embeddings)

        running_nodes = [node for node in graph.nodes() if
                         self.get_node(node).status == model.MachineStatus.Running]

        if not running_nodes: # if no running nodes, return empty embeddings
            empty_embedding = np.zeros(self.node_embeddings_dimensions, dtype=np.float32)
            concatenated_result = np.concatenate([empty_embedding for _ in self.graph_embeddings_aggregations])
            # NEW: also account for the owned_not_root slot in the empty-case shape
            concatenated_result = np.concatenate([concatenated_result, empty_embedding])
            if self.goal.endswith("node"):
                concatenated_result = np.concatenate([concatenated_result, empty_embedding])
            return node_embeddings, concatenated_result

        # Get the embeddings for the running nodes
        for node in running_nodes:
            node_index = list(graph.nodes()).index(node)
            node_embedding = z[node_index].detach().numpy()
            node_embeddings[node] = node_embedding

        embeddings_array = np.array([node_embeddings[node] for node in node_embeddings], dtype=np.float32)
        # perform aggregations across all node embeddings to get the graph embedding
        graph_embeddings = []
        for agg_type in self.graph_embeddings_aggregations:
            if agg_type == "mean":
                graph_embeddings.append(np.average(embeddings_array, axis=0))
            elif agg_type == "sum":
                graph_embeddings.append(np.sum(embeddings_array, axis=0))
            elif agg_type == "min":
                graph_embeddings.append(np.min(embeddings_array, axis=0))
            elif agg_type == "max":
                graph_embeddings.append(np.max(embeddings_array, axis=0))
            else:
                raise ValueError(f"Unknown aggregation type: {agg_type}")

        # Build the base observation FIRST
        observation_embedding = np.concatenate(graph_embeddings)

        owned_not_root_nodes = [
            node for node in running_nodes
            if node in self.owned_nodes and node != self.starter_node
            and self.get_node(node).privilege_level != model.PrivilegeLevel.ROOT
        ]
        if owned_not_root_nodes:
            # Deterministic: always point at the same one until it's resolved, then move to the next
            target_node = sorted(owned_not_root_nodes)[0]  # or pick by highest node_info.value for priority
            next_escalation_target = node_embeddings[target_node]
        else:
            next_escalation_target = np.zeros(self.node_embeddings_dimensions, dtype=np.float32)

        observation_embedding = np.concatenate([observation_embedding, next_escalation_target])

        # THEN handle the node-specific goal case (unchanged logic, now operating on the updated observation_embedding)
        if self.goal.endswith("node"):
            if self.interest_node in running_nodes:
                observation_embedding = np.concatenate([observation_embedding, node_embeddings[self.interest_node]])
            else:
                observation_embedding = np.concatenate([observation_embedding, np.zeros(self.node_embeddings_dimensions, dtype=np.float32)])
            if self.interest_node not in self.discovered_nodes and self.interest_node in node_embeddings: # remove if fictious interest node was added
                node_embeddings.pop(self.interest_node)
        return node_embeddings, observation_embedding
        
    # discrete features to be added to the observation vector to provide additional information to understand semantics of graph embedding
    def create_discrete_features(self):
        discrete_features = []
        # should be selected among these options
        if 'discovered_nodes' in self.discrete_features:
            discrete_features.append(len(self.discovered_nodes))
        if 'owned_nodes' in self.discrete_features:
            discrete_features.append(len(self.owned_nodes))
        return numpy.array(discrete_features)

    # Create the feature vector of nodes encoding properly all elements
    def convert_node_info_to_observation(self, node_info) -> Dict:
        firewall_config_array = [
            0 for _ in range(2 * self.max_services_per_node)
        ]

        if node_info.visible:
            # include firewall information if visibility acquired on the node
            for config in node_info.firewall.incoming:
                permission = config.permission.value
                if self.get_service_index(config.port, node_info) != -1 and self.get_service_index(config.port, node_info) < self.max_services_per_node:
                    firewall_config_array[self.get_service_index(config.port, node_info)] = permission
            for config in node_info.firewall.outgoing:
                permission = config.permission.value
                if self.get_service_index(config.port, node_info) != -1 and self.get_service_index(config.port, node_info) < self.max_services_per_node:
                    firewall_config_array[self.max_services_per_node + self.get_service_index(config.port,
                                                                                                           node_info)] = permission
        # include listening services information
        listening_services_running_array = [0 for _ in range(
            self.max_services_per_node)]  # array indicating if each service is listening or not
        listening_services_fv_array = [0.0 for _ in range(self.vulnerability_embeddings_dimensions)]

        if node_info.visible:
            # fill services info in case of visibility on the node
            for i, service in enumerate(node_info.services):
                if i >= self.max_services_per_node:
                    break
                feature_vector = service.feature_vector
                listening_services_running_array[i] = int(service.running)
                for j in range(self.vulnerability_embeddings_dimensions):
                    listening_services_fv_array[j] += feature_vector[j]
            if len(node_info.services) > 0:
                for i in range(self.vulnerability_embeddings_dimensions):
                    listening_services_fv_array[i] /= len(node_info.services)

        # include mean of vulnerabilities embeddings ( to have a single array independent of the number of vulnerabilities)
        # GAE requires all node feature vectors to be of the same size, hence we need to pool the vulnerabilities embeddings
        mean_vulnerabilities_embedding = [0.0 for _ in range(self.vulnerability_embeddings_dimensions)]
        if len(node_info.vulnerabilities) > 0:
            for vulnerability in node_info.vulnerabilities:
                mean_vulnerabilities_embedding = [
                    x + y for x, y in
                    zip(mean_vulnerabilities_embedding, self.vulnerabilities_embeddings[vulnerability])
                ]
            mean_vulnerabilities_embedding = [embedding / len(node_info.vulnerabilities) for embedding in
                                              mean_vulnerabilities_embedding]

        return {
            'firewall_config_array': firewall_config_array,
            'listening_services_running_array': listening_services_running_array,
            'visible': int(node_info.visible),
            'persistence': int(node_info.persistence),
            'data_collected': int(node_info.data_collected),
            'data_exfiltrated': int(node_info.data_exfiltrated),
            'defense_evasion': int(node_info.defense_evasion),
            'reimageable': int(node_info.reimageable),
            'privilege_level': int(node_info.privilege_level),
            'status': node_info.status.value,
            'value': node_info.value,
            'sla_weight': node_info.sla_weight,
            'listening_services_fv_array': listening_services_fv_array,
            'mean_vulnerabilities_embedding': mean_vulnerabilities_embedding
        }

    # Wrapper function used in case it is required to call only the step on the env without the compressed logic
    def step_env(self, source_node, target_node, vulnerability_ID, outcome):
        super().step_attacker_env(source_node, target_node, vulnerability_ID, outcome)
        return self.done or self.truncated

    # Step function that takes an action vector, calls the original step function, and update the graph
    # New observation and reward are computed and returned
    def step(self, action_vector):
        start_time = time.time()
        # find the closest action to the action vector
        source_node, target_node, vulnerability_ID, outcome, distance = self.find_closest_action_embedding(copy.deepcopy(action_vector))
        self.action_calculation_time += time.time() - start_time

        # --- Diagnostic-only drift instrumentation: decide up front whether this step is
        # sampled for logging. step_attacker_env (below) unconditionally increments
        # self.stepcount by 1 as its very first act, so self.stepcount + 1 is this step's index
        # without needing to call it first. When drift_logging is False (the default), none of
        # the drift_* blocks below run at all -- the code path is identical to before this
        # instrumentation was added. ---
        log_this_step = self.drift_logging and ((self.stepcount + 1) % self.drift_sample_rate == 0)
        # A pending (not-yet-discovered) event must always get its attribution step logged even
        # on a non-sampled step, or the registry leaks and visibility_lag_steps is wrong -- so h1/
        # h2 are forced whenever the registry is non-empty, independent of drift_sample_rate. h3
        # (needed only for this step's own "regular" event row) stays strictly log_this_step-gated.
        need_h1_h2 = self.drift_logging and (log_this_step or bool(self._drift_pending_nodes))
        if need_h1_h2:
            drift_h1 = self._drift_snapshot_from_cache()
            drift_discovered_before_step = set(self.discovered_nodes)
            action_referenced_removed_entity = (
                (source_node in self._drift_prev_step_removed_ids or target_node in self._drift_prev_step_removed_ids)
                if self._drift_prev_step_removed_ids else None
            )
            self._last_connectivity_event = None  # reset before this step's agent-action window begins
        if self.drift_logging:
            self._drift_acted_on_nodes.update([source_node, target_node])  # cheap; kept unconditional so relevance tagging stays accurate regardless of sampling

        start_time = time.time()
        super().step_attacker_env(source_node, target_node, vulnerability_ID, outcome)
        self.inner_step_time += time.time() - start_time
        start_time = time.time()
        # eventually update the evolving visible graph
        self.update_evolving_visible_graph_after_step(source_node, target_node, vulnerability_ID)
        self.update_evolving_visible_graph_time += time.time() - start_time
        action_changed_graph = self.action_changes_evolving_visible_graph(outcome)
        if action_changed_graph or self.static_defender_agent: # if the action changes the graph, re-encode the graph, or if the defender acted so we do not know what it has done
            if self.verbose > 2:
                if action_changed_graph:
                    self.logger.info("Re-encoding the graph since there was one action that changed the graph")
                elif self.static_defender_agent:
                    self.logger.info("Re-encoding the graph since the defender may have been acted with modifying actions")

            start_time = time.time()
            # need to re-encode if graph has changed
            self.node_embeddings, self.observation = self.encode(self.evolving_visible_graph)
            self.graph_encoder_time += time.time() - start_time
            self.observation = {
                "graph_embeddings": self.observation,
                "discrete_features": self.create_discrete_features()
            }
            start_time = time.time()
            # potentialy add new points in the continuous action space
            if action_changed_graph:
                if self.precise_action_space_positions:
                    self.create_continuous_action_space(nodes_to_recalculate=[source_node, target_node])
                else:
                    self.create_continuous_action_space() #nodes_to_recalculate=[source_node, target_node])
            elif self.static_defender_agent:
                if self.precise_action_space_positions:
                    self.create_continuous_action_space(nodes_to_recalculate=self.changed_nodes)
                else:
                    self.create_continuous_action_space()
            self.action_space_creation_time += time.time() - start_time

        if need_h1_h2:
            # h2: the agent's own footprint. Reuse the encode() result above if it already ran
            # (action_changed_graph or a static defender acted); otherwise force one purely for
            # logging -- the only extra encode() call this adds on a step that wouldn't
            # otherwise re-encode, and it never touches self.observation/self.node_embeddings.
            if action_changed_graph or self.static_defender_agent:
                drift_h2 = self._drift_snapshot_from_cache()
            else:
                drift_h2 = self._drift_snapshot_fresh()
            drift_agent_action_succeeded = self.reward > 0
            drift_connectivity_event = self._last_connectivity_event
            # Deferred change attribution: any node that entered self.discovered_nodes this step
            # (via this step's action, e.g. a Reconnaissance outcome -- the only place
            # discovered_nodes ever grows) either resolves a pending registry entry (a previously
            # non-visible membership_join) or is an ordinary never-removed node being discovered
            # for the first time (the ordinary-discovery control group from Step 4 of the spec).
            # Both are logged uniformly, distinguished by node_origin_is_join.
            drift_newly_discovered = set(self.discovered_nodes) - drift_discovered_before_step
            if drift_newly_discovered:
                self._log_attribution_rows(drift_h1, drift_h2, drift_newly_discovered,
                                            drift_agent_action_succeeded, action_referenced_removed_entity)

         # add term proportional to the distance (negative coefficient)
        self.reward += self.penalties_dict['distance_penalty'] * distance
        if self.verbose > 2:
            self.logger.info("Penalty (distance penalty) : += %s * %s", self.penalties_dict['distance_penalty'], distance)
        if self.verbose > 2:
            self.logger.info("Reward of the step: %s", self.reward)
        # Captured at exactly the point the original code built StepInfo (before
        # maybe_apply_dynamic_step below), so duration_in_ms is unaffected by StepInfo's
        # construction being deferred a few lines to pick up this step's change_type.
        duration_in_ms = time.time() - start_time

        nodes_changed = self.maybe_apply_dynamic_step()  # removed and/or joined node IDs this step
        # h2->h3 window events (property/membership_leave/membership_join), if any fired this step
        dynamic_events = list(self._last_dynamic_events)
        if nodes_changed:
            # a node was removed or joined after the observation/action space were already built
            # this step; re-encode and rebuild so we do not hand back a stale observation or action space
            self.node_embeddings, self.observation = self.encode(self.evolving_visible_graph)
            self.observation = {
                "graph_embeddings": self.observation,
                "discrete_features": self.create_discrete_features()
            }
            self.create_continuous_action_space()

        info = StepInfo(
            description='CyberBattleEnvCompressed step info',
            duration_in_ms=duration_in_ms,
            step_count=self.stepcount,
            source_node=source_node,
            target_node=target_node,
            source_node_tag= self.get_node(source_node).tag,
            target_node_tag= self.get_node(target_node).tag,
            vulnerability=vulnerability_ID,
            vulnerability_type=self.vulnerability_type,
            network_availability=self.network_availability,
            outcome_class=outcome,
            outcome=map_outcome_to_string(outcome),
            end_episode_reason=self.end_episode_reason,
            min_distance_action=distance,
            change_type=(",".join(event["change_type"] for event in dynamic_events) if dynamic_events else None)
        )

        if log_this_step:
            # h3: the dynamic-change event's footprint. If nodes_changed is empty this is the
            # encoder determinism sanity check (should be ~0), NOT the noise floor -- the real
            # noise floor is agent_drift (h1->h2), computed unconditionally either way.
            if nodes_changed:
                drift_h3 = self._drift_snapshot_from_cache()
            else:
                drift_h3 = self._drift_snapshot_fresh()
            self._log_drift_rows(
                drift_h1, drift_h2, drift_h3, dynamic_events, drift_connectivity_event,
                drift_agent_action_succeeded, action_referenced_removed_entity,
            )
            self._drift_prev_step_removed_ids = [
                node_id for event in dynamic_events if event["change_type"] == "membership_leave"
                for node_id in event["node_ids"]
            ]

        return self.observation, self.reward, self.done or self.truncated, info

    # =========================================================================================
    # Diagnostic-only drift instrumentation (Phase 1, thesis pivot). Everything below is only
    # ever invoked when drift_logging=True (see step() above) and has no effect on training
    # behaviour, reward, or agent decisions when off.
    # =========================================================================================

    # Splits a pooled observation vector (as returned by encode()) into its per-aggregation-slice
    # components plus the combined (concatenated) pooling-only vector, excluding the
    # next_escalation_target / interest_node tail that encode() appends after the pooled slices
    # (those are single-node lookups, not pooling output, so they are not part of h_G).
    def _split_pooling_slices(self, observation_embedding):
        dim = self.node_embeddings_dimensions
        slices = {}
        for i, agg in enumerate(self.graph_embeddings_aggregations):
            slices[agg] = observation_embedding[i * dim:(i + 1) * dim]
        combined = observation_embedding[:len(self.graph_embeddings_aggregations) * dim]
        return slices, combined

    # h1/h2/h3 snapshot "from cache": reuses node_embeddings/observation already sitting in
    # self.* (because the production code path either just computed them at reset()/the start of
    # this step, or already re-encoded this step for a real, non-diagnostic reason) rather than
    # calling encode() again. This is what keeps drift_logging from doubling the encoder cost on
    # every step that would have re-encoded anyway.
    def _drift_snapshot_from_cache(self):
        slices, combined = self._split_pooling_slices(self.observation["graph_embeddings"])
        return _DriftSnapshot(node_embeddings=dict(self.node_embeddings), combined=combined,
                               slices=slices, n_discovered=len(self.node_embeddings))

    # h1/h2/h3 snapshot "fresh": forces an extra encode() call purely for logging, for a snapshot
    # point where the production code path did NOT need to re-encode (e.g. h2 on a step whose
    # action didn't change the visible graph). Deliberately does not touch
    # self.node_embeddings/self.observation -- the returned observation must stay governed
    # exclusively by the pre-existing conditional re-encode logic above.
    def _drift_snapshot_fresh(self):
        node_embeddings, observation_embedding = self.encode(self.evolving_visible_graph)
        slices, combined = self._split_pooling_slices(observation_embedding)
        return _DriftSnapshot(node_embeddings=node_embeddings, combined=combined,
                               slices=slices, n_discovered=len(node_embeddings))

    # Relevance tagging: an event is "relevant" if ANY touched node is owned, discovered, or has
    # been acted on (as source or target) earlier this episode -- an "any", not an "all", since
    # Poisson batch leave/join events touch multiple nodes at once.
    def _is_event_relevant(self, node_ids):
        return any(
            node_id in self.owned_nodes or node_id in self.discovered_nodes or node_id in self._drift_acted_on_nodes
            for node_id in node_ids
        )

    # Node-level delta vector for an event's touched nodes, h2->h3. Conventions for a node absent
    # at one of the two snapshots (implemented explicitly rather than left to crash/produce
    # garbage): membership_join treats the pre-embedding as zero (a joined node does not exist
    # before the join); membership_leave treats the post-embedding as zero (the node is gone by
    # h3); property is a normal difference. If multiple nodes are touched by one event, the SUM
    # of their delta vectors is returned (as a vector, before any norm is taken), alongside the
    # touched-node count logged separately by the caller.
    def _node_delta_vector(self, node_ids, change_type, h2, h3):
        dim = self.node_embeddings_dimensions
        total = np.zeros(dim, dtype=np.float32)
        zero = np.zeros(dim, dtype=np.float32)
        for node_id in node_ids:
            if change_type == "membership_join":
                before, after = zero, h3.node_embeddings.get(node_id, zero)
            elif change_type == "membership_leave":
                before, after = h2.node_embeddings.get(node_id, zero), zero
            else:  # "property" (and "connectivity", if ever routed through this helper)
                before, after = h2.node_embeddings.get(node_id, zero), h3.node_embeddings.get(node_id, zero)
            total = total + (after - before)
        return total

    @staticmethod
    def _rel_drift(before_vec, after_vec):
        denom = max(float(np.linalg.norm(before_vec)), 1e-12)
        return float(np.linalg.norm(after_vec - before_vec)) / denom

    # Builds and enqueues one drift-log row per h2->h3 dynamic event fired this step (or exactly
    # one "no change" row, the encoder determinism sanity check, if none fired). Per Step 4 of
    # the instrumentation spec: attenuation_ratio is never aggregated across change types here --
    # each row carries its own change_type and its own attenuation_ratio, computed from delta
    # vectors (not from separately-logged scalar norms -- norm(a)-norm(b) != norm(a-b)).
    # Canonical row template: every drift-log row (regular event/no-change, attribution, or
    # end-of-episode flush) goes through this so DriftLogger's strict schema check never sees an
    # inconsistent key set across the different call sites below. Columns not relevant to a given
    # call site are left at their default (None/0/False as appropriate).
    def _build_drift_row(
        self, *, event_phase, change_type=None, change_fired=False, n_touched_nodes=0, relevant=None,
        event_id=None, step_fired=None, visibility_lag_steps=None, touched_node_visible=None,
        attributed=None, node_origin_is_join=None,
        n_discovered=None, n_discovered_h1=None, n_discovered_h2=None, n_discovered_h3=None,
        agent_drift_full=None, change_drift_full=None, agent_drift_slices=None, change_drift_slices=None,
        norm_h1=None, norm_h2=None, norm_h3=None,
        delta_h_v_norm=None, attenuation_ratio_full=None, attenuation_ratio_slices=None,
        action_referenced_removed_entity=None, agent_action_succeeded=None,
        connectivity_event_fired=False, connectivity_n_touched_nodes=0,
        episode_override=None, step_override=None,
        norm_h1_slices=None, norm_h2_slices=None, norm_h3_slices=None,
    ):
        aggs = self.graph_embeddings_aggregations
        row = {
            "run_id": self.drift_run_id, "seed": self.drift_seed, "scenario_id": self.drift_scenario_id,
            "episode": episode_override if episode_override is not None else self._episode_count,
            "step": step_override if step_override is not None else self.stepcount,
            "n_scenario": self.num_nodes,
            "n_discovered": n_discovered, "n_discovered_h1": n_discovered_h1,
            "n_discovered_h2": n_discovered_h2, "n_discovered_h3": n_discovered_h3,
            "change_type": change_type, "change_fired": change_fired,
            "n_touched_nodes": n_touched_nodes, "relevant": relevant,
            "event_phase": event_phase, "event_id": event_id, "step_fired": step_fired,
            "visibility_lag_steps": visibility_lag_steps, "touched_node_visible": touched_node_visible,
            "attributed": attributed, "node_origin_is_join": node_origin_is_join,
            "agent_drift_full": agent_drift_full, "change_drift_full": change_drift_full,
            "norm_h1": norm_h1, "norm_h2": norm_h2, "norm_h3": norm_h3,
            "delta_h_v_norm": delta_h_v_norm, "attenuation_ratio_full": attenuation_ratio_full,
            "action_referenced_removed_entity": action_referenced_removed_entity,
            "agent_action_succeeded": agent_action_succeeded,
            "pooling_mode": "+".join(aggs),  # stand-in for the future pooling_mode flag (separate task)
            "connectivity_event_fired": connectivity_event_fired,
            "connectivity_n_touched_nodes": connectivity_n_touched_nodes,
        }
        for agg in aggs:
            row[f"agent_drift_{agg}"] = agent_drift_slices[agg] if agent_drift_slices else None
            row[f"change_drift_{agg}"] = change_drift_slices[agg] if change_drift_slices else None
            row[f"attenuation_ratio_{agg}"] = attenuation_ratio_slices[agg] if attenuation_ratio_slices else None
        # Per-slice absolute norms (STEP 2): appended last, snapshot-major, so absolute drift
        # (norm_h1_s minus norm_h2_s is NOT the right derivation -- see note below) is
        # reconstructable per slice, not just for "full". Derivation for downstream analysis:
        # abs_drift(s) = rel_drift(s) * norm_h1_s (never norm_h2_s - norm_h1_s, which is the
        # difference of two vector magnitudes, not the magnitude of their difference).
        for snapshot_label, slices_dict in (("h1", norm_h1_slices), ("h2", norm_h2_slices), ("h3", norm_h3_slices)):
            for agg in aggs:
                row[f"norm_{snapshot_label}_{agg}"] = slices_dict[agg] if slices_dict else None
        return row

    def _log_drift_rows(self, h1, h2, h3, dynamic_events, connectivity_event,
                         agent_action_succeeded, action_referenced_removed_entity):
        aggs = self.graph_embeddings_aggregations
        agent_drift_full = self._rel_drift(h1.combined, h2.combined)
        change_drift_full = self._rel_drift(h2.combined, h3.combined)
        agent_drift_slices = {agg: self._rel_drift(h1.slices[agg], h2.slices[agg]) for agg in aggs}
        change_drift_slices = {agg: self._rel_drift(h2.slices[agg], h3.slices[agg]) for agg in aggs}
        norm_h1, norm_h2, norm_h3 = (float(np.linalg.norm(h1.combined)), float(np.linalg.norm(h2.combined)),
                                     float(np.linalg.norm(h3.combined)))
        norm_h1_slices = {agg: float(np.linalg.norm(h1.slices[agg])) for agg in aggs}
        norm_h2_slices = {agg: float(np.linalg.norm(h2.slices[agg])) for agg in aggs}
        norm_h3_slices = {agg: float(np.linalg.norm(h3.slices[agg])) for agg in aggs}

        events = dynamic_events or [None]  # None -> the "no dynamic change" sanity-check row
        for event in events:
            change_type = event["change_type"] if event else None
            node_ids = event["node_ids"] if event else []
            delta_h_v = self._node_delta_vector(node_ids, change_type, h2, h3) if event else np.zeros(
                self.node_embeddings_dimensions, dtype=np.float32)
            delta_h_v_norm = float(np.linalg.norm(delta_h_v))
            delta_h_G_full = h3.combined - h2.combined
            attenuation_ratio_full = delta_h_v_norm / max(float(np.linalg.norm(delta_h_G_full)), 1e-12)
            attenuation_ratio_slices = {
                agg: delta_h_v_norm / max(float(np.linalg.norm(h3.slices[agg] - h2.slices[agg])), 1e-12)
                for agg in aggs
            }

            if event is None:
                event_phase, touched_node_visible, event_id, attributed, step_fired = "no_change", None, None, None, None
            else:
                # "was this touched node already visible BEFORE this event fired" is checked
                # against h2 (state before the dynamic mutation), not h3 (state after) --
                # checking h3 would misclassify membership_leave's departing node (gone by h3,
                # but fully visible and immediate beforehand, not pending) as needing future
                # attribution. Only a genuinely new entity (membership_join) can be absent at h2.
                not_yet_visible = [n for n in node_ids if n not in h2.node_embeddings]
                touched_node_visible = (len(not_yet_visible) == 0)
                if touched_node_visible:
                    event_phase, event_id, attributed, step_fired = "immediate", None, True, None
                else:
                    event_phase, attributed = "fired", None  # unknown yet: resolved (attributed) or flushed unattributed at episode end
                    self._drift_event_counter += 1
                    event_id = self._drift_event_counter
                    step_fired = self.stepcount
                    for node_id in not_yet_visible:
                        self._drift_pending_nodes.append({
                            "event_id": event_id, "change_type": change_type,
                            "node_id": node_id, "step_fired": step_fired,
                        })

            row = self._build_drift_row(
                event_phase=event_phase, change_type=change_type, change_fired=event is not None,
                n_touched_nodes=len(node_ids), relevant=self._is_event_relevant(node_ids) if event else None,
                event_id=event_id, step_fired=step_fired, touched_node_visible=touched_node_visible, attributed=attributed,
                node_origin_is_join=(change_type == "membership_join") if event else None,
                n_discovered=h3.n_discovered, n_discovered_h1=h1.n_discovered, n_discovered_h2=h2.n_discovered, n_discovered_h3=h3.n_discovered,
                agent_drift_full=agent_drift_full, change_drift_full=change_drift_full,
                agent_drift_slices=agent_drift_slices, change_drift_slices=change_drift_slices,
                norm_h1=norm_h1, norm_h2=norm_h2, norm_h3=norm_h3,
                delta_h_v_norm=delta_h_v_norm, attenuation_ratio_full=attenuation_ratio_full,
                attenuation_ratio_slices=attenuation_ratio_slices,
                action_referenced_removed_entity=action_referenced_removed_entity,
                agent_action_succeeded=agent_action_succeeded,
                connectivity_event_fired=connectivity_event is not None,
                connectivity_n_touched_nodes=len(connectivity_event["node_ids"]) if connectivity_event else 0,
                norm_h1_slices=norm_h1_slices, norm_h2_slices=norm_h2_slices, norm_h3_slices=norm_h3_slices,
            )
            self._drift_logger.log(row)

    # Deferred change attribution: logs one "attributed" row per node that entered
    # self.discovered_nodes this step (h1->h2, the only window discovery can happen in). If the
    # node matches a pending registry entry (a previously non-visible membership_join), the row
    # carries that event's event_id/change_type/step_fired/visibility_lag_steps and
    # node_origin_is_join=True; otherwise it is an ordinary node being discovered for the first
    # time (never removed, never a tracked event) and node_origin_is_join=False -- the free
    # control group from Step 4: compare attenuation for these two groups at the same n_discovered.
    def _log_attribution_rows(self, h1, h2, newly_discovered_node_ids, agent_action_succeeded, action_referenced_removed_entity):
        aggs = self.graph_embeddings_aggregations
        agent_drift_full = self._rel_drift(h1.combined, h2.combined)
        agent_drift_slices = {agg: self._rel_drift(h1.slices[agg], h2.slices[agg]) for agg in aggs}
        norm_h1, norm_h2 = float(np.linalg.norm(h1.combined)), float(np.linalg.norm(h2.combined))
        norm_h1_slices = {agg: float(np.linalg.norm(h1.slices[agg])) for agg in aggs}
        norm_h2_slices = {agg: float(np.linalg.norm(h2.slices[agg])) for agg in aggs}
        delta_h_G_full = h2.combined - h1.combined

        pending_by_node = {}
        still_pending = []
        for pending in self._drift_pending_nodes:
            if pending["node_id"] in newly_discovered_node_ids and pending["node_id"] not in pending_by_node:
                pending_by_node[pending["node_id"]] = pending
            else:
                still_pending.append(pending)
        self._drift_pending_nodes = still_pending

        for node_id in newly_discovered_node_ids:
            pending = pending_by_node.get(node_id)
            # Same zero-vector convention as membership_join in _node_delta_vector: absent at h1
            # (not yet discovered), present at h2 (just discovered) -- true regardless of whether
            # this node is a joined donor node or an ordinary original one.
            delta_h_v = self._node_delta_vector([node_id], "membership_join", h1, h2)
            delta_h_v_norm = float(np.linalg.norm(delta_h_v))
            attenuation_ratio_full = delta_h_v_norm / max(float(np.linalg.norm(delta_h_G_full)), 1e-12)
            attenuation_ratio_slices = {
                agg: delta_h_v_norm / max(float(np.linalg.norm(h2.slices[agg] - h1.slices[agg])), 1e-12)
                for agg in aggs
            }
            row = self._build_drift_row(
                event_phase="attributed",
                change_type=(pending["change_type"] if pending else None), change_fired=(pending is not None),
                n_touched_nodes=1, relevant=self._is_event_relevant([node_id]),
                event_id=(pending["event_id"] if pending else None),
                step_fired=(pending["step_fired"] if pending else None),
                visibility_lag_steps=(self.stepcount - pending["step_fired"]) if pending else None,
                touched_node_visible=True, attributed=True,
                node_origin_is_join=(pending is not None and pending["change_type"] == "membership_join"),
                n_discovered=h2.n_discovered, n_discovered_h1=h1.n_discovered, n_discovered_h2=h2.n_discovered,
                agent_drift_full=agent_drift_full, agent_drift_slices=agent_drift_slices,
                norm_h1=norm_h1, norm_h2=norm_h2,
                delta_h_v_norm=delta_h_v_norm, attenuation_ratio_full=attenuation_ratio_full,
                attenuation_ratio_slices=attenuation_ratio_slices,
                action_referenced_removed_entity=action_referenced_removed_entity,
                agent_action_succeeded=agent_action_succeeded,
                norm_h1_slices=norm_h1_slices, norm_h2_slices=norm_h2_slices,
            )
            self._drift_logger.log(row)

    # Function determining if a certain outcome changes the evolving visible graph
    def action_changes_evolving_visible_graph(self, outcome):
        if self.precise_graph_encoding:
            return not (isinstance(outcome, model.InvalidAction) or isinstance(outcome, model.NoVulnerability)
                    or isinstance(outcome, model.NoEnoughPrivilege) or isinstance(outcome, model.UnsuccessfulAction)
                    or isinstance(outcome, model.OutcomeNonPresent) or isinstance(outcome, model.NonListeningPort)
                    or isinstance(outcome, model.FirewallBlock) or isinstance(outcome, model.NoNeededAction)
                    or isinstance(outcome, model.RepeatedResult) or isinstance(outcome, model.NonRunningMachine))
        else:
            return isinstance(outcome, model.LateralMove) or isinstance(outcome, model.DenialOfService) or isinstance(outcome, model.Reconnaissance)

    # Function updating the evolving visible graph after a step, adding nodes and edges if needed
    def update_evolving_visible_graph_after_step(self, source_node, target_node, vulnerability_ID):
        # Update the graph that should turn into a graph embedding
        for node in self.discovered_nodes:
            if node not in self.evolving_visible_graph.nodes() and node in self.environment.nodes:
                self.add_node_evolving_visible_graph(node)

        # If an action that should modify the node feature vectors is issued, modify the graph embedding since the graph should change
        if (isinstance(self.outcome, model.Discovery) or isinstance(self.outcome, model.Collection) or isinstance(
                self.outcome, model.Persistence)
                or isinstance(self.outcome, model.PrivilegeEscalation) or isinstance(self.outcome,
                                                                                     model.Exfiltration) or isinstance(
                    self.outcome, model.DefenseEvasion)
                or isinstance(self.outcome, model.DenialOfService) or isinstance(self.outcome,
                                                                                 model.LateralMove) or isinstance(
                    self.outcome, model.CredentialAccess)):
            self.update_node_evolving_visible_graph(target_node)

        # Add edges to the evolving visible graph
        if self.reward > 0:
            self.add_edge_evolving_visible_graph(source_node, target_node, vulnerability_ID)

    # Create continuous action space with all the actions represented as embeddings
    def create_continuous_action_space(self, nodes_to_recalculate=None):
        self.outcome_counts = defaultdict(int)  # Track counts for each outcome
        self.outcome_embeddings = defaultdict(list)  # Store action keys per outcome

        running_owned_nodes = {node: self.node_embeddings[node] for node in self.owned_nodes
                               if self.get_node(node).status == model.MachineStatus.Running}
        running_discovered_nodes = {node: self.node_embeddings[node] for node in self.discovered_nodes
                                    if self.get_node(node).status == model.MachineStatus.Running}

        for source_node, source_node_embedding in running_owned_nodes.items():
            for target_node, target_node_embedding in running_discovered_nodes.items():
                # if the current action involved some nodes, their embeddings may have changed
                if nodes_to_recalculate:
                    # check if some node between source and target is connected to node to recalculate in the evolving visible graph
                    if not (any(nx.has_path(self.evolving_visible_graph, source_node, node) for node in nodes_to_recalculate) or
                            any(nx.has_path(self.evolving_visible_graph, target_node, node) for node in nodes_to_recalculate)):
                        # in case it is not this the case we can skip calculation if already processed
                        if (source_node, target_node) in self.processed_pairs:
                            continue  # Skip redundant processing
                else: # process all if not already processed
                    if (source_node, target_node) in self.processed_pairs:
                        continue  # Skip redundant processing
                if source_node == target_node:
                    self.__add_vulnerabilities_to_action_space(source_node, source_node_embedding, target_node,
                                                   target_node_embedding, "local")

                # # in case of local vulnerability, add also all the remote ones, assuming the attacker can use a personal device as source node
                self.__add_vulnerabilities_to_action_space(source_node, source_node_embedding, target_node, target_node_embedding,
                                              "remote")
                if (source_node, target_node) not in self.processed_pairs:
                    self.processed_pairs.add((source_node, target_node))

        # Subset of actions at every timestep in order to reduce the number of points in the action space and hence distance computation
        start_time = time.time()
        if self.sample_subset_samples:
            self.__balance_action_space_by_outcome()
        self.balance_action_space_time += time.time() - start_time

    # Function to add a specific set of vulnerabilities  given a current pair of nodes to the continuous action space
    def __add_vulnerabilities_to_action_space(self, source_node, source_node_embedding, target_node, target_node_embedding,
                                 vulnerability_type):

        for vulnerability in self.vulnerabilities_embeddings_per_node_type[target_node][vulnerability_type]:
            action_key = (source_node, target_node, vulnerability["vulnerability_ID"], vulnerability["outcome"])

            if source_node == target_node and isinstance(vulnerability["outcome"], model.LateralMove) or isinstance(vulnerability["outcome"], model.CredentialAccess):
                continue  # invalid case removed that can hold since type is read by CVSS vector and outcome is instead forecasted

            # remove all obstacles that could have negative outcome, embeddings our knowledge on the action space
            if self.remove_all_obstacles and self.goal in ["control", "discovery", "control_node", "discovery_node"]:
                if isinstance(vulnerability["outcome"], model.DenialOfService):
                    continue

            # remove the main obstacles that may stop the episode
            if self.remove_main_obstacles: # regardless of the goal
                if isinstance(vulnerability["outcome"], model.DenialOfService) and target_node == self.starter_node:
                    continue

            if self.remove_main_obstacles and self.goal.endswith("node") and self.goal != "disruption_node":
                if isinstance(vulnerability["outcome"], model.DenialOfService) and target_node == self.interest_node:
                    continue

            # overwrite if changed
            self.action_embeddings[action_key] = np.concatenate((source_node_embedding, target_node_embedding, vulnerability['embedding']))

    # Keep a susbet of actions per type in the action space for distance computation issues as approximated solution, trying to keep the balance between the different outcomes.
    # Grouped by map_outcome_to_string's display categories (mechanism-level: e.g. "LateralMove-Credential"
    # combines the CredentialAccess and LateralMove classes) rather than by raw type(outcome) -- balancing
    # per raw class previously gave any goal-relevant mechanism spanning N outcome classes up to N times
    # sample_subset_samples's worth of slots (e.g. PrivilegeEscalation, a single class, capped at 100,
    # vs LateralMove-Credential's two underlying classes each independently capped at 100, i.e. up to
    # 200 combined) purely as an artifact of how many raw outcome classes happen to share that display
    # label, with no relation to the reward function or the underlying vulnerability catalogue. This
    # structural 2:1 ceiling was confirmed as the dominant driver of Compressed's low privilege-escalation
    # completion rate relative to Local (which has no such per-class multiplication at all).
    def __balance_action_space_by_outcome(self):
        outcome_counts = defaultdict(list)
        for action_key in self.action_embeddings:
            outcome = action_key[-1]
            outcome_counts[map_outcome_to_string(outcome)].append(action_key)
        reduced_embeddings = {}
        for outcome, actions in outcome_counts.items():
            if len(actions) > self.sample_subset_samples:
                actions_to_keep_indices = np.random.choice(len(actions), self.sample_subset_samples, replace=False)
                actions_to_keep = [actions[i] for i in actions_to_keep_indices]
            else:
                actions_to_keep = actions
            for action in actions_to_keep:
                reduced_embeddings[action] = self.action_embeddings[action]
        self.action_embeddings = reduced_embeddings

    # Function to find the closest action embedding to a given action vector using the specified distance metric
    def find_closest_action_embedding(self, action_vector, no_output=False):
        metric_mapping = {
            'l1': lambda x, y: np.linalg.norm(x - y, ord=1, axis=1),
            'l2': lambda x, y: np.linalg.norm(x - y, ord=2, axis=1),
            'inf': lambda x, y: np.linalg.norm(x - y, ord=np.inf, axis=1),
            'cosine': lambda x, y: distance_cosine.cdist(x, y, 'cosine').flatten()
            # 'cosine': lambda x, y: distance_cosine.cdist(
            #     np.atleast_2d(x), np.atleast_2d(y), 'cosine'
            # ).flatten()
        }

        if self.distance_metric not in metric_mapping:
            raise ValueError(f"Unsupported metric '{self.distance_metric}'. Use 'l1', 'l2', 'inf', or 'cosine'.")

        embeddings_array = np.array(list(self.action_embeddings.values()))
        vector_segment = np.atleast_2d(np.array(action_vector, dtype=np.float32))
        distances = metric_mapping[self.distance_metric](vector_segment, embeddings_array)
        min_index = np.argmin(distances)
        action, distance = list(self.action_embeddings.keys())[min_index], distances[min_index]
        closest_source_node_index, closest_target_node_index, vulnerability_index, outcome_type = action
        if self.verbose > 2 and not no_output:
            self.logger.info("Closest action -> source node: %s, target node: %s, vulnerability: %s, outcome: %s, distance: %s",
                             closest_source_node_index, closest_target_node_index, vulnerability_index, outcome_type, distance)
        return closest_source_node_index, closest_target_node_index, vulnerability_index, outcome_type, distance

    # Function to map the outcome to a one-hot encoding based on its type and vulnerability type
    def map_outcome_to_onehot(self, vulnerability_type, outcome):
        # Single canonical ordering shared between "local" and "remote" so the same outcome type
        # always maps to the same one-hot index regardless of which exploit mechanism produced it
        # (keeps e.g. local-triggered and remote-triggered PrivilegeEscalation embeddings
        # consistent in outcome-encoding space, rather than looking like different categories).
        # "remote"'s membership list previously excluded PrivilegeEscalation entirely, even though
        # attacker_actions.py's remote-exploit handler already supports and correctly gates it on
        # prior target ownership (NoEnoughPrivilege if not already partially owned) -- identical to
        # the local-exploit path, and identical to how CyberBattleLocalEnv's flat catalogue already
        # includes remote-type PrivilegeEscalation vulnerabilities with no restriction. Confirmed via
        # direct inspection this silently dropped ~95% of this topology's PrivilegeEscalation-outcome
        # vulnerabilities (58 of 61 are remote-type) from ever reaching the action space at all.
        canonical_labels = [DenialOfService, Discovery, Collection, Exfiltration,
                             Reconnaissance, DefenseEvasion, Persistence,
                             PrivilegeEscalation, CredentialAccess, LateralMove]
        valid_labels = {
            "local": [DenialOfService, Discovery, Collection, Exfiltration,
                      Reconnaissance, DefenseEvasion, Persistence,
                      PrivilegeEscalation],
            "remote": [DenialOfService, Discovery, Collection, Exfiltration,
                       Reconnaissance, DefenseEvasion, Persistence,
                       CredentialAccess, LateralMove, PrivilegeEscalation]
        }
        if isinstance(outcome, model.Execution):
            return None
        if vulnerability_type not in valid_labels:
            raise ValueError("Vulnerability type must be either 'local' or 'remote'.")
        if type(outcome) not in valid_labels[vulnerability_type]:
            return None
        index = canonical_labels.index(type(outcome))
        one_hot = [0] * self.outcome_dimensions
        one_hot[index] = 1
        return one_hot

    # Function to create the vulnerabilities embeddings from the environment nodes
    def create_vulnerabilities_embeddings(self):
        self.vulnerabilities_embeddings = {}
        for node in self.environment.nodes:
            for vulnerability_ID in self.get_node(node).vulnerabilities:
                self.vulnerabilities_embeddings[vulnerability_ID] = self.get_node(node).vulnerabilities[vulnerability_ID].embedding

    # Function used to create the vulnerabilities embeddings per node type, distinguishing between local and remote vulnerabilities
    def create_vulnerabilities_embeddings_per_node_type(self):
        # distinguish the vulns per node and per type
        self.vulnerabilities_embeddings_per_node_type = {}
        for node in self.environment.nodes:
            if node not in self.vulnerabilities_embeddings_per_node_type:
                self.vulnerabilities_embeddings_per_node_type[node] = {"local": [], "remote": []}
            for vulnerability_ID in self.get_node(node).vulnerabilities:
                embedding = self.get_node(node).vulnerabilities[vulnerability_ID].embedding
                for result in self.get_node(node).vulnerabilities[vulnerability_ID].results:
                    outcome_embedding = self.map_outcome_to_onehot(result.type_str, result.outcome)
                    if outcome_embedding is None:
                        continue
                    self.vulnerabilities_embeddings_per_node_type[node][result.type_str].append({
                            "vulnerability_ID": vulnerability_ID,
                            "outcome": result.outcome,
                            "embedding": np.concatenate((embedding, outcome_embedding))
                    })

    # Tops up vulnerabilities_embeddings and vulnerabilities_embeddings_per_node_type for a single
    # node whose vulnerability IDs may not already be known -- needed whenever a node's vulnerability
    # set is introduced or changed outside the normal one-time construction pass (a dynamically
    # joined node, or a synthesized fallback Reconnaissance vulnerability on an existing node).
    # Mirrors the per-node body of create_vulnerabilities_embeddings/_per_node_type exactly.
    def refresh_vulnerabilities_embeddings_for_node(self, node_id):
        node_info = self.get_node(node_id)
        for vulnerability_ID, vulnerability in node_info.vulnerabilities.items():
            self.vulnerabilities_embeddings.setdefault(vulnerability_ID, vulnerability.embedding)
        self.vulnerabilities_embeddings_per_node_type[node_id] = {"local": [], "remote": []}
        for vulnerability_ID, vulnerability in node_info.vulnerabilities.items():
            embedding = vulnerability.embedding
            for result in vulnerability.results:
                outcome_embedding = self.map_outcome_to_onehot(result.type_str, result.outcome)
                if outcome_embedding is None:
                    continue
                self.vulnerabilities_embeddings_per_node_type[node_id][result.type_str].append({
                    "vulnerability_ID": vulnerability_ID,
                    "outcome": result.outcome,
                    "embedding": np.concatenate((embedding, outcome_embedding))
                })

    def sample_random_action(self):
        return self.action_space.sample()

    # Function to set the graph encoder used to encode the graph and get the node embeddings
    def set_graph_encoder(self, graph_encoder):
        self.graph_encoder = graph_encoder

    # Function to set the PCA components for the vulnerability embeddings, which also updates the action space accordingly
    def set_pca_components(self, pca_components, default_value=768):
        if not pca_components:
            self.vulnerability_embeddings_dimensions = default_value
        else:
            self.vulnerability_embeddings_dimensions = pca_components
        self.action_space = spaces.Box(low=-4, high=4,
                                       shape=(self.node_embeddings_dimensions * 2 + self.vulnerability_embeddings_dimensions + self.outcome_dimensions,),
                                       dtype=numpy.float32)
