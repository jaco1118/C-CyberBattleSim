# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    cyberbattle_env.py
    Class containing the main logic of the simulation game without providing a Gym Env full interface and defining an observation and action space.
    The observation and action spaces are defined by subclasses of this class.
"""

import time
import math
from typing import Optional, List
import random
import gym as gym
import sys
import os
import copy
import numpy
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
from cyberbattle.simulation import attacker_actions, model, static_defender_actions # noqa: E402
from cyberbattle._env.static_defender import DefenderAgent, ScanAndReimageCompromisedMachines # noqa: E402
from cyberbattle.simulation.model import VulnerabilityType # noqa: E402
from cyberbattle.utils.file_utils import load_yaml # noqa: E402

class CyberBattleEnv(gym.Env):
    """ The simulation starts from a random initial node or a fixed one.
        The simulation ends if either the attacker reaches its goal, the defender reaches its goal, or a maximum number of actions is reached,
        or if one of the defender's constraints is not met (e.g. SLA).
    """

    metadata = {'render.modes': ['human']}

    @property
    def name(self) -> str:
        return "CyberBattleEnv"

    def __init__(self,
                 initial_environment: model.Model,
                 winning_reward=0,
                 losing_reward=-100,
                 random_starter_node=True,
                 absolute_reward=False, # reward = max(0, reward) at each timestep
                 stop_at_goal_reached=False, # stop the simulation when the attacker reaches its goal, done = True
                 rewards_dict=None,
                 penalties_dict=None,
                 isolation_filter_threshold=0.1, # minimum % of the network needed to be discoverable, ownable, disruptable from a node to use it as a starter node
                 max_services_per_node=10,
                 goal="control", # goal, can be general or specific node oriented
                 switch_interest_node_interval=1, # switch the interest node periodically with this period of episodes
                 interest_node_value = 1000,
                 static_defender_agent: Optional[DefenderAgent] = None,
                 static_defender_eviction_goal=True, # if the defender goal is to evict the attacker from the network
                 episode_iterations=100,  # cut-off for the episode
                 proportional_cutoff_coefficient=0, # other cut-off proportional to the number of nodes in the topology, -1 if not present
                 max_num_trials_find_feasible_starter_node=1000, # maximum number of trials to find a feasible starter node
                 verbose=0, # 0 nothing, 1 print only training information, 2 print also episode information, 3 print also single iteration information
                 logger=None,
                 change_interval=20, # calibration anchor: 1/change_interval is the expected steady-state
                                      # removal probability per step at full ramp for dynamic_mode="leave";
                                      # also still the literal period for legacy patch/service changes
                 change_type="patch", # legacy patch/service change_type: "patch" | "service" | "mixed"
                 patch_service_dynamic_enabled=True, # enables the legacy patch/service/mixed changes (unrelated to node leave/join)
                 dynamic_mode="none", # node population dynamics: "none" | "leave" | "join" | "both"
                 dynamic_min_alive_nodes=5, # absolute floor: dynamic leave never drops alive node count below this
                 dynamic_min_alive_fraction=0.5, # relative floor: fraction of the original topology size, combined with the absolute floor via max()
                 dynamic_batch_interval=150, # mean steps between low-rate Poisson "batch outage" events
                 dynamic_batch_size_mean=1.0, # mean of the Poisson term added to the guaranteed 1 when a batch event fires
                 dynamic_batch_max_fraction=0.3, # cap on batch size as a fraction of the current eligible pool
                 dynamic_degree_weighting=True, # weight per-node removal probability down for high-degree/critical nodes
                 dynamic_max_alive_nodes=None, # absolute ceiling: dynamic join never grows alive count above this; None -> derived purely from the fraction/joins-per-episode below
                 dynamic_max_alive_fraction=1.3, # relative ceiling: fraction of the original topology size, combined with the absolute ceiling via min()
                 dynamic_max_joins_per_episode=3, # hard cap on total join events per episode; also the single source of truth used to size Local/Global's pre-allocated join headroom, keeping the two mutually consistent
                 dynamic_join_rate_interval=20, # calibration anchor for join's per-parent Bernoulli rate (kept separate from change_interval, which has an unrelated meaning)
                 dynamic_join_batch_interval=150, # mean steps between low-rate Poisson batch-join events
                 dynamic_join_batch_size_mean=1.0, # mean of the Poisson term added to the guaranteed 1 when a batch join event fires
                 dynamic_join_batch_max_fraction=0.3, # cap on batch size as a fraction of the current eligible parent pool
                 dynamic_join_value_weighting=True, # weight per-parent join-sponsorship probability by ownership + node value (self.environment has no edges, so degree is not a meaningful signal here)
                 **kwargs
                 ):
        self.environment = None
        self.winning_reward = winning_reward
        self.losing_reward = losing_reward
        self.random_starter_node = random_starter_node
        self.absolute_reward = absolute_reward
        self.stop_at_goal_reached = stop_at_goal_reached
        self.rewards_dict = rewards_dict
        self.penalties_dict = penalties_dict
        self.isolation_filter_threshold = isolation_filter_threshold
        self.max_services_per_node = max_services_per_node
        if goal is not None:
            self.goal = goal.lower()
        else:
            self.goal = goal
        self.interest_node_value = interest_node_value
        self.switch_interest_node_interval = switch_interest_node_interval
        self.static_defender_agent = static_defender_agent
        self.static_defender_eviction_goal = static_defender_eviction_goal
        self.episode_iterations = episode_iterations
        self.proportional_cutoff_coefficient = proportional_cutoff_coefficient
        self.max_num_trials_find_feasible_starter_node = max_num_trials_find_feasible_starter_node
        self.verbose = verbose
        self.logger = logger
        self.change_interval = change_interval
        self.patch_service_dynamic_enabled = patch_service_dynamic_enabled
        self.change_type = change_type
        self.dynamic_mode = dynamic_mode
        self.dynamic_min_alive_nodes = dynamic_min_alive_nodes
        self.dynamic_min_alive_fraction = dynamic_min_alive_fraction
        self.dynamic_batch_interval = dynamic_batch_interval
        self.dynamic_batch_size_mean = dynamic_batch_size_mean
        self.dynamic_batch_max_fraction = dynamic_batch_max_fraction
        self.dynamic_degree_weighting = dynamic_degree_weighting
        self.dynamic_max_alive_nodes = dynamic_max_alive_nodes
        self.dynamic_max_alive_fraction = dynamic_max_alive_fraction
        self.dynamic_max_joins_per_episode = dynamic_max_joins_per_episode
        self.dynamic_join_rate_interval = dynamic_join_rate_interval
        self.dynamic_join_batch_interval = dynamic_join_batch_interval
        self.dynamic_join_batch_size_mean = dynamic_join_batch_size_mean
        self.dynamic_join_batch_max_fraction = dynamic_join_batch_max_fraction
        self.dynamic_join_value_weighting = dynamic_join_value_weighting
        self.dynamic_join_donor_pool = []  # set externally by train_agent.py's env-building loop
        self._dynamic_change_count = 0
        self.__initial_environment: model.Model = initial_environment
        self.done = False
        self.num_nodes = len(self.__initial_environment.network.nodes)
        self.episode_id = 0

        # in case of an experiment targeting a specific node, switch periodically
        if self.switch_interest_node_interval and self.goal.lower().endswith("node"):
            self.switch_interest_node()

        self.action_execution_time = 0
        self.defender_execution_time = 0
        self.check_end_time = 0

        self.action_space = gym.spaces.Discrete(1) # fictious
        self.observation_space = gym.spaces.Discrete(1) # fictious

        # Read default dicts if not provided to avoid errors (e.g. GAE not using rewards)
        if not self.rewards_dict:
            self.rewards_dict = load_yaml(os.path.join(project_root, "cyberbattle", "agents", "config", "rewards_config.yaml"))["rewards_dict"][self.goal.lower()]
        if not self.penalties_dict:
            self.penalties_dict = load_yaml(os.path.join(project_root, "cyberbattle", "agents", "config", "rewards_config.yaml"))["penalties_dict"][self.goal.lower()]

        # Reset environment to start the simulation
        self.reset_env()

    # Periodically switch the interest node if games with node of interest are played
    def switch_interest_node(self):
        if self.episode_id % self.switch_interest_node_interval == 0:
            self.interest_node = random.choice(list(self.__initial_environment.network.nodes))
            if self.verbose > 1:
                self.logger.info("Switching interest node to %s", self.interest_node)

    # Function to reset the environment, called at the beginning of each episode
    def reset_env(self):
        if self.verbose > 1:
            if self.action_execution_time != 0: # if the episode has really concluded and this is not the first time the reset is called
                self.logger.info("Action execution time in the episode: %s", self.action_execution_time)
                self.logger.info("Defender execution time in the episode: %s", self.defender_execution_time)
                self.logger.info("Check end time in the episode: %s", self.check_end_time)
        start_time = time.time()
        if self.environment:  # ensure garbage collection of the object
            del self.environment

        # deep copy because modifications could be done
        self.environment = copy.deepcopy(self.__initial_environment.network)
        self.access_graph = self.__initial_environment.access_graph
        self.knows_graph = self.__initial_environment.knows_graph
        self.dos_graph = self.__initial_environment.dos_graph
        try:
            self.pick_starter_node()
        except model.NoSuitableStarterNode:  # if no suitable starter node found, push the exception to the switcher that will choose another environment
            self.logger.info("No suitable starter node found, pushing the exception to the switcher")
            raise model.NoSuitableStarterNode("Could not find a suitable starter node")
        self.network_availability = 1.0
        self.episode_id += 1
        self.evicted = False
        self.done = False
        self.num_iterations = 0
        self.discovered_amount = 0
        self.overall_reimaged = []
        self.num_events = 0
        self._dynamic_change_count = 0
        self._dynamic_removed_this_episode: List[model.NodeID] = []
        self._dynamic_joined_this_episode: List[model.NodeID] = []
        self._used_donor_pool_indices_this_episode: set = set()
        self._dynamic_last_applied_iteration = -1
        # Diagnostic-only (drift instrumentation): events fired by the most recent
        # maybe_apply_dynamic_step() call, as [{"change_type": "property"|"membership_leave"|
        # "membership_join", "node_ids": [...]}, ...]. Populated regardless of drift_logging
        # (cheap bookkeeping) so StepInfo['change_type'] is always accurate; only consumed
        # further by subclasses that implement drift logging.
        self._last_dynamic_events: List[dict] = []
        self.discovered_nodes: List[model.NodeID] = []
        self.owned_nodes: List[model.NodeID] = []
        self.episode_rewards: List[float] = []
        self.sampled_actions = [] # in case the episode is used to sample valid actions
        # Set the actuator used to execute actions in the simulation environment
        self._actuator = attacker_actions.AttackerAgentActions(self.environment,
                                                penalties_dict=self.penalties_dict,
                                                rewards_dict=self.rewards_dict,
                                                logger=self.logger,
                                                verbose=self.verbose,
                                                )
        self._defender_actuator = static_defender_actions.StaticDefenderAgentActions(self.environment, logger=self.logger, verbose=self.verbose)
        self.stepcount = 0
        self.done = False

        # start with only the starting node as discovered/owned node
        self.discovered_nodes.append(self.starter_node)
        self.owned_nodes.append(self.starter_node)

        self.reset_time = time.time() - start_time
        if self.verbose > 1:
            self.logger.info("Reset cyberbattleenv time: %s", self.reset_time)
        self.action_execution_time = 0
        self.defender_execution_time = 0
        self.check_end_time = 0

    # Function to reset the network to its initial state
    def pick_starter_node(self) -> None:
        if not self.random_starter_node:
            self.starter_node, _ = list(self.environment.nodes(data=True))[0] # take the first node in order
        else:  # Random starter node at every episode
            # While loop to be sure to select a starter node respecting the necessary conditions
            self.num_trials = 0 # if possible within a maximum number of trials
            while True:
                # stop looking for non-isolated nodes in the topology after many attempts
                self.num_trials += 1
                if self.num_trials > self.max_num_trials_find_feasible_starter_node:
                    raise model.NoSuitableStarterNode("Could not find a suitable starter node")

                self.starter_node, _ = list(self.environment.nodes(data=True))[random.randrange(len(self.environment.nodes))]
                if self.verbose > 2:
                    self.logger.info("Selected starter node %s as trial", self.starter_node)
                # check if the node reaches the minimum number required
                self.shortest_paths_starter_control = copy.deepcopy(self.__initial_environment.access_shortest_paths[self.starter_node])
                threshold_count = self.isolation_filter_threshold * len(self.shortest_paths_starter_control) # minimum amount required to reach
                # control goal
                self.shortest_paths_starter_control.pop(self.starter_node) # remove the node itself, always reachable in 0 steps
                self.ownable_count = sum(1 for value in self.shortest_paths_starter_control.values() if value is not None)
                # discovery goal
                self.shortest_paths_starter_discovery = copy.deepcopy(self.__initial_environment.knows_shortest_paths[self.starter_node])
                self.shortest_paths_starter_discovery.pop(self.starter_node) # remove the node itself, always discoverable in 0 steps
                self.discoverable_count = sum(1 for value in self.shortest_paths_starter_discovery.values() if value is not None)
                # disruption goal
                self.shortest_paths_starter_disruption = copy.deepcopy(self.__initial_environment.dos_shortest_paths[self.starter_node])
                self.shortest_paths_starter_disruption.pop(self.starter_node)
                self.disruptable_count = sum(1 for value in self.shortest_paths_starter_disruption.values() if value is not None)
                # Check isolation based on the specific goal considered for this instance of the environment
                if self.goal == "disruption":
                    if self.disruptable_count < threshold_count:
                        # pick new starter node
                        if self.verbose > 2:
                            self.logger.info("This node is isolated, can disrupt maximum %s while it should disrupt minimum %s", self.disruptable_count, threshold_count)
                        continue
                    # set maximum number disruptable if not isolated
                    if self.proportional_cutoff_coefficient:
                        self.proportional_nodes = self.disruptable_count
                elif self.goal == "control":
                    if self.ownable_count < threshold_count:
                        # pick new starter node
                        if self.verbose > 2:
                            self.logger.info("This node is isolated, can control maximum %s while it should control minimum %s",
                                        self.ownable_count, threshold_count)
                        continue
                    # set maximum number controllable if not isolated
                    if self.proportional_cutoff_coefficient:
                        self.proportional_nodes = self.ownable_count
                elif self.goal == "discovery":
                    if self.discoverable_count < threshold_count:
                        # pick new starter node
                        if self.verbose > 2:
                            self.logger.info(
                                "This node is isolated, can discover maximum %s while it should discover minimum %s",
                                self.discoverable_count, threshold_count)
                        continue
                    # set maximum number discoverable if not isolated
                    if self.proportional_cutoff_coefficient:
                        self.proportional_nodes = self.discoverable_count
                elif self.goal.endswith("node"):
                    if self.starter_node == self.interest_node: # invalid choice, otherwise it has already won
                        continue
                    # in node-specific games the starter node must be able to control/discover/disrupt the node of interest
                    if self.goal == "control_node":
                        if self.interest_node not in self.shortest_paths_starter_control or self.shortest_paths_starter_control[self.interest_node] is None:
                            if self.verbose > 2:
                                self.logger.info("Node of interest not controllable from starter node")
                            continue
                        if self.proportional_cutoff_coefficient:
                            self.proportional_nodes = self.ownable_count
                    elif self.goal == "discovery_node":
                        if self.interest_node not in self.shortest_paths_starter_discovery or \
                            self.shortest_paths_starter_discovery[self.interest_node] is None:
                            if self.verbose > 2:
                                self.logger.info("Node of interest not discoverable from starter node")
                            continue
                        if self.proportional_cutoff_coefficient:
                            self.proportional_nodes = self.discoverable_count
                    elif self.goal == "disruption_node":
                        if self.interest_node not in self.shortest_paths_starter_disruption or \
                            self.shortest_paths_starter_disruption[self.interest_node] is None:
                            if self.verbose > 2:
                                self.logger.info("Node of interest not disruptable from starter node")
                            continue
                        if self.proportional_cutoff_coefficient:
                            self.proportional_nodes = self.disruptable_count
                    # interest node value to be set much higher than the others
                    self.set_node_property(self.interest_node, "value", self.interest_node_value)
                # determine the amount of discovery actions that can be achieved (nodes, data, visibility, ...)
                self.discoverable_amount = 0
                for node in self.environment.nodes:
                    if node in self.shortest_paths_starter_discovery or node == self.starter_node:
                        self.discoverable_amount += 1  # count the discovery of the node
                    if node in self.shortest_paths_starter_control or node == self.starter_node:
                        node_info = self.get_node(node)
                        if node_info.has_data:
                            self.discoverable_amount += 2  # data to collect and exfiltrate
                        if not node_info.visible:
                            self.discoverable_amount += 1  # need to increase the amount of visibility in the node, discovering properties
                self.num_trials = 0  # reset once here
                for node in self.environment.nodes:
                    self.set_node_property(node, "agent_installed", False)
                break
            if self.verbose > 1:
                self.logger.info("Starter node %s selected", self.starter_node)
            # property common to all games
            self.set_node_property(self.starter_node, "agent_installed", True)

    # Logic used by all step function, regardless if it is the DRL training or GAE training
    def step_attacker_env(self, source_node, target_node, vulnerability_ID, outcome_desired):
        if self.done:
            self.logger.warning("New episode must be started with env.reset()")
            raise RuntimeError("New episode must be started with env.reset()")
        self.stepcount += 1
        start_time = time.time()
        if self.verbose > 2:
            self.logger.info("Step %s: source node %s, target node %s, vulnerability %s, outcome %s", self.stepcount, source_node, target_node, vulnerability_ID, outcome_desired)
        if source_node == target_node:  # use local (or remote) vulnerability on the node in case of the same source and target node
            if self.verbose > 2:
                self.logger.info("The vulnerability selected is LOCAL")
            result, self.vulnerability_type = self._actuator.exploit_local_vulnerability(source_node, vulnerability_ID, outcome_desired), "local"
        else:  # the nodes are different, use remote vulnerability
            if self.verbose > 2:
                self.logger.info("The vulnerability selected is REMOTE")
            result, self.vulnerability_type = self._actuator.exploit_remote_vulnerability(source_node, target_node, vulnerability_ID, outcome_desired), "remote"
        self.action_execution_time += time.time() - start_time
        self.reward = result.reward

        # update lists handling the episode based on the outcome
        self.outcome = self.update_episode_by_outcome(result.outcome, target_node)

        # In case of node-specific games, the reward is zeroed if the interest node was reachable directly by any other source node and the attacker is not using it
        if self.goal.endswith("node"):
            if self.interest_node in self.discovered_nodes and target_node != self.interest_node:
                if self.verbose > 2:
                    self.logger.info("Zeroing reward since the interest node %s was reachable and not used", self.interest_node)
                self.reward = 0
            # else it remains the same

        start_time = time.time()
        # Execute the defender step if involved
        if self.static_defender_agent:
            self.static_defender_step()

        self.defender_execution_time += time.time() - start_time

        start_time = time.time()
        # Check whether there has been some ending conditions
        self.end_episode_reason = 0
        self.truncated = False
        if not self.done:
            if self.attacker_goal_reached(): # attacker goal reached (vary per goal)
                if self.verbose > 2:
                    self.logger.info("Attacker won the episode! Assigning winning reward %.1f..", self.winning_reward)
                if self.goal == "disruption" or self.stop_at_goal_reached:
                    # in case of disruption we need to stop in any case, if it won it means it has killed them all
                    self.done = True
                self.reward = self.winning_reward
                self.end_episode_reason = 1  # attacker goal reached
            elif self.check_end_game(): # losing conditions depend also on the goal
                if self.verbose > 2:
                    self.logger.info("Game lost due to end game condition reached! Assigning losing reward %.1f..", self.losing_reward)
                self.done = True
                self.reward = self.losing_reward
                self.end_episode_reason = 2  # lost game
            elif self.proportional_cutoff_coefficient and self.proportional_cutoff_reached(): # cut-off proportional to the number of nodes
                if self.verbose > 2:
                    self.logger.info("Proportional cut-off %d reached, stopping the episode..", self.proportional_cutoff_coefficient)
                self.truncated = True
                self.end_episode_reason = 3 # cut-off
            elif self.num_iterations >= self.episode_iterations: # cut-off constant regardless of the number of nodes
                if self.verbose > 2:
                    self.logger.info("Episode iterations limit %d reached, stopping the episode..", self.episode_iterations)
                self.truncated = True
                self.end_episode_reason = 3

        self.check_end_time += time.time() - start_time

        if self.absolute_reward:
            self.reward = max(0, self.reward)
            if self.verbose > 2:
                self.logger.info("Reward absolute value: %s", self.reward)

        self.source_node = source_node
        self.target_node = target_node
        self.vulnerability_ID = vulnerability_ID
        self.outcome_desired = outcome_desired
        self.outcome_obtained = result.outcome
        self.network_availability = len([node for node in self.discovered_nodes if
                                      node in self.environment.nodes and self.get_node(node).status == model.MachineStatus.Running]) / len(self.discovered_nodes)
        if self.verbose > 2:
            self.logger.info("Network availability after step: %f", self.network_availability)
        self.episode_rewards.append(self.reward)
        self.num_iterations += 1

    # Module-level, non-tunable numerical safety rail on any single node's per-step removal
    # probability. At realistic settings (small change_interval relative to eligible pool size)
    # this essentially never binds; it exists purely to bound pathological configs.
    _DYNAMIC_P_MAX = 0.25

    # Orchestrator called once per step, unconditionally, by every subclass's step() (no modulo
    # gate at the call site -- the modulo gating for legacy patch/service changes lives inside
    # here, and the new probabilistic leave/join logic has its own internal ramping/floor-or-
    # ceiling logic). Returns a single flat list combining node IDs touched by a legacy
    # patch/service change, removed, AND joined this step (empty if none happened) -- callers
    # that only care "did anything change" (e.g. Compressed's re-encode trigger) can keep
    # checking truthiness unchanged; callers that ignore the return value (Local/Global, which
    # always rebuild their observation from live state regardless) are unaffected either way.
    def maybe_apply_dynamic_step(self) -> List["model.NodeID"]:
        # Reset every call (not just on the early-return branches below) so a caller reading
        # self._last_dynamic_events after this returns never sees a stale event from a prior
        # iteration that this call itself did nothing on.
        self._last_dynamic_events = []
        if self.num_iterations == 0:
            return []
        # Local/Global "switch"/invalid-action paths don't call step_attacker_env, so
        # num_iterations doesn't advance for them; without this guard, an agent spamming such
        # actions could re-trigger the probabilistic draws multiple times at the same logical
        # timestep, inflating the calibrated rate. Compressed env always advances num_iterations
        # every step(), so this is a no-op there.
        if self.num_iterations == self._dynamic_last_applied_iteration:
            return []
        self._dynamic_last_applied_iteration = self.num_iterations
        touched: List["model.NodeID"] = []
        if self.patch_service_dynamic_enabled and self.num_iterations % self.change_interval == 0:
            touched_node = self._apply_legacy_dynamic_change()
            if touched_node is not None:
                touched = [touched_node]
        removed: List["model.NodeID"] = []
        joined: List["model.NodeID"] = []
        if self.dynamic_mode in ("leave", "both"):
            removed = self._apply_dynamic_leave()
            if removed:
                self._last_dynamic_events.append({"change_type": "membership_leave", "node_ids": list(removed)})
        if self.dynamic_mode in ("join", "both"):
            joined = self._apply_dynamic_join()
            if joined:
                self._last_dynamic_events.append({"change_type": "membership_join", "node_ids": list(joined)})
        return touched + removed + joined

    # No-op in the base class; Compressed overrides this to refresh the touched node's cached
    # feature vector in evolving_visible_graph, mirroring add_node_dynamic/remove_node_dynamic's
    # existing pattern for membership join/leave.
    def update_node_dynamic(self, node_id):
        pass

    # Legacy patch/service/mixed changes: node-count invariant, unrelated to the node
    # leave/join population dynamics. Kept as-is, just no longer dispatches "node_leave".
    # Returns the touched node id (or None) so maybe_apply_dynamic_step can fold it into the
    # value it returns, and calls update_node_dynamic on it so the node's cached feature vector
    # is refreshed in the same h2->h3 window membership's graph mutations already occupy.
    def _apply_legacy_dynamic_change(self):
        self._dynamic_change_count += 1
        touched_node = None
        if self.change_type == "patch":
            touched_node = self._patch_random_vulnerability()
        elif self.change_type == "service":
            touched_node = self._disable_random_service()
        elif self.change_type == "mixed":
            # alternate between the two
            if self._dynamic_change_count % 2 == 0:
                touched_node = self._patch_random_vulnerability()
            else:
                touched_node = self._disable_random_service()
        if touched_node is not None:
            self._last_dynamic_events.append({"change_type": "property", "node_ids": [touched_node]})
            self.update_node_dynamic(touched_node)
        return touched_node

    # Nodes eligible to be dynamically removed: discovered, running, and not one of the
    # currently-protected roles (starter/source/target/interest node).
    def _get_removal_eligible_nodes(self):
        return [
            node for node in self.discovered_nodes
            if self.get_node(node).status == model.MachineStatus.Running
            and node != self.starter_node
            and node != self.source_node
            and node != self.target_node
            and node != getattr(self, "interest_node", None)
        ]

    # Hard safety floor: the minimum number of alive nodes that dynamic leave may never drop
    # below, on top of the starter/source/target/interest-node exclusion above. Combines an
    # absolute floor and a floor proportional to the *original* topology size (self.num_nodes,
    # fixed at construction) via max(), so small topologies still keep a meaningful fraction.
    def _get_dynamic_floor(self) -> int:
        floor = max(self.dynamic_min_alive_nodes, math.ceil(self.dynamic_min_alive_fraction * self.num_nodes))
        return min(floor, self.num_nodes)

    # Count of nodes still present in the *whole* current topology graph (self.environment.nodes).
    # Deliberately counts graph presence, not running status, and deliberately not scoped to
    # self.discovered_nodes: get_alive_nodes() (discovered-only) starts at 1 right after reset, so
    # using it here would starve dynamic leave for most of an episode. Using running-status instead
    # of presence would also be wrong the other way -- ordinary gameplay (DoS/disruption actions,
    # entirely unrelated to dynamic leave) can stop a node without removing it from the graph, and
    # the floor must guard against dynamic leave depopulating the topology, not against those
    # pre-existing mechanics reducing the running-node count.
    def _count_alive_topology_nodes(self) -> int:
        return len(self.environment.nodes)

    # Mirror image of _get_dynamic_floor(): the maximum number of alive nodes that dynamic join
    # may never grow past. Combines an absolute ceiling and a ceiling proportional to the original
    # topology size via min() (the stricter of the two wins, symmetric to the floor's max()), and
    # is additionally hard-clamped to self.num_nodes + dynamic_max_joins_per_episode so the
    # config-driven ceiling can never imply more simultaneous joins than the headroom pre-allocated
    # for Local/Global (by train_agent.py, using dynamic_max_joins_per_episode as the same source
    # of truth) actually supports.
    def _get_dynamic_ceiling(self) -> int:
        fraction_ceiling = math.floor(self.dynamic_max_alive_fraction * self.num_nodes)
        absolute_ceiling = self.dynamic_max_alive_nodes if self.dynamic_max_alive_nodes is not None else fraction_ceiling
        ceiling = min(absolute_ceiling, fraction_ceiling)
        return min(ceiling, self.num_nodes + self.dynamic_max_joins_per_episode)

    # Probabilistic node-leave: per-node Bernoulli draw each step (probability weighted down for
    # high-degree/critical nodes), plus a low-rate Poisson-triggered batch removal, calibrated so
    # the expected removal rate matches change_interval as a sanity anchor, and ramped down as the
    # eligible pool approaches the safety floor. Replaces the old deterministic "exactly one node
    # every change_interval steps, forever, uncapped" trigger.
    def _apply_dynamic_leave(self) -> List["model.NodeID"]:
        if not hasattr(self, "remove_node_dynamic"):
            self.logger.warning(
                "[DynamicEnv] dynamic_mode=%r requires an environment implementing "
                "remove_node_dynamic; skipping removal", self.dynamic_mode
            )
            return []

        eligible = self._get_removal_eligible_nodes()
        if not eligible:
            return []

        alive = self._count_alive_topology_nodes()
        floor = self._get_dynamic_floor()
        room = max(0, alive - floor)
        if room == 0:
            return []

        # soft taper towards 0 as the eligible pool shrinks towards the floor, instead of a hard wall
        ramp = min(1.0, room / max(1, self.num_nodes - floor))

        # calibration anchor: at full ramp, expected removals per step equal 1/change_interval,
        # matching the old deterministic rate; this decays as attrition proceeds (the old scheme
        # never tapered, which is what let it strip the whole topology over a long episode)
        target_rate = ramp * (1.0 / self.change_interval)

        if self.dynamic_degree_weighting:
            # self.environment (the per-episode graph) never has edges -- all connectivity lives
            # on the shared Model's precomputed access_graph. Reading (not writing) it here is
            # safe, matching the established rule that only mutating access_graph/knows_graph/
            # dos_graph is the aliasing hazard. A dynamically joined node (once discovered) can be
            # leave-eligible but was never part of access_graph -- networkx's degree(n) for a node
            # not in the graph returns an (empty) DiDegreeView instead of an int or a KeyError, so
            # this must be guarded explicitly; degree 0 (lowest possible weighting protection) is
            # the correct default, since a joined node has no precomputed connectivity data at all.
            degrees = {n: (self.access_graph.degree(n) if n in self.access_graph else 0) for n in eligible}
            weights = {n: 1.0 / (1.0 + degrees[n]) for n in eligible}
        else:
            weights = {n: 1.0 for n in eligible}
        weight_sum = sum(weights.values())

        # sum(p) == target_rate by construction (before the P_MAX clip, which essentially never
        # binds at realistic settings) -- degree weighting only reallocates *which* node is likely
        # removed, not the expected *count* removed per step.
        probabilities = {
            n: min(self._DYNAMIC_P_MAX, target_rate * weights[n] / weight_sum)
            for n in eligible
        }
        hits = [n for n in eligible if random.random() < probabilities[n]]

        # low-rate Poisson "batch outage" trigger, independent of the per-node draws above
        p_batch_trigger = ramp / self.dynamic_batch_interval
        if random.random() < p_batch_trigger:
            batch_pool = [n for n in eligible if n not in hits]
            if batch_pool:
                batch_size = 1 + numpy.random.poisson(self.dynamic_batch_size_mean)
                batch_size = min(batch_size, len(batch_pool), max(1, int(self.dynamic_batch_max_fraction * len(eligible))))
                batch_weights = numpy.array([weights[n] for n in batch_pool], dtype=numpy.float64)
                batch_weights = batch_weights / batch_weights.sum()
                batch_hits = numpy.random.choice(batch_pool, size=batch_size, replace=False, p=batch_weights)
                hits += list(batch_hits)

        # belt-and-braces floor enforcement: independent Bernoulli hits plus a batch event could
        # in principle exceed room in one step even with ramping
        if len(hits) > room:
            hits = random.sample(hits, room)

        for node_id in hits:
            self.remove_node_dynamic(node_id)
            if self.verbose > 0:
                self.logger.info(
                    "[DynamicEnv] Step %d: removed node %s (floor=%d)",
                    self.num_iterations, node_id, floor
                )
        return hits

    # Probabilistic node-join: mirror image of _apply_dynamic_leave. Per-eligible-parent Bernoulli
    # draw each step (probability weighted by ownership + node value -- self.environment has no
    # edges, so degree is not a meaningful signal here, unlike leave's degree weighting), plus a
    # low-rate Poisson-triggered batch join, ramped down as the ceiling is approached, hard-capped
    # by the remaining per-episode join budget (dynamic_max_joins_per_episode -- also the exact
    # figure train_agent.py uses to size Local/Global's pre-allocated action-space headroom, so the
    # runtime ceiling here can never imply more simultaneous joins than that headroom supports).
    def _apply_dynamic_join(self) -> List["model.NodeID"]:
        if not hasattr(self, "add_node_dynamic"):
            self.logger.warning(
                "[DynamicEnv] dynamic_mode=%r requires an environment implementing "
                "add_node_dynamic; skipping join", self.dynamic_mode
            )
            return []

        remaining_budget = self.dynamic_max_joins_per_episode - len(self._dynamic_joined_this_episode)
        if remaining_budget <= 0 or not self.dynamic_join_donor_pool:
            return []

        eligible_parents = self._get_join_eligible_parents()
        if not eligible_parents:
            return []

        ceiling = self._get_dynamic_ceiling()
        room = min(max(0, ceiling - self._count_alive_topology_nodes()), remaining_budget)
        if room == 0:
            return []

        # soft ramp towards 0 as the ceiling is approached, mirroring leave's floor-ward ramp
        ramp = min(1.0, room / max(1, ceiling - self.num_nodes))
        target_rate = ramp * (1.0 / self.dynamic_join_rate_interval)

        if self.dynamic_join_value_weighting:
            weights = {
                n: (2.0 if n in self.owned_nodes else 1.0) * (1.0 + self.get_node(n).value / 100.0)
                for n in eligible_parents
            }
        else:
            weights = {n: (2.0 if n in self.owned_nodes else 1.0) for n in eligible_parents}
        weight_sum = sum(weights.values())

        probabilities = {
            n: min(self._DYNAMIC_P_MAX, target_rate * weights[n] / weight_sum)
            for n in eligible_parents
        }
        hits = [n for n in eligible_parents if random.random() < probabilities[n]]

        # low-rate Poisson "batch join" trigger, independent of the per-parent draws above
        p_batch_trigger = ramp / self.dynamic_join_batch_interval
        if random.random() < p_batch_trigger:
            batch_pool = [n for n in eligible_parents if n not in hits]
            if batch_pool:
                batch_size = 1 + numpy.random.poisson(self.dynamic_join_batch_size_mean)
                batch_size = min(batch_size, len(batch_pool), max(1, int(self.dynamic_join_batch_max_fraction * len(eligible_parents))))
                batch_weights = numpy.array([weights[n] for n in batch_pool], dtype=numpy.float64)
                batch_weights = batch_weights / batch_weights.sum()
                batch_hits = numpy.random.choice(batch_pool, size=batch_size, replace=False, p=batch_weights)
                hits += list(batch_hits)

        if len(hits) > room:
            hits = random.sample(hits, room)

        joined = []
        for parent in hits:
            new_node_id = self._spawn_node_from_pool(parent)
            if new_node_id is not None:
                joined.append(new_node_id)
        return joined

    # Shared purge logic used by every subclass's remove_node_dynamic: ground-truth graph removal,
    # discovered_nodes/owned_nodes pruning, and win-condition denominator decrement (only for the
    # sets the node actually belonged to, to avoid triggering a spurious win). This operates
    # entirely on base-class state, so it is not duplicated per subclass; only the
    # observation/action-space-specific purge (embeddings, evolving_visible_graph, ...) differs.
    def remove_node_common(self, node_id) -> bool:
        if node_id == self.starter_node:
            return False
        if node_id in self.environment.nodes():
            self.environment.remove_node(node_id)
        if node_id in self.discovered_nodes:
            self.discovered_nodes.remove(node_id)
        if node_id in self.owned_nodes:
            self.owned_nodes.remove(node_id)
        if getattr(self, "shortest_paths_starter_control", None) and \
                self.shortest_paths_starter_control.get(node_id) is not None and self.ownable_count > 0:
            self.ownable_count -= 1
        if getattr(self, "shortest_paths_starter_discovery", None) and \
                self.shortest_paths_starter_discovery.get(node_id) is not None and self.discoverable_count > 0:
            self.discoverable_count -= 1
        if getattr(self, "shortest_paths_starter_disruption", None) and \
                self.shortest_paths_starter_disruption.get(node_id) is not None and self.disruptable_count > 0:
            self.disruptable_count -= 1
        self._dynamic_removed_this_episode.append(node_id)
        return True

    # Mirror image of remove_node_common(): shared "add" logic used by every subclass's
    # add_node_dynamic. Only adds to self.environment (ground truth) -- deliberately NOT to
    # discovered_nodes/owned_nodes, since a joined node starts genuinely undiscovered until
    # _inject_reconnaissance_discovery() makes it reachable. Unconditionally increments
    # ownable_count/discoverable_count/disruptable_count by 1: unlike remove_node_common's
    # decrement (conditional on the node having been present in the per-episode
    # shortest_paths_starter_* snapshot, itself deep-copied once from the shared Model's static
    # tables), a joined node was never in that snapshot to begin with -- there is no "was it
    # counted" bit to check. Unconditional +1 is the correct semantics: a joined node is always
    # meant to be a genuine new objective for every goal type from the moment it exists.
    def add_node_common(self, node_id, node_info) -> bool:
        if node_id in self.environment.nodes():
            return False  # should never trigger given the ID scheme in _spawn_node_from_pool; defensive only
        self.environment.add_node(node_id, data=node_info)
        self.ownable_count += 1
        self.discoverable_count += 1
        self.disruptable_count += 1
        self._dynamic_joined_this_episode.append(node_id)
        return True

    # Nodes eligible to "sponsor" a join (i.e. have the new node attached to them via a
    # Reconnaissance injection, see _inject_reconnaissance_discovery). Discovered-only ensures the
    # join is actually reachable this episode -- an undiscovered parent's vulnerabilities can't be
    # exploited by the agent yet, so the injection would be inert. Running mirrors the same filter
    # _get_removal_eligible_nodes applies, for the same reason (a stopped node's vulnerabilities
    # aren't currently exploitable). Unlike leave, no starter/source/target/interest exclusion is
    # needed: appending to a Reconnaissance outcome's node list is a pure data mutation on that
    # node's VulnerabilityInfo, not a structural removal, so it's safe even on the current step's
    # active nodes.
    def _get_join_eligible_parents(self):
        return [
            n for n in self.discovered_nodes
            if self.get_node(n).status == model.MachineStatus.Running
        ]

    # Picks an unused-this-episode donor node from self.dynamic_join_donor_pool, renames it to a
    # fresh, collision-free ID, strips its own Reconnaissance outcomes (they'd otherwise reference
    # node IDs from the donor's original topology, meaningless here), adds it via add_node_dynamic
    # (subclass-implemented, representation-specific), and wires in discoverability. Returns the
    # new node's ID, or None if the pool is exhausted or the env doesn't support join.
    def _spawn_node_from_pool(self, preferred_parent) -> Optional["model.NodeID"]:
        if not hasattr(self, "add_node_dynamic"):
            self.logger.warning(
                "[DynamicEnv] dynamic_mode=%r requires an environment implementing "
                "add_node_dynamic; skipping join", self.dynamic_mode
            )
            return None
        available = [
            i for i in range(len(self.dynamic_join_donor_pool))
            if i not in self._used_donor_pool_indices_this_episode
        ]
        if not available:
            return None
        pool_index = random.choice(available)
        self._used_donor_pool_indices_this_episode.add(pool_index)
        donor_topology_id, donor_original_id, donor_node_info = self.dynamic_join_donor_pool[pool_index]

        new_node_id = f"JoinNode_{len(self._dynamic_joined_this_episode)}_{donor_topology_id}_{donor_original_id}"
        node_info = copy.deepcopy(donor_node_info)
        node_info.node_id = new_node_id
        for vuln in node_info.vulnerabilities.values():
            for result in vuln.results:
                if isinstance(result.outcome, model.Reconnaissance):
                    result.outcome.nodes = []

        self.add_node_dynamic(new_node_id, node_info)
        self._inject_reconnaissance_discovery(new_node_id, preferred_parent)
        if self.verbose > 0:
            self.logger.info(
                "[DynamicEnv] Step %d: joined node %s (from topology %s node %s, sponsored by %s)",
                self.num_iterations, new_node_id, donor_topology_id, donor_original_id, preferred_parent
            )
        return new_node_id

    # Makes a newly-joined node discoverable by appending it to an eligible parent's *existing*
    # Reconnaissance vulnerability outcome, mirroring real discovery mechanics rather than placing
    # the node pre-discovered. Tries the preferred (sponsoring) parent first, then falls back to
    # any other eligible parent with a Reconnaissance vulnerability, and only synthesizes a new one
    # if genuinely none of the eligible parents have one at all.
    def _inject_reconnaissance_discovery(self, new_node_id, preferred_parent):
        candidates = [preferred_parent] + [n for n in self._get_join_eligible_parents() if n != preferred_parent]
        for parent in candidates:
            for vuln in self.get_node(parent).vulnerabilities.values():
                for result in vuln.results:
                    if isinstance(result.outcome, model.Reconnaissance):
                        result.outcome.nodes.append(new_node_id)
                        return
        self._synthesize_recon_vulnerability(preferred_parent, new_node_id)

    # Raised only when _synthesize_recon_vulnerability cannot find ANY vulnerability embedding
    # anywhere in the entire topology to reuse (see the widened-donor search below). Deliberately
    # a hard failure, not a fabricated vector: a zero (or any invented) embedding is a distance
    # outlier in cosine-distance nearest-action matching (Task C, 0.3) and would either make the
    # synthetic action unreachable or, worse, hijack action selection via a NaN cosine distance.
    class NoVulnerabilityEmbeddingInTopology(Exception):
        pass

    # Fallback used only when no eligible parent has any Reconnaissance-outcome vulnerability at
    # all. Reuses an EXISTING embedding (never a fabricated one, e.g. zeros) so the synthetic
    # vulnerability isn't a systematic distance outlier in embedding-distance-based action
    # selection (relevant for the compressed env's cosine-distance nearest-action matching).
    # Widened donor search (Task C, 0.3/0.4 revision): the parent-only search crashed whenever
    # the parent itself had zero vulnerabilities -- confirmed reachable even with property change
    # disabled, since join-sponsor eligibility is not gated on vulnerability count (by design, see
    # below) and several real topologies already contain naturally zero-vulnerability nodes.
    # Widens to: 1) parent's own vulnerabilities, 2) any other DISCOVERED node's vulnerabilities,
    # 3) any node anywhere in the topology, 4) raise NoVulnerabilityEmbeddingInTopology if the
    # entire topology has no vulnerability embedding at all -- never fabricate one.
    # Deliberately does NOT exclude zero-vulnerability nodes from join-sponsor eligibility here
    # or anywhere else: doing so would couple the join mechanism to property change (the sponsor
    # pool would shrink as patching proceeds), entangling two change types meant to be independent.
    # Selection within tiers 2/3 (the genuinely new search tiers) is deterministic (sorted by
    # node_id then vulnerability_ID, first match with a truthy .embedding) rather than drawn from
    # self's RNG, so this fallback never consumes random state and therefore never perturbs any
    # other draw's sequence in an episode where it fires -- it also does not matter which node's
    # embedding gets reused here (only that it's a real, well-formed one), so there is nothing for
    # a seeded random choice to usefully add for the extra complexity of consuming episode RNG
    # state. Tier 1 (the parent's own vulnerabilities) deliberately keeps the EXACT original
    # insertion-order iteration (dict .values(), not sorted by ID) rather than being folded into
    # the same sorted-search helper: checked directly against a real topology and confirmed 3 of
    # 11 nodes have a different insertion-order-first vs sorted-order-first vulnerability, so
    # sorting tier 1 too would have silently changed which embedding gets reused in the
    # pre-existing, already-correct, non-crashing case (parent has >=1 vulnerability) -- that
    # would have been a real behaviour change smuggled into what must stay the byte-identical path.
    def _synthesize_recon_vulnerability(self, parent_node_id, new_node_id):
        parent_info = self.get_node(parent_node_id)
        vuln_id = f"synthetic_recon_{parent_node_id}_{new_node_id}"

        # Tier 1: parent's own vulnerabilities, EXACT original logic (insertion order), unchanged.
        existing_embeddings = [v.embedding for v in parent_info.vulnerabilities.values() if v.embedding]
        embedding = existing_embeddings[0] if existing_embeddings else None

        def first_embedding_in(node_ids):
            for node_id in sorted(node_ids):
                node_info = self.get_node(node_id)
                for candidate_vuln_id in sorted(node_info.vulnerabilities.keys()):
                    candidate_embedding = node_info.vulnerabilities[candidate_vuln_id].embedding
                    if candidate_embedding:
                        return candidate_embedding
            return None

        # Tier 2: any other discovered node (new). Tier 3: any node in the topology (new).
        if embedding is None:
            embedding = first_embedding_in(self.discovered_nodes)
        if embedding is None:
            embedding = first_embedding_in(self.environment.nodes())
        if embedding is None:
            raise self.NoVulnerabilityEmbeddingInTopology(
                f"No vulnerability embedding found anywhere in the topology while synthesizing "
                f"a reconnaissance vulnerability for parent node {parent_node_id!r} "
                f"(joining node {new_node_id!r})."
            )

        parent_info.vulnerabilities[vuln_id] = model.VulnerabilityInfo(
            vulnerability_ID=vuln_id,
            port=parent_info.services[0].name if parent_info.services else "",
            description="Synthetic reconnaissance vulnerability (dynamic node-join fallback)",
            results=[model.PredictedResult(
                type=model.VulnerabilityType.LOCAL, type_str="local",
                outcome=model.Reconnaissance([new_node_id]), outcome_str="Reconnaissance", probability=1.0
            )],
            embedding=embedding, cost=1.0,
        )
        if hasattr(self, "refresh_vulnerabilities_embeddings_for_node"):
            self.refresh_vulnerabilities_embeddings_for_node(parent_node_id)

    def _patch_random_vulnerability(self) -> Optional["model.NodeID"]:
        running_nodes = [
            node for node in self.discovered_nodes
            if self.get_node(node).status == model.MachineStatus.Running
        ]
        if not running_nodes:
            return None
        node_id = random.choice(running_nodes)
        node_info = self.get_node(node_id)
        if node_info.vulnerabilities:
            vuln_id = random.choice(list(node_info.vulnerabilities.keys()))
            del node_info.vulnerabilities[vuln_id]
            self.logger.info(
                "[DynamicEnv] Step %d: Patched vuln '%s' on node %s",
                self.num_iterations, vuln_id, node_id
            )
            return node_id
        return None

    def _disable_random_service(self) -> Optional["model.NodeID"]:
        running_nodes = [
            node for node in self.discovered_nodes
            if self.get_node(node).status == model.MachineStatus.Running
        ]
        if not running_nodes:
            return None
        node_id = random.choice(running_nodes)
        node_info = self.get_node(node_id)
        active_services = [s for s in node_info.services if s.running]
        if active_services:
            service = random.choice(active_services)
            service.running = False
            self.logger.info(
                "[DynamicEnv] Step %d: Disabled service '%s' on node %s",
                self.num_iterations, service.name, node_id
            )
            return node_id
        return None

    # Function to update the environment data structures based on the outcome of the action taken
    def update_episode_by_outcome(self, outcome, target_node):
        if isinstance(outcome, model.Reconnaissance):
            # update discovered nodes
            newly_discovered_nodes_count = 0
            outcome.new_nodes = []
            for node in outcome.nodes:
                # skip nodes that have been dynamically removed from the environment;
                # their static Reconnaissance lists may still reference them, but they
                # must not be re-added to discovered_nodes (get_node would KeyError)
                if node not in self.discovered_nodes and node in self.environment.nodes:
                    self.discovered_nodes.append(node)
                    newly_discovered_nodes_count += 1
                    outcome.new_nodes.append(node)
            self.discovered_amount += newly_discovered_nodes_count
        elif isinstance(outcome, model.CredentialAccess) or isinstance(outcome, model.LateralMove):
            # update owned nodes
            self.owned_nodes.append(target_node)
        elif isinstance(outcome, model.Collection) or isinstance(outcome, model.Exfiltration) or isinstance(outcome, model.Discovery):
            self.discovered_amount += 1  # discovered or exfiltrated some data (updated before for the number of nodes)
        return outcome

    # Function to perform one step using a static defender heuristic agent
    def static_defender_step(self):
        self._defender_actuator.on_attacker_step_taken() # the defender can reimage nodes or add external events
        new_events, self.changed_nodes = self.static_defender_agent.step(self, self._defender_actuator, self.stepcount)
        self.num_events += new_events
        if isinstance(self.static_defender_agent, ScanAndReimageCompromisedMachines):
            self.last_reimaged = self.changed_nodes
            self.overall_reimaged.extend(self.changed_nodes)
            if self.changed_nodes != []:
                for node in self.changed_nodes:
                    self.owned_nodes.remove(node)
            # Reiterate all nodes each time to see if they have been reimaged and persistence flag was set, hence we re-owned them
            for node in self.environment.nodes:
                node_info = self.get_node(node)
                if node_info.agent_installed and node_info.status == model.MachineStatus.Running and node not in self.owned_nodes:
                    self.owned_nodes.append(node)
        # no modification for ExternalRandomEvents static defender

    # Check if static defender goal is reached, i.e. all owned nodes are evicted
    def defender_goal_reached(self) -> bool:
        return self.static_defender_eviction_goal and len(self.owned_nodes) == 0

    # Function to check whether the episode has ended in a way that the agent lost
    def check_end_game(self):
        # in node-specific case, they should not kill the node of interest
        if self.goal == "control_node" or self.goal == "discovery_node":
            if self.get_node(self.interest_node).status == model.MachineStatus.Stopped:
                if self.verbose > 1:
                    self.logger.info("Episode ended since node of interest %s has been killed!", self.interest_node)
                return True
        # general condition for all cases
        # if disruption goal, it still fails if it kills all nodes including starter node before disrupting the maximum it can
        at_least_one_running_node = False
        for node in self.owned_nodes:
            if self.get_node(node).status == model.MachineStatus.Running:
                at_least_one_running_node = True
        if not at_least_one_running_node:
            if self.verbose > 1:
                self.logger.info("Episode ended since all nodes have been killed!")
        return not at_least_one_running_node

    # Function to check if the proportional cut-off has been reached, i.e. the number of iterations is greater than the number of nodes times the coefficient
    def proportional_cutoff_reached(self):
        if self.proportional_cutoff_coefficient == -1:
            return False  # No cutoff — run until goal or max_steps
        if self.num_iterations >= self.proportional_nodes * self.proportional_cutoff_coefficient:
            return True
        return False

    # Function to set the proportional cutoff coefficient
    def set_proportional_cutoff_coefficient(self, coefficient):
        self.proportional_cutoff_coefficient = coefficient

    # Function to check whether the game is ended by the attacker reaching its goal, which depends on the goal of the game
    def attacker_goal_reached(self) -> bool:
        if self.goal == "control":
            root_nodes = [node for node in self.owned_nodes if self.get_node(node).privilege_level == model.PrivilegeLevel.ROOT and node != self.starter_node]
            if len(root_nodes) == self.ownable_count:
                if self.verbose > 1:
                    self.logger.info("Attacker won the episode since all nodes controlled at root level...")
                return True # in control games, episode finished when all nodes potentially controllable are controlled
        elif self.goal == "disruption":
            stopped_nodes = [node for node in self.discovered_nodes if node in self.environment.nodes and self.get_node(node).status == model.MachineStatus.Stopped and node != self.starter_node]
            if len(stopped_nodes) == self.disruptable_count: # in disruption, if all nodes potentially disruptable are disrupted
                if self.verbose > 1:
                    self.logger.info("Attacker won the episode since all nodes except the starter node have been disrupted...")
                return True
        elif self.goal == "discovery":
            discovered_nodes = [node for node in self.discovered_nodes if node != self.starter_node]
            data_owning_nodes = [node for node in self.discovered_nodes if node in self.environment.nodes and self.get_node(node).has_data]
            data_collected_not_exfiltred = [node for node in self.discovered_nodes if node in self.environment.nodes and self.get_node(node).data_collected and not self.get_node(node).data_exfiltrated]
            if len(discovered_nodes) == self.discoverable_count and len(data_owning_nodes) == 0 and len(data_collected_not_exfiltred) == 0:
                if self.verbose > 1:
                    self.logger.info("Attacker won the episode since all nodes and data have been discovered and exfiltrated...")
                return True # in discovery games, episode finished when all nodes are discovered and no data is left
        elif self.goal == "control_node":
            if self.interest_node in self.owned_nodes and self.get_node(self.interest_node).privilege_level == model.PrivilegeLevel.ROOT:
                if self.verbose > 1:
                    self.logger.info("Attacker won the episode since the node of interest has been controlled at root level...")
                return True # the goal is to control the node of interest
        elif self.goal == "discovery_node":
            if self.interest_node in self.discovered_nodes:
                node_info = self.get_node(self.interest_node)
                if node_info.visible:
                    if node_info.has_data:
                        if node_info.data_collected and node_info.data_exfiltrated:
                            if self.verbose > 1:
                                self.logger.info(
                                    "Attacker won the episode since the node of interest has been discovered and data exfiltrated...")
                            return True # the goal is to discover the node of interest and all its elements if it has them
                    else:
                        if node_info.visible:
                            if self.verbose > 1:
                                self.logger.info("Attacker won the episode since the node of interest has been discovered (data was also not present)...")
                            return True
                        return True # case with no internal element
        elif self.goal == "disruption_node":
            if self.get_node(self.interest_node).status == model.MachineStatus.Stopped:
                if self.verbose > 1:
                    self.logger.info("Attacker won the episode since the node of interest has been disrupted...")
                return True # node to disrupt is disrupted
        return False

    # Called at the end of the episode to gather statistics:
    def get_statistics(self):
        owned_nodes = [node_id for node_id, node_data in self.environment.nodes.items() if node_data["data"].agent_installed]
        # Root-owned nodes: mirrors the exact win-condition logic in attacker_goal_reached() for "control"
        root_owned_nodes = [node for node in self.owned_nodes if self.get_node(node).privilege_level == model.PrivilegeLevel.ROOT and node != self.starter_node]
        discovered_nodes = self.discovered_nodes
        not_discovered_nodes = [node_id for node_id, node_data in self.environment.nodes.items() if node_id not in self.discovered_nodes]
        disrupted_nodes = [node_id for node_id in self.discovered_nodes if node_id in self.environment.nodes and self.get_node(node_id).status == model.MachineStatus.Stopped]
        self.network_availability = len([node for node in self.discovered_nodes if node in self.environment.nodes and self.get_node(node).status == model.MachineStatus.Running]) / len(self.discovered_nodes)
        return len(owned_nodes), len(discovered_nodes), len(not_discovered_nodes), len(disrupted_nodes), self.num_nodes, self.ownable_count, self.discoverable_count, self.disruptable_count, self.network_availability, len(self.overall_reimaged), self.num_events, self.discovered_amount, self.discoverable_amount, self.attacker_goal_reached(), len(root_owned_nodes)

    # Function to get the list of alive nodes, i.e. nodes that are running
    def get_alive_nodes(self):
        return [node_id for node_id in self.discovered_nodes if node_id in self.environment.nodes and self.get_node(node_id).status == model.MachineStatus.Running]

    # Function to get the list of nodes in the environment
    def get_nodes(self):
        return self.environment.nodes

    # Function to get the environment graph, used to access the network topology
    def get_graph(self):
        return self.environment

    # Function to get the actuator used to execute actions in the simulation environment
    def get_actuator(self):
        return self._actuator

    # Function to get the information of a specific node
    def get_node(self, node_id):
        return self.environment.nodes(data=True)[node_id]["data"]

    # Function to get the index of a service in a node's services list
    def get_service_index(self, port_name: model.PortName, node_info) -> int:
        for service in node_info.services:
            if service.name == port_name:
                return node_info.services.index(service)
        return -1

    # Function to get the list of discovered but not yet owned nodes
    def get_discovered_not_owned_nodes(self):
        discovered_not_owned_nodes = [node_id for node_id in self.discovered_nodes if node_id not in self.owned_nodes]
        return discovered_not_owned_nodes

    # Function to get the list of reachable nodes from the starter node based on the goal
    def get_reachable_nodes(self, goal="control"):
        reachable_nodes = []
        if goal == "control":
            paths = self.__initial_environment.access_shortest_paths
        elif goal == "discovery":
            paths = self.__initial_environment.knows_shortest_paths
        elif goal == "disruption":
            paths = self.__initial_environment.dos_shortest_paths
        else:
            return None # invalid goal for paths
        for node in paths[self.starter_node]:
            reachable_nodes.append(node)
        return reachable_nodes

    # Function to get the initial environment used to create the CyberBattleEnv instance
    def get_initial_environment(self):
        return self.__initial_environment

    # Function to set a property of a node, e.g. visibility, data collected, etc.
    def set_node_property(self, node_id, property_name, value):
        # Check if the node exists
        if node_id in self.environment.nodes:
            node_data = self.environment.nodes[node_id]["data"]

            # Use setattr to dynamically set the attribute on the node's data
            if hasattr(node_data, property_name):
                setattr(node_data, property_name, value)
            else:
                raise AttributeError(f"Property '{property_name}' not found in NodeInfo.")
        else:
            raise KeyError(f"Node with ID {node_id} not found in the environment.")

    # FUnction to set the cut-off for the episode, i.e. the maximum number of iterations, alternative to proportional cut-off
    def set_cut_off(self, cut_off):
        self.episode_iterations = cut_off

    # Function to set the winning reward
    def set_winning_reward(self, winning_reward):
        self.winning_reward = winning_reward

    # Function to set the goal of the attacker agent
    def set_goal(self, goal):
        self.goal = goal

    # Function to set the threshold determining a minimum amount of reachable nodes to not consider starter node isolated
    def set_isolation_filter_threshold(self, threshold):
        self.isolation_filter_threshold = threshold

    # Function to set the random starter node flag
    def set_random_starter_node(self, random_starter_node):
        self.random_starter_node = random_starter_node

    # Function to set the stop at goal reached flag
    def set_stop_at_goal_reached(self, stop_at_goal_reached):
        self.stop_at_goal_reached = stop_at_goal_reached

    # Function to print the information of a specific node
    def print_node_info(self, node_index, node_info):
        if self.get_node(node_index).agent_installed:
            print("discovery status: owned")
        elif node_index in self.discovered_nodes:
            print("discovery status: discovered")
        else:
            print("discovery status: not discovered")
        for key, value in node_info.items():
            print(f'{key}: {value}')
        print()

    # Function to print the information of all nodes in the environment
    def print_nodes_info(self):
        owned_nodes = [node_id for node_id, node_data in self.environment.nodes() if node_data.agent_installed]
        discovered_nodes = [node_id for node_id in self.discovered_nodes if
                            not self.get_node(node_id).agent_installed]
        not_discovered_nodes = [node_id for node_id, node_data in self.environment.nodes() if
                                not node_data.agent_installed and node_id not in self.discovered_nodes]
        max_width = max(len(owned_nodes), len(discovered_nodes), len(not_discovered_nodes))
        owned_texts = []
        discovered_texts = []
        not_discovered_texts = []
        for i in range(max_width):
            if i < len(owned_nodes):
                owned = owned_nodes[i]
                owned_color = "\033[0m"
                owned_texts.append(f"{owned_color}{owned}\033[0m")
            else:
                owned_texts.append("")

            if i < len(discovered_nodes):
                discovered = discovered_nodes[i]
                discovered_color =  "\033[0m"
                discovered_texts.append(f"{discovered_color}{discovered}\033[0m")
            else:
                discovered_texts.append("")

            if i < len(not_discovered_nodes):
                not_discovered = not_discovered_nodes[i]
                not_discovered_texts.append(not_discovered)
            else:
                not_discovered_texts.append("")

        owned_text = "Owned: { " + ", ".join(owned_texts) + " }"
        discovered_text = "Discovered: { " + ", ".join(discovered_texts) + " }"
        not_discovered_text = "Not discovered: { " + ", ".join(not_discovered_texts) + " }"

        print(owned_text)
        print(discovered_text)
        print(not_discovered_text)

    # Used to sample valid actions (e.g. to expose the GAE to many configurations)
    def sample_valid_action(self):
        selected = False
        valid_nodes = list(self.discovered_nodes)
        pairs = []
        vulnerabilities_selected = []
        source_node_index, target_node_index, vulnerability_ID, vulnerability_outcome_class, interested_node_index, vulnerability_type, vulnerability_outcome_str = None, None, None, None, None, None, None
        while True:
            # update list once nodes may be terminated
            for node in valid_nodes:
                if not self.get_node(node).status == model.MachineStatus.Running:
                    valid_nodes.remove(node)
            if len(valid_nodes) == 0:
                self.blocked_graph = True
                return None, None, None, None
            owned_nodes = [node for node in valid_nodes if self.get_node(node).agent_installed]
            if len(pairs) >= len(owned_nodes) * len(valid_nodes): # once all nodes selected at least once, can stop, it is enough
                break
            if len(owned_nodes) == 0:
                self.blocked_graph = True
                return None, None, None, None
            source_node_index = random.choice(owned_nodes)
            target_node_index = random.choice(valid_nodes)
            pairs.append((source_node_index, target_node_index))
            # Once selected the source and target nodes, select the vulnerability
            # Use logic to select a valid one
            if source_node_index == target_node_index:
                for vulnerability_ID in self.get_node(source_node_index).vulnerabilities:
                    vulnerability = self.get_node(source_node_index).vulnerabilities[vulnerability_ID]
                    if vulnerability.privileges_required:
                        if not self.get_node(source_node_index).privilege_level >= vulnerability.privileges_required:
                            continue
                    for result in vulnerability.results:
                        if self.get_node(source_node_index).privilege_level == model.PrivilegeLevel.ROOT and isinstance(result.outcome, model.PrivilegeEscalation):
                            continue
                        if not self.get_node(source_node_index).has_data and isinstance(result.outcome, model.Collection):
                            continue
                        if self.get_node(source_node_index).visible and isinstance(result.outcome, model.Discovery):
                            continue
                        if self.get_node(source_node_index).defense_evasion and isinstance(result.outcome, model.DefenseEvasion):
                            continue
                        if (self.get_node(source_node_index).data_exfiltrated or not self.get_node(source_node_index).data_collected) and isinstance(result.outcome, model.Exfiltration):
                            continue
                        if self.get_node(source_node_index).persistence and isinstance(result.outcome, model.Persistence):
                            continue
                        vulnerabilities_selected.append((source_node_index, target_node_index, source_node_index, vulnerability.vulnerability_ID, result.type_str, result.outcome_str, result.outcome))
            else:
                for vulnerability_ID in self.get_node(target_node_index).vulnerabilities:
                    vulnerability = self.get_node(target_node_index).vulnerabilities[vulnerability_ID]
                    if vulnerability.privileges_required:
                        if not self.get_node(target_node_index).privilege_level >= vulnerability.privileges_required:
                            continue
                    target_node_is_listening = vulnerability.port in [i.name for i in self.get_node(target_node_index).services if i.running]
                    if not target_node_is_listening:
                        continue
                    for result in vulnerability.results:
                        if result.type == VulnerabilityType.REMOTE:
                            if self.get_node(target_node_index).agent_installed and (isinstance(result.outcome, model.CredentialAccess) or isinstance(result.outcome, model.LateralMove)):
                                continue
                            if not self.get_node(target_node_index).has_data and isinstance(result.outcome, model.Collection):
                                continue
                            if self.get_node(target_node_index).visible and isinstance(result.outcome,
                                                                                                     model.Discovery):
                                continue
                            if self.get_node(target_node_index).defense_evasion and isinstance(result.outcome, model.DefenseEvasion):
                                continue
                            if (self.get_node(target_node_index).data_exfiltrated or not self.get_node(target_node_index).data_collected) and isinstance(result.outcome, model.Exfiltration):
                                continue
                            if self.get_node(target_node_index).persistence and isinstance(result.outcome, model.Persistence):
                                continue
                            if self.access_graph.has_edge(source_node_index, target_node_index):
                                edge_data = self.access_graph.get_edge_data(source_node_index, target_node_index)
                                vulnerabilities = edge_data.get('vulnerabilities', [])
                                vulnerabilities_IDs = [ vulnerability[0] for vulnerability in vulnerabilities ]
                                vulnerability_present = vulnerability_ID in vulnerabilities_IDs
                                if vulnerability_present:
                                    vulnerabilities_selected.append((source_node_index, target_node_index, target_node_index, vulnerability.vulnerability_ID, result.type_str, result.outcome_str, result.outcome))
        if vulnerabilities_selected:
            # Prioritize vulnerabilities selected according to a proper logic
            selected = False

            non_dos_vulnerabilities = []
            dos_vulnerabilities = []
            for vulnerability in vulnerabilities_selected:
                source_node_index, target_node_index, interested_node_index, vulnerability_ID, vulnerability_type, vulnerability_outcome_str, vulnerability_outcome_class = vulnerability
                if source_node_index == target_node_index and vulnerability_outcome_str.lower() == 'dos':
                    continue
                if vulnerability_outcome_str.lower() == 'dos':
                    dos_vulnerabilities.append(vulnerability)
                else:
                    non_dos_vulnerabilities.append(vulnerability)
            # prioritizing non DOS, since they disrupt elements
            while non_dos_vulnerabilities:
                vulnerability = random.choice(non_dos_vulnerabilities)
                source_node_index, target_node_index, interested_node_index, vulnerability_ID, vulnerability_type, vulnerability_outcome_str, vulnerability_outcome_class = vulnerability
                non_dos_vulnerabilities.remove(vulnerability)
                if (interested_node_index, vulnerability_ID, vulnerability_type,
                    vulnerability_outcome_str) not in self.sampled_actions:
                    selected = True
                    break
            if not selected:
                # If no non-DOS vulnerability was suitable, try DOS vulnerabilities
                while dos_vulnerabilities:
                    vulnerability = random.choice(dos_vulnerabilities)
                    source_node_index, target_node_index, interested_node_index, vulnerability_ID, vulnerability_type, vulnerability_outcome_str, vulnerability_outcome_class = vulnerability
                    dos_vulnerabilities.remove(vulnerability)
                    if (interested_node_index, vulnerability_ID, vulnerability_type,
                        vulnerability_outcome_str) not in self.sampled_actions:
                        selected = True
                        break
                if not selected:
                    self.blocked_graph = True
        else:
            self.blocked_graph = True
        if selected:
            if self.verbose > 2:
                self.logger.info("Sampled action: source node %s, target node %s, vulnerability %s, outcome %s", source_node_index, target_node_index, vulnerability_ID, vulnerability_outcome_class)
            self.sampled_actions.append((interested_node_index, vulnerability_ID, vulnerability_type, vulnerability_outcome_str))
            return source_node_index, target_node_index, vulnerability_ID, vulnerability_outcome_class
        else:
            return None, None, None, None

    # Function to update the environment used by the CyberBattleEnv instance
    def update_environment(self, environment):
        self.environment = environment

    # Function to seed the environment, setting the random seed for reproducibility
    def seed(self, seed: Optional[int] = None) -> None:
        if seed is None:
            self._seed = seed
            random.seed(seed)
            numpy.random.seed(seed)
            return

    def close(self) -> None:
        return None
