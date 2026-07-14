#!/usr/bin/env python3
"""
diagnose_stranded_nodes.py

ONE question: for nodes the agent owns but does NOT bring to ROOT, do those
nodes actually have a privilege-escalation vuln available?

Verdicts:
  (A) stranded HAS privesc  -> agent could escalate but didn't
        => learning / reward-shaping problem (expansion beats escalation,
           or privesc success_rate rolls are failing).
  (B) stranded, NO privesc, env leaves it at basic-user
        => genuine infeasibility (generator fixes matter after all).
  (C) stranded, NO privesc, but env "should" auto-root on ownership
        => env config bug OR the Table-II auto-root reading is wrong here.

Usage: fill in STEP 0 + STEP 1 for YOUR env, then run over a few episodes.
"""

import os
import yaml

PRIVESC_KEY = "privilege escalation"  # literal key in vulnerabilities_distribution.yaml

# CONFIRMED from model.py: PrivilegeLevel(IntEnum) NoAccess=0, LocalUser=1, ROOT=3.
# (2 is intentionally skipped.) So "stranded" == owned but privilege_level != 3.
ROOT_LEVEL = 3


def stranded_nodes(env):
    """
    Nodes the agent owns but has NOT brought to ROOT, at the current (ideally
    terminal) env state. Uses the real CyberBattleEnv API confirmed from source:
      env.owned_nodes                      -> list of owned NodeIDs
      env.get_node(n).privilege_level      -> PrivilegeLevel (ROOT == 3)
      env.starter_node                     -> excluded from the win condition
    Call this at episode end, exactly as attacker_goal_reached() counts root nodes.
    """
    return [n for n in env.owned_nodes
            if n != env.starter_node
            and int(env.get_node(n).privilege_level) != ROOT_LEVEL]


# ---------------------------------------------------------------------------
# YAML side -- fully working, uses the structure from collect_vulnerability_data.
# ---------------------------------------------------------------------------

def load_privesc_map(vuln_yaml_path):
    """{node_id: [privesc CVEs]} for one scenario."""
    with open(vuln_yaml_path) as f:
        stats = yaml.safe_load(f) or {}
    per_node = stats.get("outcomes_nodes_presence", {})
    return {nid: outc.get(PRIVESC_KEY, []) for nid, outc in per_node.items()}


def classify(stranded, privesc_map):
    """Return (A_list, B_or_C_list) -> nodes that had privesc vs. nodes that didn't."""
    had_privesc, no_privesc = [], []
    for nid in stranded:
        (had_privesc if privesc_map.get(nid) else no_privesc).append(nid)
    return had_privesc, no_privesc


def report(scenario_id, stranded, privesc_map):
    had, none = classify(stranded, privesc_map)
    print(f"\n=== scenario {scenario_id} ===")
    print(f"stranded (owned, not ROOT): {len(stranded)} node(s)")
    if had:
        print(f"  (A) HAD privesc, not used -> LEARNING problem: {had}")
        for nid in had:
            print(f"        {nid}: privesc CVEs available = {privesc_map[nid]}")
    if none:
        print(f"  (B/C) NO privesc available -> infeasibility OR env bug: {none}")
    if not stranded:
        print("  none stranded this episode -> full ROOT clear, score should be ~1.0")


# ---------------------------------------------------------------------------
# Wiring it into a rollout (sketch -- adapt to your sample_agent.py):
#
#   model = PPO.load(model_path)
#   privesc_map = load_privesc_map(vuln_yaml_for_this_scenario)
#   obs = env.reset()
#   done = False
#   while not done:
#       action, _ = model.predict(obs, deterministic=True)
#       obs, reward, done, info = env.step(action)
#   report(scenario_id, stranded_nodes(env), privesc_map)   # call AT episode end
#
# Run this over ~10-20 episodes across a few scenarios and tally which verdict
# dominates. That distribution is your diagnosis.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(__doc__)
    print("Fill in STEP 0/STEP 1 for your env, then wire report() into your rollout.")
