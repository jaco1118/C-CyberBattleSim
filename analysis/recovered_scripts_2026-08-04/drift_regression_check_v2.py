"""
STEP 3 (task T v2) extended regression check: byte-identical (obs+reward) comparison between
the pre-instrumentation golden code (commit 7cdfb2b, extracted via `git archive` into a scratch
tree, since the instrumentation is not on a separate branch to diff against) and the CURRENT
working tree (post STEP 1 commit 40dfc7c + STEP 2 per-slice-norm columns), with
drift_logging=False, at >=2000 steps and >=2 seeds -- extending the original 150-step/1-seed
drift_regression_check.py per the task's instruction that 150 steps is thin evidence for a
methodological claim the thesis states as a guarantee.

REPO_ROOT is a parameter (not hardcoded) specifically so the identical fixed-action-sequence
harness can be pointed at either the pre-instrumentation snapshot or the current real repo.
Topology and GAE-encoder paths always resolve against the REAL repo (both are gitignored,
so the git-archived snapshot doesn't have them) -- only cyberbattle_env.py/
cyberbattle_env_compressed.py (and everything else under cyberbattle/) differ by REPO_ROOT.

Usage:
    python drift_regression_check_v2.py capture <repo_root> <seed> <n_steps> <output.pkl>
    python drift_regression_check_v2.py compare <repo_root> <seed> <n_steps> <baseline.pkl>
"""
import sys
import os
import pickle
import random
import logging
import numpy as np
import torch
import yaml

REAL_REPO_ROOT = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim"
AGENTS_DIR = os.path.join(REAL_REPO_ROOT, "cyberbattle", "agents")
TOPOLOGY_PKL = os.path.join(REAL_REPO_ROOT, "cyberbattle", "data", "env_samples",
                             "local_baseline_single_topology", "1", "network_SecureBERT.pkl")
ENCODER_DIR = os.path.join(REAL_REPO_ROOT, "cyberbattle", "gae", "logs", "default", "SecureBERT")


def build_env(repo_root, **overrides):
    # repo_root controls which cyberbattle_env*.py gets imported; everything data/encoder-side
    # always comes from the real repo (gitignored, not present in the archived snapshot).
    # Each CLI invocation of this script is its own fresh process (see __main__ below), so a
    # plain sys.path.insert + first-time import is sufficient -- no risk of a stale cached
    # module from a different repo_root within the same process.
    sys.path.insert(0, repo_root)
    import cyberbattle._env.cyberbattle_env_compressed as ccenv
    CyberBattleCompressedEnv = ccenv.CyberBattleCompressedEnv

    with open(os.path.join(ENCODER_DIR, "train_config_encoder.yaml")) as f:
        config_encoder = yaml.safe_load(f)
    with open(os.path.join(ENCODER_DIR, "model_spec.yaml")) as f:
        spec_encoder = yaml.safe_load(f)
    config_encoder.update(spec_encoder)

    from cyberbattle.gae.model import GAEEncoder
    graph_encoder = GAEEncoder(config_encoder['node_feature_vector_size'],
                                config_encoder['model_config']['layers'],
                                config_encoder['edge_feature_vector_size'])
    graph_encoder.load_state_dict(torch.load(os.path.join(ENCODER_DIR, "encoder.pth")))
    graph_encoder.eval()
    node_embeddings_dimensions = config_encoder['model_config']['layers'][-1]['out_channels']

    with open(os.path.join(AGENTS_DIR, "config", "rewards_config.yaml")) as f:
        rewards_config = yaml.safe_load(f)

    with open(TOPOLOGY_PKL, "rb") as f:
        network = pickle.load(f)

    logger = logging.getLogger("drift_regression_check_v2")
    logger.addHandler(logging.NullHandler())

    kwargs = dict(
        initial_environment=network,
        logger=logger,
        goal="control",
        rewards_dict=rewards_config['rewards_dict']['control'],
        penalties_dict=rewards_config['penalties_dict']['control'],
        node_embeddings_dimensions=node_embeddings_dimensions,
        outcome_dimensions=10,
        discrete_features=["owned_nodes", "discovered_nodes"],
        graph_embeddings_aggregations=["mean", "max", "min"],
        episode_iterations=300,
        dynamic_mode="both",
        change_interval=20,
        dynamic_min_alive_nodes=5,
        dynamic_min_alive_fraction=0.5,
        patch_service_dynamic_enabled=True,
        change_type="mixed",
        verbose=0,
    )
    kwargs.update(overrides)
    env = CyberBattleCompressedEnv(**kwargs)
    env.set_graph_encoder(graph_encoder)
    return env


def run_fixed_sequence(env, n_steps, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    obs = env.reset()
    action_dim = env.action_space.shape[0]
    rows = []
    rng = np.random.RandomState(seed)
    for i in range(n_steps):
        action = rng.uniform(-1, 1, size=action_dim).astype(np.float32)
        obs, reward, done, info = env.step(action)
        flat_obs = np.concatenate([np.asarray(obs["graph_embeddings"]).ravel(),
                                    np.asarray(obs["discrete_features"]).ravel()])
        rows.append((flat_obs, float(reward), bool(done)))
        if done:
            random.seed(seed + i + 1)
            np.random.seed(seed + i + 1)
            obs = env.reset()
    return rows


def main():
    mode = sys.argv[1]
    repo_root = sys.argv[2]
    seed = int(sys.argv[3])
    n_steps = int(sys.argv[4])
    path = sys.argv[5]

    env = build_env(repo_root, drift_logging=False)
    rows = run_fixed_sequence(env, n_steps, seed)

    if mode == "capture":
        with open(path, "wb") as f:
            pickle.dump(rows, f)
        print(f"Captured {len(rows)} steps (seed={seed}) from repo_root={repo_root} to {path}")
    elif mode == "compare":
        with open(path, "rb") as f:
            baseline = pickle.load(f)
        assert len(baseline) == len(rows), f"length mismatch: {len(baseline)} vs {len(rows)}"
        n_obs_mismatch = 0
        n_reward_mismatch = 0
        for i, ((b_obs, b_rew, b_done), (r_obs, r_rew, r_done)) in enumerate(zip(baseline, rows)):
            if not np.array_equal(b_obs, r_obs):
                n_obs_mismatch += 1
                if n_obs_mismatch <= 3:
                    print(f"  step {i}: obs mismatch, max abs diff = {np.max(np.abs(b_obs - r_obs))}")
            if b_rew != r_rew:
                n_reward_mismatch += 1
                if n_reward_mismatch <= 3:
                    print(f"  step {i}: reward mismatch {b_rew} vs {r_rew}")
            if b_done != r_done:
                print(f"  step {i}: done mismatch {b_done} vs {r_done}")
        if n_obs_mismatch == 0 and n_reward_mismatch == 0:
            print(f"PASS: all {len(rows)} steps byte-identical (seed={seed}, drift_logging=False)")
        else:
            print(f"FAIL: {n_obs_mismatch} obs mismatches, {n_reward_mismatch} reward mismatches out of {len(rows)} steps (seed={seed})")
            sys.exit(1)
    else:
        raise ValueError(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
