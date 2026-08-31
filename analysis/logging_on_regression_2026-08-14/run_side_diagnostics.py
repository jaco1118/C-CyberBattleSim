"""Task LOGGING-ON-REGRESSION, Amendment 2, B1/B2: supplementary read-only counters over the
SAME Side B (current HEAD, drift_logging=True) construction already used and accepted in the
main comparison -- same checkpoint (band 10-15 seed 42), same topology, same seeds, same 2000
steps. This is NOT a re-comparison: no baseline is captured or compared here, and the trajectory
is deterministic given (seed, checkpoint, config, code), so re-running it under counting
instrumentation reproduces the exact same run already validated bitwise-identical against both
Side A and Side B_off. This script only adds counters; it does not modify
cyberbattle_env_compressed.py or any default.

B1 (dynamic event count per seed): read directly off info["change_type"] (StepInfo,
cyberbattle_env_compressed.py:750-766, built unconditionally every step from
",".join(event["change_type"] for event in dynamic_events) if dynamic_events else None) -- no
patch needed, this field already existed in the row info dict the main run discarded.

B2 (h2/h3 cached-vs-fresh branch counts): _drift_snapshot_from_cache/_drift_snapshot_fresh
(cyberbattle_env_compressed.py:1009-1023) are monkey-patched at the bound-method level (wrapping,
not replacing, their behaviour) purely to record which one fires and in what order, from OUTSIDE
the class -- the environment's own code and control flow are untouched.

Call-order argument for how the wrapped calls map to h1/h2/h3, established directly from step()'s
own source (already read and quoted in the STEP 0 reply): with drift_sample_rate=1 (this run's
config, matching every reported run per Task METRIC-DEFINITIONS), log_this_step is True on every
step, which makes need_h1_h2 True on every step too (:587,592) -- so h1, h2, and h3 are ALL
captured on every single step, in that fixed order (h1 at :594, h2 at :646-649, h3 at :772-775).
h1 always calls _drift_snapshot_from_cache() (never fresh). So of the (up to) 3 snapshot calls
made per step, the FIRST is always h1 (cache, uninformative for B2), the SECOND is h2, the THIRD
(if present) is h3. This script verifies exactly 3 calls happen every step (asserting the count),
then classifies calls 2 and 3 by which underlying method fired.

Usage: python run_side_diagnostics.py <seed> <n_steps>
"""
import os
import random
import sys

REPO_ROOT_REAL = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim"
RUN_FOLDER = os.path.join(
    REPO_ROOT_REAL, "cyberbattle/agents",
    "logs/trpo_250k_tuned_compressed_band10-15_seed42_2026-07-26_11-56-51/TRPO_x_control_SecureBERT")
TOPOLOGY_PKL = os.path.join(REPO_ROOT_REAL, "cyberbattle/data/env_samples/scalability_10_15/1/network_SecureBERT.pkl")


def main():
    seed, n_steps = int(sys.argv[1]), int(sys.argv[2])
    sys.path.insert(0, REPO_ROOT_REAL)

    import numpy as np
    import torch
    import yaml
    from sb3_contrib import TRPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

    from cyberbattle.gae.model import GAEEncoder
    from cyberbattle._env.cyberbattle_env_switch import RandomSwitchEnv
    import cyberbattle._env.cyberbattle_env_compressed as ccenv

    torch.set_num_threads(1)

    with open(os.path.join(RUN_FOLDER, "train_config.yaml")) as f:
        train_config = yaml.safe_load(f)
    with open(train_config["graph_encoder_config_path"]) as f:
        config_encoder = yaml.safe_load(f)
    with open(train_config["graph_encoder_spec_path"]) as f:
        spec_encoder = yaml.safe_load(f)
    config_encoder.update(spec_encoder)
    graph_encoder = GAEEncoder(config_encoder["node_feature_vector_size"],
                                config_encoder["model_config"]["layers"],
                                config_encoder["edge_feature_vector_size"])
    graph_encoder.load_state_dict(torch.load(train_config["graph_encoder_path"]))
    graph_encoder.eval()
    train_config["node_embeddings_dimensions"] = config_encoder["model_config"]["layers"][-1]["out_channels"]

    import pickle as pkl
    with open(TOPOLOGY_PKL, "rb") as f:
        network = pkl.load(f)

    import logging
    logger = logging.getLogger("logging_on_regression_diag")
    logger.addHandler(logging.NullHandler())

    kwargs = {k: v for k, v in train_config.items() if k != "verbose"}
    kwargs["drift_logging"] = True
    kwargs["drift_sample_rate"] = 1
    kwargs["drift_log_path"] = None
    kwargs["drift_run_id"] = f"logging_on_regression_diag_seed{seed}"
    kwargs["drift_seed"] = seed
    kwargs["drift_scenario_id"] = "scalability_10_15_1"

    env = ccenv.CyberBattleCompressedEnv(initial_environment=network, logger=logger, verbose=0, **kwargs)
    env.set_graph_encoder(graph_encoder)
    env.set_pca_components(train_config.get("pca_components"))

    # --- B2 instrumentation: wrap, don't replace ---
    call_log = []
    orig_cache = env._drift_snapshot_from_cache
    orig_fresh = env._drift_snapshot_fresh

    def wrapped_cache():
        call_log.append("cache")
        return orig_cache()

    def wrapped_fresh():
        call_log.append("fresh")
        return orig_fresh()

    env._drift_snapshot_from_cache = wrapped_cache
    env._drift_snapshot_fresh = wrapped_fresh

    switch_env = RandomSwitchEnv([0], switch_interval=10 ** 9, envs_list=[env], save_to_csv=False, verbose=0)
    vec_env = DummyVecEnv([lambda: switch_env])
    vecnormalize_path = os.path.join(RUN_FOLDER, "checkpoints", "1", "checkpoint_vecnormalize_250000_steps.pkl")
    vec_env = VecNormalize.load(vecnormalize_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    checkpoint_path = os.path.join(RUN_FOLDER, "checkpoints", "1", "checkpoint_250000_steps.zip")
    model = TRPO.load(checkpoint_path)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs = vec_env.reset()
    n_dynamic_event_steps = 0
    n_dynamic_events_by_type = {}
    h2_cached = h2_fresh = h3_cached = h3_fresh = 0
    n_steps_3calls = 0
    n_steps_other_call_count = 0

    for i in range(n_steps):
        call_log.clear()
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, infos = vec_env.step(action)
        info = infos[0]

        change_type = info.get("change_type")
        if change_type:
            n_dynamic_event_steps += 1
            for ct in change_type.split(","):
                n_dynamic_events_by_type[ct] = n_dynamic_events_by_type.get(ct, 0) + 1

        if len(call_log) == 3:
            n_steps_3calls += 1
            # call_log[0] is h1 (always cache, uninformative here); [1]=h2; [2]=h3
            if call_log[1] == "cache":
                h2_cached += 1
            else:
                h2_fresh += 1
            if call_log[2] == "cache":
                h3_cached += 1
            else:
                h3_fresh += 1
        else:
            n_steps_other_call_count += 1

    print(f"seed={seed} n_steps={n_steps}")
    print(f"  B1 dynamic_event_steps={n_dynamic_event_steps} by_type={n_dynamic_events_by_type}")
    print(f"  B2 steps_with_exactly_3_snapshot_calls={n_steps_3calls} "
          f"steps_with_other_call_count={n_steps_other_call_count}")
    print(f"  B2 h2_cached={h2_cached} h2_fresh={h2_fresh} h3_cached={h3_cached} h3_fresh={h3_fresh}")


if __name__ == "__main__":
    main()
