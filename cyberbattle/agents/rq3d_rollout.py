"""RQ3D: fresh inference-only CX-style rollout to obtain per-episode root_owned_departures +
final_root_owned_count (the fields event_graph_logging emits, cf. cyberbattle_env_compressed.py
Task CX B), which are absent from every existing CX output directory (event_episode.jsonl is
missing under cx_step2_static/registration/replay for all 3 bands -- STEP 0 finding).

Population: the standard 5-seed x 3-band checkpoint grid (dynamically trained, dynamic_mode='both'
during training -- verified via app.log DynamicEnv events firing from step 2 of training; NOT
static-trained). This IS Task CX PART 3's own "adapted gate checkpoints": evidence_taskCX.md:272
names them "the dynamic-trained ('adapted') GATE checkpoints"; evidence_taskF1.md:14 cites
"scalability_30_40/44 (... = gate grid slot 3)", matching grid_topology_id_map.json's
"30-40": {"3": "44"} byte-for-byte; the archived manifest folder is literally named
"2026-07-26_trpo_5seed_gate". Same population as RQ2(c) (evidence_taskRQ2C.md: "same
checkpoints/seeds/bands and the same standard attenuation config").

Inference-only: TRPO.load() + VecNormalize.load(..., training=False) + model.predict() (stochastic,
matching project convention). No .learn(), no optimizer, no checkpoint writes.

Two arms per (band, seed, topology):
  CHANGE : dynamic_mode='both', change_interval=20, allow_undiscovered_removal=True,
           uncapped_join=True -- mirrors Task CX's own evaluation condition (evidence_taskCX.md
           STEP 1's three flags; allow_undiscovered_property was removed in STEP 2 PREP, so not
           set here either -- property stays discovered-only/inherited, patch_service_dynamic_
           enabled=False as in the loaded training config, matching CX's actual condition where
           property never fired).
  STATIC : dynamic_mode='none', no relaxation flags (nothing to relax -- no disturbance at all).
           This is the undisturbed pairing baseline for departures_per_static_root (Addendum 1).

Episode counts (Addendum 1, authorized Addendum 2): 15 change-arm + 10 static-arm episodes per
(seed x topology) => 600 change / 400 static per band, 3000 total across 3 bands. Deliberately
smaller than the original's ~4410-pooled scale (STEP 0.5.1 reasoning: the original effect is large,
~43% relative, so this remains well-powered for the one comparison this task needs).

event_graph_logging=True + its prerequisite drift_logging=True (Addendum 2's only authorized
flags) write cyberbattle/utils/event_graph_logger.py's per-episode record (final_root_owned_count,
root_owned_departures, ...) to <out_dir>/<arm>/eventgraph_<band>/event_episode.jsonl. drift_log_path
is left None (in-memory only, no per-step CSV -- not needed for this task) with event_graph_log_dir
set explicitly.

Usage: python rq3d_rollout.py --out-dir <dir> [--band 10-15] [--dry-run-episodes N]
"""
import argparse
import logging
import os
import pickle

import numpy as np
import yaml
from sb3_contrib import TRPO

from cyberbattle._env.cyberbattle_env_compressed import CyberBattleCompressedEnv
from cyberbattle.gae.model import GAEEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AG = os.path.join(ROOT, "cyberbattle", "agents")
LG = logging.getLogger("rq3d")
LG.addHandler(logging.NullHandler())

SEEDS = [42, 100, 123, 200, 300]
BANDS = ["10-15", "30-40", "80-100"]
MANIFEST_PATH = os.path.join(AG, "rq3d_manifest.yaml")  # committed (standing rule: manifests, not just scripts)
N_CHANGE_EP = 15   # per (seed, topology)
N_STATIC_EP = 10   # per (seed, topology)

# config keys that are NOT CyberBattleCompressedEnv constructor kwargs
_SKIP = {"train_iterations", "nlp_extractor", "algorithm", "algorithm_hyperparams", "policy_kwargs",
         "seeds_runs", "seeds", "static_seeds", "random_seeds", "load_seeds", "load_envs", "name",
         "finetune_model", "num_runs", "verbose", "yaml", "extremal_mask", "zero_graph_embeddings",
         "switch_interval", "val_switch_interval", "checkpoints_save_freq", "early_stopping",
         "n_val_episodes", "val_freq", "save_csv_file", "save_embeddings_csv_file",
         "load_processed_envs", "pca_components", "goal", "learning_rate", "learning_rate_type",
         "node_embeddings_dimensions", "rewards_dict", "penalties_dict", "initial_environment", "logger"}


def run_folder(manifest, band, seed):
    return os.path.join(AG, manifest["bands"][band]["run_folders"][seed])


def load_encoder():
    ed = os.path.join(ROOT, "cyberbattle", "gae", "logs", "default", "SecureBERT")
    ce = yaml.safe_load(open(os.path.join(ed, "train_config_encoder.yaml")))
    ce.update(yaml.safe_load(open(os.path.join(ed, "model_spec.yaml"))))
    enc = GAEEncoder(ce['node_feature_vector_size'], ce['model_config']['layers'], ce['edge_feature_vector_size'])
    import torch
    enc.load_state_dict(torch.load(os.path.join(ed, "encoder.pth")))
    enc.eval()
    return enc, ce['model_config']['layers'][-1]['out_channels']


def make_env(cfg, network, enc, nd, arm, run_id, seed, scenario_id, egd):
    c = dict(cfg)
    if arm == "change":
        c["dynamic_mode"] = "both"
        c["change_interval"] = 20
        c["allow_undiscovered_removal"] = True
        c["uncapped_join"] = True
    else:
        c["dynamic_mode"] = "none"
        c["allow_undiscovered_removal"] = False
        c["uncapped_join"] = False
    kw = {k: v for k, v in c.items() if k not in _SKIP}
    e = CyberBattleCompressedEnv(
        initial_environment=network, logger=LG, verbose=0,
        goal=cfg.get("goal", "control"), node_embeddings_dimensions=nd,
        rewards_dict=REWARDS["rewards_dict"][cfg.get("goal", "control")],
        penalties_dict=REWARDS["penalties_dict"][cfg.get("goal", "control")],
        drift_logging=True, drift_log_path=None, drift_sample_rate=1,
        drift_run_id=run_id, drift_seed=seed, drift_scenario_id=scenario_id,
        event_graph_logging=True, event_graph_log_dir=egd,
        **kw)
    e.set_graph_encoder(enc)
    return e


def eval_episodes(env, model, vecn, n_episodes):
    """Run n_episodes; the env's own event_graph_logging writes the per-episode record as a
    side effect on every episode end -- nothing needs to be captured/returned here."""
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            nobs = vecn.normalize_obs(obs)
            action, _ = model.predict(nobs)
            obs, _, done, _ = env.step(action)


def main():
    global REWARDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--band", default=None, help="restrict to one band (10-15|30-40|80-100); default: all")
    ap.add_argument("--seed", type=int, default=None, help="restrict to one seed; default: all 5")
    ap.add_argument("--n-change", type=int, default=N_CHANGE_EP)
    ap.add_argument("--n-static", type=int, default=N_STATIC_EP)
    a = ap.parse_args()

    REWARDS = yaml.safe_load(open(os.path.join(AG, "config", "rewards_config.yaml")))
    manifest = yaml.safe_load(open(MANIFEST_PATH))
    enc, nd = load_encoder()

    bands = [a.band] if a.band else BANDS
    seeds = [a.seed] if a.seed else SEEDS

    for band in bands:
        topo_ids = manifest["bands"][band]["topology_ids"]  # 8 topology IDs for this band, standard grid order
        topo_source = manifest["bands"][band]["topology_source_folder"]
        for seed in seeds:
            rf = run_folder(manifest, band, seed)
            cfg = yaml.safe_load(open(os.path.join(rf, "train_config.yaml")))
            ck = os.path.join(rf, "checkpoints", "1", "checkpoint_250000_steps.zip")
            vp = os.path.join(rf, "checkpoints", "1", "checkpoint_vecnormalize_250000_steps.pkl")
            model = TRPO.load(ck, device="cpu")
            vecn = pickle.load(open(vp, "rb"))
            vecn.training = False
            vecn.norm_reward = False

            for topo_id in topo_ids:
                network = pickle.load(open(os.path.join(
                    ROOT, "cyberbattle", "data", "env_samples", topo_source, topo_id,
                    "network_SecureBERT.pkl"), "rb"))

                for arm, n_ep in (("change", a.n_change), ("static", a.n_static)):
                    egd = os.path.join(a.out_dir, arm, f"eventgraph_{band}")
                    os.makedirs(egd, exist_ok=True)
                    env = make_env(cfg, network, enc, nd, arm, run_id=f"{band}_seed{seed}", seed=seed,
                                    scenario_id=topo_id, egd=egd)
                    eval_episodes(env, model, vecn, n_ep)
                    print(f"[{arm}] band={band} seed={seed} topo={topo_id}: {n_ep} episodes done", flush=True)


if __name__ == "__main__":
    main()
