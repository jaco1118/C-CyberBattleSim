"""Task-Z / RQ2B ablation eval (reconstructed per evidence_taskZ.md).

Evaluates one trained cell (one --name run folder = 5 seeds under checkpoints/{run_id}/) at 10-15,
under a given change condition, and writes per-seed terminal root-owned COUNT means.

Metric: root-owned COUNT = get_statistics()[14] (owned nodes at ROOT privilege, excl. starter;
cyberbattle_env.py:1171), captured at episode termination BEFORE reset (F1 terminal-read convention).
Eval is STOCHASTIC (model.predict default; deterministic greedy collapses this continuous policy, per F1).

CRITICAL: the eval env is built from the FULL training config (same keys as training / build_band_envs
-- sample_subset_samples=100, remove_*_obstacles, goal, distance_metric, aggregations, ...), otherwise
it rebuilds the full action space every step (pathologically slow) AND is a different env than the agent
was trained on. Obs pipeline mirrors training: raw obs -> VecNormalize.normalize_obs (loaded stats) ->
[Arm 3: zero graph_embeddings[64:192]] -> model.predict.

Conditions:  static -> dynamic_mode=none ;  change -> dynamic_mode=both, change_interval=20 (30-40 setting)

Usage:
  taskZ_eval.py --run <logs/CELL_ts> --config <arm cfg yaml> --topo 44 --arm {1|2|3} \
                --condition {static|change} --episodes 200 --out <csv>
"""
import argparse, glob, os, pickle, logging, csv
import numpy as np, yaml, torch
from sb3_contrib import TRPO
from cyberbattle._env.cyberbattle_env_compressed import CyberBattleCompressedEnv
from cyberbattle.gae.model import GAEEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LG = logging.getLogger("z"); LG.addHandler(logging.NullHandler())
# config keys that are NOT env constructor kwargs (training/orchestration only)
_SKIP = {"train_iterations", "nlp_extractor", "algorithm", "algorithm_hyperparams", "policy_kwargs",
         "seeds_runs", "seeds", "static_seeds", "random_seeds", "load_seeds", "load_envs", "name",
         "finetune_model", "num_runs", "verbose", "yaml", "extremal_mask", "zero_graph_embeddings", "switch_interval",
         "val_switch_interval", "checkpoints_save_freq", "early_stopping", "n_val_episodes", "val_freq",
         "save_csv_file", "save_embeddings_csv_file", "load_processed_envs", "pca_components", "goal",
         "learning_rate", "learning_rate_type",
         # passed explicitly below -> must not also appear in **kw
         "node_embeddings_dimensions", "rewards_dict", "penalties_dict", "initial_environment", "logger"}


def load_encoder():
    ed = os.path.join(ROOT, "cyberbattle/gae/logs/default/SecureBERT")
    ce = yaml.safe_load(open(os.path.join(ed, "train_config_encoder.yaml")))
    ce.update(yaml.safe_load(open(os.path.join(ed, "model_spec.yaml"))))
    enc = GAEEncoder(ce['node_feature_vector_size'], ce['model_config']['layers'], ce['edge_feature_vector_size'])
    enc.load_state_dict(torch.load(os.path.join(ed, "encoder.pth"))); enc.eval()
    return enc, ce['model_config']['layers'][-1]['out_channels']


def make_env(cfg, network, enc, nd, condition):
    c = dict(cfg)
    c['dynamic_mode'] = "both" if condition == "change" else "none"
    if condition == "change":
        c.setdefault('change_interval', 20)
    kw = {k: v for k, v in c.items() if k not in _SKIP}
    e = CyberBattleCompressedEnv(initial_environment=network, logger=LG, verbose=0,
                                 goal=cfg.get('goal', 'control'), node_embeddings_dimensions=nd,
                                 rewards_dict=REWARDS['rewards_dict'][cfg.get('goal', 'control')],
                                 penalties_dict=REWARDS['penalties_dict'][cfg.get('goal', 'control')], **kw)
    e.set_graph_encoder(enc)
    return e


def eval_seed(ckpt, vecn_pkl, cfg, network, enc, nd, arm, condition, episodes):
    env = make_env(cfg, network, enc, nd, condition)
    model = TRPO.load(ckpt, device="cpu")
    vecn = pickle.load(open(vecn_pkl, "rb")); vecn.training = False; vecn.norm_reward = False
    roots = []
    for _ in range(episodes):
        obs = env.reset(); done = False; last_root = 0
        while not done:
            nobs = vecn.normalize_obs(obs)
            if arm == 3:
                nobs = dict(nobs); nobs["graph_embeddings"] = np.array(nobs["graph_embeddings"], copy=True)
                nobs["graph_embeddings"][64:192] = 0.0
            elif arm == 4:  # O8: zero-graph-info floor, all 256 graph_embeddings dims
                nobs = dict(nobs); nobs["graph_embeddings"] = np.array(nobs["graph_embeddings"], copy=True)
                nobs["graph_embeddings"][:] = 0.0
            action, _ = model.predict(nobs)
            obs, _, done, _ = env.step(action)
            last_root = env.get_statistics()[14]
        roots.append(last_root)
    return float(np.mean(roots)), len(roots)


def main():
    global REWARDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--topo", required=True); ap.add_argument("--arm", type=int, required=True)
    ap.add_argument("--condition", required=True); ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    REWARDS = yaml.safe_load(open(os.path.join(ROOT, "cyberbattle/agents/config/rewards_config.yaml")))
    cfg = yaml.safe_load(open(a.config))
    enc, nd = load_encoder()
    network = pickle.load(open(os.path.join(ROOT, f"cyberbattle/data/env_samples/scalability_10_15/{a.topo}/network_SecureBERT.pkl"), "rb"))
    seeds = yaml.safe_load(open(glob.glob(os.path.join(a.run, "*/seeds.yaml"))[0]))['seeds']
    ck_dir = glob.glob(os.path.join(a.run, "*/checkpoints"))[0]
    rows = []
    for i, seed in enumerate(seeds):
        sub = str(i + 1)   # checkpoint subdirs are 1-indexed; subdir (i+1) == seeds[i]
        ck = os.path.join(ck_dir, sub, "checkpoint_250000_steps.zip")
        vp = os.path.join(ck_dir, sub, "checkpoint_vecnormalize_250000_steps.pkl")
        if not (os.path.exists(ck) and os.path.exists(vp)):
            print(f"  MISSING subdir {sub} (seed {seed})"); continue
        m, n = eval_seed(ck, vp, cfg, network, enc, nd, a.arm, a.condition, a.episodes)
        rows.append([a.arm, a.topo, a.condition, sub, seed, m, n])
        print(f"  arm{a.arm} t{a.topo} {a.condition} sub{sub} seed{seed}: mean_root_owned={m:.3f} (n={n})", flush=True)
    w = csv.writer(open(a.out, "w")); w.writerow(["arm", "topo", "condition", "run_id", "seed", "mean_root_owned", "n_episodes"]); w.writerows(rows)
    print(f"wrote {a.out} ({len(rows)} seeds)")


if __name__ == "__main__":
    main()
