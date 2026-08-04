"""
Task D3 STEP 1.5 verification: (b) null control, (c) positive control, (d) action-space check,
run on the F1 static seed42 agent / topo44 with change_type="substitute". Also a change_type="patch"
sanity run to confirm the removal-only path still fires (functional half of 1.5a; the byte-identical
half is the additive git diff).
Usage: python verify_d3.py
"""
import sys, os, copy, logging, pickle
sys.path.insert(0, "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim")
os.chdir("/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents")
import numpy as np, torch, yaml, pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import TRPO
from cyberbattle.utils.envs_utils import wrap_graphs_to_compressed_envs
from cyberbattle.gae.model import GAEEncoder
from cyberbattle._env.cyberbattle_env_switch import RandomSwitchEnv

REPO = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim"
RUN = os.path.join("/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1",
                   "runs", "trpo_250k_F1_static_seed42")
SEED = 42; TOPO_ID = "44"; TOPO = f"scalability_30_40/{TOPO_ID}"
OUT = "/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1/d3_verify"
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(4)
CKPT = os.path.join(RUN, "checkpoints", "1", "checkpoint_250000_steps.zip")
VECN = os.path.join(RUN, "checkpoints", "1", "checkpoint_vecnormalize_250000_steps.pkl")
cfg = yaml.safe_load(open(os.path.join(RUN, "train_config.yaml")))
ce = yaml.safe_load(open(cfg["graph_encoder_config_path"])); ce.update(yaml.safe_load(open(cfg["graph_encoder_spec_path"])))
ge = GAEEncoder(ce["node_feature_vector_size"], ce["model_config"]["layers"], ce["edge_feature_vector_size"])
ge.load_state_dict(torch.load(cfg["graph_encoder_path"])); ge.eval()
cfg["node_embeddings_dimensions"] = ce["model_config"]["layers"][-1]["out_channels"]
logger = logging.getLogger("verify_d3"); logger.addHandler(logging.NullHandler())


def build(change_type, n_ep, tag):
    drift_csv = os.path.join(OUT, f"drift_{tag}.csv")
    if os.path.exists(drift_csv): os.remove(drift_csv)
    ec = dict(cfg)
    ec["dynamic_mode"] = "none"
    ec["patch_service_dynamic_enabled"] = True
    ec["change_type"] = change_type
    ec["change_interval"] = 20
    ec["drift_logging"] = True; ec["drift_log_path"] = drift_csv; ec["drift_sample_rate"] = 1
    ec["drift_run_id"] = f"d3_{tag}"; ec["drift_seed"] = SEED; ec["drift_scenario_id"] = f"scalability_30_40_{TOPO_ID}"
    net = pickle.load(open(os.path.join(REPO, "cyberbattle/data/env_samples", TOPO, "network_SecureBERT.pkl"), "rb"))
    env = wrap_graphs_to_compressed_envs(net, logger, **ec)
    env.set_graph_encoder(ge); env.set_pca_components(ec["pca_components"])
    switch = RandomSwitchEnv([0], switch_interval=10**9, envs_list=[env], save_to_csv=False, verbose=0)
    tmp = DummyVecEnv([lambda: Monitor(RandomSwitchEnv([0], switch_interval=10**9, envs_list=[env], save_to_csv=False, verbose=0))])
    vecn = VecNormalize.load(VECN, tmp); vecn.training = False; vecn.norm_reward = False
    model = TRPO.load(CKPT, device="cpu")
    np.random.seed(SEED); torch.manual_seed(SEED)
    import random as _r; _r.seed(SEED)

    def norm(o):
        b = {k: np.asarray(v, np.float32)[None, ...] for k, v in o.items()}
        n = vecn.normalize_obs(b); return {k: v[0] for k, v in n.items()}

    obs, _ = switch.reset(); ep = 0; guard = 0
    d_checks = []  # (d) action-space checks
    while ep < n_ep and guard < n_ep * 8000 + 50000:
        guard += 1
        a, _ = model.predict(norm(obs), deterministic=False)
        obs, r, done, trunc, info = switch.step(a)
        cur = switch.current_env
        for evn in cur._last_dynamic_events:
            if evn["change_type"] == "property_substitution":
                X = evn["node_ids"][0]; added = evn["added_vuln"]; removed = evn["removed_vuln"]
                # authoritative catalogue the action space is built from
                cat = cur.vulnerabilities_embeddings_per_node_type.get(X, {"local": [], "remote": []})
                cat_ids = {e["vulnerability_ID"] for t in ("local", "remote") for e in cat[t]}
                # searchable action space (post-rebuild, this step). A vuln lives on the TARGET, so a
                # substituted vuln on X appears in keys with target==X (k[1]).
                ak = cur.action_embeddings
                added_in_actions = any(k[1] == X and k[2] == added for k in ak)
                removed_in_actions = any(k[1] == X and k[2] == removed for k in ak)  # stale key may persist (OK)
                removed_on_node = removed in cur.get_node(X).vulnerabilities  # authoritative: gone from node?
                # NaN / degeneracy guard on this node's action vectors
                any_nan = any(not np.all(np.isfinite(v)) for k, v in ak.items() if (k[0] == X or k[1] == X))
                d_checks.append(dict(node=X, added=added, removed=removed,
                                     added_in_catalogue=added in cat_ids, removed_in_catalogue=removed in cat_ids,
                                     removed_on_node=removed_on_node,
                                     added_in_actions=added_in_actions, removed_in_actions=removed_in_actions,
                                     node_owned=(X in cur.owned_nodes), any_nan_action=any_nan))
        if done or trunc:
            obs, _ = switch.reset(); ep += 1
    switch.current_env._drift_logger.close()
    return drift_csv, d_checks


print("=" * 70)
print("D3 1.5(a) functional: change_type=patch (removal-only path still fires)")
pcsv, _ = build("patch", 2, "patch")
pdf = pd.read_csv(pcsv)
prop = pdf[pdf.change_type == "property"]
print(f"  patch run: rows={len(pdf)} property(removal) events={len(prop)} "
      f"mean change_drift_full(removal)={prop.change_drift_full.mean():.5f}")

print("=" * 70)
print("D3 1.5(b/c/d): change_type=substitute")
scsv, dchecks = build("substitute", 3, "substitute")
sdf = pd.read_csv(scsv)
noch = sdf[sdf.change_type.isna() | (sdf.event_phase == "no_change")]
sub = sdf[sdf.change_type == "property_substitution"]

print("-" * 70)
print("(b) NULL CONTROL — no-change rows drift must be ~0:")
print(f"    no_change rows={len(noch)}  max|change_drift_full|={noch.change_drift_full.abs().max():.2e}  "
      f"mean={noch.change_drift_full.abs().mean():.2e}")

print("(c) POSITIVE CONTROL — substitution rows must have non-zero change drift, no NaN:")
print(f"    property_substitution rows={len(sub)}  "
      f"nonzero_frac={(sub.change_drift_full.abs() > 1e-9).mean():.3f}  "
      f"mean change_drift_full={sub.change_drift_full.mean():.5f}  "
      f"min={sub.change_drift_full.min():.5f} max={sub.change_drift_full.max():.5f}")
print(f"    any NaN in substitution change_drift_full? {bool(sub.change_drift_full.isna().any())}")
print(f"    all n_touched_nodes==1? {bool((sub.n_touched_nodes == 1).all())} (single-event, 0.3)")

print("(d) ACTION-SPACE CHECK — added appears, removed gone (per substitution event):")
dc = pd.DataFrame(dchecks)
if len(dc):
    print(f"    events checked={len(dc)}")
    print(f"    added in node catalogue all True? {bool(dc.added_in_catalogue.all())} (added is a usable capability)")
    print(f"    removed gone from node.vulnerabilities all True? {bool((~dc.removed_on_node).all())} (authoritative removal)")
    print(f"    removed gone from node catalogue all True? {bool((~dc.removed_in_catalogue).all())}")
    print(f"    added_in_actions frac True={dc.added_in_actions.mean():.3f} "
          f"(present when its outcome survives action-space filters; subsampling may drop some)")
    print(f"    removed_in_actions frac True={dc.removed_in_actions.mean():.3f} "
          f"(stale key MAY persist, exactly like the removal condition -- not a failure)")
    print(f"    any NaN in touched-node action vectors? {bool(dc.any_nan_action.any())} (must be False)")
    # Authoritative failures: added not in catalogue, removed still on node/catalogue, or NaN.
    bad = dc[(~dc.added_in_catalogue) | dc.removed_in_catalogue | dc.removed_on_node | dc.any_nan_action]
    print(f"    FAILING events={len(bad)}")
    if len(bad): print(bad.to_string())
else:
    print("    NO substitution events captured (would be a problem)")
print("=" * 70)
