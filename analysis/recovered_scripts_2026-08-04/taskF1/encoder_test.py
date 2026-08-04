"""Task H 0.2 empirical: encode a full scenario with the FROZEN encoder and report the per-node
embedding-norm distribution. Compares a 225-node scenario against an 80-100 one to detect
degeneration/saturation/collapse at the larger size. Uses the env's own encode() path.
Usage: python encoder_test.py <scenario_pkl> <label>"""
import sys, os, logging, pickle
sys.path.insert(0, "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim")
os.chdir("/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents")
import numpy as np, torch, yaml
from cyberbattle.utils.envs_utils import wrap_graphs_to_compressed_envs
from cyberbattle.gae.model import GAEEncoder

PKL = sys.argv[1]; LABEL = sys.argv[2]
torch.set_num_threads(4)
REF_CFG = "/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1/runs/trpo_250k_F1_static_seed42/train_config.yaml"
cfg = yaml.safe_load(open(REF_CFG))
ce = yaml.safe_load(open(cfg["graph_encoder_config_path"])); ce.update(yaml.safe_load(open(cfg["graph_encoder_spec_path"])))
ge = GAEEncoder(ce["node_feature_vector_size"], ce["model_config"]["layers"], ce["edge_feature_vector_size"])
ge.load_state_dict(torch.load(cfg["graph_encoder_path"])); ge.eval()
cfg["node_embeddings_dimensions"] = ce["model_config"]["layers"][-1]["out_channels"]
logger = logging.getLogger("enc"); logger.addHandler(logging.NullHandler())
ec = dict(cfg); ec["dynamic_mode"] = "none"; ec["patch_service_dynamic_enabled"] = False; ec["drift_logging"] = False
net = pickle.load(open(PKL, "rb"))
env = wrap_graphs_to_compressed_envs(net, logger, **ec)
env.set_graph_encoder(ge); env.set_pca_components(ec["pca_components"])
env.reset()

# Build the FULL discovered graph: every running node + every access-graph edge (the ground-truth
# reachability the agent would eventually traverse), so encode() sees the whole topology.
import cyberbattle.simulation.model as model
running = [n for n in env.environment.nodes if env.get_node(n).status == model.MachineStatus.Running]
for n in running:
    env.add_node_evolving_visible_graph(n)
edges_added = 0
for u, v, data in net.access_graph.edges(data=True):
    if u not in running or v not in running:
        continue
    vulns = data.get("vulnerabilities", [])
    vid = None
    for cand, _ in vulns:
        if cand in env.vulnerabilities_embeddings:
            vid = cand; break
    if vid is None:  # fallback: any vuln on the target that has an embedding
        for cand in env.get_node(v).vulnerabilities:
            if cand in env.vulnerabilities_embeddings:
                vid = cand; break
    if vid is not None:
        env.add_edge_evolving_visible_graph(u, v, vid); edges_added += 1

node_embeddings, obs = env.encode(env.evolving_visible_graph)
norms = np.array([float(np.linalg.norm(v)) for v in node_embeddings.values()])
zero_frac = float((norms < 1e-6).mean())
print(f"[{LABEL}] n_nodes={len(running)} edges_added={edges_added} n_embeddings={len(norms)}")
print(f"  per-node embedding norm: mean={norms.mean():.4f} sd={norms.std():.4f} "
      f"min={norms.min():.4f} p10={np.percentile(norms,10):.4f} p50={np.percentile(norms,50):.4f} "
      f"p90={np.percentile(norms,90):.4f} max={norms.max():.4f}")
print(f"  coeff_of_variation(sd/mean)={norms.std()/max(norms.mean(),1e-9):.4f}  frac_near_zero={zero_frac:.4f}")
# pairwise cosine spread: are embeddings collapsing toward a single direction?
M = np.array(list(node_embeddings.values()), dtype=np.float64)
Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
mean_dir = Mn.mean(axis=0); mean_dir /= (np.linalg.norm(mean_dir) + 1e-9)
align = Mn @ mean_dir  # cosine of each node to the mean direction; ->1 means collapsed
print(f"  cosine-to-mean-direction: mean={align.mean():.4f} p90={np.percentile(align,90):.4f} "
      f"(->1.0 = collapsed toward a constant direction)")
