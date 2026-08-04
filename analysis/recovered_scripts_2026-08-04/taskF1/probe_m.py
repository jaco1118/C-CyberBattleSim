"""Task M: two-hop coverage vs the degree effect, at the TRIAL level. Extends probe_p.py's probe() with
per-trial COVERAGE (1-hop and 2-hop, as a fraction of the graph) computed on the SAME sparse graph the
propagation runs on (the DFS spanning tree, probe_p.py:60), reusing the existing hop machinery
(single_source_shortest_path_length, probe_p.py:100). Saves every trial to CSV for the pooled trial-level
test. Sparse mode only (the encoder's real input). 2 threads (runs in idle headroom beside Task Z eval).
No training, no episodes, no env changes."""
import sys, os, logging, pickle, random, collections, glob
sys.path.insert(0, "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim")
os.chdir("/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents")
import numpy as np, torch, yaml, networkx as nx, pandas as pd
from cyberbattle.utils.envs_utils import wrap_graphs_to_compressed_envs
from cyberbattle.gae.model import GAEEncoder
REPO = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim"
REF = "/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1/runs/trpo_250k_F1_static_seed42/train_config.yaml"
OUT = "/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1/m_out"
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(2); random.seed(0); np.random.seed(0)   # seed 0, IDENTICAL to probe_p.py for reproducibility
cfg = yaml.safe_load(open(REF))
ce = yaml.safe_load(open(cfg["graph_encoder_config_path"])); ce.update(yaml.safe_load(open(cfg["graph_encoder_spec_path"])))
ge = GAEEncoder(ce["node_feature_vector_size"], ce["model_config"]["layers"], ce["edge_feature_vector_size"])
ge.load_state_dict(torch.load(cfg["graph_encoder_path"])); ge.eval()
cfg["node_embeddings_dimensions"] = ce["model_config"]["layers"][-1]["out_channels"]
logger = logging.getLogger("m"); logger.addHandler(logging.NullHandler())
ec = dict(cfg); ec["dynamic_mode"] = "none"; ec["patch_service_dynamic_enabled"] = False; ec["drift_logging"] = False

def make_env(pkl):
    net = pickle.load(open(pkl, "rb"))
    env = wrap_graphs_to_compressed_envs(net, logger, **ec)
    env.set_graph_encoder(ge); env.set_pca_components(ec["pca_components"]); env.reset()
    return env, net
def bfs_nodes(access, root, k):
    seen=[root]; q=collections.deque([root]); s={root}
    while q and len(seen)<k:
        u=q.popleft()
        for v in access.successors(u):
            if v not in s:
                s.add(v); seen.append(v); q.append(v)
                if len(seen)>=k: break
    return seen
def edge_vuln(net, env, u, v):
    d = net.access_graph.get_edge_data(u, v)
    if d and d.get("vulnerabilities"):
        for cand,_ in d["vulnerabilities"]:
            if cand in env.vulnerabilities_embeddings: return cand
    for cand in env.get_node(v).vulnerabilities:
        if cand in env.vulnerabilities_embeddings: return cand
    return None
def build_visible(env, net, nodes):   # SPARSE = DFS spanning tree (probe_p.py:60), the correct graph
    env.reset_evolving_visible_graph(); g = env.evolving_visible_graph
    for n in nodes:
        if n not in g: env.add_node_evolving_visible_graph(n)
    A = net.access_graph.subgraph(nodes)
    for u, v in list(nx.dfs_tree(A, nodes[0]).edges()):
        vid = edge_vuln(net, env, u, v)
        if vid is not None: env.add_edge_evolving_visible_graph(u, v, vid)
    return g
def encode_h(env):
    ne, _ = env.encode(env.evolving_visible_graph); return ne

def probe(pkl, k, band, sc, n_trials=30):
    env, net = make_env(pkl)
    feas = [n for n in net.access_graph.nodes() if net.access_graph.out_degree(n) > 0]
    root = random.choice(feas)
    nodes = bfs_nodes(net.access_graph, root, k)
    if len(nodes) < k*0.8: return []
    build_visible(env, net, nodes)
    G = env.evolving_visible_graph.copy(); h = encode_h(env)
    present = [n for n in nodes if n in h]; N = len(present)
    if N < 5: return []
    Gu = G.to_undirected()
    deg = {n: Gu.degree(n) for n in present}
    order = sorted(present, key=lambda n: deg[n])
    idx = np.linspace(0, len(order)-1, min(n_trials, len(order))).astype(int)
    trials = [order[i] for i in sorted(set(idx))]
    H = np.array([h[n] for n in present]); hbar = H.mean(0)
    rows = []
    for v in trials:
        surv = [n for n in present if n != v]
        direct = (hbar - h[v]) / (N-1)
        env.evolving_visible_graph = G.copy(); env.evolving_visible_graph.remove_node(v)
        hp = encode_h(env)
        surv2 = [n for n in surv if n in hp]
        if len(surv2) < 3: continue
        dh = np.array([hp[n]-h[n] for n in surv2]); prop = float(np.linalg.norm(dh.mean(0)))
        # COVERAGE on the SAME sparse graph, from the removed node v (pre-removal Gu), reusing the hop machinery
        hops = nx.single_source_shortest_path_length(Gu, v)   # dist v -> every node (incl v at 0)
        within1 = sum(1 for d in hops.values() if d <= 1)      # v + its neighbours
        within2 = sum(1 for d in hops.values() if d <= 2)
        rows.append(dict(band=band, scenario=sc, v=str(v), N=N, deg=int(deg[v]),
                         prop=prop, direct=float(np.linalg.norm(direct)),
                         cov1=within1 / N, cov2=within2 / N,
                         n1=within1, n2=within2))
    return rows

if __name__ == "__main__":
    BANDS = {"10-15": ("scalability_10_15", 12), "30-40": ("scalability_30_40", 25), "80-100": ("scalability_80_100", 69)}
    NREP = 4
    allrows = []
    for band, (sub, k) in BANDS.items():
        pkls = sorted(glob.glob(f"{REPO}/cyberbattle/data/env_samples/{sub}/*/network_SecureBERT.pkl"))[:NREP]
        for sc, pkl in enumerate(pkls):
            allrows += probe(pkl, k, band, sc, n_trials=30)
    df = pd.DataFrame(allrows)
    df.to_csv(os.path.join(OUT, "m_trials.csv"), index=False)
    print(f"saved {len(df)} trials to {OUT}/m_trials.csv")
    print("per-band trial counts:", df.groupby('band').size().to_dict())
    print("\n### reproducibility check vs Task P (sparse prop median per band) ###")
    for band in BANDS:
        d = df[df.band==band]
        print(f"  {band:7s}: n={len(d)} prop_med={d.prop.median():.4f} corr(prop,deg)={d.prop.corr(d.deg):+.2f} "
              f"cov2 mean={d.cov2.mean():.3f} var={d.cov2.var():.4f}")
