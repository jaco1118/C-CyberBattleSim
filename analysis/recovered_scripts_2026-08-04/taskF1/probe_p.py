"""Task P: offline encoder probe. Decompose pooled-mean change on node removal into DIRECT (removed
node leaving the pool) vs PROPAGATION (survivors' embeddings moving because the graph changed).
Frozen encoder forward passes only. Two graph structures per band:
  SPARSE = BFS tree over access_graph, size n_discovered (matches the real evolving_visible_graph:
           edges only where the agent traversed -> tree-like, diameter>2, lets 3.1 test 2-hop).
  DENSE  = access-graph-induced subgraph on the same node set (task's-premise density; upper bound).
"""
import sys, os, logging, pickle, random, collections
sys.path.insert(0, "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim")
os.chdir("/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents")
import numpy as np, torch, yaml, networkx as nx
from cyberbattle.utils.envs_utils import wrap_graphs_to_compressed_envs
from cyberbattle.gae.model import GAEEncoder
REPO = "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim"
REF = "/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1/runs/trpo_250k_F1_static_seed42/train_config.yaml"
torch.set_num_threads(4); random.seed(0); np.random.seed(0)
cfg = yaml.safe_load(open(REF))
ce = yaml.safe_load(open(cfg["graph_encoder_config_path"])); ce.update(yaml.safe_load(open(cfg["graph_encoder_spec_path"])))
ge = GAEEncoder(ce["node_feature_vector_size"], ce["model_config"]["layers"], ce["edge_feature_vector_size"])
ge.load_state_dict(torch.load(cfg["graph_encoder_path"])); ge.eval()
cfg["node_embeddings_dimensions"] = ce["model_config"]["layers"][-1]["out_channels"]
logger = logging.getLogger("p"); logger.addHandler(logging.NullHandler())
ec = dict(cfg); ec["dynamic_mode"] = "none"; ec["patch_service_dynamic_enabled"] = False; ec["drift_logging"] = False

def make_env(pkl):
    net = pickle.load(open(pkl, "rb"))
    env = wrap_graphs_to_compressed_envs(net, logger, **ec)
    env.set_graph_encoder(ge); env.set_pca_components(ec["pca_components"]); env.reset()
    return env, net

def bfs_nodes(access, root, k):
    seen = [root]; q = collections.deque([root]); s = {root}
    while q and len(seen) < k:
        u = q.popleft()
        for v in access.successors(u):
            if v not in s:
                s.add(v); seen.append(v); q.append(v)
                if len(seen) >= k: break
    return seen

def edge_vuln(net, env, u, v):
    d = net.access_graph.get_edge_data(u, v)
    if d and d.get("vulnerabilities"):
        for cand, _ in d["vulnerabilities"]:
            if cand in env.vulnerabilities_embeddings: return cand
    for cand in env.get_node(v).vulnerabilities:
        if cand in env.vulnerabilities_embeddings: return cand
    return None

def build_visible(env, net, nodes, mode):
    """Return an nx.DiGraph (env.evolving_visible_graph style) on `nodes`; SPARSE=BFS tree, DENSE=access subgraph."""
    env.reset_evolving_visible_graph(); g = env.evolving_visible_graph
    for n in nodes:
        if n not in g: env.add_node_evolving_visible_graph(n)
    A = net.access_graph.subgraph(nodes)
    if mode == "sparse":
        # DFS spanning tree edges from the first node: DEEP (chain-like, diameter>2, mean deg ~2),
        # modelling the agent pivoting node-to-node rather than exploiting everything from the
        # starter (which a BFS tree over the ~1-hop-complete dense access graph collapses to a star).
        edges = list(nx.dfs_tree(A, nodes[0]).edges())
    else:
        edges = list(A.edges())
    for u, v in edges:
        vid = edge_vuln(net, env, u, v)
        if vid is not None: env.add_edge_evolving_visible_graph(u, v, vid)
    return g

def encode_h(env):
    ne, _ = env.encode(env.evolving_visible_graph)
    return ne  # dict node->64d

def probe(pkl, k, mode, n_trials=40):
    env, net = make_env(pkl)
    feas = [n for n in net.access_graph.nodes() if net.access_graph.out_degree(n) > 0]
    root = random.choice(feas)
    nodes = bfs_nodes(net.access_graph, root, k)
    if len(nodes) < k * 0.8: return None
    build_visible(env, net, nodes, mode)
    G = env.evolving_visible_graph.copy()
    h = encode_h(env); present = [n for n in nodes if n in h]
    N = len(present)
    if N < 5: return None
    H = np.array([h[n] for n in present]); hbar = H.mean(0)
    Gu = G.to_undirected()
    deg = {n: Gu.degree(n) for n in present}
    # sample removal nodes across degree range
    order = sorted(present, key=lambda n: deg[n])
    idx = np.linspace(0, len(order) - 1, min(n_trials, len(order))).astype(int)
    trials = [order[i] for i in sorted(set(idx))]
    Hmax = H.max(0); Hmin = H.min(0)  # extremal pooled slices (over the 64 dims)
    rows = []
    for v in trials:
        surv = [n for n in present if n != v]
        direct = (hbar - h[v]) / (N - 1)
        env.evolving_visible_graph = G.copy(); env.evolving_visible_graph.remove_node(v)
        hp = encode_h(env)
        surv2 = [n for n in surv if n in hp]
        if len(surv2) < 3: continue
        dh = np.array([hp[n] - h[n] for n in surv2]); prop = dh.mean(0)
        try: hops = nx.single_source_shortest_path_length(Gu, v)
        except Exception: hops = {}
        by_hop = collections.defaultdict(list)
        for j, n in enumerate(surv2):
            by_hop[min(hops.get(n, 99), 99)].append(float(np.linalg.norm(dh[j])))
        # STEP 2: extremal slices. v holds a coord-wise max iff h_v == Hmax on >=1 dim.
        v_held_max = bool(np.any(np.isclose(h[v], Hmax))); v_held_min = bool(np.any(np.isclose(h[v], Hmin)))
        Hs = np.array([h[n] for n in surv2]); Hsp = np.array([hp[n] for n in surv2])
        newmax = Hsp.max(0); newmin = Hsp.min(0)
        max_changed = float(np.linalg.norm(newmax - Hmax)); min_changed = float(np.linalg.norm(newmin - Hmin))
        # max change from propagation ALONE: recompute max over SAME (un-re-encoded) survivor embeddings
        max_direct_only = float(np.linalg.norm(Hs.max(0) - Hmax))  # v removed, survivors NOT re-encoded
        rows.append(dict(v=v, N=N, deg=deg[v], hv_norm=float(np.linalg.norm(h[v])),
                         dist_hbar=float(np.linalg.norm(h[v] - hbar)),
                         direct=float(np.linalg.norm(direct)), prop=float(np.linalg.norm(prop)),
                         by_hop={hh: (float(np.mean(vals)), len(vals)) for hh, vals in by_hop.items()},
                         v_held_max=v_held_max, v_held_min=v_held_min,
                         max_changed=max_changed, min_changed=min_changed, max_direct_only=max_direct_only))
    return dict(mode=mode, N=N, mean_deg=float(np.mean(list(deg.values()))), rows=rows)

if __name__ == "__main__":
    import glob
    BANDS = {"10-15": ("scalability_10_15", 12), "30-40": ("scalability_30_40", 25), "80-100": ("scalability_80_100", 69)}
    NREP = 4  # scenarios per band
    results = {}
    for band, (sub, k) in BANDS.items():
        pkls = sorted(glob.glob(f"{REPO}/cyberbattle/data/env_samples/{sub}/*/network_SecureBERT.pkl"))[:NREP]
        for mode in ["sparse", "dense"]:
            allrows = []; Ns = []; degs = []
            for pkl in pkls:
                r = probe(pkl, k, mode, n_trials=30)
                if r: allrows += r["rows"]; Ns.append(r["N"]); degs.append(r["mean_deg"])
            results[(band, mode)] = (allrows, np.mean(Ns) if Ns else 0, np.mean(degs) if degs else 0)

    print("\n### STEP 1.1 direct vs propagation magnitude & ratio (SPARSE = real encoder input) [FINDING] ###")
    for band in BANDS:
        for mode in ["sparse", "dense"]:
            rows, N, dg = results[(band, mode)]
            if not rows: continue
            d = np.array([r["direct"] for r in rows]); p = np.array([r["prop"] for r in rows]); rt = p / np.maximum(d, 1e-12)
            print(f"  {band:7s} {mode:6s}: N~{N:.0f} deg~{dg:.1f} n={len(rows)} | direct med={np.median(d):.4f} | prop med={np.median(p):.4f} p90={np.percentile(p,90):.4f} | prop/direct med={np.median(rt):.3f} mean={rt.mean():.2f}")

    print("\n### STEP 1.2/1.3 slopes vs N (sparse), and prop/direct share vs N [FINDING] ###")
    for term in ["direct", "prop"]:
        xs, ys = [], []
        for band in BANDS:
            rows, N, dg = results[(band, "sparse")]
            if rows: xs.append(N); ys.append(np.median([r[term] for r in rows]))
        sl = np.polyfit(np.log(xs), np.log(ys), 1)[0]
        print(f"  {term:6s} median vs N: values={[round(y,4) for y in ys]} at N={[round(x) for x in xs]} -> log-log slope={sl:+.2f}")
    for band in BANDS:
        rows, N, dg = results[(band, "sparse")]
        if rows:
            rt = np.array([r["prop"] for r in rows]) / np.maximum([r["direct"] for r in rows], 1e-12)
            print(f"  {band:7s}: prop/direct  median={np.median(rt):.3f}  mean={rt.mean():.2f}  frac(prop>direct)={np.mean(rt>1):.2f}")

    print("\n### STEP 3.1 propagation magnitude by HOP distance from removed node (sparse) [FINDING] ###")
    for band in BANDS:
        rows, N, dg = results[(band, "sparse")]
        agg = collections.defaultdict(list)
        for r in rows:
            for hh, (m, n) in r["by_hop"].items(): agg[hh].append(m)
        hops_sorted = sorted(h for h in agg if h < 99)
        s = "  " + band + ": " + "  ".join(f"{hh}hop:mean|Δh_i|={np.mean(agg[hh]):.4f}(n_events={len(agg[hh])})" for hh in hops_sorted[:5])
        ge3 = [m for hh in agg for m in agg[hh] if hh >= 3 and hh < 99]
        s += f"  | >=3hop: max={max(ge3) if ge3 else 0:.2e} (should be 0)"
        print(s)

    print("\n### STEP 3.2/3.3 propagation vs removed node's degree / norm / distance-from-mean (sparse) [FINDING] ###")
    for band in BANDS:
        rows, N, dg = results[(band, "sparse")]
        if len(rows) < 5: continue
        deg = np.array([r["deg"] for r in rows]); pr = np.array([r["prop"] for r in rows])
        nrm = np.array([r["hv_norm"] for r in rows]); dm = np.array([r["dist_hbar"] for r in rows])
        def corr(a, b): return np.corrcoef(a, b)[0,1] if np.std(a)>0 and np.std(b)>0 else float('nan')
        print(f"  {band:7s}: corr(prop, degree)={corr(deg,pr):+.2f}  corr(prop, ||h_v||)={corr(nrm,pr):+.2f}  corr(prop, dist_from_mean)={corr(dm,pr):+.2f}")
        lo = pr[deg <= np.median(deg)]; hi = pr[deg > np.median(deg)]
        print(f"           prop median: low-degree removals={np.median(lo):.4f}  high-degree removals={np.median(hi):.4f}")

    print("\n### STEP 2 extremal slices: how often v held an extreme, and propagation-only changes (sparse) [FINDING] ###")
    for band in BANDS:
        rows, N, dg = results[(band, "sparse")]
        if not rows: continue
        held_max = np.mean([r["v_held_max"] for r in rows])
        no_max_rows = [r for r in rows if not r["v_held_max"]]
        # among removals where v held NO coord max: did the max slice still change (propagation only)?
        prop_only_max = np.mean([r["max_changed"] > 1e-6 for r in no_max_rows]) if no_max_rows else float('nan')
        # decompose max change: direct-only (survivors not re-encoded) vs full
        mc = np.array([r["max_changed"] for r in rows]); mdo = np.array([r["max_direct_only"] for r in rows])
        print(f"  {band:7s}: v_held_max frac={held_max:.2f} | among v-held-NO-max: max-slice-changed(prop only) frac={prop_only_max:.2f} | "
              f"max change: full med={np.median(mc):.4f} direct-only med={np.median(mdo):.4f}")
