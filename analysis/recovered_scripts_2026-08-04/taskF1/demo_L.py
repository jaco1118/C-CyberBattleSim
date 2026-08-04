"""Task L STEP 1 acceptance demo (Amendment 1): reproduce a change event's EXACT policy input offline
from the side log alone. VecNormalize sits OUTSIDE the env; the logged obs is PRE-normalisation; the
vecnorm stats are saved with the checkpoint. So policy_input = vecn.normalize_obs(logged_raw_obs)."""
import sys, os, json, logging, pickle, glob
sys.path.insert(0, "/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim")
os.chdir("/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents")
import numpy as np, torch, yaml
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import TRPO
from cyberbattle.utils.envs_utils import wrap_graphs_to_compressed_envs
from cyberbattle.gae.model import GAEEncoder
from cyberbattle._env.cyberbattle_env_switch import RandomSwitchEnv
torch.set_num_threads(2)
REPO="/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim"
RUN=f"/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1/runs/trpo_250k_F1_static_seed42"
OUTDIR="/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1/L_demo"
os.system(f"rm -rf {OUTDIR}"); os.makedirs(OUTDIR, exist_ok=True)
SEED=42; TOPO="scalability_30_40/44"
cfg=yaml.safe_load(open(os.path.join(RUN,"train_config.yaml")))
ce=yaml.safe_load(open(cfg["graph_encoder_config_path"])); ce.update(yaml.safe_load(open(cfg["graph_encoder_spec_path"])))
ge=GAEEncoder(ce["node_feature_vector_size"],ce["model_config"]["layers"],ce["edge_feature_vector_size"])
ge.load_state_dict(torch.load(cfg["graph_encoder_path"])); ge.eval()
cfg["node_embeddings_dimensions"]=ce["model_config"]["layers"][-1]["out_channels"]
lg=logging.getLogger("L"); lg.addHandler(logging.NullHandler())
c=dict(cfg); c["dynamic_mode"]="both"; c["patch_service_dynamic_enabled"]=False; c["change_interval"]=20
c["drift_logging"]=True; c["drift_log_path"]=os.path.join(OUTDIR,"drift.csv"); c["drift_sample_rate"]=1
c["drift_run_id"]="Ldemo"; c["drift_seed"]=SEED; c["drift_scenario_id"]="scalability_30_40_44"
c["event_graph_logging"]=True; c["event_graph_log_dir"]=os.path.join(OUTDIR,"eventgraph")
net=pickle.load(open(os.path.join(REPO,"cyberbattle/data/env_samples",TOPO,"network_SecureBERT.pkl"),"rb"))
env=wrap_graphs_to_compressed_envs(net,lg,**c); env.set_graph_encoder(ge); env.set_pca_components(c["pca_components"])
# donor pool for join
DONOR=os.path.join(REPO,"cyberbattle/data/env_samples/join_donor_pool_20_topologies")
import copy as _c
pool=[]
for sub in sorted(os.listdir(DONOR)):
    sp=os.path.join(DONOR,sub)
    if os.path.isdir(sp) and sub.isdigit():
        dn=pickle.load(open(os.path.join(sp,f"network_{c['nlp_extractor']}.pkl"),"rb"))
        pool+=[(sub,nid,_c.deepcopy(nd["data"])) for nid,nd in dn.network.nodes(data=True)]
env.dynamic_join_donor_pool=pool
switch=RandomSwitchEnv([0],switch_interval=10**9,envs_list=[env],save_to_csv=False,verbose=0)
tmp=DummyVecEnv([lambda: Monitor(RandomSwitchEnv([0],switch_interval=10**9,envs_list=[env],save_to_csv=False,verbose=0))])
vecn=VecNormalize.load(os.path.join(RUN,"checkpoints/1/checkpoint_vecnormalize_250000_steps.pkl"),tmp)
vecn.training=False; vecn.norm_reward=False
mdl=TRPO.load(os.path.join(RUN,"checkpoints/1/checkpoint_250000_steps.zip"),device="cpu")
np.random.seed(SEED); torch.manual_seed(SEED)
import random as _r; _r.seed(SEED)
def norm(o):
    b={k:np.asarray(v,np.float32)[None,...] for k,v in o.items()}
    n=vecn.normalize_obs(b); return {k:v[0] for k,v in n.items()}
# capture, per (episode,step) AFTER each env step, the raw obs the env returned and its normalised form
captured={}
obs,_=switch.reset(); ep=0
while ep<3:
    a,_=mdl.predict(norm(obs),deterministic=False)
    obs,r,done,trunc,info=switch.step(a)
    cur=switch.current_env
    key=(int(cur._episode_count),int(cur.stepcount))
    normed=norm(obs)
    captured[key]=(np.asarray(obs["graph_embeddings"],np.float32).copy(),
                   normed["graph_embeddings"].copy(), normed["discrete_features"].copy())
    if done or trunc: obs,_=switch.reset(); ep+=1
env._event_graph_logger  # exists
# --- offline reproduction from the side log ALONE ---
egdir=os.path.join(OUTDIR,"eventgraph")
recs=[json.loads(l) for l in open(os.path.join(egdir,"event_graph.jsonl"))]
f32=np.fromfile(os.path.join(egdir,"event_obs.f32"),dtype=np.float32)
def readvec(off,n): return f32[off//4:off//4+n]
print(f"side log: {len(recs)} change-steps logged; obs store {f32.nbytes} bytes")
# pick the first change-step that we also captured in the eval loop
checked=0; maxdiff=0.0
for rec in recs:
    key=(rec["episode"],rec["step"])
    if key not in captured: continue
    pg_off,pg_n=rec["obs"]["post_graph"]; pd_off,pd_n=rec["obs"]["post_discrete"]
    log_graph=readvec(pg_off,pg_n); log_discrete=readvec(pd_off,pd_n)
    # reproduce policy input offline: normalise the logged RAW obs with the saved vecnorm stats
    repro=norm({"graph_embeddings":log_graph,"discrete_features":log_discrete})
    raw_eval, normed_eval_g, normed_eval_d = captured[key]
    d_rawobs=float(np.max(np.abs(log_graph-raw_eval)))       # logged raw obs vs env-returned raw obs
    d_policy=float(np.max(np.abs(repro["graph_embeddings"]-normed_eval_g)))  # reproduced policy input vs actual
    d_disc=float(np.max(np.abs(repro["discrete_features"]-normed_eval_d)))
    maxdiff=max(maxdiff,d_policy,d_disc)
    if checked<3:
        print(f"  event-step {key} type={[e['change_type'] for e in rec['events']]}: "
              f"max|logged_raw - env_raw|={d_rawobs:.3e}  max|repro_policy_input - actual|={max(d_policy,d_disc):.3e}")
    checked+=1
print(f"\nACCEPTANCE (Amendment 1): change-steps cross-checked={checked}; "
      f"MAX |reproduced policy input - value eval used| = {maxdiff:.3e}  -> {'PASS (==0)' if maxdiff==0.0 else 'FAIL'}")
# quick schema dump
r0=recs[0]
print(f"\nside-log record keys: {list(r0.keys())}")
print(f"  events[0] keys: {list(r0['events'][0].keys())}  sample: {r0['events'][0]}")
print(f"  pre_edges n={len(r0['pre_edges'])} post_edges n={len(r0['post_edges'])} "
      f"action_keys pre_count={r0['action_keys_pre_count']} post_count={r0['action_keys_post_count']} "
      f"added={len(r0['action_keys_added'])} removed={len(r0['action_keys_removed'])}")
