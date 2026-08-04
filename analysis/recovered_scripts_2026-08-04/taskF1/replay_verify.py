"""Task L DECISION 1 replay verification. For one FULL evaluation episode with the POLICY IN THE LOOP:
(a) run it twice identically -> drift CSVs must be byte-identical (determinism-based replay).
(b) replay by FEEDING the recorded raw action sequence instead of calling the policy -> must also match
    run 1 (replay independent of policy determinism). Eval uses model.predict(..., deterministic=False)
    (stochastic) -- quoted. PYTHONHASHSEED pinned by the caller."""
import sys, os, logging, pickle, yaml, random
sys.path.insert(0,"/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim")
os.chdir("/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim/cyberbattle/agents")
import numpy as np, torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import TRPO
from cyberbattle.utils.envs_utils import wrap_graphs_to_compressed_envs
from cyberbattle.gae.model import GAEEncoder
from cyberbattle._env.cyberbattle_env_switch import RandomSwitchEnv
BAND=sys.argv[1]; MODE=sys.argv[2]; OUT=os.path.abspath(sys.argv[3]); ACTFILE=sys.argv[4] if len(sys.argv)>4 else None
os.makedirs(OUT,exist_ok=True); torch.set_num_threads(1)
REPO="/cs/student/project_msc/2025/sec/slchan/C-CyberBattleSim"; SEED=42
if BAND=="30-40": RUN=f"/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1/runs/trpo_250k_F1_static_seed42"; TOPO="scalability_30_40/44"
else: RUN=f"/cs/student/project_msc/2025/sec/slchan/claude_home/.claude/jobs/0dfa230d/tmp/taskF1/f2_runs/trpo_250k_F2_static_band80-100_seed42"; TOPO="scalability_80_100/5"
cfg=yaml.safe_load(open(os.path.join(RUN,"train_config.yaml")))
ce=yaml.safe_load(open(cfg["graph_encoder_config_path"])); ce.update(yaml.safe_load(open(cfg["graph_encoder_spec_path"])))
ge=GAEEncoder(ce["node_feature_vector_size"],ce["model_config"]["layers"],ce["edge_feature_vector_size"]); ge.load_state_dict(torch.load(cfg["graph_encoder_path"])); ge.eval()
cfg["node_embeddings_dimensions"]=ce["model_config"]["layers"][-1]["out_channels"]
lg=logging.getLogger("rv"); lg.addHandler(logging.NullHandler())
c=dict(cfg); c["dynamic_mode"]="both"; c["patch_service_dynamic_enabled"]=False; c["change_interval"]=20
c["drift_logging"]=True; c["drift_log_path"]=os.path.join(OUT,"drift.csv"); c["drift_sample_rate"]=1
c["drift_run_id"]="rv"; c["drift_seed"]=SEED; c["drift_scenario_id"]=BAND
net=pickle.load(open(os.path.join(REPO,"cyberbattle/data/env_samples",TOPO,"network_SecureBERT.pkl"),"rb"))
env=wrap_graphs_to_compressed_envs(net,lg,**c); env.set_graph_encoder(ge); env.set_pca_components(c["pca_components"])
import copy as _cp; DONOR=os.path.join(REPO,"cyberbattle/data/env_samples/join_donor_pool_20_topologies"); pool=[]
for sub in sorted(os.listdir(DONOR)):
    sp=os.path.join(DONOR,sub)
    if os.path.isdir(sp) and sub.isdigit():
        dn=pickle.load(open(os.path.join(sp,f"network_{c['nlp_extractor']}.pkl"),"rb")); pool+=[(sub,nid,_cp.deepcopy(nd["data"])) for nid,nd in dn.network.nodes(data=True)]
env.dynamic_join_donor_pool=pool
sw=RandomSwitchEnv([0],switch_interval=10**9,envs_list=[env],save_to_csv=False,verbose=0)
tmp=DummyVecEnv([lambda: Monitor(RandomSwitchEnv([0],switch_interval=10**9,envs_list=[env],save_to_csv=False,verbose=0))])
vecn=VecNormalize.load(os.path.join(RUN,"checkpoints/1/checkpoint_vecnormalize_250000_steps.pkl"),tmp); vecn.training=False; vecn.norm_reward=False
mdl=TRPO.load(os.path.join(RUN,"checkpoints/1/checkpoint_250000_steps.zip"),device="cpu")
np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)
def norm(o):
    b={k:np.asarray(v,np.float32)[None,...] for k,v in o.items()}; n=vecn.normalize_obs(b); return {k:v[0] for k,v in n.items()}
replay_actions = np.load(ACTFILE) if (MODE=="replay" and ACTFILE) else None
obs,_=sw.reset(); acts=[]; step=0
while True:
    if MODE=="replay": a=replay_actions[step]
    else: a,_=mdl.predict(norm(obs),deterministic=False)   # <-- STOCHASTIC action selection (quoted)
    acts.append(np.asarray(a,np.float32))
    obs,r,d,t,i=sw.step(a); step+=1
    if d or t: break
env._drift_logger.close()
np.save(os.path.join(OUT,"actions.npy"), np.array(acts))
print(f"[{BAND} {MODE}] episode length={step} steps; wrote drift.csv + actions.npy")
