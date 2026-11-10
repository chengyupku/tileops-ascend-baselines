from pathlib import Path
import json,time,base64,subprocess
r=Path.cwd();h=r/'reports/baseline-retest32-20261030/server1';files={};pilots=('54e8ae423c0618c74fc8','b186fb5881845f1725d1')
s=(r/'tools/baseline_retest32_server1_softmax_v41.py').read_text().replace('SERVER1_SOFTMAX_GRAPH_ROOT','SERVER1_FUSED_GRAPH_ROOT').replace("('MoeGatingTopKSoftmax','MoeGatingTopKSoftmaxV2')","('MoeGatingTopKSoftmax','MoeGatingTopK','MoeFusedTopk')").replace('import adapters,native,gating_normalization','import adapters,native,gating_normalization,routing').replace('Path(gating_normalization.__file__),','Path(gating_normalization.__file__),Path(routing.__file__),').replace("{'SoftmaxFwdOp':gating_normalization.softmax_dispatch}","{'FusedTopKFwdOp':gating_normalization.fused_topk_dispatch}").replace('lambda: gating_normalization.softmax_dispatch(case)','lambda: gating_normalization.fused_topk_dispatch(case)');files['tools/baseline_retest32_server1_fusedtopk_v42.py']=s
s=(r/'tools/baseline_retest32_server1_entry_v41.py').read_text().replace("if task['key'][:2]==['ops-transformer','SoftmaxFwdOp']:","if task['key'][:2]==['ops-transformer','FusedTopKFwdOp']:\n    import baseline_retest32_server1_fusedtopk_v42\nelif task['key'][:2]==['ops-transformer','SoftmaxFwdOp']:");files['tools/baseline_retest32_server1_entry_v42.py']=s
s=(r/'tools/baseline_retest32_server1_python_v41.sh').read_text().replace('entry_v41','entry_v42');needle=' export ASCEND_CUSTOM_OPP_PATH="$OLD1076_TRANSFORMER_VENDOR"';s=s.replace(needle,''' if [ "$case_op" = FusedTopKFwdOp ]; then
  selection=$(python3 -c 'import json,sys;a=sys.argv[1:];p=json.load(open(a[a.index("--task")+1]))["case"]["params"];print(int(not ((p["scoring_func"]=="softmax" and p["num_expert_group"]==1 and p["topk_group"]==1 and p["routed_scaling_factor"]==1.) or (p["scoring_func"]=="sigmoid" and not p["renormalize"]))))' "${@:2}")
  export SERVER1_FUSED_GRAPH_ROOT="$root/reports/baseline-retest32-20261030/server1/deployments/ops_graph/root"
  export OLD1076_TRANSFORMER_VENDOR="$root/reports/baseline-retest32-20261030/server1/deployments/ops_graph/vendors/allpairs_transformer"
  if [ "$selection" = 1 ]; then
   export SERVER1_FUSED_GRAPH_ROOT="$root/reports/baseline-retest32-20261030/server1/deployments/ops_selection/root"
   export OLD1076_TRANSFORMER_VENDOR="$root/reports/baseline-retest32-20261030/server1/deployments/ops_selection/vendors/selectionpairs_transformer"
  fi
 fi
'''+needle);files['tools/baseline_retest32_server1_python_v42.sh']=s
p=r/'tools/run_baseline_retest32_server1.py';s=p.read_text();(h/'runner.before_v42.py').write_text(s);s=s.replace('key=lambda t: (',"key=lambda t: (t['id'] not in "+repr(pilots)+",",1);s=s.replace("or (t['key'][1]=='SoftmaxFwdOp'","or (t['key'][1]=='FusedTopKFwdOp' and (t['id'] in "+repr(pilots)+" or all(latest.get(i,{}).get('status')=='cached' for i in "+repr(pilots)+"))) or (t['key'][1]=='SoftmaxFwdOp'",1);s=s.replace('if len(chosen)==4 or',"if len(chosen)==4 or t['id'] in "+repr(pilots)+" or",1).replace('python_v41.sh','python_v42.sh').replace("'revision':'softmax_original_graph_v41'","'revision':'fusedtopk_original_graph_v42'");files['tools/run_baseline_retest32_server1.py']=s
for p,s in files.items():
 if p.endswith('.py'):compile(s,p,'exec')
 (r/p).write_text(s)
row={'at':time.time(),'revision':'fusedtopk_original_graph_v42','pilots':list(pilots),'new_condition':'Reuse original fused_topk_dispatch/routing via original GraphCall and existing verified ARMvendors; two original API paths tested before releasing other15'}
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
code='files='+repr({p:base64.b64encode(s.encode()).decode() for p,s in files.items()})+'\nrow='+repr(row)+'''
import base64,json,time
from pathlib import Path
r=Path('/opt/tileops-baseline-cache/workspace');h=r/'reports/baseline-retest32-20261030/server1'
for p,s in files.items():
 dest=r/p
 if dest.exists():(h/(dest.name+'.before_v42')).write_bytes(dest.read_bytes())
 tmp=dest.with_suffix('.deploy');tmp.write_bytes(base64.b64decode(s));tmp.replace(dest)
(h/'STOP').write_text(json.dumps({'reason':'drain for FusedTopK v42','at':time.time()}))
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\\n')
print('deployed v42; draining')
'''
subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,check=True)
