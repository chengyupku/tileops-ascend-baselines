from pathlib import Path
import json,time,base64,subprocess
r=Path.cwd();h=r/'reports/baseline-retest32-20261030/server1';files={}
s=(r/'tools/baseline_retest32_server1_transformer_mhc_v40.py').read_text();s=s.replace("graph.WORKSPACE=ROOT/'reports/baseline-retest32-20261030/server1/deployments/ops_graph/root'","graph.WORKSPACE=Path(os.environ['SERVER1_SOFTMAX_GRAPH_ROOT'])")
a=s.index('  assert op==');b=s.index('  self.binding.call(op,inputs,outputs,attrs)',a);s=s[:a]+"  assert op in ('MoeGatingTopKSoftmax','MoeGatingTopKSoftmaxV2')\n"+s[b:]
s=s.replace('import adapters,native','import adapters,native,gating_normalization').replace('Path(native.__file__),','Path(native.__file__),Path(gating_normalization.__file__),').replace("package.PROVIDERS={'MHCPostFwdOp':adapters.mhc_post}","package.PROVIDERS={'SoftmaxFwdOp':gating_normalization.softmax_dispatch}").replace('lambda: adapters.mhc_post(case)','lambda: gating_normalization.softmax_dispatch(case)').replace('via public ACLNN','via original GraphCall').replace('Original public ACLNN binding for ARM host.','Original graph API binding for ARM host.')
files['tools/baseline_retest32_server1_softmax_v41.py']=s
s=(r/'tools/baseline_retest32_server1_entry_v40.py').read_text().replace("if task['key'][:2]==['ops-transformer','MHCPostFwdOp']:","if task['key'][:2]==['ops-transformer','SoftmaxFwdOp']:\n    import baseline_retest32_server1_softmax_v41\nelif task['key'][:2]==['ops-transformer','MHCPostFwdOp']:");files['tools/baseline_retest32_server1_entry_v41.py']=s
s=(r/'tools/baseline_retest32_server1_python_v40.sh').read_text().replace('entry_v40','entry_v41');needle=' export ASCEND_CUSTOM_OPP_PATH="$OLD1076_TRANSFORMER_VENDOR"';s=s.replace(needle,''' if [ "$case_op" = SoftmaxFwdOp ]; then
  width=$(python3 -c 'import json,sys;a=sys.argv[1:];p=json.load(open(a[a.index("--task")+1]))["case"]["params"];print(p["x"][0][p["dim"]])' "${@:2}")
  export SERVER1_SOFTMAX_GRAPH_ROOT="$root/reports/baseline-retest32-20261030/server1/deployments/ops_graph/root"
  export OLD1076_TRANSFORMER_VENDOR="$root/reports/baseline-retest32-20261030/server1/deployments/ops_graph/vendors/allpairs_transformer"
  if [ "$width" -gt 1024 ]; then
   export SERVER1_SOFTMAX_GRAPH_ROOT="$root/reports/baseline-retest32-20261030/server1/deployments/ops_selection/root"
   export OLD1076_TRANSFORMER_VENDOR="$root/reports/baseline-retest32-20261030/server1/deployments/ops_selection/vendors/selectionpairs_transformer"
  fi
 fi
'''+needle);files['tools/baseline_retest32_server1_python_v41.sh']=s
p=r/'tools/run_baseline_retest32_server1.py';s=p.read_text();(h/'runner.before_v41.py').write_text(s);s=s.replace('key=lambda t: (',"key=lambda t: (t['id'] not in ('28287f4939a34659c4c7','2c09433f0f9006dcd105'),",1);s=s.replace("or (t['key'][1]=='MHCPostFwdOp'","or (t['key'][1]=='SoftmaxFwdOp' and (t['id'] in ('28287f4939a34659c4c7','2c09433f0f9006dcd105') or all(latest.get(i,{}).get('status')=='cached' for i in ('28287f4939a34659c4c7','2c09433f0f9006dcd105')))) or (t['key'][1]=='MHCPostFwdOp'",1);s=s.replace('if len(chosen)==4 or',"if len(chosen)==4 or t['id'] in ('28287f4939a34659c4c7','2c09433f0f9006dcd105') or",1).replace('python_v40.sh','python_v41.sh').replace("'revision':'mhc_original_graph_v40'","'revision':'softmax_original_graph_v41'");files['tools/run_baseline_retest32_server1.py']=s
for p,s in files.items():
 if p.endswith('.py'):compile(s,p,'exec')
 (r/p).write_text(s)
row={'at':time.time(),'revision':'softmax_original_graph_v41','pilots':['28287f4939a34659c4c7','2c09433f0f9006dcd105'],'new_condition':'ARM selection package built verified; use original softmax_dispatch full-outputV2 for longaxis and original topk permutation shortaxis via original GraphCall; no arithmetic/kernel edits'}
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
code='files='+repr({p:base64.b64encode(s.encode()).decode() for p,s in files.items()})+'\nrow='+repr(row)+'''
import base64,json,time
from pathlib import Path
r=Path('/opt/tileops-baseline-cache/workspace');h=r/'reports/baseline-retest32-20261030/server1';base=h/'deployments/ops_selection';link=base/'root/.cache/ascend-classic/native-pilot-20261020/ops-transformer/install/vendors/classicpilot_transformer';link.parent.mkdir(parents=True,exist_ok=True)
if not link.exists():link.symlink_to(base/'vendors/selectionpairs_transformer',target_is_directory=True)
for p,s in files.items():
 dest=r/p
 if dest.exists():(h/(dest.name+'.before_v41')).write_bytes(dest.read_bytes())
 tmp=dest.with_suffix('.deploy');tmp.write_bytes(base64.b64decode(s));tmp.replace(dest)
(h/'STOP').write_text(json.dumps({'reason':'drain for Softmax v41','at':time.time()}))
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\\n')
print('deployed v41; draining')
'''
subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,check=True)
