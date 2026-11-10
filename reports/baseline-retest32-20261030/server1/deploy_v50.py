from pathlib import Path
import json,time,subprocess,base64
r=Path.cwd();h=r/'reports/baseline-retest32-20261030/server1';files={};pilots={'AmaxFwdOp':'353136a712a0c95b3497','ReluFwdOp':'69ba7307d680f7c905a1','ClampScalarFwdOp':'ab34786d300bb0408433'};ids=tuple(pilots.values())
s=(r/'tools/baseline_retest32_server1_entry_v49.py').read_text().replace("if task['key'][:2]==['ops-transformer','InstanceNormFwdOp']:","if task['key'][0]=='ops-transformer' and task['key'][1] in ('AmaxFwdOp','ReluFwdOp','ClampScalarFwdOp'):\n    import baseline_retest32_server1_extrema_v50\nelif task['key'][:2]==['ops-transformer','InstanceNormFwdOp']:");files['tools/baseline_retest32_server1_entry_v50.py']=s
s=(r/'tools/baseline_retest32_server1_python_v49.sh').read_text().replace('entry_v49','entry_v50').replace(' export ASCEND_CUSTOM_OPP_PATH="$OLD1076_TRANSFORMER_VENDOR"',' if [ "$case_op" = AmaxFwdOp ] || [ "$case_op" = ReluFwdOp ] || [ "$case_op" = ClampScalarFwdOp ]; then\n  export OLD1076_TRANSFORMER_VENDOR="$root/reports/baseline-retest32-20261030/server1/deployments/ops_attention/vendors/attentionpairs_transformer"\n fi\n export ASCEND_CUSTOM_OPP_PATH="$OLD1076_TRANSFORMER_VENDOR"');files['tools/baseline_retest32_server1_python_v50.sh']=s
p=r/'tools/run_baseline_retest32_server1.py';s=p.read_text();(h/'runner.before_v50.py').write_text(s);s=s.replace('key=lambda t: (',"key=lambda t: (t['id'] not in "+repr(ids)+",",1)
for op,i in pilots.items():s=s.replace("or (t['key'][1]=='MoEPrePermuteFwdOp'",f"or (t['key'][1]=='{op}' and (t['id']=='{i}' or latest.get('{i}',{{}}).get('status')=='cached')) or (t['key'][1]=='MoEPrePermuteFwdOp'",1)
s=s.replace('if len(chosen)==4 or',"if len(chosen)==4 or t['id'] in "+repr(ids)+" or",1).replace('python_v49.sh','python_v50.sh').replace("'revision':'instancenorm_original_graph_v49'","'revision':'extrema_original_graph_v50'");files['tools/run_baseline_retest32_server1.py']=s
for p,s in files.items():
 if p.endswith('.py'):compile(s,p,'exec')
 (r/p).write_text(s)
row={'at':time.time(),'revision':'extrema_original_graph_v50','pilots':pilots,'new_condition':'Original attentionpairs ARM environment and original extrema/pointwise dispatch; three scope pilots gate remaining5'}
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
code='files='+repr({p:base64.b64encode(s.encode()).decode() for p,s in files.items()})+'\nrow='+repr(row)+'''
import base64,json,time
from pathlib import Path
r=Path('/opt/tileops-baseline-cache/workspace');h=r/'reports/baseline-retest32-20261030/server1'
for p,s in files.items():
 dest=r/p
 if dest.exists():(h/(dest.name+'.before_v50')).write_bytes(dest.read_bytes())
 tmp=dest.with_suffix('.deploy');tmp.write_bytes(base64.b64decode(s));tmp.replace(dest)
(h/'STOP').write_text(json.dumps({'reason':'drain for Extrema v50','at':time.time()}))
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\\n')
print('deployed v50; draining')
'''
subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,check=True)
