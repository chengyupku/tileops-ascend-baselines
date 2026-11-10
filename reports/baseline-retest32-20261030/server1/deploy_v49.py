from pathlib import Path
import json,time,subprocess,base64
r=Path.cwd();h=r/'reports/baseline-retest32-20261030/server1';files={};pilots=('7654e938681555f89daf','b09d4fc7b995919e37c3')
s=(r/'tools/baseline_retest32_server1_entry_v47.py').read_text().replace("if task['key'][0]=='ops-transformer' and task['key'][1] in ('RMSNormFwdOp','RoPEFwdOp'):","if task['key'][:2]==['ops-transformer','InstanceNormFwdOp']:\n    import baseline_retest32_server1_instancenorm_v49\nelif task['key'][0]=='ops-transformer' and task['key'][1] in ('RMSNormFwdOp','RoPEFwdOp'):");files['tools/baseline_retest32_server1_entry_v49.py']=s
s=(r/'tools/baseline_retest32_server1_python_v47.sh').read_text().replace('entry_v47','entry_v49').replace('[ "$case_op" = RMSNormFwdOp ] ||','[ "$case_op" = InstanceNormFwdOp ] || [ "$case_op" = RMSNormFwdOp ] ||');files['tools/baseline_retest32_server1_python_v49.sh']=s
p=r/'tools/run_baseline_retest32_server1.py';s=p.read_text();(h/'runner.before_v49.py').write_text(s);s=s.replace('key=lambda t: (',"key=lambda t: (t['id'] not in "+repr(pilots)+",",1);s=s.replace("or (t['key'][1]=='MoEPrePermuteFwdOp'","or (t['key'][1]=='InstanceNormFwdOp' and (t['id'] in "+repr(pilots)+" or all(latest.get(i,{}).get('status')=='cached' for i in "+repr(pilots)+"))) or (t['key'][1]=='MoEPrePermuteFwdOp'",1);s=s.replace('if len(chosen)==4 or',"if len(chosen)==4 or t['id'] in "+repr(pilots)+" or",1).replace('python_v47.sh','python_v49.sh').replace("'revision':'gdn_guard_teardown_timeout_v48'","'revision':'instancenorm_original_graph_v49'");files['tools/run_baseline_retest32_server1.py']=s
for p,s in files.items():
 if p.endswith('.py'):compile(s,p,'exec')
 (r/p).write_text(s)
row={'at':time.time(),'revision':'instancenorm_original_graph_v49','pilots':pilots,'new_condition':'Original channel_norm with already verified NormRopeConcat ARM package; affine and non-affine full pilots gate remaining7'}
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
code='files='+repr({p:base64.b64encode(s.encode()).decode() for p,s in files.items()})+'\nrow='+repr(row)+'''
import base64,json,time
from pathlib import Path
r=Path('/opt/tileops-baseline-cache/workspace');h=r/'reports/baseline-retest32-20261030/server1'
for p,s in files.items():
 dest=r/p
 if dest.exists():(h/(dest.name+'.before_v49')).write_bytes(dest.read_bytes())
 tmp=dest.with_suffix('.deploy');tmp.write_bytes(base64.b64decode(s));tmp.replace(dest)
(h/'STOP').write_text(json.dumps({'reason':'drain for InstanceNorm v49','at':time.time()}))
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\\n')
print('deployed v49; draining')
'''
subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,check=True)
