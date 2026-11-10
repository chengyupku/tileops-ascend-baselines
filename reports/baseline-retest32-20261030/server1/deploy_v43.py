from pathlib import Path
import json,time,base64,subprocess
r=Path.cwd();h=r/'reports/baseline-retest32-20261030/server1';files={};pilot='9c348ca90e14563ea98f'
s=(r/'tools/baseline_retest32_server1_transformer_add_v35.py').read_text().replace("assert attrs=={'drop_pad_mode':0,'k':1}","assert attrs['drop_pad_mode']==0 and attrs['k']>0")
a=s.index("    assert case._call.signature.name=='AddFwdOp'");s=s[:a]+"    assert case._call.signature.name=='MoEPostPermuteFwdOp'\n    return lambda:adapters.postpermute(case)\nregistry.candidate_factory=factory\n";files['tools/baseline_retest32_server1_postpermute_v43.py']=s
s=(r/'tools/baseline_retest32_server1_entry_v42.py').read_text().replace("if task['key'][:2]==['ops-transformer','FusedTopKFwdOp']:","if task['key'][:2]==['ops-transformer','MoEPostPermuteFwdOp']:\n    import baseline_retest32_server1_postpermute_v43\nelif task['key'][:2]==['ops-transformer','FusedTopKFwdOp']:");files['tools/baseline_retest32_server1_entry_v43.py']=s
s=(r/'tools/baseline_retest32_server1_python_v42.sh').read_text().replace('entry_v42','entry_v43').replace('[ "$case_op" = MHCPostFwdOp ];','[ "$case_op" = MHCPostFwdOp ] || [ "$case_op" = MoEPostPermuteFwdOp ];');files['tools/baseline_retest32_server1_python_v43.sh']=s
p=r/'tools/run_baseline_retest32_server1.py';s=p.read_text();(h/'runner.before_v43.py').write_text(s);s=s.replace('key=lambda t: (',"key=lambda t: (t['id']!='"+pilot+"',",1);s=s.replace("or (t['key'][1]=='FusedTopKFwdOp'","or (t['key'][1]=='MoEPostPermuteFwdOp' and (t['id']=='"+pilot+"' or latest.get('"+pilot+"',{}).get('status')=='cached')) or (t['key'][1]=='FusedTopKFwdOp'",1);s=s.replace('if len(chosen)==4 or',"if len(chosen)==4 or t['id']=='"+pilot+"' or",1).replace('python_v42.sh','python_v43.sh').replace("'revision':'fusedtopk_original_graph_v42'","'revision':'postpermute_original_acl_v43'");files['tools/run_baseline_retest32_server1.py']=s
for p,s in files.items():
 if p.endswith('.py'):compile(s,p,'exec')
 (r/p).write_text(s)
row={'at':time.time(),'revision':'postpermute_original_acl_v43','pilot':pilot,'new_condition':'Original postpermute provider via verified originalMoeFinalizeRoutingV2 ACLNN ABI; k derived from weights/indices by originalAPI, no arithmetic changes; remaining8afterpilot'}
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
code='files='+repr({p:base64.b64encode(s.encode()).decode() for p,s in files.items()})+'\nrow='+repr(row)+'''
import base64,json,time
from pathlib import Path
r=Path('/opt/tileops-baseline-cache/workspace');h=r/'reports/baseline-retest32-20261030/server1'
for p,s in files.items():
 dest=r/p
 if dest.exists():(h/(dest.name+'.before_v43')).write_bytes(dest.read_bytes())
 tmp=dest.with_suffix('.deploy');tmp.write_bytes(base64.b64decode(s));tmp.replace(dest)
(h/'STOP').write_text(json.dumps({'reason':'drain for PostPermute v43','at':time.time()}))
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\\n')
print('deployed v43; draining')
'''
subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,check=True)
