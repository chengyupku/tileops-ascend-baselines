from pathlib import Path
import json,time,subprocess,base64
r=Path.cwd();h=r/'reports/baseline-retest32-20261030/server1';files={};pilots=('02c2823a84e3ff3a4782','98a90f325e8e9657d2ba')
s=(r/'tools/baseline_retest32_server1_prepermute_v44.py').read_text().replace('ops_prepermute/root','ops_normrope/root').replace("package.PROVIDERS={'MoEPrePermuteFwdOp':original.transformer_graph_prepermute}","from benchmarks.ascend_rms_cases import transformer_rms_cases\nfrom benchmarks.ascend_rope_cases import transformer_rope_cases\npackage.PROVIDERS={'RMSNormFwdOp':transformer_rms_cases,'RoPEFwdOp':transformer_rope_cases}").replace("assert case._call.signature.name=='MoEPrePermuteFwdOp'\n  return lambda:original.transformer_graph_prepermute(case)","assert case._call.signature.name in package.PROVIDERS\n  return lambda:package.PROVIDERS[case._call.signature.name](case)");files['tools/baseline_retest32_server1_normrope_v46.py']=s
s=(r/'tools/baseline_retest32_server1_entry_v44.py').read_text().replace("if task['key'][:2]==['ops-transformer','MoEPrePermuteFwdOp']:","if task['key'][0]=='ops-transformer' and task['key'][1] in ('RMSNormFwdOp','RoPEFwdOp'):\n    import baseline_retest32_server1_normrope_v46\nelif task['key'][:2]==['ops-transformer','MoEPrePermuteFwdOp']:");files['tools/baseline_retest32_server1_entry_v46.py']=s
s=(r/'tools/baseline_retest32_server1_python_v44.sh').read_text().replace('entry_v44','entry_v46').replace(' export ASCEND_CUSTOM_OPP_PATH="$OLD1076_TRANSFORMER_VENDOR"',' if [ "$case_op" = RMSNormFwdOp ] || [ "$case_op" = RoPEFwdOp ]; then\n  export OLD1076_TRANSFORMER_VENDOR="$root/reports/baseline-retest32-20261030/server1/deployments/ops_normrope/vendors/normropepairs_transformer"\n fi\n export ASCEND_CUSTOM_OPP_PATH="$OLD1076_TRANSFORMER_VENDOR"');files['tools/baseline_retest32_server1_python_v46.sh']=s
p=r/'tools/run_baseline_retest32_server1.py';s=p.read_text();(h/'runner.before_v46.py').write_text(s);s=s.replace('key=lambda t: (',"key=lambda t: (t['id'] not in "+repr(pilots)+",",1)
for op,ident in zip(('RMSNormFwdOp','RoPEFwdOp'),pilots):
 s=s.replace("or (t['key'][1]=='MoEPrePermuteFwdOp'",f"or (t['key'][1]=='{op}' and (t['id']=='{ident}' or latest.get('{ident}',{{}}).get('status')=='cached')) or (t['key'][1]=='MoEPrePermuteFwdOp'",1)
s=s.replace('if len(chosen)==4 or',"if len(chosen)==4 or t['id'] in "+repr(pilots)+" or",1).replace('python_v44.sh','python_v46.sh').replace("'revision':'gdn_profile_timeout_v45'","'revision':'normrope_original_graph_v46'");files['tools/run_baseline_retest32_server1.py']=s
for p,s in files.items():
 if p.endswith('.py'):compile(s,p,'exec')
 (r/p).write_text(s)
row={'at':time.time(),'revision':'normrope_original_graph_v46','pilots':pilots,'new_condition':'Original normrope ARM host/source package deployed; original RMS/RoPE dispatch unchanged; each scope gated on own full pilot'}
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
code='files='+repr({p:base64.b64encode(s.encode()).decode() for p,s in files.items()})+'\nrow='+repr(row)+'''
import base64,json,time
from pathlib import Path
r=Path('/opt/tileops-baseline-cache/workspace');h=r/'reports/baseline-retest32-20261030/server1'
for p,s in files.items():
 dest=r/p
 if dest.exists():(h/(dest.name+'.before_v46')).write_bytes(dest.read_bytes())
 tmp=dest.with_suffix('.deploy');tmp.write_bytes(base64.b64decode(s));tmp.replace(dest)
(h/'STOP').write_text(json.dumps({'reason':'drain for NormRoPE v46','at':time.time()}))
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\\n')
print('deployed v46; draining')
'''
subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,check=True)
