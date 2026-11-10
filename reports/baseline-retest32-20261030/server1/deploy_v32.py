import json,time,base64,hashlib,subprocess
from pathlib import Path
r=Path.cwd();h=r/'reports/baseline-retest32-20261030/server1';sdk=r/'.tools/ascend/cann/aarch64-linux/asc'
manifest={str(p.relative_to(sdk)):hashlib.sha256(p.read_bytes()).hexdigest() for top in ['include','impl'] for p in (sdk/top).rglob('*') if p.is_file()}
files={};s=(r/'tools/baseline_retest32_server1_entry_v31.py').read_text().replace("if task['key'][0]=='DeepGEMM-Ascend':","if task['key'][0]=='DeepGEMM-Ascend':\n    os.environ['ASCEND_HOME_PATH']=str(root/'reports/baseline-retest32-20261030/server1/deployments/deepgemm_local_sdk')\n    os.environ['ASCEND_TOOLKIT_HOME']=os.environ['ASCEND_HOME_PATH']",1);files['tools/baseline_retest32_server1_entry_v32.py']=s
files['tools/baseline_retest32_server1_python_v32.sh']=(r/'tools/baseline_retest32_server1_python_v31.sh').read_text().replace('entry_v31.py','entry_v32.py')
p=r/'tools/run_baseline_retest32_server1.py';s=p.read_text();(h/'runner.before_v32.py').write_text(s);s=s.replace("return (x['key'][0]=='DeepGEMM-Ascend'", "return (x['key'][0]=='DeepGEMM-Ascend' and x.get('revision')=='deepgemm_dependencies_v31' and 'asc_load_l2_cache_mode' in str(x.get('error'))) or (x['key'][0]=='DeepGEMM-Ascend'",1).replace("in ('Conv1dFwdOp','BmmFwdOp'","in ('Conv1dFwdOp','Conv3dFwdOp','BmmFwdOp'").replace('python_v31.sh','python_v32.sh').replace("'revision':'deepgemm_dependencies_v31'","'revision':'deepgemm_sdk_v32'");files['tools/run_baseline_retest32_server1.py']=s
for p,s in files.items():
 if p.endswith('.py'):compile(s,p,'exec')
 (r/p).write_text(s)
row={'at':time.time(),'revision':'deepgemm_sdk_v32','new_condition':'v31 original JIT compilation needs local SDK c_api types; deployed exact local successful include+impl; unmodified remote bisheng CPU compilation now succeeds','conv3d':'pilot originalchecker+valid32core audited;release remaining7','sdk_files':len(manifest)}
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
code='files='+repr({p:base64.b64encode(s.encode()).decode() for p,s in files.items()})+'\nmanifest='+repr(manifest)+'\nrow='+repr(row)+'''
from pathlib import Path
import base64,json,hashlib,time
r=Path('/opt/tileops-baseline-cache/workspace');h=r/'reports/baseline-retest32-20261030/server1';sdk=h/'deployments/deepgemm_local_sdk/aarch64-linux/asc'
for p,digest in manifest.items():assert hashlib.sha256((sdk/p).read_bytes()).hexdigest()==digest,p
(h/'DEEPGEMM_SDK_DEPLOYMENT.json').write_text(json.dumps({'at':time.time(),'files':manifest,'cpu_compile_log':str(h/'deepgemm_compile_probe_complete_sdk.log'),'compiler':'unmodified /usr/local/Ascend/cann-9.1.0/bin/bisheng','new_condition':row},indent=2))
for p,s in files.items():
 dest=r/p
 if dest.exists():(h/(dest.name+'.before_v32')).write_bytes(dest.read_bytes())
 tmp=dest.with_suffix('.deploy');tmp.write_bytes(base64.b64decode(s));tmp.replace(dest)
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\\n')
print(json.dumps(row))
'''
p=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,capture_output=True,check=True);print(p.stdout);(h/'DEEPGEMM_SDK_DEPLOYMENT.json').write_text(json.dumps({'manifest':manifest,'deployment':row},indent=2))
