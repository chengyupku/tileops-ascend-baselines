"""CPU-only original native extension build on server1; no torch/device imports."""
import json,subprocess,base64
from pathlib import Path
r=Path.cwd();h=Path(__file__).resolve().parent;old=r/'.agent_memory/THREADS/tileops-progress-report-20261019/CLASSIC_BASELINE_PILOT/PHASE2/transformer_add_compile_audit_180'
a=json.loads((old/'object_attempt.json').read_text());b=json.loads((old/'link_attempt.json').read_text());receipt=json.loads((r/'.cache/ascend-classic/native-pilot-20261020/ops-transformer-add/build_receipt.json').read_text())
code='compile_command='+repr(a['command'])+'\nlink_command='+repr(b['command'])+'\noriginal='+repr(receipt['original_source_sha256'])+r'''
import os,subprocess,time,json,hashlib
from pathlib import Path
root=Path('/opt/tileops-baseline-cache/workspace');h=root/'reports/baseline-retest32-20261030/server1';build=h/'native_build/fp16_add';build.mkdir(parents=True,exist_ok=True)
sdk=h/'deployments/deepgemm_local_sdk';overlay=Path('/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages');py=next(Path('/opt/tileops-ci/python').glob('*/include/python3.12'))
for p,d in original.items():assert hashlib.sha256((root/p).read_bytes()).hexdigest()==d,p
assert __import__('shutil').disk_usage(root).free>54223962112
prefix='/root/dyq/workspace';old=prefix+'/.agent_memory/THREADS/tileops-progress-report-20261019/CLASSIC_BASELINE_PILOT/PHASE2/transformer_add_compile_audit_180'
def mapped(arg):
 return arg.replace(old,str(build)).replace(prefix+'/.tools/ascend/compiler/bin/bisheng','/usr/local/Ascend/cann-9.1.0/bin/bisheng').replace(prefix+'/.venv/lib/python3.12/site-packages',str(overlay)).replace('/usr/include/python3.12',str(py)).replace(prefix+'/.tools/ascend/cann',str(sdk)).replace('/x86_64-linux/', '/aarch64-linux/').replace(prefix,str(root))
env=dict(os.environ,ASCEND_HOME_PATH=str(sdk),ASCEND_TOOLKIT_HOME=str(sdk))
results=[]
for name,cmd in [('object',compile_command),('link',link_command)]:
 args=[mapped(s) for s in cmd];start=time.time();p=subprocess.run(args,env=env,capture_output=True,text=True,timeout=300);(build/(name+'.log')).write_text(p.stdout+p.stderr);row={'at':start,'finished_at':time.time(),'command':args,'exit_code':p.returncode,'cpu_only':True};results.append(row);(build/(name+'.json')).write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
 if p.returncode:print((p.stdout+p.stderr)[-4000:],flush=True);break
if results[-1]['exit_code']==0 and len(results)==2:
 import shutil
 dest=root/'.cache/ascend-classic/native-pilot-20261020/ops-transformer-add';dest.mkdir(parents=True,exist_ok=True);binary=dest/'libclassic_transformer_add_original.so';assert not binary.exists();shutil.copy2(build/binary.name,binary)
 (dest/'build_receipt.json').write_text(json.dumps({'at':time.time(),'original_source_sha256':original,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'native_source_modified':False,'device_validated':False,'build_evidence':results},indent=2))
print(json.dumps({'complete':len(results)==2 and results[-1]['exit_code']==0,'build':str(build)}))
'''
p=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','nice -n 10 python3 -'],input=code,text=True,capture_output=True)
(h/'FP16_ADD_BUILD.log').write_text(p.stdout+p.stderr);print(p.stdout[-5000:]);print(p.stderr[-1000:]);raise SystemExit(p.returncode)
