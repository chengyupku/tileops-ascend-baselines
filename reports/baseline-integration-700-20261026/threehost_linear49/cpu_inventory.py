import pathlib,os,json,hashlib,platform,subprocess
W=pathlib.Path('/opt/tileops-all-pairs/workspace'); P=W/'reports/baseline-integration-700-20261026/threehost_linear49';P.mkdir(parents=True,exist_ok=True)
roots=[W/'.cache/all-pairs/remote_native_tla/python-deps',pathlib.Path('/opt/tileops-ci/runtime/lib/python3.12/site-packages'),W/'.cache/all-pairs/full_rollout_20261024/remote_ready/server2/group1/recovery_1622/tasks/fla-bf16/runtime/deps',pathlib.Path('/usr/local/Ascend/cann-9.1.0')]
files=[]
for root in roots:
 for d,ds,fs in os.walk(root):
  for f in fs:
   p=pathlib.Path(d)/f;s=str(p)
   if (f=='__init__.py' and p.parent.name=='fla') or f in ['ascend_toolkit_install.info','ascend_all_cann_install.info','aclnn_chunk_gated_delta_rule.h'] or ('chunk_gated_delta_rule' in s and p.suffix in ['.json','.o']):
    files.append({'path':s,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size':p.stat().st_size})
r={'arch':platform.machine(),'files':files,'status':json.loads(subprocess.run(['npu-guard','status','--json'],capture_output=True,text=True).stdout)}
(P/'CPU_INVENTORY.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
