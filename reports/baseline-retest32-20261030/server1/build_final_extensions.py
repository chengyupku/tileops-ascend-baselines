from pathlib import Path
import subprocess,json,base64,hashlib
h=Path(__file__).resolve().parent;r=Path.cwd();old=r/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_transformer'
receipts={k:json.loads((old/(k+'_build.json')).read_text()) for k in ('state_passing','gelu')}
files={}
for d in ['experimental/mamba/mamba2_chunk_state_passing','experimental/mamba/common','experimental/moe/biasgategelu']:
 for p in (r/'ascend_baselines/ops-transformer'/d).rglob('*'):
  if p.is_file():files[str(p.relative_to(r))]=base64.b64encode(p.read_bytes()).decode()
code='receipts='+repr(receipts)+'\nfiles='+repr(files)+r'''
from pathlib import Path
import subprocess,json,hashlib,base64,os,time,shutil
root=Path('/opt/tileops-baseline-cache/workspace');h=root/'reports/baseline-retest32-20261030/server1'
assert shutil.disk_usage(root).free>50.5*2**30
for name,data in files.items():
 p=root/name;b=base64.b64decode(data)
 if p.exists():assert p.read_bytes()==b,name
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
sdk=h/'deployments/deepgemm_local_sdk';overlay=Path('/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages');py=next(Path('/opt/tileops-ci/python').glob('*/include/python3.12'))
prefix='/root/dyq/workspace';results={}
for key,receipt in receipts.items():
 build=h/'native_build'/('final_'+key);build.mkdir(exist_ok=True,parents=True)
 def mapped(arg):
  return arg.replace(prefix+'/.cache/all-pairs/ops_transformer/'+key+'_build',str(build)).replace(prefix+'/.tools/ascend/compiler/bin/bisheng','/usr/local/Ascend/cann-9.1.0/bin/bisheng').replace(prefix+'/.venv/lib/python3.12/site-packages',str(overlay)).replace('/usr/include/python3.12',str(py)).replace(prefix+'/.tools/ascend/cann',str(sdk)).replace('/x86_64-linux/', '/aarch64-linux/').replace(prefix,str(root))
 rows=[]
 for step in ('compile','link'):
  assert shutil.disk_usage(root).free>50.5*2**30
  cmd=[mapped(s) for s in receipt[step]];start=time.time();p=subprocess.run(cmd,env=dict(os.environ,ASCEND_HOME_PATH=str(sdk),ASCEND_TOOLKIT_HOME=str(sdk)),capture_output=True,text=True,timeout=300)
  (build/(step+'.log')).write_text(p.stdout+p.stderr);rows.append(dict(step=step,command=cmd,at=start,returncode=p.returncode,cpu_only=True));print(key,step,p.returncode,(p.stdout+p.stderr)[-1800:],flush=True)
  if p.returncode:break
 results[key]=rows
 if len(rows)==2 and rows[-1]['returncode']==0:
  name=Path(receipt['link'][3]).name;binary=build/name;data=binary.read_bytes();assert data[:4]==b'\x7fELF' and int.from_bytes(data[18:20],'little')==183
  dst=root/'.cache/all-pairs/ops_transformer'/(key+'_build')/name;dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists();shutil.copy2(binary,dst)
  out=dict(receipt,binary_sha256=hashlib.sha256(data).hexdigest(),compile=rows[0]['command'],link=rows[1]['command'],arm_build_evidence=rows)
  rp=root/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_transformer'/(key+'_build.json');rp.parent.mkdir(exist_ok=True,parents=True);assert not rp.exists();rp.write_text(json.dumps(out,indent=2))
(h/'FINAL_EXTENSIONS_ARM_BUILD.json').write_text(json.dumps(results,indent=2))
'''
p=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','nice -n 10 python3 -'],input=code,text=True,capture_output=True);(h/'FINAL_EXTENSIONS_BUILD.log').write_text(p.stdout+p.stderr);print(p.stdout,p.stderr);raise SystemExit(p.returncode)
