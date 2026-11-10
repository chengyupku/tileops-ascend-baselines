from pathlib import Path
import json,subprocess,time,sys,os
P=Path(__file__).resolve().parent
op,cid=sys.argv[1:];out=P/'local_runs'/(cid+'-original-default');out.mkdir(parents=True,exist_ok=True)
env=os.environ.copy()
for n in ['TMPDIR','XDG_CACHE_HOME','ASCEND_WORK_PATH','ASCEND_CACHE_PATH','ASCEND_PROCESS_LOG_PATH','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR']:
 d=P/'c'/n;d.mkdir(parents=True,exist_ok=True);env[n]=str(d)
cmd=['npu-guard','run','--owner','baseline700-20261026/surge_tla_attention97/worker01','--wait','120','--timeout','600','--on-conflict','terminate','--',str(P/'local-python'),str(P/'run_local.py'),op,cid]
r=dict(argv=cmd,start=time.time());(out/'command.json').write_text(json.dumps(r,indent=2))
with (out/'guard.log').open('x') as f:r['returncode']=subprocess.run(cmd,env=env,cwd=out,stdout=f,stderr=subprocess.STDOUT).returncode
r['end']=time.time()
try:
 m=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
 if m.get('owner')=='baseline700-20261026/surge_tla_attention97/worker01':r['metadata']=m
except Exception:pass
(out/'guard.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
