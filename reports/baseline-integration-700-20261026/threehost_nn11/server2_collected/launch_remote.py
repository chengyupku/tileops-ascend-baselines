import os,sys,json,subprocess,time,datetime
from pathlib import Path
P=Path(__file__).resolve().parent
cid,attempt=sys.argv[1:];out=P/attempt;out.mkdir(exist_ok=False)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
cmd=['npu-guard','run','--owner','baseline700-20261026/threehost_nn11','--wait','120','--timeout','900','--','bash',str(P/'environment.sh'),'/opt/tileops-ci/runtime/bin/python','-I','-S','-B',str(P/'private_python.py'),str(P/'run_remote.py'),cid,attempt]
(out/'COMMAND.json').write_text(json.dumps(cmd));env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['FLA_NPU_DISABLE_PTH']='1'
t=time.time();at=now()
with (out/'guard.log').open('w') as f:
 proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 while proc.poll() is None:
  stage={}
  for n in ['STAGE.json','progress.json']:
   try:stage[n]=json.loads((out/n).read_text())
   except (OSError,ValueError):pass
  (P/'HEARTBEAT.json').write_text(json.dumps(dict(phase='guard_or_fullchecker',currentkey=['ops-nn','GroupedGemmFwdOp',cid],host='server2',lastprogress=now(),blockedreason=None,elapsed_s=time.time()-t,stage=stage),indent=2))
  time.sleep(15)
rc=proc.returncode
m=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());(out/'GUARD_FINAL.json').write_text(json.dumps(m,indent=2))
(out/'GUARD_RESULT.json').write_text(json.dumps(dict(returncode=rc,requested_at=at,finished_at=now(),elapsed_s=time.time()-t),indent=2))
print(cid,rc,flush=True)
