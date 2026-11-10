import os,sys,json,subprocess,time,datetime
from pathlib import Path
P=Path(__file__).resolve().parent;cid,attempt=sys.argv[1:];out=P/attempt;out.mkdir(exist_ok=False)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
cmd=['npu-guard','run','--owner','baseline700-20261026/nn_dense_resume8','--wait','600','--timeout','300','--','/root/dyq/workspace/.tools/baselines/baseline-run','ops-nn','python',str(P/'run_case.py'),cid,attempt]
(out/'COMMAND.json').write_text(json.dumps(cmd));(P/'HEARTBEAT.json').write_text(json.dumps({'at':now(),'phase':'guard','case_id':cid}))
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['FLA_NPU_DISABLE_PTH']='1'
start=time.time();at=now()
with (out/'guard.log').open('w') as f:rc=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
(out/'GUARD_RESULT.json').write_text(json.dumps({'returncode':rc,'requested_at':at,'finished_at':now(),'elapsed_s':time.time()-start},indent=2))
print(cid,rc,flush=True)
