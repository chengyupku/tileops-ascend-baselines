import datetime,json,os,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;base,op,cid,attempt=sys.argv[1:];out=P/attempt;out.mkdir(exist_ok=False)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
if (P/'FEEDBACK.txt').exists():print((P/'FEEDBACK.txt').read_text(),flush=True)
(P/'HEARTBEAT.json').write_text(json.dumps(dict(utc=now(),phase='guard_queue_or_check',current_keys=[[base,op,cid]],last_progress_at=now(),blocked_reason=None)))
(out/'adapter.py').write_bytes((P/'adapter.py').read_bytes())
cmd=['npu-guard','run','--owner','baseline700-20261026/nn_math','--wait','3600','--timeout','300','--','/root/dyq/workspace/.tools/baselines/baseline-run',base,'python',str(P/'run_case.py'),base,op,cid,attempt]
(out/'COMMAND.json').write_text(json.dumps(cmd));start=time.time();env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['FLA_NPU_DISABLE_PTH']='1'
with (out/'guard.log').open('w') as log:rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
(out/'GUARD_RESULT.json').write_text(json.dumps(dict(returncode=rc,elapsed_s=time.time()-start,finished_at=now())));print(rc,flush=True)
