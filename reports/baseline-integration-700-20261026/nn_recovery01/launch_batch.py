import datetime,json,os,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;name=sys.argv[1];jobs=json.loads((P/(name+'.json')).read_text());out=P/name;out.mkdir(exist_ok=False)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
if (P/'FEEDBACK.txt').exists():print((P/'FEEDBACK.txt').read_text(),flush=True)
for base,op,cid,a in jobs:
 d=P/a;d.mkdir(exist_ok=False);(d/'adapter.py').write_bytes((P/'adapter.py').read_bytes())
(P/'HEARTBEAT.json').write_text(json.dumps(dict(utc=now(),phase='guard_batch',host='local',current_keys=[j[:3] for j in jobs],last_progress_at=now(),blocked_reason=None)))
cmd=['npu-guard','run','--owner','baseline700-20261026/nn_recovery01','--wait','3600','--timeout','300','--','/root/dyq/workspace/.tools/baselines/baseline-run',jobs[0][0],'python',str(P/('batch_process.py')),str(P/(name+'.json'))]
(out/'COMMAND.json').write_text(json.dumps(cmd));start=time.time();requested_at=now();env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['FLA_NPU_DISABLE_PTH']='1'
with (out/'guard.log').open('w') as f:rc=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
receipt=dict(requested_at=requested_at,returncode=rc,elapsed_s=time.time()-start,finished_at=now());(out/'GUARD_RESULT.json').write_text(json.dumps(receipt))
for *_,a in jobs:
 d=P/a
 if (d/'CHILD_RESULT.json').exists():
  child=json.loads((d/'CHILD_RESULT.json').read_text());(d/'GUARD_RESULT.json').write_text(json.dumps({**receipt,'returncode':rc or child['returncode'],'batch_guard':str(out/'GUARD_RESULT.json'),'child_elapsed_s':child['elapsed_s']}));(d/'guard.log').write_bytes((out/'guard.log').read_bytes())
print(rc)
