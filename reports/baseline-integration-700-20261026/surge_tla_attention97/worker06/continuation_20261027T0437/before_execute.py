from pathlib import Path
import os,subprocess,json,datetime,time,sys,concurrent.futures
P=Path(__file__).resolve().parent
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
OWNER='baseline700-20261026/surge_tla_attention97/worker06'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
op,cid=sys.argv[1:]; key=['tilelang-ascend',op,cid]
out=P/'local_runs'/(cid+os.environ.get('WORKER_ATTEMPT_SUFFIX','-pto-A5-original-branch'));out.mkdir(parents=True,exist_ok=True)
assert not (out/'guard.json').exists()
def status(h):
 cmd=['npu-guard','status','--json'] if h=='local' else [SSH,h,'npu-guard status --json']
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=40)
 d=dict(collected_at=now(),command=cmd,exit=r.returncode,stdout=r.stdout,stderr=r.stderr)
 (out/(h+'_status.json')).write_text(json.dumps(d,indent=2));return h,r.returncode
with concurrent.futures.ThreadPoolExecutor(3) as e: print(list(e.map(status,['local','openlibing-DevEnv-661495','openlibing-DevEnv-055409'])),flush=True)
cmd=['npu-guard','run','--owner',OWNER,'--wait','120','--timeout','180','--on-conflict','terminate','--',str(P/'local-python'),str(P/'run_local.py'),op,cid]
d=dict(key=key,requested=now(),start=time.time(),command=cmd,host='local',changed_condition='First device run of this exact key; sibling timeout is not a result for this key.')
(out/'command.json').write_text(json.dumps(d,indent=2));(P/'HEARTBEAT.json').write_text(json.dumps(dict(at=now(),key=key,phase='guard_submitted',last_output=str(out/'command.json'),next='original checker'),indent=2))
with (out/'guard.log').open('x') as f:d['guard_exit']=subprocess.run(cmd,cwd=P,stdout=f,stderr=subprocess.STDOUT).returncode
d.update(end=time.time(),ended=now());d['wall_seconds']=d['end']-d['start'];(out/'guard.json').write_text(json.dumps(d,indent=2))
row=dict(key=key,host='local',status='device_timeout' if d['guard_exit']==124 else 'not_started_guard_wait' if d['guard_exit']==75 else 'guard_visibility_or_external_failure' if d['guard_exit'] in [76,78] else 'checker_receipt_available' if (out/'checker.json').exists() else 'execution_failed' if d['guard_exit'] else 'requires_review',guard=str(out/'guard.json'),checker=str(out/'checker.json') if (out/'checker.json').exists() else None,native=str(out/'native.json') if (out/'native.json').exists() else None,accepted_added=0,registered=False,wall_seconds=d['wall_seconds'])
(out/'RESULT.json').write_text(json.dumps(row,indent=2));(P/'HEARTBEAT.json').write_text(json.dumps(dict(at=now(),key=key,phase='case_finished',last_output=str(out/'RESULT.json'),next='review and next exact key'),indent=2));print(json.dumps(row),flush=True)
