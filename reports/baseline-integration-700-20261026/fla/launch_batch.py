import sys,json,subprocess,time,datetime,pathlib,os
p=pathlib.Path(__file__).resolve().parent
batch,*ids=sys.argv[1:]
if (p/'FEEDBACK.txt').exists():print((p/'FEEDBACK.txt').read_text(),flush=True)
t=datetime.datetime.now(datetime.timezone.utc).isoformat();start=time.monotonic()
(p/'HEARTBEAT.json').write_text(json.dumps(dict(utc=t,phase=batch,current_keys=ids,last_progress_at=t,blocked_reason=None),indent=2))
cmd=['npu-guard','run','--owner','baseline700-20261026/fla','--wait','3600','--timeout','300','--','/root/dyq/workspace/.tools/baselines/baseline-run','flash-linear-attention-npu','python',str(p/'run_gdn_batch.py'),*ids]
(p/(batch+'.COMMAND.json')).write_text(json.dumps(cmd,indent=2))
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
with (p/(batch+'.guard.log')).open('w') as log: rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=p).returncode
m=json.loads(pathlib.Path('/run/lock/ascend-npu/device-0.lock').read_text());valid=m.get('owner')=='baseline700-20261026/fla' and m.get('requested_at','')>=t
receipt=dict(returncode=rc,started_at=t,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),total_seconds=time.monotonic()-start,metadata=m if valid else None,queue_seconds=None,device_seconds=None,compile_seconds=None)
if valid:
 for a,b,name in [('requested_at','acquired_at','queue_seconds')]:
  if m.get(a) and m.get(b):receipt[name]=(datetime.datetime.fromisoformat(m[b])-datetime.datetime.fromisoformat(m[a])).total_seconds()
 if receipt['queue_seconds'] is not None:receipt['device_seconds']=receipt['total_seconds']-receipt['queue_seconds']
(p/(batch+'.GUARD_RESULT.json')).write_text(json.dumps(receipt,indent=2));print(receipt,flush=True)
raise SystemExit(rc)
