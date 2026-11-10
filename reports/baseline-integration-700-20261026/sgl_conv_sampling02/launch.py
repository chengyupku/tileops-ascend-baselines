from pathlib import Path
import sys,json,os,subprocess,datetime,time
R=Path(__file__).resolve().parent
op,cid,attempt=sys.argv[1:]
a=R/'runs'/attempt;a.mkdir(parents=True,exist_ok=False)
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
t=time.monotonic(); started=now()
(R/'HEARTBEAT.json').write_text(json.dumps(dict(utc=started,phase='guard_wait_or_run',current_keys=[cid],last_progress_at=started,blocked_reason=None)))
cmd=[str(R/'RUN_GUARDED.sh'),str(R/'run_case.py'),'sgl-kernel-npu',op,cid,str(a/'output')]
(a/'COMMAND.json').write_text(json.dumps(cmd)); env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
probe=subprocess.run(['npu-guard','status','--json'],capture_output=True,text=True)
(a/'PRE_GUARD_STATUS.json').write_text(probe.stdout)
state=json.loads(probe.stdout)
reservation=state.get('reservation',{}); group=reservation.get('metadata',{}).get('child',{}).get('pid') if reservation.get('locked') else None
external=[]
for item in state.get('pids',[]):
 try: cooperative=group is not None and os.getpgid(item['pid'])==group
 except ProcessLookupError: cooperative=True
 if not cooperative:external.append(item)
if external or state.get('health',{}).get('value')!=0 or state.get('errors'):
 (a/'GUARD_RESULT.json').write_text(json.dumps({'returncode':75,'command_started':False,'reason':'Pre-request status detects external users or unhealthy/unknown device; no guard request submitted','status':state,'started':started,'finished':now(),'elapsed_seconds':time.monotonic()-t}))
 raise SystemExit(75)
with (a/'guard.log').open('x') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
(a/'GUARD_RESULT.json').write_text(json.dumps(dict(returncode=p.returncode,started=started,finished=now(),elapsed_seconds=time.monotonic()-t)))
print(a,p.returncode)
