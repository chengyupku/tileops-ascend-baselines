from pathlib import Path
import subprocess,json,os,time,sys,datetime,concurrent.futures,tarfile,io,shlex
P=Path(__file__).resolve().parent;W=P.parents[3]
SSH=str(W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh')
RP=Path('/opt/tileops-all-pairs/workspace')/P.relative_to(W)
HOSTS=['local','openlibing-DevEnv-661495','openlibing-DevEnv-055409']
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def status(h):
 c=['npu-guard','status','--json'] if h=='local' else [SSH,h,'npu-guard status --json']
 r=subprocess.run(c,capture_output=True,text=True,timeout=40)
 return dict(host=h,rc=r.returncode,stdout=r.stdout,stderr=r.stderr,received_at=now())
def run(op,cid,host,change):
 key=['tilelang-ascend',op,cid];assert key in json.loads((P/'REMAINING.json').read_text())
 attempt=datetime.datetime.now(datetime.timezone.utc).strftime('continuation_%Y%m%dT%H%M%S')+'-'+host+'-'+cid
 out=P/'resume_runs'/attempt;out.mkdir(parents=True)
 with concurrent.futures.ThreadPoolExecutor(3) as pool:states=list(pool.map(status,HOSTS))
 (out/'threehost_status.json').write_text(json.dumps(states,indent=2))
 for s in states:
  try:d=json.loads(s['stdout']);print(s['host'],d.get('state'),d.get('reservation',{}).get('metadata',{}).get('owner'),flush=True)
  except:pass
 runtime=P if host=='local' else RP
 env=os.environ.copy();env.update(INTEGRATION700_TLA_ROUTE='legacy',PYTHONDONTWRITEBYTECODE='1',W07_ATTEMPT=attempt)
 base=['npu-guard','run','--owner','baseline700-20261026/surge_tla_attention97/worker07','--wait','600','--timeout','600','--',str(P/'local-python') if host=='local' else '/opt/tileops-all-pairs/bin/allpairs-python',str(runtime/('run_local_resume.py' if host=='local' else 'run_remote_resume.py')),op,cid]
 if host!='local':
  b=io.BytesIO()
  with tarfile.open(fileobj=b,mode='w') as t:
   for name in ['run_remote_resume.py','adapter.py','element_adapter.py','capture.py','SCOPE.json','private_tla']:
    t.add(P/name,arcname=name)
  subprocess.run([SSH,host,'mkdir -p '+shlex.quote(str(RP))+' && tar -xf - -C '+shlex.quote(str(RP))],input=b.getvalue(),check=True)
  cmd=[SSH,host,'env INTEGRATION700_TLA_ROUTE=legacy PYTHONDONTWRITEBYTECODE=1 W07_ATTEMPT='+shlex.quote(attempt)+' '+shlex.join(base)]
 else:cmd=base
 rec=dict(key=key,host=host,argv=cmd,guard_argv=base,requested_at=now(),start=time.time(),changed_condition=change,prior_evidence=str(P/'first_delivery/RESULT.json'),compatibility_receipts=[str(P/f) for f in ['local_cpu.json','openlibing-DevEnv-661495_cpu.json','openlibing-DevEnv-055409_cpu.json']])
 (out/'command.json').write_text(json.dumps(rec,indent=2))
 def heartbeat():
  (P/'HEARTBEAT.json').write_text(json.dumps(dict(utc=now(),current_key=key,phase='guard_wait_or_original_checker',last_substantive_output=str(out/'command.json'),blocker=None,next='capture exact checker and native evidence'),indent=2))
 heartbeat()
 with (out/'guard.log').open('w') as f:
  proc=subprocess.Popen(cmd,cwd=P,env=env,stdout=f,stderr=subprocess.STDOUT)
  while proc.poll() is None:
   try:proc.wait(timeout=45)
   except subprocess.TimeoutExpired:heartbeat()
 rec.update(returncode=proc.returncode,end=time.time(),ended_at=now())
 if host!='local':
  r=subprocess.run([SSH,host,'if test -d '+shlex.quote(str(RP/'resume_runs'/attempt))+'; then tar -cf - -C '+shlex.quote(str(RP/'resume_runs'/attempt))+' .; fi'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  if r.stdout:
   with tarfile.open(fileobj=io.BytesIO(r.stdout)) as t:t.extractall(out)
 (out/'guard.json').write_text(json.dumps(rec,indent=2))
 m=json.loads((out/'GUARD_RUNNING.json').read_text()) if (out/'GUARD_RUNNING.json').exists() else {}
 rec.update(acquired_at=m.get('acquired_at'),child_start=m.get('updated_at'),child_identity=m.get('child'),wall_seconds=rec['end']-rec['start'],held_cpu_seconds=None,device_seconds=None)
 (out/'ATTEMPT.json').write_text(json.dumps(rec,indent=2))
 print(json.dumps(dict(out=str(out),rc=rec['returncode'],child=m.get('child'),checker=(out/'checker.json').exists())),flush=True)
if __name__=='__main__':
 run(*sys.argv[1:])
 from collect_resume import collect
 collect()
