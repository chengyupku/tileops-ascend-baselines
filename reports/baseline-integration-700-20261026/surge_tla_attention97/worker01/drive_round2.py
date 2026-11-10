from pathlib import Path
_prior=Path('/proc/3269411/cmdline')
if _prior.exists() and b'worker01/drive.py' in _prior.read_bytes():
 raise SystemExit('Prior orchestrator still alive; refusing concurrent ownership')
from pathlib import Path
import json,subprocess,time,datetime,concurrent.futures,tarfile,io
P=Path(__file__).resolve().parent;R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/surge_tla_attention97/worker01'
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
HOSTS=['openlibing-DevEnv-055409','openlibing-DevEnv-661495','local'];tried={('ds-v3-single-request-float16',HOSTS[1])};iteration=500
S=json.loads((P/'RESULT.json').read_text());rows=S['exactkeys']
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def hb(phase,key=None,**kw):(P/'HEARTBEAT.json').write_text(json.dumps(dict(utc=utc(),phase=phase,key=key,**kw),indent=2))
def status(h):
 cmd=['npu-guard','status','--json'] if h=='local' else [SSH,h,'npu-guard status --json']
 try:
  z=subprocess.run(cmd,capture_output=True,text=True,timeout=40);r=dict(host=h,observed_at=utc(),rc=z.returncode,stdout=z.stdout,stderr=z.stderr)
  try:r['status']=json.loads(z.stdout)
  except Exception:pass
 except Exception as e:r=dict(host=h,observed_at=utc(),rc=None,error=str(e))
 return r
end=datetime.datetime(2026,10,27,5,19,tzinfo=datetime.timezone.utc).timestamp()
for row in rows:
 if row['status'].startswith('reused') or row['status']=='checker_completed_pending_review':continue
 for a in row.get('new_attempts',[]):tried.add((row['key'][2],a['host']))
 key=row['key'];cid=key[2]
 while time.time()<end:
  iteration+=1;hb('three_host_admission',key,last_output='private recipe and exact prior RESULT',next='choose available compatible host')
  with concurrent.futures.ThreadPoolExecutor() as ex:ss=list(ex.map(status,HOSTS))
  (P/f'status_{iteration:03d}.json').write_text(json.dumps(ss,indent=2))
  available=[x['host'] for x in ss if x['rc']==0 and (cid,x['host']) not in tried]
  if not available:
   available=[x['host'] for x in ss if x['rc']==75 and x['host']!='local' and (cid,x['host']) not in tried]
  if not available:
   if all((cid,h) in tried for h in HOSTS):break
   time.sleep(20);continue
  h=available[0];tried.add((cid,h));print(utc(),'START',h,cid,flush=True)
  hb('guarded_checker',key,host=h,admission=f'status_{iteration:03d}.json',last_output='compatible idle snapshot',next='original complete checker')
  if h=='local':
   cmd=['python3',str(P/'launch_local.py'),key[1],cid]
  else:cmd=[SSH,h,f'python3 {R}/launch.py {key[1]} {cid}']
  with (P/f'{h}_{cid}_launch.log').open('w') as f:
   proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
   while proc.poll() is None:
    time.sleep(15);hb('guarded_checker',key,host=h,subprocess_pid=proc.pid,last_output='guard/checker log streaming',next='collect result')
  if h=='local':out=P/'local_runs'/(cid+'-original-default')
  else:
   z=subprocess.run([SSH,h,f'tar -cf - -C {R} runs/{cid}-original-default'],capture_output=True)
   dest=P/h;dest.mkdir(exist_ok=True)
   if z.returncode==0:
    with tarfile.open(fileobj=io.BytesIO(z.stdout)) as t:t.extractall(dest,filter='data')
   out=dest/'runs'/(cid+'-original-default')
  guard=json.loads((out/'guard.json').read_text()) if (out/'guard.json').exists() else {}
  checker=json.loads((out/'checker.json').read_text()) if (out/'checker.json').exists() else None
  rc=guard.get('returncode');row.setdefault('new_attempts',[]).append(dict(host=h,guard=str(out/'guard.json'),checker=str(out/'checker.json') if checker is not None else None,guard_exit=rc,admission=f'status_{iteration:03d}.json',elapsed_with_queue=guard.get('end',0)-guard.get('start',0)))
  row['host']=h;row['checker']=checker;row['guard']=guard;row['evidence']=str(out);row['numerical_success']=False;row['accepted_added']=0
  if rc==75:row['status']='not_started_guard75'
  elif rc==78:row['status']='not_started_guard78'
  elif rc==76:row['status']='invalidated_external_process_guard76'
  elif rc==124:row['status']='guard_timeout_not_numerical_failure'
  elif checker is not None:row['status']='checker_completed_pending_review'
  else:row['status']='execution_error_pending_review'
  S['utc']=utc();(P/'RESULT.json').write_text(json.dumps(S,indent=2));(P/(cid+'.RESULT.json')).write_text(json.dumps(row,indent=2))
  print(utc(),'END',h,cid,'guard',rc,'checker',checker is not None,flush=True)
  if rc not in [75,76,78]:break
 else:row['status']='not_started_time_budget';continue
hb('batch_finished',last_output='per-key RESULT and raw receipts',next='review exact errors')
