from pathlib import Path
import subprocess,json,datetime,time,concurrent.futures,sys
P=Path(__file__).resolve().parent;SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';HOSTS=['local','openlibing-DevEnv-661495','openlibing-DevEnv-055409']
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sample(h):
 c=['npu-guard','status','--json'] if h=='local' else [SSH,h,'npu-guard status --json'];r=subprocess.run(c,capture_output=True,text=True,timeout=40)
 try:d=json.loads(r.stdout)
 except Exception:d={'state':'unknown','error':r.stderr}
 return h,d
def aggregate():
 rows=json.loads((P/'PRESERVED_RESULTS.json').read_text());rows += [json.loads(f.read_text()) for f in P.glob('local_runs/*/RESULT.json')];rows += [json.loads(f.read_text()) for f in P.glob('openlibing-*/runs/*/RESULT.json')];(P/'RESULT.json').write_text(json.dumps(rows,indent=2))
sc=json.loads((P/'SCOPE.json').read_text());keys=[r['key'] for r in sc if r['operator'] in ['GQAPagedFwdOp','GQAVarlenFwdOp'] and r['case_id']!='gemma4-31b-uniform-float16']
for key in keys:
 prior=[f for f in list(P.glob('local_runs/*/RESULT.json'))+list(P.glob('openlibing-*/runs/*/RESULT.json')) if json.loads(f.read_text())['key']==key]
 if prior and any(json.loads(f.read_text()).get('status') not in ['not_started_guard_wait','invalidated_guard_visibility_lost'] and json.loads(Path(json.loads(f.read_text())['guard']).read_text())['guard_exit'] not in [75,76,78] for f in prior):continue
 tried=set(json.loads(f.read_text())['host'] for f in prior)
 for attempt in range(3):
  with concurrent.futures.ThreadPoolExecutor(3) as e: states=dict(e.map(sample,HOSTS))
  available=[h for h in HOSTS if h not in tried]
  if not available:break
  h=min(available,key=lambda h:(states[h].get('state')!='idle',len(states[h].get('pids',[])),HOSTS.index(h)))
  (P/'latest_host_selection.json').write_text(json.dumps(dict(at=now(),key=key,selected=h,states=states),indent=2))
  cmd=[sys.executable,str(P/'execute.py'),key[1],key[2]] if h=='local' else [sys.executable,str(P/'remote_execute.py'),h,key[1],key[2]]
  print(now(),'START',key,h,flush=True)
  with (P/'batch.log').open('a') as log:
   child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
   while True:
    try:rc=child.wait(timeout=45);break
    except subprocess.TimeoutExpired:
     (P/'HEARTBEAT.json').write_text(json.dumps(dict(at=now(),key=key,host=h,phase='foreground_guard_or_checker',last_output='batch.log and per-case guard.log',next='wait bounded guard completion'),indent=2));print(now(),'ACTIVE',key,h,flush=True)
  aggregate();tried.add(h)
  files=list(P.glob('local_runs/*/RESULT.json'))+list(P.glob('openlibing-*/runs/*/RESULT.json'))
  row=next((json.loads(f.read_text()) for f in files if json.loads(f.read_text())['key']==key and json.loads(f.read_text())['host']==h),None)
  print(now(),'END',key,h,row,flush=True)
  if row and json.loads(Path(row['guard']).read_text())['guard_exit'] not in [75,76,78]:break
aggregate()
(P/'HEARTBEAT.json').write_text(json.dumps(dict(at=now(),phase='batch_finished_review_pending',last_output='RESULT.json',next='final evidence review'),indent=2))
