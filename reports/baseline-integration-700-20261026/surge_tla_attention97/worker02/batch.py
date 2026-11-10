from pathlib import Path
import subprocess,json,time,datetime,concurrent.futures,os,shlex
P=Path(__file__).resolve().parent
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
REMOTE='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/surge_tla_attention97/worker02'
HOSTS=['local','openlibing-DevEnv-661495','openlibing-DevEnv-055409']
END=datetime.datetime.fromisoformat('2026-10-27T05:25:00+00:00').timestamp()
rows=json.loads((P/'RESULT.json').read_text());route={tuple(x['key']):x for x in json.loads((P/'CPU_ROUTE.json').read_text())};seq=max([int(f.stem) for f in (P/'status_samples').glob('*.json')]+[0])

def heartbeat(key,phase,last,blocker=None):
 (P/'HEARTBEAT.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),key=key,phase=phase,last_output=last,blocker=blocker,next='independent original checker for remaining exactkeys'),indent=2))
def status(h):
 c=['npu-guard','status','--json'] if h=='local' else [SSH,h,'npu-guard status --json']
 try:
  r=subprocess.run(c,capture_output=True,text=True,timeout=20)
  disk=__import__('shutil').disk_usage(P).free if h=='local' else int(subprocess.run([SSH,h,'df -B1 --output=avail '+REMOTE+' | tail -1'],capture_output=True,text=True,timeout=20).stdout.strip())
  return dict(host=h,rc=r.returncode,data=json.loads(r.stdout),stderr=r.stderr,free_disk_bytes=disk)
 except Exception as e:return dict(host=h,rc=None,error=str(e))
def statuses():
 global seq
 seq+=1
 with concurrent.futures.ThreadPoolExecutor() as e:s=list(e.map(status,HOSTS))
 (P/'status_samples').mkdir(exist_ok=True);(P/'status_samples'/('%04d.json'%seq)).write_text(json.dumps(dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hosts=s),indent=2));return s

def process(row,host,attempt):
 key=row['key'];op,cid=key[1:];suffix='-batch%d'%attempt;name=cid+'-pto-A5-original-branch'+suffix
 budget=180 if cid=='mla-tp-b1-bfloat16' or op=='GQADenseFwdOp' else 300
 env=os.environ.copy();env.update(ATTEMPT_SUFFIX=suffix,CASE_TIMEOUT=str(budget))
 cmd=['python3',str(P/'launch_local.py'),op,cid] if host=='local' else [SSH,host,'ATTEMPT_SUFFIX='+suffix+' CASE_TIMEOUT='+str(budget)+' python3 '+REMOTE+'/launch.py '+shlex.quote(op)+' '+shlex.quote(cid)]
 heartbeat(key,'guard_submission',cmd)
 start=time.time();r=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 while r.poll() is None:
  try:stdout,stderr=r.communicate(timeout=30)
  except subprocess.TimeoutExpired:heartbeat(key,'guarded_foreground_command',cmd,'queued or executing; inspect guard log');continue
  break
 else:stdout,stderr=r.communicate()
 (P/'launch_logs').mkdir(exist_ok=True);(P/'launch_logs'/(cid+suffix+'.json')).write_text(json.dumps(dict(command=cmd,local_start=start,local_end=time.time(),launch_rc=r.returncode,stdout=stdout,stderr=stderr),indent=2))
 if host!='local':subprocess.run(['python3',str(P/'collect.py'),host],check=True)
 out=P/'local_runs'/name if host=='local' else P/host/'runs'/name
 receipt=json.loads((out/'guard.json').read_text()) if (out/'guard.json').exists() else {}
 gm=json.loads((out/'GUARD_RUNNING.json').read_text()) if (out/'GUARD_RUNNING.json').exists() else {}
 checker=json.loads((out/'checker.json').read_text()) if (out/'checker.json').exists() else None
 rc=receipt.get('returncode');state='guard_queue_timeout_not_run' if rc==75 else 'external_interference_invalid' if rc==76 else 'command_timeout' if rc==124 else 'execution_failed'
 if checker:state='verified_pending_parent_registration' if rc==0 and checker.get('status')=='verified' else checker.get('status','execution_failed')
 verified=state=='verified_pending_parent_registration'
 timing=dict(requested_at=gm.get('requested_at'),acquired_at=gm.get('acquired_at'),child_started_at=gm.get('updated_at'),end_epoch=receipt.get('end'),queue_seconds=None,physical_idle_wait_seconds=None,guarded_command_seconds=None,held_cpu_seconds=None,device_seconds=None)
 if gm:
  requested=datetime.datetime.fromisoformat(gm['requested_at']).timestamp();acquired=datetime.datetime.fromisoformat(gm['acquired_at']).timestamp();child=datetime.datetime.fromisoformat(gm['updated_at']).timestamp()
  timing.update(queue_seconds=acquired-requested,physical_idle_wait_seconds=child-acquired,guarded_command_seconds=receipt['end']-child)
 a=dict(host=host,status=state,guard_exit=rc,receipt=str(out/'guard.json'),command=receipt.get('argv'),checker=str(out/'checker.json') if checker else None,native_evidence=str(out/'native.json') if (out/'native.json').exists() else None,actual_native_launch=(out/'launch_01/native.json').exists(),timings=timing)
 row.setdefault('attempts',[]).append(a);row.update(status=state,numerical_success=verified,numerical_failure_established=bool(checker and checker.get('stage')=='verify' and checker.get('error_type')=='AssertionError'),accepted_added=0,host=host,guard=a['receipt'],checker=a['checker'],native_evidence=a['native_evidence'],timings=timing,registration_ready=verified)
 (P/'RESULT.json').write_text(json.dumps(rows,indent=2));(P/'results').mkdir(exist_ok=True);(P/'results'/(cid+'.json')).write_text(json.dumps(row,indent=2));print(json.dumps(dict(key=key,status=state,guard_exit=rc,host=host)),flush=True)
 return rc

pending=[r for r in rows if r['status']=='pending_actual_check']
pending.sort(key=lambda r:(r['key'][2]!='mla-tp-b1-bfloat16',route[tuple(r['key'])]['manifest'].get('B',1)*route[tuple(r['key'])]['manifest'].get('N_kv',route[tuple(r['key'])]['manifest'].get('S_kv',1))))
for row in pending:
 attempt=0;queued_reservations=set()
 while time.time()<END:
  ss=statuses();idle=[s['host'] for s in ss if s.get('data',{}).get('state')=='idle' and s['rc']==0]
  if not idle:
   near=[]
   for st in ss:
    data=st.get('data',{});m=data.get('reservation',{}).get('metadata',{});owner=m.get('owner','');ident=(st['host'],m.get('pid'),m.get('start_ticks'))
    if m.get('phase')!='running' or ident in queued_reservations or not owner.startswith('baseline700-20261026/surge_tla_attention97/'):continue
    budget=300 if owner.endswith('worker04') else 180 if owner.endswith('worker01') and m.get('command')=='timeout' else 600
    elapsed=(datetime.datetime.fromisoformat(data['time'])-datetime.datetime.fromisoformat(m['updated_at'])).total_seconds()
    if budget-100<=elapsed<=budget+30:near.append((budget-elapsed,st['host'],ident))
   if near:
    near.sort();idle=[near[0][1]];queued_reservations.add(near[0][2])
  shape=route[tuple(row['key'])]['manifest']
  input_bytes=shape['B']*shape['N_kv']*(shape['D']+shape['PE'])*2 if row['key'][1]=='MLADecodeWithKVCacheFwdOp' else shape['B']*shape['S_kv']*shape['H_kv']*shape['D']*4
  storage_required=input_bytes*2+3*1024**3
  idle=[h for h in idle if next(s for s in ss if s['host']==h).get('free_disk_bytes',0)>storage_required]
  if not idle:
   heartbeat(row['key'],'waiting_for_available_host','CPU preparation complete; all three latest statuses saved','No currently idle compatible host; avoiding repeated 120-second queue expiry');time.sleep(20);continue
  attempt+=1
  (P/'storage_decisions').mkdir(exist_ok=True)
  (P/'storage_decisions'/(row['key'][2]+'-'+str(attempt)+'.json')).write_text(json.dumps(dict(key=row['key'],input_bytes=input_bytes,required_free_bytes=storage_required,chosen_host=idle[0],hosts=ss),indent=2))
  rc=process(row,idle[0],attempt)
  if rc!=75:break
 if time.time()>=END:break
heartbeat(None,'batch_finished','RESULT.json updated per key')
