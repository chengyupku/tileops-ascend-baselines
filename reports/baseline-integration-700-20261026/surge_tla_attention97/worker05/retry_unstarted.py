from pathlib import Path
import subprocess,json,time,datetime,concurrent.futures,sys,shlex
P=Path(__file__).resolve().parent;SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';REMOTE='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/surge_tla_attention97/worker05'
HOSTS=['local','openlibing-DevEnv-661495','openlibing-DevEnv-055409'];DEADLINE=datetime.datetime.fromisoformat('2026-10-27T05:18:00+00:00').timestamp()
todo=[r for r in json.loads((P/'RESULT.json').read_text()) if r['status']=='guard_not_started']
(P/'attempts').mkdir(exist_ok=True)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def beat(key,phase,last,blocker=None):
 (P/'HEARTBEAT.json').write_text(json.dumps(dict(utc=now(),current_key=key,phase=phase,last_substantive_output=last,blocker=blocker,next='Complete original checker, preserve receipts, then next scoped key'),indent=2))
def status(h):
 cmd=['npu-guard','status','--json'] if h=='local' else [SSH,h,'npu-guard status --json']
 try:
  r=subprocess.run(cmd,capture_output=True,text=True,timeout=25);s=json.loads(r.stdout);return h,dict(received_utc=now(),command=cmd,exit=r.returncode,status=s,stderr=r.stderr)
 except Exception as e:return h,dict(received_utc=now(),error=str(e))
for index,row in enumerate(todo):
 key=row['key'];op,cid=key[1:];attempt='round02';tag='retry_'+str(index).zfill(2)+'_'+cid
 used={a['host'] for a in row.get('attempts',[])}
 mapping={'local':'local_runs','openlibing-DevEnv-661495':'server1_runs','openlibing-DevEnv-055409':'server2_runs'}
 preferred=[h for h in HOSTS if mapping[h] not in used] or HOSTS
 while time.time()<DEADLINE:
  with concurrent.futures.ThreadPoolExecutor(3) as e:states=dict(e.map(status,HOSTS))
  (P/'attempts'/(tag+'_latest_threehost.json')).write_text(json.dumps(states,indent=2))
  idle=[h for h in preferred if states[h].get('status',{}).get('state')=='idle' and not states[h].get('status',{}).get('reservation',{}).get('locked')]
  if idle:host=idle[0];break
  # No idle gap is required: join an existing cooperative queue with bounded wait.
  compatible=[h for h in preferred if states[h].get('status',{}).get('reservation',{}).get('locked')]
  if compatible:host=compatible[index%len(compatible)];break
  beat(key,'waiting_for_available_compatible_host',str(P/'attempts'/(tag+'_latest_threehost.json')),'All three current host status samples busy/unknown; no repeated guard submission');print(now(),'WAIT',cid,flush=True);time.sleep(30)
 else:break
 (P/'attempts'/(tag+'_pre_submit.json')).write_text(json.dumps(states,indent=2))
 cmd=[sys.executable,str(P/'attempt_launcher.py'),op,cid,attempt] if host=='local' else [SSH,host,'python3 '+REMOTE+'/attempt_launcher.py '+shlex.quote(op)+' '+shlex.quote(cid)+' '+attempt]
 beat(key,'guard_wait_or_run',str(P/'attempts'/(tag+'_pre_submit.json')));print(now(),'START',host,key,flush=True)
 with (P/'attempts'/(tag+'_launcher.log')).open('x') as f:
  proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
  while proc.poll() is None:
   beat(key,'guard_wait_or_run',str(P/'attempts'/(tag+'_launcher.log')));time.sleep(20)
 if host!='local':subprocess.run([sys.executable,str(P/'harvest.py'),host,cid+'-'+attempt],check=True)
 subprocess.run([sys.executable,str(P/'summarize.py')],check=True)
 beat(key,'case_receipts_saved',str(P/'RESULT.json'));print(now(),'FINISHED',host,key,flush=True)
beat(None,'scheduled_scope_finished',str(P/'RESULT.json'))
