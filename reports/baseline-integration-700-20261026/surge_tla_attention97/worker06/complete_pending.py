from pathlib import Path
import os,sys,json,time,datetime,subprocess,concurrent.futures
P=Path(__file__).resolve().parent;SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';HOSTS=['local','openlibing-DevEnv-661495','openlibing-DevEnv-055409']
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sample(h):
 c=['npu-guard','status','--json'] if h=='local' else [SSH,h,'npu-guard status --json'];r=subprocess.run(c,capture_output=True,text=True,timeout=40)
 try:d=json.loads(r.stdout)
 except Exception:d={'state':'unknown','error':r.stderr}
 return h,d
def attempts(key):
 return [(f,json.loads(f.read_text())) for f in list(P.glob('local_runs/*/RESULT.json'))+list(P.glob('openlibing-*/runs/*/RESULT.json')) if json.loads(f.read_text())['key']==key]
def complete(key):
 for f,r in attempts(key):
  g=json.loads(Path(r['guard']).read_text());o=f.parent
  if g['guard_exit'] in [0,124] and list(o.glob('launch_*/native.json')):return True
  if (o/'checker.json').exists():
   c=json.loads((o/'checker.json').read_text())
   if c.get('status')=='verified':return True
   if c.get('error_type')!='TypeError' or 'NoneType' not in c.get('error',''):return True
 return False
scope=json.loads((P/'SCOPE.json').read_text());keys=[r['key'] for r in scope if r['operator'] in ['GQAPagedFwdOp','GQAVarlenFwdOp'] and r['case_id']!='gemma4-31b-uniform-float16']
keys.sort(key=lambda k:k[2]!='llama-8b-p256-float16-float16')
deadline=datetime.datetime.fromisoformat('2026-10-27T05:23:00+00:00').timestamp();serial=0
while time.time()<deadline:
 pending=[k for k in keys if not complete(k)]
 if not pending:break
 with concurrent.futures.ThreadPoolExecutor(3) as e:states=dict(e.map(sample,HOSTS))
 idle=[h for h in HOSTS if states[h].get('state')=='idle'];serial+=1
 snapshot=dict(at=now(),pending=pending,states=states,phase='idle_window_check')
 (P/('pending_window_%03d.json'%serial)).write_text(json.dumps(snapshot,indent=2))
 if not idle:
  (P/'HEARTBEAT.json').write_text(json.dumps(dict(at=now(),phase='await_idle_window_without_reservation',pending=pending,last_output='pending_window_%03d.json'%serial,next='threehost idle sample in 45 seconds'),indent=2));print(now(),'WAIT_IDLE',len(pending),flush=True);time.sleep(45);continue
 h=idle[0];key=pending[0];suffix='-idle-window-%03d'%serial
 reason='New three-host sample demonstrates idle device after prior reservation-only failures; exact key still has no completed original-native attempt.'
 if key[2]=='llama-8b-p256-float16-float16':reason='Per-case fresh kernel cache fixes prior cached adapter libpath=None before native launch; public API/checker unchanged. See CACHE_REMEDY.json.'
 (P/('retry_condition_%03d.json'%serial)).write_text(json.dumps(dict(at=now(),key=key,host=h,suffix=suffix,changed_condition=reason,status_snapshot='pending_window_%03d.json'%serial),indent=2))
 env=os.environ.copy();env['WORKER_ATTEMPT_SUFFIX']=suffix
 cmd=[sys.executable,str(P/'execute.py'),key[1],key[2]] if h=='local' else [sys.executable,str(P/'remote_execute.py'),h,key[1],key[2]]
 print(now(),'START',key,h,suffix,flush=True)
 with (P/'pending.log').open('a') as log:
  child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
  while True:
   try:rc=child.wait(timeout=45);break
   except subprocess.TimeoutExpired:
    (P/'HEARTBEAT.json').write_text(json.dumps(dict(at=now(),key=key,host=h,phase='foreground_pending_attempt',changed_condition=reason,last_output='pending.log and per-case receipts',next='bounded guard completion'),indent=2));print(now(),'ACTIVE',key,h,flush=True)
 print(now(),'END',key,h,'child_exit',rc,flush=True)
 subprocess.run([sys.executable,str(P/'finalize.py')],stdout=subprocess.DEVNULL,check=True)
print(now(),'PENDING_REMAINING',[k for k in keys if not complete(k)],flush=True)
