from pathlib import Path
import subprocess,concurrent.futures,json,datetime,time,os
P=Path(__file__).resolve().parent
D=P/'continuation_20261027T0437'
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
HOSTS=['local','openlibing-DevEnv-661495','openlibing-DevEnv-055409']
CASES=[('GQAVarlenFwdOp','llama-8b-uniform-float16'),('GQAVarlenFwdOp','gpt-oss-q-lt-kv-float16'),('GQAPagedFwdOp','llama-8b-p256-cut-float16-float16'),('GQAPagedFwdOp','llama-8b-serving-step-float16-float16')]
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def status(h):
 cmd=['npu-guard','status','--json'] if h=='local' else [SSH,h,'npu-guard status --json']
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=40)
 try:s=json.loads(r.stdout)
 except Exception:s={}
 return {'host':h,'utc':now(),'exit':r.returncode,'sample':s,'stderr':r.stderr}
def score(r):
 s=r['sample'];m=s.get('reservation',{}).get('metadata',{}); external=bool(m.get('external_pids')) or (s.get('pids') and not s.get('reservation',{}).get('locked'))
 if r['exit']==0:return 0
 if r['exit'] not in [0,75] or s.get('errors'):return 100
 if external:return 80
 return 10+(1 if r['host']=='local' else 0)
for index,(op,cid) in enumerate(CASES):
 if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,10,27,5,17,tzinfo=datetime.timezone.utc):break
 with concurrent.futures.ThreadPoolExecutor(3) as e:states=list(e.map(status,HOSTS))
 chosen=min(states,key=score);h=chosen['host']
 (D/(cid+'_host_selection.json')).write_text(json.dumps({'at':now(),'selected':h,'states':states,'recipe':'reuse previously verified deployed runtime; unchanged source hashes rechecked; only adapter metadata delta'},indent=2))
 # Every planned key is exactly in the original scope, and no completed device timeout exists.
 prior=[]
 for base in ['local_runs','openlibing-DevEnv-661495/runs','openlibing-DevEnv-055409/runs']:
  for f in (P/base).glob(cid+'-*/guard.json'):
   g=json.loads(f.read_text());prior.append(g)
 assert not any(g.get('guard_exit')==124 for g in prior),cid
 env=dict(os.environ,WORKER_ATTEMPT_SUFFIX='-continuation-20261027T0437-unstarted',WORKER_CHANGED_CONDITION='Exact case had no completed device run; reuse metadata binding and validated original host recipe; wait increased from 120 to 600; host selected from fresh three-host status')
 cmd=['python3',str(P/'execute.py'),op,cid] if h=='local' else ['python3',str(P/'remote_execute.py'),h,op,cid]
 print(json.dumps({'at':now(),'starting':cid,'host':h,'status_exits':[(r['host'],r['exit']) for r in states]}),flush=True)
 proc=subprocess.Popen(cmd,env=env)
 while proc.poll() is None:
  heartbeat={'at':now(),'key':['tilelang-ascend',op,cid],'phase':'foreground_guard_request_active','launcher_pid':proc.pid,'last_output':str(D/(cid+'_host_selection.json')),'next':'collect complete checker/guard, immediately write per-key RESULT','blocker':'guard queue or original execution; consult current guard.log'}
  (P/'HEARTBEAT.json').write_text(json.dumps(heartbeat,indent=2));print(json.dumps(heartbeat),flush=True)
  try:proc.wait(timeout=50)
  except subprocess.TimeoutExpired:pass
 print(json.dumps({'at':now(),'finished':cid,'launcher_exit':proc.returncode}),flush=True)
 subprocess.run(['python3',str(P/'finalize.py')],check=True,stdout=subprocess.DEVNULL)
