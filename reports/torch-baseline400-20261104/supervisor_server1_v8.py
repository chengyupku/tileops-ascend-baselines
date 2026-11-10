"""One host, latest pipeline lookahead2, independent archival by local collector."""
import json,pathlib,subprocess,time,os,sys,shutil,signal,fcntl,hashlib
P=pathlib.Path(__file__).resolve().parent;host=sys.argv[1];H=P/host;H.mkdir(exist_ok=True)
lock=(H/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def read(p,default):
 try:return json.loads(p.read_text())
 except FileNotFoundError:return default
def put(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def clean_handoff(tid):
 for directory in [pathlib.Path('/dev/shm/torch-baseline400'),P/'spool']:
  for suffix in ['.pt','.partial']:
   (directory/(tid+suffix)).unlink(missing_ok=True)
  for shard in directory.glob(tid+'.shard*'):
   if shard.suffix in ('.pt','.partial','.gz'):shard.unlink(missing_ok=True)
rows=read(H/'PLAN.json',{})['rows'];state=read(H/'STATE.json',{});child=None;stop=False
def shutdown(sig,frame):
 global stop
 stop=True
 if child and child.poll() is None:child.terminate()
for s in [signal.SIGTERM,signal.SIGINT]:signal.signal(s,shutdown)
while not stop and not (H/'STOP').exists():
 blocked=read(H/'BLOCKED.json',{})
 retry=read(H/'RETRY.json',{});todo=[r for r in rows if r['id'] not in blocked and state.get(r['id'],{}).get('status') not in ('completed','accepted') and (r['id'] not in state or retry.get(r['id'],{}).get('at',0)>state[r['id']].get('at',0))]
 free=shutil.disk_usage(P).free/2**30
 put(H/'HEARTBEAT.json',{'at':time.time(),'pid':os.getpid(),'start_ticks':pathlib.Path('/proc/self/stat').read_text().split()[21],'free_GiB':free,'todo':len(todo),'status':'waiting_disk' if free<50.5 else ('running' if todo else 'waiting_repairs')})
 if not todo or free<50.5:time.sleep(10);continue
 # Preserve diagnostic ordering in PLAN.
 group=todo[:26];batch='b'+str(time.time_ns());jobs=[];destinations={}
 for r in group:
  attempts=list((H/'runs'/r['id']).glob('a*'));dest=H/'runs'/r['id']/('a'+str(max([int(x.name[1:]) for x in attempts]+[0])+1));dest.mkdir(parents=True);put(dest/'task.json',r);destinations[r['id']]=dest
  jobs.append({'id':r['id'],'prepare':{'cpu_only':True,'timeout':3600,'argv':['bash',str(P/'launch_server1_v8.sh'),'prepare',str(dest/'task.json'),str(dest)]},'device':{'timeout':3600,'argv':['bash',str(P/'launch_server1_v8.sh'),'device',str(dest/'task.json'),str(dest)]}})
 plan={'owner':'torch400/'+host,'wait':1800,'continue_on_job_error':True,'jobs':jobs};pf=H/(batch+'.json');put(pf,plan);bo=H/'pipeline'/batch;bo.parent.mkdir(exist_ok=True)
 cmd=['/usr/bin/python3',str(P/'tools/npu_pipeline_ready.py'),'--plan',str(pf),'--out',str(bo),'--lookahead','2']
 with (H/'pipeline.log').open('a') as log:
  child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
  while child.poll() is None:
   free=shutil.disk_usage(P).free/2**30
   put(H/'HEARTBEAT.json',{'at':time.time(),'pid':os.getpid(),'start_ticks':pathlib.Path('/proc/self/stat').read_text().split()[21],'child_pid':child.pid,'batch':batch,'free_GiB':free,'status':'running'})
   # A closed per-job guard is sufficient for immediate archival; do not wait for batch end.
   ev=bo/'events.jsonl'
   if ev.exists():
    for line in ev.read_text().splitlines():
     try:e=json.loads(line)
     except json.JSONDecodeError:continue
     if e.get('event')!='phase_end':continue
     tid=e['id'];dest=destinations[tid]
     if (dest/'DONE.json').exists():continue
     if e['phase']=='prepare' and e['returncode']==0:continue
     pd=bo/tid;g=read(pd/'guard.json',{})
     status='failed'
     if e['phase']=='device' and e['returncode']==0:
      status='completed' if g.get('performance_valid') and g.get('phase')=='finished' and g.get('workload_exit_code')==0 else 'invalid_guard'
     for name in ['guard.json','prepare.log','device.log']:
      if (pd/name).exists():shutil.copy2(pd/name,dest/name)
     result={'status':status,'at':time.time(),'attempt':dest.name,'key':read(dest/'task.json',{})['key'],'pipeline':str(bo),'error':None if status=='completed' else str(e)}
     clean_handoff(tid)
     state[tid]=result;put(dest/'DONE.json',result);put(H/'STATE.json',state)
   if free<50:child.terminate()
   time.sleep(2)
 summary=read(bo/'summary.json',{})
 for x in summary.get('jobs',[]):
  dest=destinations[x['id']];pd=bo/x['id']
  clean_handoff(x['id'])
  if (dest/'DONE.json').exists():continue
  for name in ['guard.json','prepare.log','device.log']:
   if (pd/name).exists():shutil.copy2(pd/name,dest/name)
  if x['status'] in ('completed','failed','invalid_guard'):
   result={'status':x['status'],'at':time.time(),'attempt':dest.name,'key':read(dest/'task.json',{})['key'],'pipeline':str(bo),'error':summary.get('error') if x['status']!='completed' else None};state[x['id']]=result;put(dest/'DONE.json',result)
  elif x['status']=='cancelled':
   # No receipt is promoted. A canceled preparation is eligible to run afresh.
   put(dest/'CANCELLED.json',x)
  payload=read(dest/'payload.json',{})
  payload_path=pathlib.Path(payload.get('path','/dev/shm/torch-baseline400/'+x['id']+'.pt'))
  if payload_path.exists():payload_path.unlink()
 if not summary:
  put(H/'SUPERVISOR_ERROR.json',{'at':time.time(),'returncode':child.returncode,'batch':batch});time.sleep(15)
 put(H/'STATE.json',state)

put(H/'STOPPED.json',{'at':time.time(),'pid':os.getpid()})
