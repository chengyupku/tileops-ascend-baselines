"""Stdlib CPU controller: lookahead2 prepare, one guarded device, concurrent archival."""
import concurrent.futures,fcntl,hashlib,importlib.util,json,os,pathlib,queue,shlex,subprocess,tarfile,tempfile,threading,time,traceback
P=pathlib.Path(__file__).resolve().parent;ROOT=pathlib.Path('/root/dyq/workspace')
REMOTE='/opt/baseline-flex-20261106/server1'
SSH=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495']
mutex=threading.RLock();archive_queue=queue.Queue();archive_pending=set()
def read(p,default=None):
 try:return json.loads(p.read_text())
 except FileNotFoundError:return default

def put(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2));q.replace(p)
def ssh(argv,**kw):return subprocess.run(SSH+[shlex.join(argv)],**kw)
def event(kind,**kw):
 with mutex:
  with (P/'journal.jsonl').open('a') as f:f.write(json.dumps({'at':time.time(),'event':kind,**kw},ensure_ascii=False)+'\n');f.flush()
def state(**kw):
 with mutex:
  s=read(P/'STATE.json',{});s.update(kw,at=time.time(),pid=os.getpid(),start_ticks=pathlib.Path('/proc/self/stat').read_text().split()[21],total=len(read(P/'PLAN.json')['rows']));put(P/'STATE.json',s)
def prepare(t):
 # No imports of torch, baseline providers, compiler or NPU runtime.
 task=dict(t);task.setdefault('timeout',3600);tid=t['id']
 for existing in sorted((P/'runs'/tid).glob('a*')):
  if not (existing/'DEVICE_DONE.json').exists() and (existing/'prepare.json').exists():
   return task,existing,REMOTE+'/runs/'+str(existing.relative_to(P/'runs'))
 prior=list((P/'runs'/tid).glob('a*'));attempt='a'+str(max([int(x.name[1:]) for x in prior]+[0])+1)
 dest=P/'runs'/tid/attempt;dest.mkdir(parents=True,exist_ok=False);out=REMOTE+'/runs/'+tid+'/'+attempt
 put(dest/'task.json',task)
 proof={'cpu_only':True,'kind':'manifest and checker/measurement fingerprints; installed native package reuse','at':time.time(),'sources':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [P/'bootstrap.py',P/'v3/measure_baselines.py',ROOT/'TileOPs/scripts/cache_frozen_pair_check.py']}}
 put(dest/'prepare.json',proof)
 code='import pathlib,json; p=pathlib.Path('+repr(out)+');p.mkdir(parents=True,exist_ok=False);d=json.load(__import__("sys").stdin);[(p/k).write_text(json.dumps(v,ensure_ascii=False,indent=2)) for k,v in d.items()]'
 ssh(['python3','-c',code],input=json.dumps({'task.json':task,'prepare.json':proof}),text=True,capture_output=True,check=True,timeout=60)
 event('prepared',id=tid,attempt=attempt,key=t['key']);return task,dest,out

def validate(dest):
 m=read(dest/'measurement.json',{});g=read(dest/'guard.json',{});c=read(dest/'correctness.json',{})
 assert m.get('status')=='measured_pending_guard',m.get('error',m.get('status','measurement missing'))
 assert g.get('performance_valid') is True and g.get('workload_exit_code')==0 and g.get('exit_code')==0,'guard invalid'
 assert g.get('owner')=='flex/server1/'+m['task_id'],'guard owner mismatch'
 assert c.get('status')=='verified' and c.get('inputs_preserved') and not c.get('provider_source_changed') and not c.get('oracle_source_changed'),'full checker invalid'
 assert hashlib.sha256((dest/'correctness.json').read_bytes()).hexdigest()==m['correctness_sha256']
 assert m['identity']['environment']['aicore_count']==32 and m['identity']['environment']['vectorcore_count']==64
 assert m['identity']['protocol']['id']=='ascend950-isolated-ffts-v3'
 assert all(m['experiment_v3_sources'][n]==hashlib.sha256((P/'v3'/n).read_bytes()).hexdigest() for n in m['experiment_v3_sources']), 'measurement code identity mismatch'
 assert len(m['timing']['latency_samples_ms'])==15 and len(m['timing']['retained_profiles'])==15
 spec=importlib.util.spec_from_file_location('v3_replay',P/'v3/kernel_task_profile.py');parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
 remote=REMOTE+'/runs/'+str(dest.relative_to(P/'runs'))
 for row in m['timing']['retained_profiles']:
  directory=dest/pathlib.Path(row['profile_path']).relative_to(remote)
  assert parser.trace_digest(directory)==row['profile_trace_sha256'],'raw digest mismatch'
  replay=parser.parse_profile(directory,1)
  assert row.get('parser_revision')==parser.PARSER_REVISION, 'parser revision mismatch'
  assert replay.get('profile_named_task_coverage_complete',True), 'incomplete task coverage'
  assert abs(replay['latency_ms']-sum(x['duration_us'] for x in row['profile_kernel_tasks'])/1000)<1e-6,'raw replay mismatch'
 return m

def archive_one(task,dest,out):
 ssh(['python3',REMOTE+'/remote_job.py','manifest',out],capture_output=True,check=True,timeout=180)
 partial=dest/'full-receipt.tar.gz.partial';archive=dest/'full-receipt.tar.gz'
 with partial.open('wb') as f:ssh(['nice','-n','10','tar','-C',out,'-czf','-','.'],stdout=f,stderr=subprocess.PIPE,check=True,timeout=600)
 # Extract only regular files into the private attempt directory; reject traversal/links.
 with tarfile.open(partial,'r:gz') as tar:
  members=tar.getmembers()
  for item in members:
   rel=pathlib.PurePosixPath(item.name)
   assert not rel.is_absolute() and '..' not in rel.parts
   assert item.isfile() or item.isdir(),item.name
  tar.extractall(dest,filter='data')
 manifest=read(dest/'FILES.json')['files']
 for name,meta in manifest.items():
  f=dest/name;assert f.stat().st_size==meta['size']
  assert hashlib.sha256(f.read_bytes()).hexdigest()==meta['sha256'],name
 partial.replace(archive);receipt={'verified':True,'at':time.time(),'remote':out,'local':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(manifest),'bytes':archive.stat().st_size}
 try:
  m=validate(dest);accepted=True;error=None
 except Exception as e:accepted=False;error=str(e)
 receipt.update(accepted=accepted,error=error)
 # Remote cleanup is after full archive verification, including failed attempts.
 ssh(['python3',REMOTE+'/remote_job.py','clean',out],input=json.dumps(receipt),text=True,capture_output=True,check=True,timeout=300)
 # Compressed archive is authoritative; keep small receipts immediately browsable locally.
 import shutil
 for name in ('profiles','jit_kernel_cache','compile_cache','compile_debug'):
  if (dest/name).exists():shutil.rmtree(dest/name)
 with mutex:
  progress=read(P/'PROGRESS.json',{})
  progress[task['id']]={'key':task['key'],'status':'accepted' if accepted else 'pending','attempt':dest.name,'at':time.time(),'error':error,'receipt':str(dest/'measurement.json'),'archive':receipt}
  put(P/'PROGRESS.json',progress);put(dest/'ARCHIVE.json',receipt)
 event('archived',id=task['id'],accepted=accepted,error=error,archive=str(archive))

def archiver():
 while True:
  item=archive_queue.get()
  try:archive_one(*item)
  except Exception as e:
   event('archive_error',id=item[0]['id'],error=str(e));time.sleep(15);archive_queue.put(item)
  finally:
   with mutex:archive_pending.discard(item[0]['id'])
   archive_queue.task_done()

def enqueue(item):
 with mutex:
  if item[0]['id'] in archive_pending:return
  archive_pending.add(item[0]['id']);archive_queue.put(item)

def main():
 lock=(P/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 tasks=read(P/'PLAN.json')['rows'];lookup={t['id']:t for t in tasks}
 threading.Thread(target=archiver,daemon=True).start()
 # Recover closed attempts after controller interruption, without dropping raw.
 for done in (P/'runs').glob('*/a*/DEVICE_DONE.json'):
  dest=done.parent;tid=dest.parent.name
  if not read(dest/'ARCHIVE.json',{}).get('verified'):enqueue((lookup[tid],dest,REMOTE+'/runs/'+str(dest.relative_to(P/'runs'))))
 state(phase='running',started_at=time.time())
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  while True:
   if (P/'STOP').exists():archive_queue.join();state(phase='stopped');return
   progress=read(P/'PROGRESS.json',{});attempted={p.parent.parent.name for p in (P/'runs').glob('*/a*/DEVICE_DONE.json')};retry=read(P/'RETRY.json',{})
   todo=[t for t in tasks if progress.get(t['id'],{}).get('status')!='accepted' and (t['id'] not in attempted or retry.get(t['id'],{}).get('at',0)>max([read(f,{}).get('at',0) for f in (P/'runs'/t['id']).glob('a*/DEVICE_DONE.json')]+[0]))]
   if not todo:
    state(phase='complete' if len([x for x in progress.values() if x['status']=='accepted'])==len(tasks) else 'awaiting_review_or_archive');time.sleep(15);continue
   todo=todo[:8]
   futures={}
   for i,t in enumerate(todo):
    if (P/'STOP').exists():
     archive_queue.join();state(phase='stopped');return
    for j in range(i,min(i+2,len(todo))):
     if j not in futures:futures[j]=pool.submit(prepare,todo[j])
    task,dest,out=futures.pop(i).result()
    while True:
     s=ssh(['python3','-c','import os,json;s=os.statvfs("/opt");print(json.dumps({"free_GiB":s.f_bavail*s.f_frsize/2**30}))'],capture_output=True,text=True,timeout=30)
     if s.returncode==0:
      disk=json.loads(s.stdout);put(P/'DISK_STATUS.json',{**disk,'at':time.time(),'floor_GiB':50,'admission_GiB':50.5})
      if disk['free_GiB']>=50.5 and archive_queue.qsize()<3:break
     state(phase='waiting_disk_or_archive',current=task['key']);time.sleep(10)
    while True:
     if (P/'STOP').exists():archive_queue.join();state(phase='stopped');return
     # Nightly admission cutoff: device job maximum3600sec + margin, UTC19=UTC+8 03.
     tm=time.gmtime();nightly=False
     probe=ssh(['env','LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64','/opt/tileops-ci/runtime/bin/python','-I','-S','/opt/torch-v010-20261106/tools/npu_guard.py','status','--json'],capture_output=True,text=True,timeout=60)
     try:g=json.loads(probe.stdout)
     except ValueError:g={'state':'unknown'}
     put(P/'DEVICE_STATUS.json',{'at':time.time(),'guard':g,'nightly_reservation':nightly})
     if not nightly and g.get('state')=='idle':break
     state(phase='waiting_nightly' if nightly else 'waiting_device',current=task['key']);time.sleep(30)
    state(phase='device_running',current=task['key'],current_id=task['id'],attempt=dest.name)
    event('device_start',id=task['id'],key=task['key'],attempt=dest.name)
    with (dest/'console.log').open('w') as log:
     rc=ssh(['python3',REMOTE+'/remote_job.py','run',out],stdout=log,stderr=subprocess.STDOUT).returncode
    if rc==255:
     state(phase='connection_lost_needs_process_audit',current=task['key']);return
    if rc in (73,75):
     event('deferred_before_device',id=task['id'],returncode=rc);state(phase='waiting_device',current=task['key']);time.sleep(30);break
    put(dest/'DEVICE_DONE.json',{'at':time.time(),'returncode':rc});event('device_done',id=task['id'],returncode=rc)
    enqueue((task,dest,out))
if __name__=='__main__':main()
