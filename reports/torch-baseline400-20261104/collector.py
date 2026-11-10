"""CPU archive transfer/replay overlaps both remote pipelines. Never runs device code."""
import pathlib,json,subprocess,tarfile,tempfile,hashlib,importlib.util,time,threading,concurrent.futures,shlex,fcntl,statistics,traceback
P=pathlib.Path(__file__).resolve().parent;REMOTE='/opt/torch-baseline400-20261104';HOSTS=json.loads((P/'ASSIGNMENT.json').read_text())['hosts']
spec=importlib.util.spec_from_file_location('parser',P/'tools/kernel_task_profile.py');parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
spec32=importlib.util.spec_from_file_location('parser32',P/'tools/kernel_task_profile_full32.py');parser32=importlib.util.module_from_spec(spec32);spec32.loader.exec_module(parser32)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
def ssh(host,args,**kwargs):return subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-o','ClearAllForwardings=yes',HOSTS[host],shlex.join(args)],**kwargs)
def collect(host):
 H=P/host
 code="import pathlib,json; p=pathlib.Path("+repr(REMOTE+'/'+host)+");print(json.dumps({'done':[str(f.parent.relative_to(p/'runs')) for f in (p/'runs').glob('*/a*/DONE.json')], 'state':json.load(open(p/'STATE.json')) if (p/'STATE.json').exists() else {}, 'heartbeat':json.load(open(p/'HEARTBEAT.json')) if (p/'HEARTBEAT.json').exists() else {}}))"
 scan=ssh(host,['python3','-c',code],capture_output=True,text=True,check=True,timeout=45);scan=json.loads(scan.stdout);scan['heartbeat']['completed']=sum(x.get('status') in ('completed','accepted') for x in scan.get('state',{}).values());scan['heartbeat']['received_at']=time.time();put(H/'REMOTE_STATUS.json',scan['heartbeat'])
 for rel in scan['done']:
  tid,attempt=rel.split('/');dest=H/'runs'/tid/attempt;dest.mkdir(parents=True,exist_ok=True);report=dest/'ARCHIVE.json'
  previous=json.loads(report.read_text()) if report.exists() else {}
  if previous.get('cleaned') and (previous.get('accepted') or 'SOURCES.json' not in (previous.get('error') or '')):continue
  archive=dest/'full-receipt.tar.gz'
  if not archive.exists():
   with archive.with_suffix('.partial').open('wb') as fp:ssh(host,['python3',REMOTE+'/archive_remote.py','pack',host,tid,attempt],stdout=fp,stderr=subprocess.PIPE,check=True,timeout=3600)
   archive.with_suffix('.partial').replace(archive)
  error=None;measurement=None
  with tempfile.TemporaryDirectory(prefix='torch400-replay-') as tmp:
   root=pathlib.Path(tmp)
   with tarfile.open(archive) as tar:
    for m in tar.getmembers():
     r=pathlib.PurePosixPath(m.name);assert not r.is_absolute() and '..' not in r.parts and (m.isfile() or m.isdir())
    tar.extractall(root,filter='data')
   manifest=json.loads((root/'FILES.json').read_text())['files']
   for n,d in manifest.items():assert (root/n).stat().st_size==d['size'] and sha(root/n)==d['sha256'],n
   for f in root.iterdir():
    if f.is_file() and f.suffix in ('.json','.log'):(dest/f.name).write_bytes(f.read_bytes())
   try:
    m=json.loads((root/'measurement.json').read_text());g=json.loads((root/'guard.json').read_text());c=json.loads((root/'correctness.json').read_text());task=json.loads((root/'task.json').read_text())
    if m['key'][1] in ('DeltaNetChunkBwdOp','GLAChunkBwdOp'):
     route=json.loads((root/'device_route.json').read_text());assert route.get('device_arithmetic') and route.get('original_cpu_checker')
    device_log=(root/'device.log').read_text(errors='replace');assert 'fall back to run on the CPU' not in device_log and 'npu_cpu_fallback' not in device_log,'NPU backend used CPU fallback; not admissible device timing'
    assert m['status']=='measured_pending_guard' and m['key']==task['key'] and m['task_id']==tid
    assert g['performance_valid'] and g['phase']=='finished' and g['workload_exit_code']==0 and g['exit_code']==0
    assert g['owner']=='torch400/'+host+'/'+tid
    assert c['status']=='verified' and c['inputs_preserved'] and m['inputs_preserved'] and sha(root/'correctness.json')==m['correctness_sha256']
    assert m['source_sha256'] in {sha(f) for f in P.glob('SOURCES.v*.json')}
    assert (m['hardware']['cube'],m['hardware']['vector'])==(32,64)
    samples=m['timing']['latency_samples_ms'];profiles=m['timing']['retained_profiles'];assert len(samples)==len(profiles)==15
    values=[]
    for row,sample in zip(profiles,samples):
     relative=pathlib.Path(row['profile_path']).relative_to(pathlib.Path(REMOTE)/host/'runs'/tid/attempt);directory=root/relative
     assert parser.trace_digest(directory)==row['profile_trace_sha256']
     decoder=parser32 if row['parser_revision']==parser32.PARSER_REVISION else parser
     parsed=decoder.parse_profile(directory,1);assert parsed['parser_revision']==row['parser_revision'];assert abs(parsed['latency_ms']-sample)<1e-9;values.append(parsed['latency_ms'])
    assert abs(statistics.mean(values)-m['timing']['latency_ms'])<1e-9
    measurement={'latency_ms':m['timing']['latency_ms'],'key':m['key'],'raw_windows':15}
   except Exception as e:error=str(e)+'\n'+traceback.format_exc(limit=3)
  receipt={'verified':True,'remote':REMOTE+'/'+host+'/runs/'+rel,'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'accepted':error is None,'error':error,'measurement':measurement,'at':time.time(),'cleaned':False}
  put(report,receipt)
  if not previous.get('cleaned'):
   ssh(host,['python3',REMOTE+'/archive_remote.py','clean',host,tid,attempt],input=json.dumps(receipt),text=True,capture_output=True,check=True,timeout=180)
  receipt['cleaned']=True;put(report,receipt)
  print(host,tid,'accepted' if error is None else 'pending',flush=True)
 while False:pass
 progress={}
 for f in (H/'runs').glob('*/a*/ARCHIVE.json'):
  d=json.loads(f.read_text());tid=f.parent.parent.name
  if d.get('accepted') or tid not in progress or (not progress[tid].get('accepted') and d['at']>progress[tid]['at']):progress[tid]=d
 put(H/'PROGRESS.json',progress)
def main():
 lock=(P/'collector.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def host_loop(h):
  while not (P/'STOP_COLLECTOR').exists():
   try:collect(h)
   except Exception:put(P/h/'COLLECTOR_ERROR.json',{'at':time.time(),'traceback':traceback.format_exc()})
   time.sleep(10)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  list(pool.map(host_loop,HOSTS))
if __name__=='__main__':main()
