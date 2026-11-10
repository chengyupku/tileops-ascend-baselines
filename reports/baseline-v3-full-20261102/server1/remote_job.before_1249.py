"""CPU-only guard launcher, disk monitor and immutable artifact manifest."""
import sys,json,os,time,pathlib,subprocess,signal,hashlib
P=pathlib.Path(__file__).resolve().parent;ROOT=P.parents[2]
def put(p,d):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
mode=sys.argv[1];out=pathlib.Path(sys.argv[2]).resolve()
assert out.is_relative_to(P/'runs')
if mode=='run':
 task=json.load(open(out/'task.json'));free=os.statvfs('/opt');free=free.f_bavail*free.f_frsize/2**30
 if free<50.5:print('disk admission refused',free,flush=True);sys.exit(73)
 cmd=['python3',str(ROOT/'tools/baseline_retest32_server1_guard_receipt.py'),'--owner','v3-full/server1/'+task['id'],'--wait','600','--timeout',str(task.get('timeout',3600)),'--receipt',str(out/'guard.json'),'--','bash',str(P/'launch.sh'),task['key'][0],str(P/'bootstrap.py'),'--task',str(out/'task.json'),'--out',str(out)]
 child=subprocess.Popen(cmd,cwd=ROOT);stopped=False
 def stop(sig,frame):
  global stopped
  if child.poll() is None:child.send_signal(signal.SIGTERM);stopped=True
 for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop)
 while child.poll() is None:
  s=os.statvfs('/opt');free=s.f_bavail*s.f_frsize/2**30
  put(out/'DISK_STATUS.json',{'at':time.time(),'free_GiB':free,'floor_GiB':50,'child':child.pid,'stopped_for_disk':stopped})
  if free<50 and not stopped:stop(signal.SIGTERM,None)
  time.sleep(2)
 sys.exit(child.returncode if child.returncode>=0 else 128-child.returncode)
elif mode=='manifest':
 files={}
 for f in sorted(out.rglob('*')):
  if f.is_symlink():raise ValueError('symlink in artifact: '+str(f))
  if f.is_file() and f.name not in ('FILES.json','ARCHIVED.json'):
   h=hashlib.sha256()
   with f.open('rb') as s:
    for b in iter(lambda:s.read(1048576),b''):h.update(b)
   files[str(f.relative_to(out))]={'size':f.stat().st_size,'sha256':h.hexdigest()}
 put(out/'FILES.json',{'at':time.time(),'files':files})
elif mode=='clean':
 # Only task-owned generated directories, after the local archive has been verified.
 import shutil
 receipt=json.load(sys.stdin);assert receipt['verified'] and receipt['remote']==str(out)
 manifest=json.load(open(out/'FILES.json'))['files']
 for folder in ('profiles','jit_kernel_cache','compile_cache','compile_debug','pto_cache'):
  d=out/folder
  if d.exists():
   for f in d.rglob('*'):
    if f.is_file():
     rel=str(f.relative_to(out));assert rel in manifest
     assert f.stat().st_size==manifest[rel]['size']
     assert hashlib.sha256(f.read_bytes()).hexdigest()==manifest[rel]['sha256']
   shutil.rmtree(d)
 # Complete correctness/measurement remain in the verified local full archive.
 for name in ('correctness.json','measurement.json'):
  f=out/name
  if f.exists():
   assert name in manifest
   assert hashlib.sha256(f.read_bytes()).hexdigest()==manifest[name]['sha256']
   f.unlink()
 put(out/'ARCHIVED.json',receipt)
 # FILES.json is retained inside the verified full tar and locally extracted.
 (out/'FILES.json').unlink()
