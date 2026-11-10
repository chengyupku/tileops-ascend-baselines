"""Read-only metadata observer. Original subprocess args/env and native calls forwarded unchanged."""
import pathlib,os,json,datetime,hashlib,subprocess,shutil,mmap

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2))
def stable(path):
 r=FREEZE['files'][path];st=pathlib.Path(path).stat()
 assert (st.st_ino,st.st_dev,st.st_size,st.st_mtime_ns,st.st_ctime_ns)==tuple(r[k] for k in ['inode','device','size','mtime_ns','ctime_ns']),path
 return r

def mappings():
 lines=pathlib.Path('/proc/self/maps').read_text().splitlines();result=[]
 for path in PATHS:
  real=str(pathlib.Path(path).resolve());matching=[s for s in lines if len(s.split(maxsplit=5))==6 and s.split(maxsplit=5)[5]==real]
  if not matching:continue
  r=stable(path);st=pathlib.Path(path).stat()
  with open(path,'rb') as f:
   before=set(pathlib.Path('/proc/self/maps').read_text().splitlines())
   with mmap.mmap(f.fileno(),min(st.st_size,4096),access=mmap.ACCESS_READ):
    delta=set(pathlib.Path('/proc/self/maps').read_text().splitlines())-before
    mm=[s for s in delta if len(s.split(maxsplit=5))==6 and s.split(maxsplit=5)[5]==real and int(s.split()[4])==st.st_ino]
  assert mm, path
  devices={s.split()[3] for s in mm}
  assert all(int(s.split()[4])==st.st_ino and s.split()[3] in devices for s in matching),path
  result.append(dict(path=path,sha256=r['sha256'],stat=r,maps=matching,passive_mmap=mm,utc=now()))
 return result

def install(root,out):
 global ROOT,OUT,FREEZE,PATHS
 ROOT=root;OUT=out;FREEZE=json.loads((root/'FREEZE.json').read_text());PATHS=json.loads((root/'RUNTIME_PATHS.json').read_text())
 # One stat-only preflight; large hashes already frozen and verified outside guard.
 for p in FREEZE['files']:stable(p)
 write('identity_start.json',dict(utc=now(),freeze_sha256=hashlib.sha256((root/'FREEZE.json').read_bytes()).hexdigest(),files=len(FREEZE['files'])))
 original=subprocess.Popen
 class ObservedPopen(original):
  def __init__(self,args,*a,**kw):
   env=kw.get('env') or os.environ;argv=args if isinstance(args,(list,tuple)) else [args]
   executable=kw.get('executable') or argv[0];resolved=shutil.which(str(executable),path=env.get('PATH'))
   rec=dict(utc=now(),argv=list(map(str,argv)),cwd=str(kw.get('cwd') or os.getcwd()),environment=dict(env),executable=resolved)
   if resolved:
    candidates=[p for p,r in FREEZE['files'].items() if r['realpath']==str(pathlib.Path(resolved).resolve())]
    if candidates:rec['frozen_executable']=stable(candidates[0])
    else:
     st=pathlib.Path(resolved).stat();rec['observed_executable']=dict(sha256=hashlib.sha256(pathlib.Path(resolved).read_bytes()).hexdigest(),inode=st.st_ino,device=st.st_dev)
   with (OUT/'compiler_subprocess.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
   super().__init__(args,*a,**kw)
 subprocess.Popen=ObservedPopen

def finish():
 write('runtime_mappings.json',dict(utc=now(),records=mappings()))
 for p in FREEZE['files']:stable(p)
 write('identity_end.json',dict(utc=now(),files=len(FREEZE['files'])))
