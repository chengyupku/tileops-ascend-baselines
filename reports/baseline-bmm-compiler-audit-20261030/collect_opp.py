import os,sys,json,hashlib,subprocess,time,csv,base64
from pathlib import Path
base=Path('/usr/local/Ascend/cann-9.1.0');workspace=Path(sys.argv[1]);files={};errors=[];cache={};links={}
def sha(p):
 s=p.stat();key=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
 if key not in cache:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  cache[key]=h.hexdigest()
 return cache[key]
def tree(p,label):
 if not p.exists():errors.append('missing '+str(p));return
 if p.is_file():files[label]={'sha256':sha(p),'size':p.stat().st_size};return
 for folder,dirs,names in os.walk(p,followlinks=True):
  dirs[:]=[d for d in dirs if d!='__pycache__' and not (Path(folder)/d).resolve() in Path(folder).resolve().parents]
  for n in names:
   q=Path(folder)/n
   if q.suffix=='.pyc':continue
   rel=label+'/'+str(q.relative_to(p))
   try:
    if q.is_symlink():links[rel]=os.readlink(q)
    if q.is_file():files[rel]={'sha256':sha(q),'size':q.stat().st_size}
   except (OSError,ValueError) as e:errors.append([rel,str(e)])

for rel in ['opp/built-in/op_impl/ai_core/tbe/impl','opp/built-in/op_impl/ai_core/tbe/impl_mode','opp/built-in/op_impl/ai_core/tbe/config','opp/built-in/op_impl/ai_core/tbe/op_tiling','opp/built-in/op_impl/ai_core/tbe/op_host','opp/built-in/op_impl/ai_core/tbe/op_api','opp/built-in/op_proto']:
 tree(base/rel,'CANN/'+rel)
print(json.dumps({'at':time.time(),'files':files,'links':links,'errors':errors,'unique_file_hashes':len(cache)}))
