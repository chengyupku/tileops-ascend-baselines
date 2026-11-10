"""Restore missing cache-path metadata from the actual loaded CDLL, before evidence hooks."""
from pathlib import Path
import hashlib,json,os,time

def bind_loaded_path(kernel):
 a=kernel.adapter
 loaded=getattr(getattr(a,'lib',None),'_name',None)
 before=getattr(a,'libpath',None)
 record={'time':time.time(),'pid':os.getpid(),'adapter_class':str(type(a)),'reported_libpath':before,'loaded_name':loaded,'changed':False}
 try:
  if not loaded: raise RuntimeError('Original adapter lacks actual CDLL identity')
  path=Path(loaded).resolve(strict=True)
  maps=Path('/proc/self/maps').read_text()
  mapped=[line for line in maps.splitlines() if line.split() and line.split()[-1]==str(path)]
  if not mapped: raise RuntimeError('CDLL path absent from process maps')
  stat=path.stat()
  if not any(int(line.split()[4])==stat.st_ino for line in mapped): raise RuntimeError('Mapped library inode differs')
  record.update(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),inode=stat.st_ino,maps=mapped)
  if before is None:
   a.lib_generator.set_lib_path(str(path))
   record['changed']=True
  elif Path(before).resolve()!=path: raise RuntimeError('Reported and actually loaded library paths differ')
 finally:
  with Path('metadata_binding.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
