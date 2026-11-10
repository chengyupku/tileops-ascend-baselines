# Instrument the actual function objects held by _MathCall. Never load a target library.
import ctypes as C
from benchmarks.ascend_math_more import _MathCall
native_events=[]
_original_init=_MathCall.__init__
def mapping(fn):
 address=C.cast(fn,C.c_void_p).value
 lines=Path('/proc/self/maps').read_text().splitlines()
 matches=[line for line in lines if int(line.split()[0].split('-')[0],16)<=address<int(line.split()[0].split('-')[1],16)]
 assert len(matches)==1
 fields=matches[0].split();path=Path(fields[-1]);st=path.stat()
 return dict(address=address,map_line=matches[0],path=str(path),map_device=fields[3],map_inode=int(fields[4]),stat_device=st.st_dev,stat_inode=st.st_ino,sha256=sha(path))
def observed_init(self,*args,**kwargs):
 _original_init(self,*args,**kwargs)
 assert self.api=='aclnnGroupedMatmulV5'
 for attr in ('size','launch'):
  fn=getattr(self,attr)
  def wrapper(*args,_fn=fn,_attr=attr):
   before=dict(event='enter',function=self.api+('GetWorkspaceSize' if _attr=='size' else ''),handle=self.lib._handle,library_name=str(self.lib._name),time=time.time(),**mapping(_fn))
   native_events.append(before);dump('NATIVE_CALLS.json',native_events)
   result=_fn(*args)
   native_events.append(dict(event='return',function=before['function'],return_code=result,time=time.time()))
   dump('NATIVE_CALLS.json',native_events)
   return result
  setattr(self,attr,wrapper)
_MathCall.__init__=observed_init
