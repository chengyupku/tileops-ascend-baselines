"""Observation only: log real ABI calls, runtime mappings, and bound inputs."""
import os,json,datetime,hashlib
from pathlib import Path
from direct_acl import DirectACL

def install():
 old=DirectACL.call
 def observed(self,inputs,outputs,attributes):
  d=Path(os.environ['CAPTURE_DIR'])
  def log(stage,**kw):
   with (d/'dispatch.jsonl').open('a') as f:f.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),api=self.op,stage=stage,**kw))+'\n')
  def meta(x):
   if isinstance(x,self.torch.Tensor):return dict(shape=list(x.shape),dtype=str(x.dtype),stride=list(x.stride()),device=str(x.device))
   return x
  self.torch.save(tuple(x.cpu().clone() if isinstance(x,self.torch.Tensor) else x for x in inputs),d/'bound_inputs.pt')
  log('call',inputs=[meta(x) for x in inputs],outputs=[meta(x) for x in outputs],attributes=[x.decode() if isinstance(x,bytes) else x for x in attributes])
  try:
   result=old(self,inputs,outputs,attributes)
   log('synchronized_return')
   (d/'runtime.maps').write_text(Path('/proc/self/maps').read_text())
   return result
  except BaseException as e:
   log('exception',error=repr(e));raise
 DirectACL.call=observed
