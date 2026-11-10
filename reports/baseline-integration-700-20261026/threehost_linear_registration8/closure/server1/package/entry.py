import sys,os,json,runpy,hashlib,ctypes,time
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;S=R/'snapshot';W=R.parents[2]
from direct_acl import require_guard,DirectACL
require_guard()
run=Path(os.environ['GDN_RUN_DIR']);(run/'GUARD_RUNNING.json').write_text(Path('/run/lock/ascend-npu/device-0.lock').read_text())
sys.path[:0]=[str(S/'TileOPs'),str(S/'TileOPs/src'),str(S),str(P)]
sys.path.extend(['/opt/tileops-ci/runtime/lib/python3.12/site-packages',str(W/'.cache/all-pairs/remote_native_vllm/python-deps'),str(W/'.cache/all-pairs/remote_native_tla/python-deps')])
import setuptools
import benchmarks.ascend_all_pairs.registry as registry
from binding import build
orig=registry.candidate_factory
registry.candidate_factory=lambda case,lib: (lambda:build(case)) if lib=='ops-transformer' and case._call.signature.name=='DeltaNetFwdOp' else orig(case,lib)
old=DirectACL.call
def call(self,ins,outs,attrs):
 t=time.monotonic();result=old(self,ins,outs,attrs)
 rec={'op':self.op,'symbol':hex(ctypes.cast(self.launch,ctypes.c_void_p).value),'workspace_symbol':hex(ctypes.cast(self.workspace,ctypes.c_void_p).value),'library':self.library._name,'inode':os.stat(self.library._name).st_ino,'device':os.stat(self.library._name).st_dev,'inputs':[None if x is None else {'shape':list(x.shape),'dtype':str(x.dtype)} for x in ins],'outputs':[{'shape':list(x.shape),'dtype':str(x.dtype)} for x in outs],'wall_seconds':time.monotonic()-t}
 with (run/'dispatch.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
 return result
DirectACL.call=call
try:runpy.run_path(str(S/'TileOPs/scripts/ascend_pair_check.py'),run_name='__main__')
finally:
 import torch
 if torch.npu.is_initialized():torch.npu.synchronize()
 maps=Path('/proc/self/maps').read_text();(run/'maps.txt').write_text(maps)
 files=set(line.split()[-1] for line in maps.splitlines() if '/' in line)
 files.update(getattr(m,'__file__','') or '' for m in list(sys.modules.values()))
 hashes={}
 for name in files:
  p=Path(name)
  if p.is_file():
   try:hashes[name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'inode':p.stat().st_ino,'device':p.stat().st_dev}
   except OSError:pass
 (run/'LOADED_HASHES.json').write_text(json.dumps(hashes,indent=2))
