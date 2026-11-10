from pathlib import Path
import sys,types,importlib,json,hashlib
R=Path(__file__).resolve().parent;W=R.parents[2];sys.path[:0]=[str(W/'TileOPs'),str(R)]
import importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','triton'):raise RuntimeError('CPU preflight blocked '+fullname)
sys.meta_path.insert(0,Block())
from benchmarks.ascend_all_pairs import registry
sgl=types.ModuleType('benchmarks.ascend_all_pairs.sgl');sgl.PROVIDERS={};sys.modules[sgl.__name__]=sgl
from route_install import install
rows=[]
class Reached(Exception):pass
for name in ['attention_adapter','attention_page16','dsa_adapter','gather_adapter','integration700_sgl_attention32']:
 module=importlib.import_module(name);install(module)
 for op,fn in module.PROVIDERS.items():
  def trace(frame,event,arg):
   if event=='call' and frame.f_code is fn.__code__:raise Reached()
   return trace
  case=types.SimpleNamespace(_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)))
  sys.settrace(trace)
  try:registry.candidate_factory(case,'sgl-kernel-npu')()
  except Reached:passed=True
  else:passed=False
  finally:sys.settrace(None)
  assert passed
  rows.append(dict(operator=op,factory=name+'.'+fn.__name__,actual_factory_entered=passed,sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()))
(R/'OTHER_DISPATCH_PREFLIGHT.json').write_text(json.dumps(dict(passed=True,rows=rows,method='Actual registry and actual private factory code entry intercepted before runtime imports; no torch import'),indent=2))
