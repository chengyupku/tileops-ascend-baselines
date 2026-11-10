import os,sys,pathlib,importlib.abc,json
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
P=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(P/'source/src'),str(P/'source'),str(P)]
class NoDevice(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0,NoDevice())
import torch
from types import SimpleNamespace
from workloads import rope
from workloads.device import set_run_device
from adapters.rope_cached import bind
set_run_device('cpu')
results=[]
for name,params in [('RoPEFwdOp',{'input_layout':'1d','rope_layout':'neox','base':10000.}),('RoPEFwdOp',{'input_layout':'1d','rope_layout':'interleaved','base':10000.}),('RoPENeoxPositionIdsFwdOp',{'base':10000.,'max_position':32,'rotary_dim':4})]:
 x=torch.randn(8,8) if name=='RoPEFwdOp' else torch.randn(8,2,8)
 args=(x,) if name=='RoPEFwdOp' else (x,torch.arange(8))
 w=SimpleNamespace(call=SimpleNamespace(signature=SimpleNamespace(name=name),params=params,specs={}),tensors={})
 b=bind(w,args)
 assert torch.equal(b.execute(*args),rope.RoPECall.ref_program(w,*args))
 changed=(x*2,)+args[1:]
 assert torch.equal(b.execute(*changed),rope.RoPECall.ref_program(w,*changed))
 if len(args)>1:
  moved=(x,args[1].flip(0));assert torch.equal(b.execute(*moved),rope.RoPECall.ref_program(w,*moved))
 results.append({'op':name,'params':params,'exact_match':True,'runtime_inputs_not_cached':True})
(P/'ROPE_BOUNDARY_CPU_TEST.json').write_text(json.dumps(results,indent=2));print('passed',len(results))
