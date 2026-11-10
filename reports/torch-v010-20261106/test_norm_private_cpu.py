import os,sys,pathlib,json
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
P=pathlib.Path(__file__).resolve().parent;sys.path[:0]=[str(P/'source/src'),str(P/'source'),str(P)]
import torch
from types import SimpleNamespace as NS
from workloads.norm import _norm_reference
from adapters.norm_private import bind
passed=[]
for name in ('BatchNormFwdOp','InstanceNormFwdOp'):
 for training in (True,False):
  p={'training':training,'use_input_stats':training,'momentum':.1,'eps':1e-5};c=NS(_call=NS(params=p,signature=NS(name=name)))
  args=(torch.randn(2,4,8),torch.zeros(4),torch.ones(4),torch.randn(4),torch.randn(4));original=[x.clone() for x in args]
  expected=_norm_reference(name,args,p);b=bind(c,args)
  for _ in range(3):
   result=b.check_once(lambda:None)
   e=expected if isinstance(expected,tuple) else (expected,);r=result if isinstance(result,tuple) else (result,)
   assert all((x is None and y is None) or torch.equal(x,y) for x,y in zip(e,r))
  assert all(torch.equal(x,y) for x,y in zip(args,original));passed.append([name,training])
(P/'NORM_PRIVATE_CPU_TEST.json').write_text(json.dumps({'passed':passed,'repeats':3,'device_validation':'pending'},indent=2));print('passed')
