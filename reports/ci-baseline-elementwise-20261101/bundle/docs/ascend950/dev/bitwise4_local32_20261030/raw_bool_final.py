"""Exact noncanonical bool storage validation for final public implementations."""
import importlib
import json
from pathlib import Path
import torch
import torch_npu
from tileops.ops import elementwise
for name in ['bitwise_binary','bitwise_or','bitwise_xor','bitwise_not']:
 importlib.import_module('tileops.kernels.ascend.elementwise.'+name)
a=torch.arange(256,dtype=torch.uint8).repeat_interleave(256)
b=torch.arange(256,dtype=torch.uint8).repeat(256)
checks=[]
for name,fn in [('BitwiseAndFwdOp',torch.logical_and),('BitwiseOrFwdOp',torch.logical_or),('BitwiseXorFwdOp',torch.logical_xor),('BitwiseNotFwdOp',torch.logical_not)]:
 for n in [1,7,129,65536]:
  x=a[:n].npu().view(torch.bool);y=b[:n].npu().view(torch.bool)
  if name=='BitwiseNotFwdOp':out=getattr(elementwise,name)()(x);ref=fn(a[:n]!=0)
  else:out=getattr(elementwise,name)()(x,y);ref=fn(a[:n]!=0,b[:n]!=0)
  torch.npu.synchronize()
  torch.testing.assert_close(out.cpu(),ref,rtol=0,atol=0)
  torch.testing.assert_close(out.view(torch.uint8).cpu(),ref.to(torch.uint8),rtol=0,atol=0)
  checks.append({'op':name,'n':n,'passed':True})
Path(__file__).with_name('raw-bool-final.json').write_text(json.dumps({'runtime_limit':torch.npu.get_device_limit(0),'checks':checks},indent=2)+'\n')
print(len(checks),'checks passed',flush=True)
