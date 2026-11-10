"""Evaluate scalar-base power using the native FP64-base path, round to input dtype."""
import torch
from torch.utils._python_dispatch import TorchDispatchMode
class RopePower(TorchDispatchMode):
 def __torch_dispatch__(self,f,types,args=(),kwargs=None):
  kwargs=kwargs or {}
  if f==torch.ops.aten.pow.Scalar and isinstance(args[1],torch.Tensor) and args[1].device.type=='npu' and args[1].dtype==torch.float32:
   e=args[1];b=torch.full_like(e,args[0],dtype=torch.float64)
   return torch.pow(b,e).float()
  return f(*args,**kwargs)
