"""Match this campaign's CPU FP8 cast for finite overflow, on device and timed."""
import torch
from torch.utils._python_dispatch import TorchDispatchMode
class FiniteFP8Saturation(TorchDispatchMode):
 def __torch_dispatch__(self,func,types,args=(),kwargs=None):
  kwargs=kwargs or {}
  if func==torch.ops.aten._to_copy.default and kwargs.get('dtype')==torch.float8_e4m3fn and args[0].dtype in (torch.float16,torch.bfloat16,torch.float32):
   x=args[0];x=torch.where(torch.isfinite(x),x.clamp(-448.,448.),x)
   return func(x,**kwargs)
  return func(*args,**kwargs)
