"""Bit-preserving FP8 indexing through uint8 storage; no numerical conversion."""
import torch
from torch.utils._python_dispatch import TorchDispatchMode
class FP8Index(TorchDispatchMode):
 def __torch_dispatch__(self,func,types,args=(),kwargs=None):
  kwargs=kwargs or {}
  if args and isinstance(args[0],torch.Tensor) and args[0].dtype in (torch.float8_e4m3fn,torch.float8_e5m2):
   a=list(args);dtype=a[0].dtype
   if func in (torch.ops.aten.index.Tensor,torch.ops.aten.index_select.default):
    a[0]=a[0].view(torch.uint8);return func(*a,**kwargs).view(dtype)
   if func in (torch.ops.aten.index_put_.default,torch.ops.aten.index_put.default):
    assert not kwargs.get('accumulate',a[3] if len(a)>3 else False),'FP8 accumulated scatter is not a byte copy'
    a[0]=a[0].view(torch.uint8);a[2]=a[2].view(torch.uint8);return func(*a,**kwargs).view(dtype)
   if func in (torch.ops.aten.index_copy_.default,torch.ops.aten.index_copy.default):
    a[0]=a[0].view(torch.uint8);a[3]=a[3].view(torch.uint8);return func(*a,**kwargs).view(dtype)
  return func(*args,**kwargs)
