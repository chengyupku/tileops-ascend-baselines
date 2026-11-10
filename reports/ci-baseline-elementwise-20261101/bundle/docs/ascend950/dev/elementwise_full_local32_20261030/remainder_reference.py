"""Original TileOPs CUDA Remainder semantics with scalar libm remainder.

See kernels/elementwise/arithmetic.py RemainderFwdKernel._remainder:
FP32 fmod, add divisor on a nonzero opposite-sign remainder, narrow once.
Scalar libm avoids this CPU installation's extreme-ratio vector fmod NaNs.
"""
import math
import torch

def remainder_reference(a,b):
 assert a.device.type==b.device.type=='cpu'
 x,y=torch.broadcast_tensors(a.float(),b.float())
 def rem(u,v):
  if math.isinf(u) or v==0:return float('nan')
  return math.fmod(u,v)
 mod=torch.tensor([rem(u,v) for u,v in zip(x.flatten().tolist(),y.flatten().tolist())],dtype=torch.float32).reshape(x.shape)
 return torch.where((mod!=0)&((mod<0)!=(y<0)),mod+y,mod).to(a.dtype)
