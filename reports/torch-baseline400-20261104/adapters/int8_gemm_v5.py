"""Exact INT32 dot via bounded exact-integer FP32 products on NPU.

Each 512-term partial dot is <=2**23 in absolute value even for -128 inputs;
FP32 integer arithmetic is exact in this range. Convert each partial to INT32
before summation, preserving the declared INT32 contraction without CPU fallback.
"""
import torch

def reference(call,a,b,scale_a,scale_b,bias=None):
 accumulator=torch.zeros((a.shape[0],b.shape[0]),dtype=torch.int32,device=a.device)
 for start in range(0,a.shape[1],512):
  product=a[:,start:start+512].float() @ b[:,start:start+512].float().T
  accumulator=accumulator+product.to(torch.int32)
 result=accumulator.float()*scale_a.float()*scale_b.float()
 if bias is not None:result=result+bias.float()
 return result.to(getattr(torch,call.params['out_dtype']))
