"""Decode the original W4 layout with exact byte arithmetic; public torch only."""
import torch
from adapters.execution_boundary import ExecutionBoundary

def decode(prepacked,scale,zero):
 n,kp=prepacked.shape
 # Original words: [step, lane, word], low half then high half.
 x=prepacked.float().reshape(n,kp//64,4,4,2,2)
 high=torch.floor(x/16);low=x-high*16
 q=torch.stack((low,high),-1).flatten(-2).permute(0,1,3,5,2,4).reshape(n,kp*2)
 size=kp*2//scale.shape[1]
 return ((q.reshape(n,-1,size)-zero.float()[...,None])*scale.float()[...,None]).reshape(n,kp*2)

def bind(case,args,previous):
 assert case._call.signature.name=='GemmW4A16FwdOp'
 def run(a,packed,scale,zero):
  weight=decode(packed,scale,zero).to(a.dtype)
  return (a.float()@weight.float().T).to(a.dtype)
 return ExecutionBoundary(run,tuple(args),lambda:None,{'candidate':'w4-byte-arithmetic-v1','change':'exact byte/nibble permutation; remove scalar word loops and bit shifts; original FP32 contraction and activation-dtype weight rounding','input_dependent_work_timed':True})
