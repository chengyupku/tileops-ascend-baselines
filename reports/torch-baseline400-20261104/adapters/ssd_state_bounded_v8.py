"""Evaluate unchanged independent chunks with bounded temporary memory."""
import torch

def reference(original,x,Bmat,dt,dA_cumsum,n_groups,seq_idx=None):
 b,seq,h,p=x.shape;c,Q=dt.shape[2:];n=Bmat.shape[-1]
 out=torch.empty((b,c,h,p,n),dtype=torch.float32,device=x.device)
 for batch in range(b):
  for start in range(0,c,4):
   end=min(start+4,c);sl=slice(start*Q,end*Q)
   value=original(x[batch:batch+1,sl],Bmat[batch:batch+1,sl],dt[batch:batch+1,:,start:end],dA_cumsum[batch:batch+1,:,start:end],n_groups,seq_idx=None if seq_idx is None else seq_idx[batch:batch+1,sl])
   out[batch:batch+1,start:end]=value
 return out
