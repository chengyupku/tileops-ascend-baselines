"""Bound independent chunk dimension; original per-chunk arithmetic stays unchanged."""
def bind(module):
 original=module.ssd_chunk_state_fwd_ref
 def reference(x,Bmat,dt,dA_cumsum,n_groups,seq_idx=None):
  import torch
  q=dt.shape[-1];nc=dt.shape[2];out=[]
  for first in range(0,nc,4):
   last=min(first+4,nc);sl=slice(first*q,last*q)
   out.append(original(x[:,sl],Bmat[:,sl],dt[:,:,first:last],dA_cumsum[:,:,first:last],n_groups,None if seq_idx is None else seq_idx[:,sl]))
  return torch.cat(out,dim=1)
 module.ssd_chunk_state_fwd_ref=reference
