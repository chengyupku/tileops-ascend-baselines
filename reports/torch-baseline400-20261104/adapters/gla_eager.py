"""Stable, chunked PyTorch GLA counterpart, qualified against the FP64 recurrence.

Pairwise cumulative-gate differences avoid exp(-prefix) overflow. No kernel
under test is used here. Chunking bounds both eager launch count and storage.
"""
import torch

def reference(q,k,v,g,initial_state=None,cu_seqlens=None,cu_seqlens_cpu=None,*,scale=None):
 b,s,h,dk=q.shape;dv=v.shape[-1];scale=dk**-.5 if scale is None else scale
 offsets=None if cu_seqlens is None else cu_seqlens.cpu().tolist()
 sequences=b if offsets is None else len(offsets)-1
 outputs=torch.empty_like(v);final=[]
 for seq in range(sequences):
  batch=seq if offsets is None else 0
  start,end=(0,s) if offsets is None else offsets[seq:seq+2]
  state=torch.zeros((h,dk,dv),dtype=torch.float32,device=q.device) if initial_state is None else initial_state[seq].float().clone()
  for first in range(start,end,32):
   last=min(first+32,end);n=last-first
   qc,kc,vc,gc=(x[batch,first:last].float().transpose(0,1) for x in (q,k,v,g))
   prefix=gc.cumsum(dim=1)
   mask=torch.arange(n,device=q.device)[:,None]>=torch.arange(n,device=q.device)[None,:]
   difference=prefix[:,:,None,:]-prefix[:,None,:,:]
   difference=torch.where(mask[None,:,:,None],difference,0.)
   weights=(qc[:,:,None,:]*kc[:,None,:,:]*difference.exp()).sum(dim=-1)
   weights=weights.masked_fill(~mask[None,:,:],0.)
   value=((qc*prefix.exp())@state + weights@vc)*scale
   outputs[batch,first:last]=value.transpose(0,1).to(q.dtype)
   decay=prefix[:,-1,:]
   state=decay.exp()[...,None]*state+(kc*(decay[:,None,:]-prefix).exp()).transpose(-1,-2)@vc
  final.append(state)
 return outputs,torch.stack(final)
