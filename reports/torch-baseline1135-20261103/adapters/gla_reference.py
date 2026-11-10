"""Independent direct GLA recurrence, FP64 on CPU and FP32 for NPU eager."""
import torch

def reference(q,k,v,g,initial_state=None,cu_seqlens=None,cu_seqlens_cpu=None,*,scale=None):
 dtype=q.dtype;device=q.device;acc=torch.float64 if device.type=='cpu' else torch.float32
 batch,length,heads,dk=q.shape;dv=v.shape[-1]
 scale=dk**-.5 if scale is None else scale
 qq,kk,vv,gg=(x.to(acc) for x in (q,k,v,g))
 offsets=[0,length] if cu_seqlens is None else cu_seqlens.cpu().tolist()
 sequences=batch if cu_seqlens is None else len(offsets)-1
 out=torch.empty((batch,length,heads,dv),dtype=dtype,device=device);states=[]
 for seq in range(sequences):
  b=seq if cu_seqlens is None else 0
  begin,end=(0,length) if cu_seqlens is None else offsets[seq:seq+2]
  state=torch.zeros((heads,dk,dv),dtype=acc,device=device) if initial_state is None else initial_state[seq].to(acc).clone()
  for t in range(begin,end):
   state=gg[b,t].exp()[...,None]*state+kk[b,t][...,None]*vv[b,t][:,None,:]
   out[b,t]=(scale*(qq[b,t][...,None]*state).sum(dim=-2)).to(dtype)
  states.append(state.float())
 return out,torch.stack(states)
