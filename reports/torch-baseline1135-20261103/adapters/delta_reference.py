"""Independent direct ungated delta recurrence, FP64 on CPU."""
import torch

def reference(q,k,v,beta,initial_state=None,cu_seqlens=None,cu_seqlens_cpu=None,*,
              scale=None,use_qk_l2norm_in_kernel=False):
    dtype=q.dtype;B,S,H,K=q.shape;V=v.shape[-1]
    acc=torch.float64 if q.device.type=='cpu' else torch.float32
    qq,kk,vv,bb=(x.to(acc) for x in (q,k,v,beta))
    if use_qk_l2norm_in_kernel:
        qq=qq*torch.rsqrt((qq*qq).sum(-1,keepdim=True)+1e-6)
        kk=kk*torch.rsqrt((kk*kk).sum(-1,keepdim=True)+1e-6)
    scale=K**-.5 if scale is None else scale
    offsets=None if cu_seqlens is None else cu_seqlens.cpu().tolist()
    count=B if offsets is None else len(offsets)-1
    output=torch.empty_like(v);states=[]
    for n in range(count):
        b=n if offsets is None else 0
        start,end=(0,S) if offsets is None else offsets[n:n+2]
        state=torch.zeros(H,K,V,dtype=acc,device=q.device) if initial_state is None else initial_state[n].to(acc).clone()
        for t in range(start,end):
            residual=bb[b,t,:,None]*(vv[b,t]-(state*kk[b,t,:, :,None]).sum(-2))
            state=state+kk[b,t,:,:,None]*residual[:,None,:]
            output[b,t]=(scale*(qq[b,t,:,:,None]*state).sum(-2)).to(dtype)
        states.append(state.float())
    return output,torch.stack(states)
