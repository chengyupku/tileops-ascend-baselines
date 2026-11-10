"""All-six-output NPU PyTorch reference; qualify against independent CPU solves."""
import torch

def reference(q,k,v,beta,chunk_size):
    B,H,L,K=q.shape;V=v.shape[-1];C=chunk_size;N=L//C;dtype=q.dtype
    qq,kk,vv=(x.float().reshape(B,H,N,C,x.shape[-1]) for x in (q,k,v))
    bb=beta.float().reshape(B,H,N,C)
    gram=kk@kk.transpose(-1,-2)
    lower=-torch.tril(bb[...,None]*gram,diagonal=-1)
    eye=torch.eye(C,device=q.device,dtype=torch.float32)
    inverse=eye.expand_as(lower).clone();power=lower;width=1
    while width<C:
        inverse=inverse+power@inverse
        power=power@power;width*=2
    w=inverse@(bb[...,None]*kk);u=inverse@(bb[...,None]*vv)
    state=torch.zeros(B,H,K,V,device=q.device,dtype=torch.float32);states=[state];outputs=[]
    for n in range(N):
        residual=u[:,:,n]-w[:,:,n]@state
        causal=torch.tril(qq[:,:,n]@kk[:,:,n].transpose(-1,-2))
        outputs.append(qq[:,:,n]@state+causal@residual)
        state=state+kk[:,:,n].transpose(-1,-2)@residual;states.append(state)
    a=inverse.reshape(B,H,L,C).to(dtype)
    return (torch.cat(outputs,dim=2).to(dtype),torch.stack(states,dim=2),a,a.clone(),w.reshape(B,H,L,K).to(dtype),u.reshape(B,H,L,V).to(dtype))
