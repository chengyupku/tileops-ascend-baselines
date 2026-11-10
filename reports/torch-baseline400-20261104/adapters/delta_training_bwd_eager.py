"""Independent NPU autograd reference, qualified against explicit CPU derivatives."""
import torch
from adapters.delta_training_eager import reference as forward

def reference(do,q,k,v,beta,S,Aw,Au,w,u,*,chunk_size):
    with torch.enable_grad():
        variables=tuple(x.detach().float().requires_grad_(True) for x in (q,k,v,beta))
        output=forward(*variables,chunk_size)[0]
        gradients=torch.autograd.grad(output,variables,grad_outputs=do.float())
    return tuple(x.to(q.dtype) for x in gradients)
