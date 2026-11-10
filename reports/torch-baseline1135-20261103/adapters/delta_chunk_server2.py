"""Six-output torch device baseline; independent FP64 CPU checker remains unchanged."""
import torch

def reference(q,k,v,beta,chunk_size):
    dtype=q.dtype
    q,k,v,beta=(x.float() for x in (q,k,v,beta))
    B,H,L,K=q.shape; V=v.shape[-1]; C=chunk_size
    assert L % C == 0
    eye=torch.eye(C,device=q.device,dtype=torch.float32).expand(B,H,C,C)
    state=torch.zeros((B,H,K,V),device=q.device,dtype=torch.float32)
    states=[state]; outputs=[]; inverses=[]; ws=[]; us=[]
    for start in range(0,L,C):
        qc=q[:,:,start:start+C];kc=k[:,:,start:start+C];vc=v[:,:,start:start+C];bc=beta[:,:,start:start+C]
        lower=torch.tril(bc[...,None]*(kc@kc.transpose(-1,-2)),diagonal=-1)
        # Nilpotent strict-lower L: (I+L)^-1=(I-L)(I+L^2)(I+L^4)...
        power=-lower; inverse=eye+power; degree=1
        while degree*2<C:
            power=power@power; inverse=inverse@(eye+power); degree*=2
        w=inverse@(bc[...,None]*kc);u=inverse@(bc[...,None]*vc)
        residual=u-w@state
        outputs.append(qc@state+torch.tril(qc@kc.transpose(-1,-2))@residual)
        state=state+kc.transpose(-1,-2)@residual
        states.append(state);inverses.append(inverse);ws.append(w);us.append(u)
    A=torch.cat(inverses,dim=2).to(dtype)
    return (torch.cat(outputs,dim=2).to(dtype).contiguous(),torch.stack(states,dim=2).contiguous(),A.contiguous(),A.clone().contiguous(),torch.cat(ws,dim=2).to(dtype).contiguous(),torch.cat(us,dim=2).to(dtype).contiguous())
