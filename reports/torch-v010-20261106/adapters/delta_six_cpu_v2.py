"""Independent CPU triangular solves for all declared DeltaNetChunk outputs."""
import torch
@torch.no_grad()
def reference(q,k,v,beta,chunk_size):
 assert q.device.type=='cpu'
 dtype=q.dtype;B,H,L,K=q.shape;V=v.shape[-1];C=chunk_size
 q,k,v,beta=(x.double() for x in (q,k,v,beta))
 state=torch.zeros(B,H,K,V,dtype=torch.float64);states=[state];outputs=[];inverses=[];ws=[];us=[]
 eye=torch.eye(C,dtype=torch.float64).expand(B,H,C,C)
 for i in range(0,L,C):
  qc,kc,vc,bc=q[:,:,i:i+C],k[:,:,i:i+C],v[:,:,i:i+C],beta[:,:,i:i+C]
  matrix=eye+torch.tril(bc[...,None]*(kc@kc.transpose(-1,-2)),diagonal=-1)
  inverse=torch.linalg.solve_triangular(matrix,eye,upper=False,unitriangular=True)
  w=inverse@(bc[...,None]*kc);u=inverse@(bc[...,None]*vc)
  residual=u-w@state
  outputs.append(qc@state+torch.tril(qc@kc.transpose(-1,-2))@residual)
  state=state+kc.transpose(-1,-2)@residual
  states.append(state);inverses.append(inverse);ws.append(w);us.append(u)
 a=torch.cat(inverses,dim=2).to(dtype)
 return torch.cat(outputs,dim=2).to(dtype),torch.stack(states,dim=2).float(),a,a.clone(),torch.cat(ws,dim=2).to(dtype),torch.cat(us,dim=2).to(dtype)
