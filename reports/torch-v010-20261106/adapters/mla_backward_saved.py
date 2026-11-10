"""MLA backward uses supplied forward LSE; QK probability recomputation remains timed."""
def reference(do,q,k,v,o,lse,*offsets,is_causal=True,sm_scale=None):
 import torch
 dense=q.ndim==4
 if dense:
  batch,sq=q.shape[:2];sk=k.shape[1]
  bq=list(range(0,batch*sq+1,sq));bk=list(range(0,batch*sk+1,sk))
  qf,kf,vf,dof=(t.flatten(0,1) for t in (q,k,v,do))
  lf=lse.transpose(0,1).reshape(q.shape[2],batch*sq)
 else:
  qf,kf,vf,dof=q,k,v,do;lf=lse
  bq,bk=(x.tolist() for x in offsets)
 scale=sm_scale if sm_scale is not None else q.shape[-1]**-.5
 dq,dk,dv=(torch.zeros(t.shape,dtype=torch.float32,device=t.device) for t in (qf,kf,vf))
 for q0,q1,k0,k1 in zip(bq,bq[1:],bk,bk[1:]):
  key,value=kf[k0:k1].float(),vf[k0:k1].float()
  pos=torch.arange(q1-q0,device=q.device)+(k1-k0)-(q1-q0)
  ki=torch.arange(k1-k0,device=q.device)
  for h0 in range(0,qf.shape[1],8):
   heads=slice(h0,h0+8)
   for r0 in range(0,q1-q0,1024):
    rows=slice(r0,min(r0+1024,q1-q0));absolute=slice(q0+r0,min(q0+r0+1024,q1))
    scores=torch.einsum('shd,nhd->hsn',qf[absolute,heads].float(),key[:,heads])*scale
    if is_causal:scores=scores.masked_fill((ki[None,:]>pos[rows,None])[None],float('-inf'))
    probs=torch.exp(scores-lf[heads,absolute].float().unsqueeze(-1)).nan_to_num(0.)
    grad=dof[absolute,heads].float()
    dp=torch.einsum('shd,nhd->hsn',grad,value[:,heads])
    ds=probs*(dp-(probs*dp).sum(-1,keepdim=True))
    dv[k0:k1,heads]+=torch.einsum('hsn,shd->nhd',probs,grad)
    dk[k0:k1,heads]+=torch.einsum('hsn,shd->nhd',ds,qf[absolute,heads].float())
    dq[absolute,heads]=torch.einsum('hsn,nhd->shd',ds,key[:,heads])
 return tuple(g.reshape(t.shape).to(q.dtype) for g,t in zip((dq*scale,dk*scale,dv),(q,k,v)))
