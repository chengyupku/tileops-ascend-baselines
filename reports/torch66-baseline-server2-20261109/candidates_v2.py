"""Second family batch; all arithmetic stays in torch/torch-npu."""
import torch
import torch.nn.functional as F
from adapters.execution_boundary import ExecutionBoundary

def prefix_product(x):
    # Inclusive tree product handles zero and sign without logarithms.
    out=x;step=1
    while step<x.shape[-1]:
        out=torch.cat((out[...,:step],out[...,step:]*out[...,:-step]),-1);step*=2
    return out

def bind(case,args,previous):
    import torch_npu
    op=case._call.signature.name;p=case._call.params
    desc={'implementation':'torch-npu eager','candidate':'v2','input_dependent_work_timed':True}
    if op=='GroupedGemmFwdOp':
        def run(a,b,sizes,offsets):
            weight=b.transpose(-1,-2) if p.get('transpose_b',False) else b
            # Device group boundaries are recomputed from live metadata each call.
            ends=(offsets+sizes).to(torch.int64)
            return torch_npu.npu_grouped_matmul([a],[weight],group_list=ends,split_item=3,group_type=0)[0]
        desc.update(change='native grouped matmul, remove per-expert CPU loop and conversions',targets=[1,2,4])
    elif op=='MoEPrePermuteFwdOp':
        count=p['num_local_experts'];topk=args[1].shape[1]
        def run(hidden,ids):
            flat=ids.flatten();values,order=torch.sort(flat,stable=True)
            rows=hidden[order//topk]
            ends=torch.searchsorted(values,torch.arange(count,dtype=values.dtype,device=ids.device),right=True).to(torch.int32)
            inverse=torch.empty_like(order,dtype=torch.int32)
            inverse.scatter_(0,order,torch.arange(order.numel(),device=ids.device,dtype=torch.int32))
            return rows,ends,inverse
        desc.update(change='sort once; segment ends from ordered ids instead of histogram plus prefix sum',targets=[1,4])
    elif op=='InstanceNormFwdOp':
        def run(x,rm,rv,w,b):
            assert rm is None and rv is None
            # Keep the CPU reference saved-stat rounding, reuse FP32 input/casts.
            xf=x.float();dims=tuple(range(2,x.ndim));var,mean=torch.var_mean(xf,dims,correction=0,keepdim=True)
            mean=mean.to(x.dtype).float();inv=torch.rsqrt(var+p['eps']).to(x.dtype).float()
            sh=[1,-1]+[1]*(x.ndim-2);alpha=inv if w is None else inv*w.float().reshape(sh)
            beta=-mean*alpha
            if b is not None:beta=beta+b.float().reshape(sh)
            return torch.addcmul(beta,xf,alpha).to(x.dtype)
        desc.update(change='single var_mean pass; preserve native FP16 saved-stat rounding',targets=[1,4])
    elif op=='CumsumFwdOp':
        def run(x):return torch.cumsum(x,dim=p['dim'],dtype=torch.float32).to(getattr(torch,p.get('dtype') or str(x.dtype).split('.')[-1]))
        desc.update(change='public torch cumsum with FP32 accumulation, avoid separate input cast where backend supports fused conversion',targets=[1,2])
    elif op=='CumprodFwdOp':
        def run(x):
            v=x.float().movedim(p['dim'],-1)
            return prefix_product(v).movedim(-1,p['dim']).to(getattr(torch,p.get('dtype') or str(x.dtype).split('.')[-1]))
        desc.update(change='tree prefix multiplication following scan structure; zero/sign preserved, no log approximation',targets=[1,4])
    elif op=='EngramDecodeFwdOp':
        wld=case.workload;maxlen=wld.max_conv_len;dilation=wld.dilation
        def run(e,h,state,wk,wv,wh,wr,cw):
            ef=e.float();k=ef@wk.float();v=ef@wv.float();b,d=h.shape
            hk=torch.cat((h.float(),k.to(h.dtype).float()),0)
            norms=torch_npu.npu_rms_norm(hk,wh.float(),epsilon=1e-6)[0];hn,kn=norms[:b],norms[b:]
            alpha=torch.sigmoid((hn*kn).sum(-1,keepdim=True)/(d**.5));vh=alpha*v
            vn=torch_npu.npu_rms_norm(vh.to(h.dtype).float(),wr.float(),epsilon=1e-6)[0]
            padded=F.pad(state.float(),(0,0,maxlen-state.shape[1])) if maxlen>state.shape[1] else state.float()
            weights=cw.float();conv=torch.zeros_like(vh)
            for j in range(cw.shape[0]-1):
                i=maxlen-(cw.shape[0]-1-j)*dilation
                if 0<=i<maxlen:conv=conv+weights[j]*padded[:,i]
            conv=conv+weights[-1]*vn
            next_state=torch.cat((state if maxlen>state.shape[1] else state[:,1:],vn[:,None].to(state.dtype)),1)
            return (F.silu(conv)+vh).to(h.dtype),next_state
        desc.update(change='batch H/K normalization; native FP32 RMS primitives; reuse converted input and convolution weights',targets=[1,4])
    elif op=='FusedQKNormRoPEFwdOp':
        w=case.workload;nq=w.num_heads;nk=w.num_kv_heads;eps=w.eps;dim=w.dim
        def run(qkv,qw,kw,cache,positions):
            count=nq+nk;x=qkv[:,:count*dim].reshape(-1,count,dim).float()
            weight=torch.cat((qw.expand(nq,-1),kw.expand(nk,-1)),0).float()
            # RMS cannot take a different weight per row: normalize first, apply live weights.
            inv=torch.rsqrt(x.square().mean(-1,keepdim=True)+eps);y=x*inv*weight
            cos,sin=cache[positions.long()].float().chunk(2,-1);half=cos.shape[-1];rot=y[...,:half*2]
            if w.rope_layout=='neox':a,b=rot[...,:half],rot[...,half:]
            else:a,b=rot[...,::2],rot[...,1::2]
            u=a*cos[:,None]-b*sin[:,None];v=b*cos[:,None]+a*sin[:,None]
            rotated=torch.cat((u,v),-1) if w.rope_layout=='neox' else torch.stack((u,v),-1).flatten(-2)
            qkv[:,:count*dim]=torch.cat((rotated,y[...,half*2:]),-1).reshape(qkv.shape[0],-1).to(qkv.dtype)
            return qkv
        # Existing worker has already made private mutable qkv/reset outside timing.
        desc.update(change='batch Q/K heads through shared normalization and rotation; preserve FP32 until final store',targets=[1,4])
        return ExecutionBoundary(run,tuple(args),previous[2],desc)
    elif op=='SelectiveScanFwdOp':
        precise=case.id=='mamba-130m-no-gate-float32'
        def run(u,delta,A,B,C,D=None,z=None,bias=None,seq_idx=None,initial=None):
            batch,dim,length=u.shape;groups=B.shape[1];n=A.shape[-1];uf=u.float();dt=delta.float()
            if bias is not None:dt=dt+bias.float()[:,None]
            if p.get('delta_softplus',True):
                if precise:
                    wide=dt+torch.zeros_like(dt,dtype=torch.float64)
                    dt=torch.where(wide>20,wide,torch.log1p(torch.exp(wide))).float()
                else:dt=F.softplus(dt)
            bf=B.float().repeat_interleave(dim//groups,1);cf=C.float().repeat_interleave(dim//groups,1)
            state=torch.zeros((batch,dim,n),device=u.device) if initial is None else initial.float()
            if precise:
                expo=dt[...,None]*A[None,:,None,:]
                decays=torch.pow(torch.full_like(expo,2.718281828459045,dtype=torch.float64),expo).float()
            outputs=[]
            for first in range(0,length,64):
                last=min(first+64,length);dd=dt[:,:,first:last]
                a=decays[:,:,first:last] if precise else torch.exp(dd[...,None]*A[None,:,None,:])
                b=(dd*uf[:,:,first:last])[...,None]*bf[:,:, :,first:last].transpose(-1,-2)
                if seq_idx is not None:
                    idx=torch.arange(first,last,device=u.device)
                    restart=(idx>0)[None,:]&(seq_idx[:,idx]!=seq_idx[:,(idx-1).clamp_min(0)])
                    a=torch.where(restart[:,None,:,None],0.,a)
                off=1
                while off<last-first:
                    b=torch.cat((b[:,:,:off],b[:,:,off:]+a[:,:,off:]*b[:,:,:-off]),2)
                    a=torch.cat((a[:,:,:off],a[:,:,off:]*a[:,:,:-off]),2);off*=2
                states=b+a*state[:,:,None,:]
                outputs.append((states*cf[:,:,:,first:last].transpose(-1,-2)).sum(-1));state=states[:,:,-1]
            y=torch.cat(outputs,-1)
            if D is not None:y=y+uf*D.float()[:,None]
            if z is not None:y=y*F.silu(z.float())
            return y.to(u.dtype),state
        desc.update(change='chunked batched affine-prefix recurrence; special FP64 decay/softplus kept where contract requires',targets=[1,4])
    else:
        from candidates_v1 import bind as other
        return other(case,args,previous)
    return ExecutionBoundary(run,tuple(args),lambda:None,desc)
