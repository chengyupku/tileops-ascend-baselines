"""Engineering baselines: torch/torch-npu operations only, no custom device kernel."""
import torch
from adapters.execution_boundary import ExecutionBoundary

def bind(case, args, previous):
    import torch_npu
    op=case._call.signature.name;p=case._call.params
    desc={'implementation':'torch-npu eager','candidate':'v1','input_dependent_work_timed':True}
    if op=='RMSNormFwdOp':
        eps=p.get('eps');eps=torch.finfo(torch.float32).eps if eps is None else eps
        def run(x,weight):
            # Keep FP32 accumulation; public fused RMS primitive accepts stored dtype.
            if weight is None:weight=torch.ones(x.shape[-1],dtype=x.dtype,device=x.device)
            return torch_npu.npu_rms_norm(x,weight,epsilon=eps)[0]
        desc.update(change='native torch_npu RMSNorm replaces FP32 eager expansion',targets=[1,2,4])
    elif op=='MaxPool1dFwdOp' and p['kernel_size']==args[0].shape[-1] and p.get('padding',0)==0:
        def run(x):return x.amax(dim=-1,keepdim=True)
        desc.update(change='global pooling is one row maximum, matching TileOPs reduction',targets=[1,4])
    elif op=='MeanPoolingFwdOp':
        chunk=p['chunk_size'];dtype=getattr(torch,p.get('accum_dtype','float32'))
        def run(x,*metadata):
            if not metadata or metadata[0] is None:
                b,s,h,d=x.shape
                if s%chunk:raise ValueError('uniform tail needs explicit bounds')
                return x.reshape(b,s//chunk,chunk,h,d).mean(2,dtype=dtype).to(x.dtype)
            offsets,indices=metadata
            # Input metadata remains live: gather each indexed chunk, mask its tail.
            starts=offsets[indices[:,0].long()].long()+indices[:,1].long()*chunk
            ends=offsets[indices[:,0].long()+1].long()
            loc=starts[:,None]+torch.arange(chunk,device=x.device)
            valid=loc<ends[:,None]
            values=x[:,loc.clamp_max(x.shape[1]-1)].to(dtype)
            sums=(values*valid[None,:,:,None,None]).sum(2)
            return (sums/valid.sum(1)[None,:,None,None]).to(x.dtype)
        desc.update(change='batch all chunk reductions; avoid one Python/device call per chunk',targets=[1,4])
    elif op in ('RoPEFwdOp','LongRoPEFwdOp','RoPELlama31FwdOp','YaRNFwdOp','RoPENeoxPositionIdsFwdOp'):
        from workloads import rope
        from adapters.rope_pow_v10 import RopePower
        x=args[0];pos=op=='RoPENeoxPositionIdsFwdOp';inter=p.get('rope_layout','neox')=='interleaved'
        with RopePower():
            if pos:
                rd=p.get('rotary_dim') or x.shape[-1]
                cos,sin=rope.rope_frequency_tables(rd,p['max_position'],base=p['base'],dtype=x.dtype,device=x.device)
            else:
                rd=x.shape[-1];seq=x.shape[0] if p['input_layout']=='1d' else x.shape[1]
                fun={'RoPEFwdOp':rope.rope_frequency_tables,'LongRoPEFwdOp':rope.longrope_frequency_tables,'RoPELlama31FwdOp':rope.llama31_frequency_tables,'YaRNFwdOp':rope.yarn_frequency_tables}[op]
                kw={k:v for k,v in p.items() if k not in ('input_layout','rope_layout')}
                if 'rescale_factors' in case.workload.tensors:kw['rescale_factors']=case.workload.tensors['rescale_factors']
                cos,sin=fun(rd,seq,dtype=x.dtype,device=x.device,**kw)
        cos=cos.repeat_interleave(2,-1) if inter else torch.cat((cos,cos),-1)
        sin=sin.repeat_interleave(2,-1) if inter else torch.cat((sin,sin),-1)
        def run(x,positions=None):
            if pos:
                c=cos[positions.long()][:,None,None,:];s=sin[positions.long()][:,None,None,:]
                z=x[...,:rd].unsqueeze(1)
            else:
                z=x[None,:,None,:] if x.ndim==2 else x
                c=cos[None,:,None,:];s=sin[None,:,None,:]
            out=torch_npu.npu_rotary_mul(z,c,s,rotary_mode='interleave' if inter else 'half')
            out=out.squeeze(1) if pos else out.reshape_as(x)
            return torch.cat((out,x[...,rd:]),-1) if pos and rd<x.shape[-1] else out
        desc.update(change='cache constant table representation; native torch_npu rotary primitive; runtime positions gather stays timed',targets=[1,2,4],tables_outside_steady_state=True)
    elif op=='SSDChunkStateFwdOp':
        def run(x,B,dt,dA,seq_idx=None):
            b,seq,h,hd=x.shape;g=B.shape[2];n=B.shape[-1];chunks=dt.shape[2];q=dt.shape[-1]
            w=torch.exp(torch.clamp(dA[:,:,:,-1:]-dA,max=0.))*dt
            if seq_idx is not None:
                ids=seq_idx.reshape(b,chunks,q);w=w*((ids>=0)&(ids==ids[:,:,-1:]))[:,None]
            xc=x.float().reshape(b,chunks,q,h,hd).permute(0,1,3,4,2)
            bh=B.float().reshape(b,chunks,q,g,n).repeat_interleave(h//g,3).permute(0,1,3,2,4)
            out=torch.matmul(xc*w.permute(0,2,1,3).unsqueeze(-2),bh)
            return out
        desc.update(change='weighted batched contraction; no Q*P*N outer-product tensor',targets=[1,4])
    elif op=='SSDStatePassingFwdOp':
        def run(states,dA,initial=None):
            b,c,h,d=states.shape;aa=torch.exp(dA).permute(0,2,1).unsqueeze(-1);bb=states.float()
            step=1
            while step<c:
                bb=torch.cat((bb[:,:step],bb[:,step:]+aa[:,step:]*bb[:,:-step]),1)
                aa=torch.cat((aa[:,:step],aa[:,step:]*aa[:,:-step]),1);step*=2
            if initial is not None:bb=bb+aa*initial.float()[:,None]
            first=torch.zeros_like(states[:,0],dtype=torch.float32) if initial is None else initial.float()
            return torch.cat((first[:,None],bb[:,:-1]),1),bb[:,-1]
        desc.update(change='batched affine-prefix recurrence rather than one kernel chain per chunk',targets=[1,4])
    else:
        raise NotImplementedError('candidate not implemented: '+op)
    return ExecutionBoundary(run,tuple(args),lambda:None,desc)
