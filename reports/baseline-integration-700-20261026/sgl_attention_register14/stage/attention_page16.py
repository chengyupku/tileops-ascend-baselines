from pathlib import Path
import math

def _build(case,paged=False):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,LIBRARIES,UnsupportedCase,local_function,source_hashes
    p=case.params
    if p.get('softcap') not in (None,0):
        raise UnsupportedCase('Original SGL decode attention has no logits softcap.')
    if case.inputs[0].dtype not in (torch.float16,torch.bfloat16):
        raise UnsupportedCase('This original SGL unquantized decode API has no external FP8 Q/K/V scale inputs.')
    root=LIBRARIES/'sgl-kernel-npu'
    host='python/sgl_kernel_npu/sgl_kernel_npu/attention/decode_attention.py'
    fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.attention.decode_attention','decode_gqa',host)
    sinkhost='python/sgl_kernel_npu/sgl_kernel_npu/attention/sinks_attention.py'
    sinkfn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.attention.sinks_attention','attention_sinks_triton',sinkhost)
    prefillfn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.attention.sinks_attention','attention_sinks_prefill_triton',sinkhost)
    ropehost='python/sgl_kernel_npu/sgl_kernel_npu/norm/fused_rope_qk_mqa.py'
    rotate=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.norm.fused_rope_qk_mqa','fused_rope_qk_mqa',ropehost)
    def request(q,k,v,cos,sin,sinks):
        nq,heads,dim=q.shape
        nk,kvheads,vdim=v.shape
        if p.get('pos_encoding_mode','none')!='none':
            rd=p.get('rotary_dim') or dim
            cache=torch.cat((cos,sin),dim=-1).float()
            q,_=rotate(q.float().contiguous(),q.float().contiguous(),cache[nk-nq:nk],rd,p.get('rope_layout','neox')=='neox')
            k,_=rotate(k.float().contiguous(),k.float().contiguous(),cache[:nk],rd,p.get('rope_layout','neox')=='neox')
            q,k=q.to(v.dtype),k.to(v.dtype)
        group=heads//kvheads
        padgroup=1<<(group-1).bit_length()
        scale=p.get('sm_scale')
        scale=dim**-0.5 if scale is None else scale
        if 1<nq<=nk and dim==vdim and p.get('is_causal',True):
            pages=math.ceil(nk/16)
            kk=F.pad(k,(0,0,0,0,0,pages*16-nk)).reshape(pages,16,kvheads,dim)
            vv=F.pad(v,(0,0,0,0,0,pages*16-nk)).reshape(pages,16,kvheads,dim)
            table=torch.arange(pages,device=q.device,dtype=torch.int32).reshape(1,pages)
            lengths=torch.tensor([nk],device=q.device,dtype=torch.int32)
            queries=torch.tensor([nq],device=q.device,dtype=torch.int32)
            ss=torch.full((heads,),float('-inf'),device=q.device,dtype=torch.float32) if sinks is None else sinks
            window=p.get('window_size_left',-1)
            window=-1 if window<0 else window+1
            return prefillfn(q.reshape(nq,-1).contiguous(),kk,vv,ss,queries,table,lengths,scale,window,heads,kvheads).reshape(nq,heads,vdim)
        result=[]
        for i in range(nq):
            pos=nk-nq+i
            begin=max(0,pos-p['window_size_left']) if p.get('window_size_left',-1)>=0 else 0
            end=min(nk,pos+1) if p.get('is_causal',True) else nk
            if p.get('window_size_right',-1)>=0:end=min(end,pos+p['window_size_right']+1)
            if end<=begin:
                result.append(torch.zeros((1,heads,vdim),device=q.device,dtype=q.dtype));continue
            n=end-begin
            pages=math.ceil(n/128)
            kk=F.pad(k[begin:end],(0,0,0,0,0,pages*128-n)).reshape(pages,128,kvheads,dim)
            vv=F.pad(v[begin:end],(0,0,0,0,0,pages*128-n)).reshape(pages,128,kvheads,vdim)
            table=torch.arange(pages,device=q.device,dtype=torch.int32).reshape(1,pages)
            lengths=torch.tensor([n],device=q.device,dtype=torch.int32)
            if sinks is not None:
                if dim!=vdim:raise UnsupportedCase('SGL sink attention requires equal Q/K/V feature width.')
                qq=F.pad(q[i:i+1].reshape(1,kvheads,group,dim),(0,0,0,padgroup-group)).reshape(1,-1).contiguous()
                ss=F.pad(sinks.reshape(kvheads,group),(0,padgroup-group)).reshape(-1).contiguous()
                out=sinkfn(qq,kk,vv,ss,table,lengths,scale,-1,kvheads*padgroup,kvheads)
                result.append(out.reshape(1,kvheads,padgroup,vdim)[:,:,:group].reshape(1,heads,vdim))
            else:
                qq=F.pad(q[i:i+1].reshape(1,kvheads,group,dim),(0,0,0,padgroup-group)).reshape(1,kvheads*padgroup,dim)
                out=torch.empty((1,kvheads*padgroup,vdim),device=q.device,dtype=q.dtype)
                fn(qq,kk,vv,out,lengths,scale,128,table)
                result.append(out.reshape(1,kvheads,padgroup,vdim)[:,:,:group].reshape(1,heads,vdim))
        return torch.cat(result,dim=0) if result else torch.empty((0,heads,vdim),device=q.device,dtype=q.dtype)
    def run_varlen(q,k,v,cuq,cuk,qs=None,ks=None,vs=None,cos=None,sin=None,sinks=None):
        if any(x is not None for x in (qs,ks,vs)):
            raise UnsupportedCase('Unquantized SGL API has no supplied scale parameters; input dequantization is not a layout adaptation.')
        qb,kb=cuq.cpu().tolist(),cuk.cpu().tolist()
        out=[request(q[qb[i]:qb[i+1]],k[kb[i]:kb[i+1]],v[kb[i]:kb[i+1]],cos,sin,sinks) for i in range(len(qb)-1)]
        return torch.cat(out,dim=0)
    def run_paged(q,kp,vp,table,lengths,cuq,qs=None,ks=None,vs=None,cos=None,sin=None,sinks=None):
        if any(x is not None for x in (qs,ks,vs)):
            raise UnsupportedCase('Unquantized SGL API has no supplied FP8 scales.')
        qb,kl=cuq.cpu().tolist(),lengths.cpu().tolist()
        out=[]
        for i,n in enumerate(kl):
            pages=table[i,:math.ceil(n/kp.shape[1])].long()
            k,v=[x[pages].flatten(0,1)[:n] for x in (kp,vp)]
            out.append(request(q[qb[i]:qb[i+1]],k,v,cos,sin,sinks))
        return torch.cat(out,dim=0)
    return Candidate(Implementation(run_paged if paged else run_varlen),'sgl_kernel_npu::decode_gqa / attention_sinks_triton / attention_sinks_prefill_triton + native RoPE',
                     'Read packed boundaries and actual lengths per call; gather paged KV when needed, select original causal prefill API for multi-query equal-feature rows (negative-infinity sink for no sink), otherwise decode selected causal/window intervals. Native optional sinks/rotation; pad pages/independent heads only.',
                     source_hashes(Path(__file__),root/host,root/sinkhost,root/ropehost),timing_mode='host_roundtrip')

def varlen(case):return _build(case,False)

def paged(case):return _build(case,True)

def mla_decode(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,LIBRARIES,local_function,source_hashes
    host='python/sgl_kernel_npu/sgl_kernel_npu/attention/decode_attention.py'
    fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.attention.decode_attention','decode_mla',host)
    q,qp,k,kp=case.inputs
    b,n,h,d=k.shape
    pages=math.ceil(n/128)
    table=torch.arange(b*pages,device=q.device,dtype=torch.int32).reshape(b,pages)
    lengths=torch.full((b,),n,device=q.device,dtype=torch.int32)
    def run(q,qp,k,kp):
        kk=F.pad(k,(0,0,0,0,0,pages*128-n)).contiguous().reshape(b*pages,128,h,d)
        pp=F.pad(kp,(0,0,0,0,0,pages*128-n)).contiguous().reshape(b*pages,128,h,kp.shape[-1])
        out=torch.empty_like(q)
        fn(torch.cat((q,qp),dim=-1),kk,pp,out,lengths,(d+qp.shape[-1])**-0.5,128,table)
        return out
    return Candidate(Implementation(run),'sgl_kernel_npu.attention.decode_attention.decode_mla',
                     'Concatenate supplied latent/positional queries; pad and view supplied independent latent/positional keys as pages; native full MLA and actual lengths, including tail pages.',
                     source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/host))

def mha_paged(case):
    from dataclasses import replace
    from types import SimpleNamespace
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import source_hashes
    q,k,v,lengths,table=case.inputs
    page=case.params['page_size']
    b,sq,h,d=q.shape
    qflat=q.reshape(b*sq,h,d)
    cu=torch.arange(b+1,device=q.device,dtype=torch.int32)*sq
    def pages(x):return x.reshape(-1,page,h,d)
    native=paged(SimpleNamespace(inputs=(qflat,pages(k),pages(v),table,lengths,cu,None,None,None,None,None,None),params=case.params))
    def run(q,k,v,lengths,table):
        cu=torch.arange(q.shape[0]+1,device=q.device,dtype=torch.int32)*q.shape[1]
        return native.implementation.run(q.reshape(-1,h,d),pages(k),pages(v),table,lengths,cu).reshape_as(q)
    return replace(native,implementation=Implementation(run),source_sha256=native.source_sha256|source_hashes(Path(__file__)),
                   adaptation='Dense batch queries become packed rows with explicit offsets; flat physical KV cache becomes page views; original SGL GQA with one query per KV head.')
PROVIDERS={"MLADecodeWithKVCacheFwdOp":mla_decode,"MHADecodePagedWithKVCacheFwdOp":mha_paged,"GQAVarlenFwdOp":varlen}
