from pathlib import Path
import hashlib
from benchmarks.ascend_baselines import Candidate, LIBRARIES, local_function, source_hashes

def build(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    p=case.params
    if not p['is_causal'] or p.get('pos_encoding_mode','none')!='none' or p.get('softcap') is not None or p.get('window_size_right',-1)!=-1:
        raise ValueError('Unsupported exact batched prefill contract')
    host='python/sgl_kernel_npu/sgl_kernel_npu/attention/sinks_attention.py'
    fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.attention.sinks_attention','attention_sinks_prefill_triton',host)
    def run(q,k,v,cuq,cuk,*extra):
        if any(x is not None for x in extra):raise ValueError('No extra scale/RoPE/sink admitted')
        qb,kb=cuq.cpu().tolist(),cuk.cpu().tolist()
        ql=[b-a for a,b in zip(qb,qb[1:])];kl=[b-a for a,b in zip(kb,kb[1:])]
        if len(ql)!=len(kl) or qb[0]!=0 or kb[0]!=0 or qb[-1]!=q.shape[0] or kb[-1]!=k.shape[0] or any(not (0<a<=b) for a,b in zip(ql,kl)):
            raise ValueError('Invalid actual prefill boundaries')
        h,d=q.shape[-2:];hk=k.shape[-2]
        if d!=v.shape[-1] or d not in (64,128) or h%hk:raise ValueError('Unsupported feature layout')
        pages=[(n+127)//128 for n in kl];width=max(pages);tables=[];offset=0
        for n in pages:
            tables.append(list(range(offset,offset+n))+[offset]*(width-n));offset+=n
        def pack(x):
            return torch.cat([F.pad(x[kb[j]:kb[j+1]],(0,0,0,0,0,pages[j]*128-kl[j])).reshape(pages[j],128,hk,d) for j in range(len(kl))],dim=0).contiguous()
        kk,vv=pack(k),pack(v)
        table=torch.tensor(tables,device=q.device,dtype=torch.int32)
        lengths=torch.tensor(kl,device=q.device,dtype=torch.int32)
        queries=torch.tensor(ql,device=q.device,dtype=torch.int32)
        sinks=torch.full((h,),float('-inf'),device=q.device,dtype=torch.float32)
        scale=d**-0.5 if p.get('sm_scale') is None else p['sm_scale'];window=p.get('window_size_left',-1)
        return fn(q.reshape(q.shape[0],h*d).contiguous(),kk,vv,sinks,queries,table,lengths,scale,-1 if window<0 else window+1,h,hk).reshape_as(q)
    hashes=source_hashes(LIBRARIES/'sgl-kernel-npu'/host)
    hashes[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return Candidate(Implementation(run),'sgl_kernel_npu::attention_sinks_prefill_triton','Actual packed boundaries read per run, original KV sliced/padded into pages, one original batched causal/window API; all metadata/layout timed.',hashes,timing_mode='host_roundtrip')


def build_rope(case):
    import torch
    from types import SimpleNamespace
    from dataclasses import replace
    from benchmarks.api import Implementation
    from benchmarks.ascend_rope_attention_cases import _rotation
    import benchmarks.ascend_rope_attention_cases as rotation_adapter
    p=case.params
    rd=p.get('rotary_dim') or p['q'][0][-1]
    rotate,rhash,cleanup=_rotation('sgl-kernel-npu',p.get('rope_layout','neox')=='neox',rd)
    native=build(SimpleNamespace(params=dict(p,pos_encoding_mode='none',rotary_dim=None)))
    def run(q,k,v,cuq,cuk,qs=None,ks=None,vs=None,cos=None,sin=None,*extra):
        if any(x is not None for x in (qs,ks,vs,*extra)) or cos is None or sin is None:raise ValueError('Exact unscaled RoPE required')
        qb,kb=cuq.cpu().tolist(),cuk.cpu().tolist()
        if len(qb)!=len(kb) or qb[0]!=0 or kb[0]!=0 or qb[-1]!=q.shape[0] or kb[-1]!=k.shape[0]:raise ValueError('Invalid packed bounds')
        ql=[b-a for a,b in zip(qb,qb[1:])];kl=[b-a for a,b in zip(kb,kb[1:])]
        if any(not (0<a<=b<=cos.shape[0]) for a,b in zip(ql,kl)):raise ValueError('Invalid positions')
        qp=torch.cat([torch.arange(b-a,b,device=q.device) for a,b in zip(ql,kl)])
        kp=torch.cat([torch.arange(b,device=q.device) for b in kl])
        qq=rotate(q.unsqueeze(0),cos[qp],sin[qp]).squeeze(0)
        kk=rotate(k.unsqueeze(0),cos[kp],sin[kp]).squeeze(0)
        return native.implementation.run(qq,kk,v,cuq,cuk)
    return replace(native,implementation=Implementation(run),source_sha256=native.source_sha256|rhash|source_hashes(Path(rotation_adapter.__file__)),cleanup=cleanup,api='sgl native RoPE twice + native batched prefill',adaptation='Read actual packed bounds and gather supplied per-request position tables each run; two original rotations across packed rows plus one original batched prefill; all timed.')
