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
    def run(q,k,v,cuq,cuk,qs=None,ks=None,vs=None,cos=None,sin=None,sinks=None):
        if any(x is not None for x in (qs,ks,vs,cos,sin)) or sinks is None:raise ValueError('Exact unscaled sinks input required')
        qb,kb=cuq.cpu().tolist(),cuk.cpu().tolist()
        ql=[b-a for a,b in zip(qb,qb[1:])];kl=[b-a for a,b in zip(kb,kb[1:])]
        if len(ql)!=len(kl) or qb[0]!=0 or kb[0]!=0 or qb[-1]!=q.shape[0] or kb[-1]!=k.shape[0] or any(not (0<=a<=b) for a,b in zip(ql,kl)):
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
        if tuple(sinks.shape)!=(h,):raise ValueError('Invalid actual sink shape')
        scale=d**-0.5 if p.get('sm_scale') is None else p['sm_scale'];window=p.get('window_size_left',-1)
        return fn(q.reshape(q.shape[0],h*d).contiguous(),kk,vv,sinks,queries,table,lengths,scale,-1 if window<0 else window+1,h,hk).reshape_as(q)
    hashes=source_hashes(LIBRARIES/'sgl-kernel-npu'/host)
    hashes[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return Candidate(Implementation(run),'sgl_kernel_npu::attention_sinks_prefill_triton','Actual packed boundaries read per run, original KV sliced/padded into pages, original supplied sinks and empty requests, one original batched causal/window API; all metadata/layout timed.',hashes,timing_mode='host_roundtrip')
