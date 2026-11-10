from pathlib import Path
import hashlib
from benchmarks.ascend_baselines import Candidate,LIBRARIES,local_function,source_hashes

def build(case):
    import torch
    from benchmarks.api import Implementation
    p=case.params
    if not p['is_causal'] or p.get('pos_encoding_mode','none')!='none' or p.get('softcap') is not None or p.get('window_size_right',-1)!=-1:raise ValueError('Unsupported exact paged prefill')
    host='python/sgl_kernel_npu/sgl_kernel_npu/attention/sinks_attention.py'
    fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.attention.sinks_attention','attention_sinks_prefill_triton',host)
    def run(q,k,v,table,lengths,cuq,qs=None,ks=None,vs=None,cos=None,sin=None,sinks=None):
        if any(x is not None for x in (qs,ks,vs,cos,sin)):raise ValueError('Unsupported extras')
        qb=cuq.cpu().tolist();ls=lengths.cpu().tolist();ql=[b-a for a,b in zip(qb,qb[1:])]
        if qb[0]!=0 or qb[-1]!=q.shape[0] or len(ql)!=len(ls) or any(not (0<=a<=b<=table.shape[1]*k.shape[1]) for a,b in zip(ql,ls)):raise ValueError('Actual query/cache bounds mismatch')
        h,d=q.shape[-2:];page=k.shape[1]
        if page&(page-1) or d!=v.shape[-1]:raise ValueError('Unadmitted page/feature layout')
        if sinks is None:sinks=torch.full((h,),float('-inf'),device=q.device,dtype=torch.float32)
        queries=torch.tensor(ql,device=q.device,dtype=torch.int32)
        scale=d**-0.5 if p.get('sm_scale') is None else p['sm_scale'];w=p.get('window_size_left',-1)
        return fn(q.reshape(q.shape[0],h*d).contiguous(),k,v,sinks,queries,table,lengths,scale,-1 if w<0 else w+1,h,k.shape[2]).reshape_as(q)
    hashes=source_hashes(LIBRARIES/'sgl-kernel-npu'/host);hashes[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return Candidate(Implementation(run),'sgl_kernel_npu::attention_sinks_prefill_triton','Original paged cache/table/lengths and supplied sinks passed unchanged; actual query metadata and optional negative-infinity sinks constructed in timedrun; one original batched causal/window API.',hashes,timing_mode='host_roundtrip')
