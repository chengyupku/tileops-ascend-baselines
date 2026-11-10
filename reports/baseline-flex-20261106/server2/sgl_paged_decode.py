from pathlib import Path
import hashlib
from benchmarks.ascend_baselines import Candidate,LIBRARIES,local_function,source_hashes

def build(case):
    import torch
    from benchmarks.api import Implementation
    p=case.params
    if p.get('pos_encoding_mode','none')!='none' or p.get('softcap') is not None or p.get('window_size_left',-1)!=-1 or p.get('window_size_right',-1)!=-1:raise ValueError('Unsupported exact paged contract')
    host='python/sgl_kernel_npu/sgl_kernel_npu/attention/decode_attention.py'
    fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.attention.decode_attention','decode_gqa',host)
    def run(q,k,v,table,lengths,cuq,*extra):
        if any(x is not None for x in extra):raise ValueError('No scale/rotation/sinks admitted')
        qb=cuq.cpu().tolist();ls=lengths.cpu().tolist()
        if qb!=list(range(q.shape[0]+1)) or len(ls)!=q.shape[0] or any(n<=0 or n>table.shape[1]*k.shape[1] for n in ls):raise ValueError('Actual singlequery/positivecache contract mismatch')
        out=torch.empty((q.shape[0],q.shape[1],v.shape[-1]),device=q.device,dtype=q.dtype)
        scale=q.shape[-1]**-0.5 if p.get('sm_scale') is None else p['sm_scale']
        fn(q,k,v,out,lengths,scale,k.shape[1],table)
        return out
    hashes=source_hashes(LIBRARIES/'sgl-kernel-npu'/host);hashes[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return Candidate(Implementation(run),'sgl_kernel_npu::decode_gqa','Original paged cache and table passed unchanged; actual one-query boundaries checked inside timedrun; one original native API for whole batch.',hashes,timing_mode='host_roundtrip')
