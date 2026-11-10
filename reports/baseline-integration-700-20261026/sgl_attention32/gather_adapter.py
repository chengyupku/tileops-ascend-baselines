from pathlib import Path
from benchmarks.ascend_baselines import Candidate,LIBRARIES,local_function,source_hashes

def build(case):
    import torch
    from benchmarks.api import Implementation
    host='python/sgl_kernel_npu/sgl_kernel_npu/sparsity_driven_kv_offload/ops.py'
    fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.sparsity_driven_kv_offload.ops','unidex_copy_inplace',host)
    def run(dst,cache,table,cu,starts=None,scale=None):
        assert scale is None
        offsets=cu.cpu().tolist();bases=[0]*(len(offsets)-1) if starts is None else starts.cpu().tolist()
        src=[]
        for i,(begin,end) in enumerate(zip(offsets[:-1],offsets[1:])):
            pos=torch.arange(bases[i],bases[i]+end-begin,device=cache.device,dtype=torch.int64)
            src.append(table[i,pos//cache.shape[1]].long()*cache.shape[1]+pos%cache.shape[1])
        src=torch.cat(src).contiguous();out=torch.empty_like(dst)
        ix=torch.arange(dst.shape[0],device=dst.device,dtype=torch.int64)
        valid=torch.ones(dst.shape[0],device=dst.device,dtype=torch.bool)
        return fn(cache,out,src,ix,valid,2,1)
    root=LIBRARIES/'sgl-kernel-npu'
    return Candidate(Implementation(run),'sgl_kernel_npu.sparsity_driven_kv_offload.ops.unidex_copy_inplace','Original complete Python API and native registered binary; actual page/sequence/start metadata maps full rows. No compatibility bridge.',source_hashes(Path(__file__),root/host,root/'python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so',root/'csrc/sparsity_driven_kv_offload/unidex_copy/op_host/unidex_copy.cpp',root/'csrc/sparsity_driven_kv_offload/unidex_copy/op_kernel/unidex_copy_kernel.cpp'),timing_mode='host_roundtrip')
PROVIDERS={'PagedKVCacheGatherFwdOp':build}
