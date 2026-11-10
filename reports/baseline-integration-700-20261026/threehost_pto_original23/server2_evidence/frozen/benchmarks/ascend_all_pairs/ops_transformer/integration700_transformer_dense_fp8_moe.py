"""Original V5 no-group complete TensorList API; layout-only dense/batch binding."""
from pathlib import Path
import ctypes as C
P=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_dense')
def fp8_moe(case):
    import torch
    from benchmarks.ascend_math_more import _MathCall
    from benchmarks.ascend_baselines import Candidate,source_hashes
    from benchmarks.api import Implementation
    vendor=P/'gemm/private_provider'
    b=_MathCall('aclnnGroupedMatmulV5',[C.c_void_p]*12+[C.c_int64]*4+[C.c_void_p]*4,library='ops-transformer',host_library=vendor/'op_api/lib/libcust_opapi.so')
    b.codes[torch.int8]=2
    base=b.base;base.aclCreateTensorList.argtypes=[C.POINTER(C.c_void_p),C.c_uint64];base.aclCreateTensorList.restype=C.c_void_p;base.aclDestroyTensorList.argtypes=[C.c_void_p];base.aclDestroyTensorList.restype=C.c_int
    lists=[]
    def tensorlist(ts,record):
        ptrs=[b.tensor(t,record) for t in ts];arr=(C.c_void_p*len(ptrs))(*ptrs);record['keep'].append(arr);out=base.aclCreateTensorList(arr,len(ptrs))
        if not out:raise RuntimeError('aclCreateTensorList failed')
        for ptr in ptrs:record['tensors'].remove(ptr)
        lists.append(out);return out
    def run(a,sa,w,sb,metadata):
        counts=[int(v) for v in metadata.cpu().tolist()];e,max_m,k=a.shape;n=w.shape[1]
        lhs=torch.cat([a[i,:c] for i,c in enumerate(counts)],0).contiguous();rhs=w.transpose(1,2).contiguous()
        scale_a=torch.cat([sa[i,:c] for i,c in enumerate(counts)],0).contiguous();scale_b=sb.transpose(1,2).contiguous()
        groups=metadata.to(torch.int64).cumsum(0).contiguous();out=torch.empty((sum(counts),n),dtype=torch.bfloat16,device=a.device)
        record=b.record();args=[tensorlist([lhs],record),tensorlist([rhs],record),None,tensorlist([scale_b],record),None,None,None,tensorlist([scale_a],record),b.tensor(groups,record),None,None,None,3,0,0,0,None,tensorlist([out],record),None,None]
        b.call(args,record,a.device)
        result=torch.zeros((e,max_m,n),dtype=torch.bfloat16,device=a.device);offset=0
        for i,c in enumerate(counts):result[i,:c].copy_(out[offset:offset+c]);offset+=c
        return result
    def cleanup():
        torch.npu.synchronize()
        while lists:b.check(base.aclDestroyTensorList(lists.pop()),'aclDestroyTensorList')
        b.cleanup()
    files=[Path(__file__),vendor/'op_api/lib/libcust_opapi.so']+[f for f in vendor.rglob('*') if f.is_file() and f.suffix in ('.py','.cpp','.h') and 'backup' not in str(f)]
    return Candidate(Implementation(run),'ops-transformer::aclnnGroupedMatmulV5','One original no-group TensorList call for every batch; canonical dtypes, pure transpose/packing/output stack. No extra arithmetic.',source_hashes(*files),cleanup=cleanup)
