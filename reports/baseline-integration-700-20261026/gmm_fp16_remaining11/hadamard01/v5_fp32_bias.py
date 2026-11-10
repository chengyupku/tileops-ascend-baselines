"""Original V5 no-group complete TensorList API; layout-only dense/batch binding."""
from pathlib import Path
import ctypes as C
P=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_dense')
def dense_fp32(case):
    import torch
    from benchmarks.ascend_math_more import _MathCall
    from benchmarks.ascend_baselines import Candidate,source_hashes
    from benchmarks.api import Implementation
    vendor=P/'gemm/private_provider'
    b=_MathCall('aclnnGroupedMatmulV5',[C.c_void_p]*12+[C.c_int64]*4+[C.c_void_p]*4,library='ops-transformer',host_library=vendor/'op_api/lib/libcust_opapi.so')
    base=b.base;base.aclCreateTensorList.argtypes=[C.POINTER(C.c_void_p),C.c_uint64];base.aclCreateTensorList.restype=C.c_void_p;base.aclDestroyTensorList.argtypes=[C.c_void_p];base.aclDestroyTensorList.restype=C.c_int
    lists=[]
    def tensorlist(ts,record):
        ptrs=[b.tensor(t,record) for t in ts];arr=(C.c_void_p*len(ptrs))(*ptrs);record['keep'].append(arr);out=base.aclCreateTensorList(arr,len(ptrs))
        if not out:raise RuntimeError('aclCreateTensorList failed')
        for ptr in ptrs:record['tensors'].remove(ptr)
        lists.append(out);return out
    def run(a,w,bias=None):
        original_dtype=a.dtype;a=a.float();w=w.float()
        if case._call.signature.name=='BmmFwdOp':lhs=[t.contiguous() for t in a];rhs=[t.contiguous() for t in w]
        else:lhs=[(a.T if case.params['trans_a'] else a).contiguous()];rhs=[(w.T if case.params['trans_b'] else w).contiguous()]
        ys=[torch.empty((x.shape[0],y.shape[1]),dtype=a.dtype,device=a.device) for x,y in zip(lhs,rhs,strict=True)]
        record=b.record();args=[tensorlist(lhs,record),tensorlist(rhs,record),None if bias is None else tensorlist([bias.float().contiguous()],record)]+[None]*9+[0,-1,0,0,None,tensorlist(ys,record),None,None]
        b.call(args,record,a.device)
        return (torch.stack(ys) if case._call.signature.name=='BmmFwdOp' else ys[0]).to(original_dtype)
    def cleanup():
        torch.npu.synchronize()
        while lists:b.check(base.aclDestroyTensorList(lists.pop()),'aclDestroyTensorList')
        b.cleanup()
    files=[Path(__file__),vendor/'op_api/lib/libcust_opapi.so']+[f for f in vendor.rglob('*') if f.is_file() and f.suffix in ('.py','.cpp','.h') and 'backup' not in str(f)]
    return Candidate(Implementation(run),'ops-transformer::aclnnGroupedMatmulV5','One original no-group TensorList call for every batch; canonical dtypes, pure transpose/packing/output stack. No extra arithmetic.',source_hashes(*files),cleanup=cleanup)
