"""Original V5 no-group complete TensorList API; layout-only dense/batch binding."""
from pathlib import Path
import ctypes as C
P=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_dense')
def quant(case):
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
    def run(a,w,sa,sb,bias=None):
        lhs=a.contiguous();rhs=w.T.contiguous();m=lhs.shape[0];n=rhs.shape[1]
        scale_a=sa.reshape(-1).expand(m).contiguous() if sa.numel()==1 else sa.reshape(-1).contiguous()
        scale_b=sb.reshape(-1).expand(n).contiguous() if sb.numel()==1 else sb.reshape(-1).contiguous()
        out=torch.empty((m,n),dtype=getattr(torch,case.params['out_dtype']),device=a.device)
        record=b.record();groups=torch.tensor([m],dtype=torch.int64,device=a.device)
        args=[tensorlist([lhs],record),tensorlist([rhs.unsqueeze(0)],record),None if bias is None else tensorlist([bias],record),tensorlist([scale_b.reshape(1,-1)],record),None,None,None,tensorlist([scale_a],record),b.tensor(groups,record),None,None,None,3,0,0,0,None,tensorlist([out],record),None,None]
        b.call(args,record,a.device);return out
    def cleanup():
        torch.npu.synchronize()
        while lists:b.check(base.aclDestroyTensorList(lists.pop()),'aclDestroyTensorList')
        b.cleanup()
    files=[Path(__file__),vendor/'op_api/lib/libcust_opapi.so']+[f for f in vendor.rglob('*') if f.is_file() and f.suffix in ('.py','.cpp','.h') and 'backup' not in str(f)]
    return Candidate(Implementation(run),'ops-transformer::aclnnGroupedMatmulV5','One original no-group TensorList call for every batch; canonical dtypes, pure transpose/packing/output stack. No extra arithmetic.',source_hashes(*files),cleanup=cleanup)
