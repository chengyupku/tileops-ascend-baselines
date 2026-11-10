"""Original complete grouped_matmul_no_quant executable, unchanged example launch settings."""
from pathlib import Path

def factory(case):
    import torch,numpy,ctypes
    from benchmarks.ascend_baselines import Candidate,source_hashes
    from benchmarks.api import Implementation
    P=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_attention_misc')
    binary=P/'grouped_bridge.so';data=P/'grouped_data';data.mkdir(exist_ok=True)
    assert case.id=='ds-v3-decode-mlp-up-float16' and case.params.get('activation') is None
    def run(a,b,metadata):
        a.detach().cpu().numpy().tofile(data/'input_a.bin')
        # Stream independent experts to avoid a second full host weight allocation.
        with (data/'input_b.bin').open('wb') as f:
            for expert in b:expert.detach().T.contiguous().cpu().numpy().tofile(f)
        metadata.long().cpu().numpy().tofile(data/'group_list.bin')
        # baseM/baseN/baseK/coreNum are exactly the original CSV example (16,64,64,2).
        args=[str(binary),str(b.shape[0]),'0','0','1','1','1','nd','float16',str(a.shape[0]),str(b.shape[1]),str(a.shape[1]),'0','1','16','64','64','2',str(data),str(data/'out.bin')]
        lib=ctypes.CDLL(str(binary));fn=lib.integration700_grouped_host;fn.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_char_p)];fn.restype=ctypes.c_int
        argv=(ctypes.c_char_p*len(args))(*(x.encode() for x in args));rc=fn(len(args),argv)
        if rc:raise RuntimeError('Original grouped host returned '+str(rc))
        result=torch.from_numpy(numpy.fromfile(data/'out.bin',dtype=numpy.float16).reshape(a.shape[0],b.shape[1])).to(a.device)
        torch.npu.synchronize()
        return result
    return Candidate(Implementation(run),'ops-tensor::grouped_matmul_no_quant original executable','Only ND weight transpose, physical prefix int64 metadata, binary input/output transport; unmodified original host/kernel with existing CSV launch settings.',source_hashes(Path(__file__),binary),timing_mode='host_completed')
