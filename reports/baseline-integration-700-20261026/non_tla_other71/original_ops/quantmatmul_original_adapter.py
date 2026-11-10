"""Private complete QuantMatmulV4 API adapter; no graph decomposition."""
import ctypes as C
from pathlib import Path

def int8_gemm(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, LIBRARIES, source_hashes
    from benchmarks.ascend_math_more import _MathCall
    b = _MathCall('aclnnQuantMatmulV4', [C.c_void_p]*6+[C.c_bool]*2+[C.c_void_p], library='ops-nn')
    b.codes[torch.int8] = 2
    def run(a, weight, scale_a, scale_b, bias=None):
        out = torch.empty((a.shape[0], weight.shape[0]), device=a.device, dtype=getattr(torch,case.params['out_dtype']))
        r = b.record()
        args = [b.tensor(a.contiguous(),r), b.tensor(weight.contiguous(),r), b.tensor(scale_b.reshape(-1).contiguous(),r), None, b.tensor(scale_a.reshape(-1).contiguous(),r), None if bias is None else b.tensor(bias.contiguous(),r), False, True, b.tensor(out,r)]
        b.call(args,r,a.device)
        return out
    return Candidate(Implementation(run),'ops-nn::aclnnQuantMatmulV4','Original complete API; flatten independent FP32 scale views, native NT flags, unchanged INT8 inputs and BF16 output.', b.hashes(LIBRARIES/'ops-nn/matmul/quant_batch_matmul_v3') | source_hashes(Path(__file__)),cleanup=b.cleanup)
