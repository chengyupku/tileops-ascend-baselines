"""Complete installed dynamic-quant API; no original kernel or compiler changes."""
import torch

def factory(op, params):
    if op != 'INT8QuantPerChannelFwdOp':
        raise NotImplementedError('Only exact per-row quantization contract reviewed')
    import torch_npu
    def run(x):
        if x.ndim!=2 or x.dtype not in (torch.float16,torch.bfloat16):
            raise ValueError('Dynamic quant native documented input dtype/shape contract')
        q,scale=torch_npu.npu_dynamic_quant(x)
        # Fixed workload defines scale=1 for zero rows; preserve native codes.
        return q,torch.where(scale>0,scale,torch.ones_like(scale))
    return run
