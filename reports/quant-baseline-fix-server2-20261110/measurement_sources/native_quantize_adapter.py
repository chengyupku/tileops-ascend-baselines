"""Exact eager scale calculation plus installed complete division-mode quantize API."""
import torch
from workloads.quantization import quantize as ref

def factory(op,params):
    if op!='INT8QuantPerChannelFwdOp':raise NotImplementedError(op)
    import torch_npu
    def run(x):
        xf=x.float()
        scale=ref._int8_scale(xf.abs().amax(dim=1))
        q=torch_npu.npu_quantize(xf,scale,None,torch.qint8,0,True)
        return q,scale
    return run
