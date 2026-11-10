"""Original BiasGateGelu; zero bias binding, original example block count 40."""
from pathlib import Path
P=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_dense')
_schema=None
_loaded=False

def gelu(case):
    import torch
    from benchmarks.ascend_baselines import Candidate,source_hashes
    from benchmarks.api import Implementation
    global _schema,_loaded
    binary=P/'liboriginal_bias_gate_gelu.so'
    if not _loaded:
        _schema=torch.library.Library('ascend_ops','FRAGMENT')
        _schema.define('bias_gate_gelu(int block_dim, Tensor input, Tensor bias, Tensor(a!) output, int rows, int columns) -> int')
        torch.ops.load_library(str(binary))
        _loaded=True
    def run(x):
        inp=x.reshape(-1,x.shape[-1]).contiguous()
        bias=torch.zeros(inp.shape[-1],dtype=x.dtype,device=x.device)
        out=torch.empty((inp.shape[0],inp.shape[1]//2),dtype=x.dtype,device=x.device)
        rc=torch.ops.ascend_ops.bias_gate_gelu(40,inp,bias,out,*inp.shape)
        if rc:raise RuntimeError('BiasGateGelu return code '+str(rc))
        return out.reshape(*x.shape[:-1],x.shape[-1]//2)
    return Candidate(Implementation(run),'ops-transformer::BiasGateGelu','Original complete BiasGateGelu with zero bias and unchanged example block_dim=40; complete canonical output.',source_hashes(Path(__file__),P/'BIAS_GATE_GELU.cpp',binary))
