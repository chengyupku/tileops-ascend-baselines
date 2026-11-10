"""Complete original RotaryPositionEmbedding with manifest frequency state."""
from pathlib import Path
import ctypes as C
P=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_dense')
def rope(case):
    import torch
    from workloads import rope as tables
    try:from .integration700_transformer_dense_graph import GraphCall
    except ImportError:from integration700_transformer_dense_graph import GraphCall
    from benchmarks.ascend_baselines import Candidate,source_hashes
    from benchmarks.api import Implementation
    vendor=P/'rope/classicpilot_attention_transformer'
    binding=GraphCall('ops-transformer')
    name=case._call.signature.name
    fn={'YaRNFwdOp':tables.yarn_frequency_tables,'LongRoPEFwdOp':tables.longrope_frequency_tables}[name]
    def run(x):
        kwargs={k:v for k,v in case._call.params.items() if k not in ('input_layout','rope_layout')}
        if case.workload.tensors.get('rescale_factors') is not None:kwargs['rescale_factors']=case.workload.tensors['rescale_factors'].cpu()
        cs=fn(x.shape[-1],x.shape[0],dtype=x.dtype,device='cpu',**kwargs)
        cs=tuple(t.to(x.device) for t in cs)
        inp=x.contiguous().reshape(1,x.shape[0],1,x.shape[-1])
        cos,sin=[torch.cat((t,t),-1).contiguous().reshape_as(inp) for t in cs]
        out=torch.empty_like(inp)
        binding.call('RotaryPositionEmbedding',[inp,cos,sin,None],[out],{'mode':0},names=['x','cos','sin','seq_len','y'])
        return out.reshape_as(x)
    files=[Path(__file__),vendor/'op_api/lib/libcust_opapi.so']+[f for f in vendor.rglob('*') if f.is_file() and f.suffix in ('.py','.cpp','.h')]
    return Candidate(Implementation(run),'ops-transformer::aclnnRotaryPositionEmbedding','Original complete mode0 rotation; manifest frequency state and pure 1d-to-BSND metadata binding; all outputs retained.',source_hashes(*files),cleanup=binding.cleanup)
